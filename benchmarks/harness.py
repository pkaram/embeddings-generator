"""Latency, throughput, and resource benchmark against a running embeddings API.

Clocks, kept separate on purpose:

- cache_miss load_time: download plus load into RAM, only when the cache is absent
- cold_load load_time: read a cached model into RAM
- warmup_encode: processing_time of the request that loaded the model
- steady encode: processing_time after two discarded warmup requests
- client e2e: time until the HTTP client finishes reading the JSON body
"""

import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

from app.measurement import model_is_cached, rates, summarize_latencies
from benchmarks.client import EmbeddingClient
from benchmarks.report import fmt_mib, fmt_percent, fmt_rate, fmt_seconds, markdown_table

STEADY_TEXT_COUNT = 64
WARMUP_TEXT_COUNT = 8
DEFAULT_BATCH_SIZES = (1, 8, 16, 32, 64)

_SENTENCES = (
    "Semantic search ranks documents by the meaning of a query.",
    "CPU inference keeps the embedding service inexpensive to run.",
    "Batch size changes throughput more than it changes vector quality.",
    "Normalized embeddings turn cosine similarity into a dot product.",
)


def workload(count: int) -> list:
    """Fixed sentences so runs differ by batch size, not by input length."""
    return [_SENTENCES[index % len(_SENTENCES)] for index in range(count)]


class ResourceMonitor:
    """Poll GET /system on a background thread with its own HTTP client."""

    def __init__(self, base_url: str, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.client = EmbeddingClient(base_url, timeout_s=10.0)
        self.samples = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.client.close()

    def mark(self) -> int:
        with self._lock:
            return len(self.samples)

    def stats_since(self, mark: int) -> dict:
        with self._lock:
            chunk = list(self.samples[mark:])
        rss = [sample["rss_bytes"] for sample in chunk]
        # The first sample in a longer window still includes work from before
        # the window. A single sample is the only reading we have.
        cpu_samples = chunk if len(chunk) < 2 else chunk[1:]
        cpu = [sample["cpu_percent"] for sample in cpu_samples]
        return {
            "peak_rss_bytes": max(rss) if rss else None,
            "mean_cpu_percent": (sum(cpu) / len(cpu)) if cpu else None,
            "samples": len(chunk),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self.client.system()
                sample["t"] = time.perf_counter()
                with self._lock:
                    self.samples.append(sample)
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def benchmark_model(
    client: EmbeddingClient,
    model_name: str,
    batch_sizes: Sequence[int],
    requests: int,
    cache_dir: Optional[str],
    monitor: ResourceMonitor,
) -> dict:
    """Cold load, warmup encode, and a steady-state batch sweep for one model."""
    if requests < 1:
        raise ValueError("requests must be at least 1")

    _log(f"Unloading any resident model before {model_name}")
    client.unload()
    cache_root = Path(cache_dir) if cache_dir else None
    cache_visible = cache_root is not None and cache_root.is_dir()
    cache_existed = cache_visible and model_is_cached(cache_dir, model_name)
    phases = {}

    first = client.embed(workload(WARMUP_TEXT_COUNT), model_name, batch_size=WARMUP_TEXT_COUNT)
    after_first = client.system()
    if not cache_visible or cache_existed:
        phases["cold_load"] = _load_phase(first, after_first)
        phases["warmup_encode"] = _encode_phase(first)
        if not cache_visible:
            phases["cold_load"]["note"] = (
                "Host cache directory was not visible, so load_time includes a download if the model was not cached"
            )
    else:
        phases["cache_miss"] = _load_phase(first, after_first)
        phases["cache_miss"]["note"] = "load_time includes the download into the model cache"
        _log(f"Cache miss recorded for {model_name}; measuring cold load from disk next")
        client.unload()
        second = client.embed(workload(WARMUP_TEXT_COUNT), model_name, batch_size=WARMUP_TEXT_COUNT)
        after_second = client.system()
        phases["cold_load"] = _load_phase(second, after_second)
        phases["warmup_encode"] = _encode_phase(second)

    info = client.model_info()
    steady_texts = workload(STEADY_TEXT_COUNT)
    for _ in range(2):
        client.embed(steady_texts, model_name, batch_size=batch_sizes[0])

    sweep = []
    measured = set()
    for requested in batch_sizes:
        row = _steady_state(
            client, monitor, model_name, steady_texts, requested, requests, measured
        )
        if row is not None:
            sweep.append(row)
            measured.add(row["batch_size"])

    if not sweep:
        raise RuntimeError(f"No batch sizes were measured for {model_name}")

    return {
        "model_name": model_name,
        "dimensions": info.get("embedding_dimensions"),
        "max_sequence_length": info.get("max_sequence_length"),
        "model_size_bytes": info.get("model_size_bytes"),
        "phases": phases,
        "batch_sweep": sweep,
        "steady_text_count": STEADY_TEXT_COUNT,
        "requests_per_batch_size": requests,
    }


def render_benchmark_markdown(results: Sequence[dict], base_url: str) -> str:
    """Markdown report for one or more model benchmarks."""
    sections = [
        "# Embedding benchmark",
        "",
        f"Service: `{base_url}`",
        "",
        "Encode time is `processing_time` from the API. End-to-end time is measured by the client and includes JSON serialization. "
        "Each steady-state request sends 64 texts. p99 with a short sample is a coarse tail estimate.",
        "",
        "## Model summary",
        "",
        markdown_table(
            [
                "Model",
                "Dim",
                "Disk (MiB)",
                "Cold load (s)",
                "Warmup encode (s)",
                "Steady batch",
                "Encode p50 (s)",
                "Encode p95 (s)",
                "E2E p50 (s)",
                "E2E p95 (s)",
                "Texts/s",
                "Peak RSS (MiB)",
            ],
            [_summary_row(result) for result in results],
        ),
        "",
        "## Batch sweep",
        "",
    ]
    for result in results:
        sections.append(f"### {result['model_name']}")
        sections.append("")
        sections.append(
            markdown_table(
                [
                    "Batch",
                    "Encode p50 (s)",
                    "Encode p95 (s)",
                    "Encode p99 (s)",
                    "E2E p50 (s)",
                    "E2E p95 (s)",
                    "E2E p99 (s)",
                    "Texts/s",
                    "Req/s",
                    "Peak RSS (MiB)",
                    "CPU %",
                ],
                [_sweep_row(row) for row in result["batch_sweep"]],
            )
        )
        sections.append("")
        cache_miss = result["phases"].get("cache_miss")
        if cache_miss:
            sections.append(
                f"Cache-miss load time: {fmt_seconds(cache_miss['load_time_s'])} s. "
                "That figure includes the download."
            )
            sections.append("")
    return "\n".join(sections).rstrip()


def headline_batch(result: dict) -> dict:
    """Steady-state row used in the model summary. Prefer batch size 32."""
    for row in result["batch_sweep"]:
        if row["batch_size"] == 32:
            return row
    return result["batch_sweep"][0]


def _steady_state(client, monitor, model_name, texts, requested, requests, already_measured):
    mark = monitor.mark()
    started = time.perf_counter()
    probe = client.embed(texts, model_name, batch_size=requested)
    effective = int(probe["batch_size"])
    if effective != requested:
        _log(f"Server clamped batch size {requested} to {effective}")
    if effective in already_measured:
        _log(f"Skipping requested batch {requested}; effective batch {effective} is already measured")
        return None

    e2e = [probe["client_e2e_s"]]
    encode = [probe["processing_time"]]
    for _ in range(requests - 1):
        body = client.embed(texts, model_name, batch_size=requested)
        e2e.append(body["client_e2e_s"])
        encode.append(body["processing_time"])
    elapsed = time.perf_counter() - started
    throughput = rates(len(texts) * requests, requests, elapsed)
    resources = monitor.stats_since(mark)
    _log(
        f"{model_name} batch {effective}: "
        f"encode p50={encode and summarize_latencies(encode)['p50']:.4f}s "
        f"texts/s={throughput['texts_per_s']:.2f}"
    )
    return {
        "batch_size": effective,
        "requested_batch_size": requested,
        "request_texts": len(texts),
        "requests": requests,
        "encode": summarize_latencies(encode),
        "e2e": summarize_latencies(e2e),
        "texts_per_s": throughput["texts_per_s"],
        "requests_per_s": throughput["requests_per_s"],
        "peak_rss_bytes": resources["peak_rss_bytes"],
        "mean_cpu_percent": resources["mean_cpu_percent"],
    }


def _load_phase(body: dict, system: dict) -> dict:
    return {
        "load_time_s": body["load_time"],
        "client_e2e_s": body["client_e2e_s"],
        "rss_bytes": system.get("rss_bytes"),
    }


def _encode_phase(body: dict) -> dict:
    return {
        "encode_time_s": body["processing_time"],
        "client_e2e_s": body["client_e2e_s"],
        "texts": body["total_texts"],
    }


def _summary_row(result: dict) -> list:
    steady = headline_batch(result)
    cold = result["phases"].get("cold_load", {})
    warmup = result["phases"].get("warmup_encode", {})
    return [
        result["model_name"],
        result.get("dimensions"),
        fmt_mib(result.get("model_size_bytes")),
        fmt_seconds(cold.get("load_time_s")),
        fmt_seconds(warmup.get("encode_time_s")),
        steady["batch_size"],
        fmt_seconds(steady["encode"]["p50"]),
        fmt_seconds(steady["encode"]["p95"]),
        fmt_seconds(steady["e2e"]["p50"]),
        fmt_seconds(steady["e2e"]["p95"]),
        fmt_rate(steady["texts_per_s"]),
        fmt_mib(steady.get("peak_rss_bytes")),
    ]


def _sweep_row(row: dict) -> list:
    return [
        row["batch_size"],
        fmt_seconds(row["encode"]["p50"]),
        fmt_seconds(row["encode"]["p95"]),
        fmt_seconds(row["encode"]["p99"]),
        fmt_seconds(row["e2e"]["p50"]),
        fmt_seconds(row["e2e"]["p95"]),
        fmt_seconds(row["e2e"]["p99"]),
        fmt_rate(row["texts_per_s"]),
        fmt_rate(row["requests_per_s"]),
        fmt_mib(row.get("peak_rss_bytes")),
        fmt_percent(row.get("mean_cpu_percent")),
    ]


def _log(message: str) -> None:
    print(message, file=sys.stderr)
