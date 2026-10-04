"""Closed-loop concurrency sweep against a warm embeddings API.

The API encodes on the event loop and the container starts one worker.
Latency climbs once concurrency exceeds what that single forward pass can
absorb. The saturation concurrency is the first level whose throughput is
not at least 10 percent above the best throughput seen so far.
"""

import asyncio
import sys
import time
from typing import Sequence

import httpx

from app.measurement import saturation_concurrency, summarize_latencies
from benchmarks.client import EmbeddingClient
from benchmarks.harness import workload
from benchmarks.report import fmt_percent, fmt_rate, fmt_seconds, markdown_table

DEFAULT_LEVELS = (1, 2, 4, 8, 16)
REQUEST_TEXTS = 8


async def run_level(
    base_url: str,
    model_name: str,
    concurrency: int,
    duration_s: float,
    batch_size: int,
) -> dict:
    """Hold ``concurrency`` requests in flight for ``duration_s`` seconds."""
    base_url = base_url.rstrip("/")
    texts = workload(REQUEST_TEXTS)
    successes = []
    encode_times = []
    errors = 0
    stop_at = time.perf_counter() + duration_s
    timeout = httpx.Timeout(120.0, connect=10.0)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as http:
        async def worker() -> None:
            nonlocal errors
            while time.perf_counter() < stop_at:
                started = time.perf_counter()
                try:
                    response = await http.post(
                        "/embeddings",
                        json={
                            "texts": texts,
                            "model_name": model_name,
                            "normalize": True,
                            "batch_size": batch_size,
                        },
                    )
                    elapsed = time.perf_counter() - started
                    if response.status_code != 200:
                        errors += 1
                        continue
                    body = response.json()
                    successes.append(elapsed)
                    encode_times.append(float(body["processing_time"]))
                except Exception:
                    errors += 1

        await asyncio.gather(*(worker() for _ in range(concurrency)))

    total = len(successes) + errors
    summary = summarize_latencies(successes) if successes else None
    mean_encode = (sum(encode_times) / len(encode_times)) if encode_times else None
    return {
        "concurrency": concurrency,
        "duration_s": duration_s,
        "requests": total,
        "errors": errors,
        "error_rate": (errors / total) if total else 1.0,
        "e2e": summary,
        "mean_encode_s": mean_encode,
        "texts_per_s": (len(successes) * REQUEST_TEXTS / duration_s) if duration_s else 0.0,
        "requests_per_s": (len(successes) / duration_s) if duration_s else 0.0,
    }


def run_sweep(
    base_url: str,
    model_name: str,
    levels: Sequence[int],
    duration_s: float,
    batch_size: int,
    timeout_s: float = 3600.0,
) -> dict:
    """Warm the model, then run each concurrency level."""
    client = EmbeddingClient(base_url, timeout_s=timeout_s)
    try:
        client.health()
        _log(f"Warming {model_name} before the concurrency sweep")
        client.embed(workload(REQUEST_TEXTS), model_name, batch_size=batch_size)
    finally:
        client.close()

    rows = []
    for concurrency in levels:
        _log(f"Load level concurrency={concurrency} duration={duration_s}s")
        row = asyncio.run(
            run_level(base_url, model_name, concurrency, duration_s, batch_size)
        )
        rows.append(row)
        e2e = row["e2e"]
        p95 = f"{e2e['p95']:.4f}s" if e2e else "n/a"
        _log(f"  p95={p95} texts/s={row['texts_per_s']:.2f} errors={row['errors']}")

    return {
        "model_name": model_name,
        "base_url": base_url,
        "batch_size": batch_size,
        "request_texts": REQUEST_TEXTS,
        "levels": rows,
        "saturation_concurrency": saturation_concurrency(rows),
        "note": (
            "Encode runs on the event loop in a single worker. "
            "Saturation is the first concurrency whose texts/s failed to beat the best lower level by 10 percent."
        ),
    }


def render_load_markdown(result: dict) -> str:
    """Markdown report for a concurrency sweep."""
    saturation = result["saturation_concurrency"]
    saturation_text = str(saturation) if saturation is not None else "not reached"
    lines = [
        "# Load test",
        "",
        f"Service: `{result['base_url']}`",
        f"Model: `{result['model_name']}`",
        "",
        result["note"],
        "",
        f"Saturation concurrency: **{saturation_text}**",
        "",
        markdown_table(
            [
                "Concurrency",
                "Requests",
                "Errors",
                "Error %",
                "E2E p50 (s)",
                "E2E p95 (s)",
                "E2E p99 (s)",
                "Mean encode (s)",
                "Texts/s",
                "Req/s",
            ],
            [_load_row(row) for row in result["levels"]],
        ),
        "",
        "Mean encode is the server `processing_time`. The gap between that and end-to-end p50 is queueing plus JSON.",
    ]
    return "\n".join(lines)


def _load_row(row: dict) -> list:
    e2e = row["e2e"] or {}
    return [
        row["concurrency"],
        row["requests"],
        row["errors"],
        fmt_percent((row["error_rate"] or 0) * 100),
        fmt_seconds(e2e.get("p50")),
        fmt_seconds(e2e.get("p95")),
        fmt_seconds(e2e.get("p99")),
        fmt_seconds(row.get("mean_encode_s")),
        fmt_rate(row["texts_per_s"]),
        fmt_rate(row["requests_per_s"]),
    ]


def _log(message: str) -> None:
    print(message, file=sys.stderr)
