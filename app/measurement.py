"""Latency summaries, retrieval metrics, and model-cache helpers.

These functions are the shared definition of the numbers reported by the
benchmark harness, the retrieval evaluation, and the load test.
"""

import math
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile. ``p`` is in the closed range 0 to 100."""
    if not values:
        raise ValueError("values must not be empty")
    if p < 0 or p > 100:
        raise ValueError("p must be between 0 and 100")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_latencies(values: Sequence[float]) -> dict:
    """Return count, mean, and p50/p95/p99 for a latency sample."""
    if not values:
        raise ValueError("values must not be empty")
    total = sum(float(value) for value in values)
    return {
        "count": len(values),
        "mean": total / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def rates(total_texts: int, total_requests: int, elapsed_s: float) -> dict:
    """Texts per second and requests per second over a measured window."""
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive")
    return {
        "texts_per_s": total_texts / elapsed_s,
        "requests_per_s": total_requests / elapsed_s,
    }


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top ``k``."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Reciprocal rank of the first relevant hit inside the top ``k``."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` using gains of ``2^rel - 1``."""
    if k <= 0:
        raise ValueError("k must be positive")
    gains = [float(relevance.get(doc_id, 0.0)) for doc_id in ranked_ids[:k]]
    ideal = sorted(
        (float(value) for value in relevance.values() if value > 0),
        reverse=True,
    )[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        raise ValueError("relevance must contain a positive gain")
    return _dcg(gains) / ideal_dcg


def mean_metric(values: Sequence[float]) -> float:
    """Arithmetic mean. Empty input is a failed evaluation, not a zero score."""
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def saturation_concurrency(levels: Sequence[Mapping[str, float]]) -> Optional[int]:
    """Lowest concurrency where throughput stops growing by at least 10 percent.

    ``levels`` must be ordered by increasing concurrency and each item needs
    ``concurrency`` and ``texts_per_s``. Returns ``None`` when throughput is
    still climbing at the highest concurrency that was measured.
    """
    if not levels:
        raise ValueError("levels must not be empty")

    best = float(levels[0]["texts_per_s"])
    for row in levels[1:]:
        rate = float(row["texts_per_s"])
        if rate < best * 1.10:
            return int(row["concurrency"])
        best = max(best, rate)
    return None


def model_cache_dirname(model_name: str) -> str:
    """Hugging Face hub cache directory name for a repository id."""
    return "models--" + model_name.replace("/", "--")


def model_cache_path(cache_dir: str, model_name: str) -> Path:
    """Path of the hub cache directory for ``model_name`` under ``cache_dir``."""
    return Path(cache_dir) / model_cache_dirname(model_name)


def model_cache_candidates(cache_dir: str, model_name: str) -> list:
    """Directories where this stack may store one model.

    sentence-transformers 2.2 writes the repository id with ``/`` replaced by
    ``_``. The Hub client writes ``models--{repo}``.
    """
    root = Path(cache_dir)
    return [
        root / model_name.replace("/", "_"),
        root / model_cache_dirname(model_name),
    ]


def model_is_cached(cache_dir: str, model_name: str) -> bool:
    """Whether a local cache directory for ``model_name`` is present."""
    return any(path.is_dir() for path in model_cache_candidates(cache_dir, model_name))


def model_disk_size_bytes(cache_dir: str, model_name: str) -> Optional[int]:
    """Size of the inference weights, not ONNX or OpenVINO exports.

    Prefers ``model.safetensors``, then ``pytorch_model.bin``. If neither is
    present, sums the cache directory and counts each inode once so Hub
    snapshot symlinks are not double-counted.
    """
    for candidate in model_cache_candidates(cache_dir, model_name):
        weight = _preferred_weight(candidate)
        if weight is not None:
            return weight.stat().st_size
    for candidate in model_cache_candidates(cache_dir, model_name):
        if candidate.is_dir():
            return _unique_file_bytes(candidate)
    return None


def evict_model_cache(cache_dir: str, model_name: str) -> bool:
    """Delete one model's cache directories. Returns whether anything was removed."""
    removed = False
    for path in model_cache_candidates(cache_dir, model_name):
        if path.exists():
            shutil.rmtree(path)
            removed = True
    lock = Path(cache_dir) / ".locks" / model_cache_dirname(model_name)
    if lock.exists():
        if lock.is_dir():
            shutil.rmtree(lock)
        else:
            lock.unlink()
        removed = True
    return removed


def _preferred_weight(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    for name in ("model.safetensors", "pytorch_model.bin"):
        for path in root.rglob(name):
            if {"onnx", "openvino"}.intersection(path.parts):
                continue
            if path.is_file():
                return path
    return None


def _unique_file_bytes(root: Path) -> int:
    seen = set()
    total = 0
    for path in root.rglob("*"):
        if not path.is_symlink() and not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += stat.st_size
    return total


def _dcg(gains: Sequence[float]) -> float:
    return sum((2.0 ** gain - 1.0) / math.log2(index + 2) for index, gain in enumerate(gains))
