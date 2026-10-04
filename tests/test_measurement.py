"""Tests for latency summaries, retrieval metrics, and cache helpers."""

import math

import pytest

from app.measurement import (
    evict_model_cache,
    mean_metric,
    model_cache_path,
    model_disk_size_bytes,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    rates,
    recall_at_k,
    saturation_concurrency,
    summarize_latencies,
)


def test_percentile_interpolates_linearly():
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([5], 99) == 5


def test_summarize_latencies_reports_tails():
    summary = summarize_latencies([1, 2, 3, 4])
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["p50"] == 2.5
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["p99"] == pytest.approx(3.97)


def test_rates_use_the_elapsed_window():
    assert rates(64, 2, 4) == {"texts_per_s": 16, "requests_per_s": 0.5}


def test_recall_and_mrr():
    ranked = ["b", "a", "c"]
    relevant = {"a", "c"}
    assert recall_at_k(ranked, relevant, 1) == 0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 3) == 1
    assert mrr_at_k(ranked, relevant, 1) == 0
    assert mrr_at_k(ranked, relevant, 3) == 0.5


def test_ndcg_penalizes_a_relevant_hit_below_rank_one():
    score = ndcg_at_k(["b", "a"], {"a": 1.0}, 2)
    assert score == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["a"], {"a": 1.0}, 2) == 1


def test_mean_metric_rejects_an_empty_evaluation():
    with pytest.raises(ValueError):
        mean_metric([])


def test_saturation_is_the_first_level_that_stops_growing():
    levels = [
        {"concurrency": 1, "texts_per_s": 10},
        {"concurrency": 2, "texts_per_s": 18},
        {"concurrency": 4, "texts_per_s": 19},
    ]
    assert saturation_concurrency(levels) == 4
    still_climbing = [
        {"concurrency": 1, "texts_per_s": 10},
        {"concurrency": 2, "texts_per_s": 20},
    ]
    assert saturation_concurrency(still_climbing) is None


def test_disk_size_counts_a_snapshotted_blob_once(tmp_path):
    root = model_cache_path(str(tmp_path), "org/name")
    blobs = root / "blobs"
    snapshot = root / "snapshots" / "hash"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / "weights"
    blob.write_bytes(b"0123456789")
    (snapshot / "weights").symlink_to(blob)
    (root / "config.json").write_bytes(b"abcde")

    assert model_disk_size_bytes(str(tmp_path), "org/name") == 15
    assert model_disk_size_bytes(str(tmp_path), "missing/model") is None


def test_model_size_prefers_safetensors_over_onnx_exports(tmp_path):
    root = tmp_path / "org_name"
    (root / "onnx").mkdir(parents=True)
    (root / "onnx" / "model.onnx").write_bytes(b"x" * 1000)
    (root / "model.safetensors").write_bytes(b"y" * 20)

    assert model_disk_size_bytes(str(tmp_path), "org/name") == 20


def test_evict_model_cache_removes_one_model(tmp_path):
    root = model_cache_path(str(tmp_path), "org/name")
    root.mkdir(parents=True)
    (root / "weights.bin").write_bytes(b"x")

    alternate = tmp_path / "org_name"
    alternate.mkdir()
    (alternate / "model.safetensors").write_bytes(b"y")

    assert evict_model_cache(str(tmp_path), "org/name") is True
    assert not root.exists()
    assert not alternate.exists()
    assert evict_model_cache(str(tmp_path), "org/name") is False
