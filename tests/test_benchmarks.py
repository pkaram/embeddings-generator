"""Tests for retrieval scoring and the benchmark report."""

import numpy as np

from benchmarks.client import DEFAULT_EMBED_TIMEOUT_S, EmbeddingClient
from benchmarks.harness import headline_batch, render_benchmark_markdown, workload
from benchmarks.loadtest import render_load_markdown
from benchmarks.retrieval import (
    document_text,
    ensure_scifact,
    load_qrels,
    select_corpus,
    select_queries,
    summarize_retrieval,
    top_k_ids,
)


def test_embed_timeout_allows_a_slow_model_download():
    client = EmbeddingClient("http://127.0.0.1:9")
    try:
        assert client.timeout_s == DEFAULT_EMBED_TIMEOUT_S
        assert client._embed_timeout.read == DEFAULT_EMBED_TIMEOUT_S
        assert DEFAULT_EMBED_TIMEOUT_S > 600
    finally:
        client.close()


def test_workload_is_stable_and_long_enough_for_the_batch_sweep():
    texts = workload(64)
    assert len(texts) == 64
    assert texts == workload(64)


def test_top_k_ids_returns_the_highest_scores_first():
    scores = np.array([0.1, 0.9, 0.3, 0.8], dtype=np.float32)
    assert top_k_ids(scores, ["a", "b", "c", "d"], 2) == ["b", "d"]


def test_summarize_retrieval_on_a_perfect_and_a_missed_query():
    rankings = {"q1": ["a", "b"], "q2": ["c", "a"]}
    qrels = {"q1": {"a": 1.0}, "q2": {"a": 1.0}}
    summary = summarize_retrieval(rankings, qrels, k=10, corpus_ids=["a", "b", "c"])
    assert summary["queries_scored"] == 2
    assert summary["recall_at_k"] == 1
    assert summary["mrr_at_k"] == (1.0 + 0.5) / 2


def test_qrels_and_subset_selection(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\nq2\td3\t1\n")
    qrels = load_qrels(path)
    assert qrels == {"q1": {"d1": 1.0}, "q2": {"d3": 1.0}}

    rows = [{"_id": "d1", "title": "T", "text": "body"}, {"_id": "x", "title": "", "text": "other"}]
    selected = select_corpus(rows, qrels, max_docs=1)
    assert [row["_id"] for row in selected] == ["d1"]
    assert document_text(rows[0]) == "T. body"
    assert document_text(rows[1]) == "other"

    queries = select_queries(
        [{"_id": "q1", "text": "one"}, {"_id": "q9", "text": "ignored"}, {"_id": "q2", "text": "two"}],
        qrels,
        max_queries=1,
    )
    assert [row["_id"] for row in queries] == ["q1"]


def test_ensure_scifact_does_not_download_when_files_exist(tmp_path):
    root = tmp_path / "scifact"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text('{"_id": "1"}\n')
    (root / "queries.jsonl").write_text('{"_id": "q"}\n')
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n")
    assert ensure_scifact(root) == root.resolve()


def test_benchmark_report_prefers_batch_32_and_names_the_clocks():
    result = {
        "model_name": "org/model",
        "dimensions": 384,
        "model_size_bytes": 80 * 1024 * 1024,
        "phases": {
            "cold_load": {"load_time_s": 1.5, "client_e2e_s": 1.7, "rss_bytes": 1000},
            "warmup_encode": {"encode_time_s": 0.2, "client_e2e_s": 0.3, "texts": 8},
        },
        "batch_sweep": [
            _sweep_row(1, p50=0.4),
            _sweep_row(32, p50=0.1),
        ],
    }
    assert headline_batch(result)["batch_size"] == 32
    markdown = render_benchmark_markdown([result], "http://localhost:8000")
    assert "Encode p50" in markdown
    assert "org/model" in markdown
    assert "0.1000" in markdown


def test_load_report_states_saturation():
    markdown = render_load_markdown({
        "base_url": "http://localhost:8000",
        "model_name": "org/model",
        "note": "Encode runs on the event loop in a single worker.",
        "saturation_concurrency": 4,
        "levels": [
            {
                "concurrency": 1,
                "requests": 10,
                "errors": 0,
                "error_rate": 0.0,
                "e2e": {"p50": 0.1, "p95": 0.2, "p99": 0.3},
                "mean_encode_s": 0.08,
                "texts_per_s": 80,
                "requests_per_s": 10,
            }
        ],
    })
    assert "Saturation concurrency: **4**" in markdown
    assert "Mean encode" in markdown


def _sweep_row(batch_size, p50):
    latency = {"count": 30, "mean": p50, "p50": p50, "p95": p50, "p99": p50}
    return {
        "batch_size": batch_size,
        "encode": latency,
        "e2e": latency,
        "texts_per_s": 100,
        "requests_per_s": 2,
        "peak_rss_bytes": 200 * 1024 * 1024,
        "mean_cpu_percent": 90,
    }
