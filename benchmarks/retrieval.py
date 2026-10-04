"""Semantic-search evaluation through the embeddings API.

The service stays an embedding server. This client embeds a corpus and the
queries, ranks with a dot product (vectors are normalized), and scores
Recall@k, MRR@k, and nDCG@k against published qrels.

Default dataset is SciFact from BEIR: a few thousand abstracts and a few
hundred test queries, small enough to finish on CPU.
"""

import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from app.measurement import mean_metric, mrr_at_k, ndcg_at_k, recall_at_k
from benchmarks.client import EmbeddingClient
from benchmarks.report import fmt_score, markdown_table

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DEFAULT_DATASET_DIR = Path("data/beir/scifact")


def ensure_scifact(root: Path = DEFAULT_DATASET_DIR) -> Path:
    """Download and extract SciFact if the three BEIR files are not already present."""
    root = root.resolve()
    if _dataset_ready(root):
        return root

    zip_path = root.parent / "scifact.zip"
    root.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        _log(f"Downloading SciFact from {SCIFACT_URL}")
        request = urllib.request.Request(SCIFACT_URL, headers={"User-Agent": "embeddings-generator"})
        with urllib.request.urlopen(request, timeout=120) as response, zip_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    with zipfile.ZipFile(zip_path) as archive:
        _safe_extract(archive, root.parent)
    if not _dataset_ready(root):
        raise FileNotFoundError(
            f"Expected {root / 'corpus.jsonl'}, {root / 'queries.jsonl'}, and {root / 'qrels' / 'test.tsv'} "
            "after extracting SciFact."
        )
    return root


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(path: Path) -> dict:
    """Map query id to {doc id: relevance}. Non-positive grades are dropped."""
    qrels = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] == "query-id":
                continue
            score = float(parts[2])
            if score <= 0:
                continue
            qrels.setdefault(parts[0], {})[parts[1]] = score
    return qrels


def document_text(row: Mapping) -> str:
    """Title plus abstract, which is the text SciFact embeddings should see."""
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def select_corpus(rows: Sequence[Mapping], qrels: Mapping, max_docs: Optional[int]) -> list:
    """Full corpus, or a subset that keeps judged documents first."""
    if max_docs is None or max_docs >= len(rows):
        return list(rows)
    relevant = {doc_id for grades in qrels.values() for doc_id in grades}
    judged = []
    background = []
    for row in rows:
        if str(row["_id"]) in relevant:
            judged.append(row)
        else:
            background.append(row)
    if len(judged) >= max_docs:
        return judged[:max_docs]
    return judged + background[: max_docs - len(judged)]


def select_queries(queries: Sequence[Mapping], qrels: Mapping, max_queries: Optional[int]) -> list:
    """Test queries that have qrels, optionally truncated."""
    chosen = [row for row in queries if str(row["_id"]) in qrels]
    if max_queries is not None:
        chosen = chosen[:max_queries]
    return chosen


def embed_texts(client: EmbeddingClient, pairs: Sequence[tuple], model_name: str, batch_size: int) -> dict:
    """Embed ``(id, text)`` pairs. Returns id to vector."""
    vectors = {}
    batch_size = max(1, min(batch_size, 100))
    total = len(pairs)
    for start in range(0, total, batch_size):
        chunk = list(pairs[start:start + batch_size])
        body = client.embed([text for _, text in chunk], model_name, batch_size=batch_size)
        for (doc_id, _), vector in zip(chunk, body["embeddings"]):
            vectors[doc_id] = vector
        done = min(start + batch_size, total)
        _log(f"Embedded {done}/{total} with {model_name}")
    return vectors


def rank_queries(query_vectors: Mapping, corpus_ids: Sequence[str], corpus_matrix: np.ndarray, k: int) -> dict:
    """Dot-product top-k. Callers pass normalized vectors, so this is cosine."""
    rankings = {}
    for query_id, vector in query_vectors.items():
        scores = corpus_matrix @ np.asarray(vector, dtype=np.float32)
        rankings[query_id] = top_k_ids(scores, corpus_ids, k)
    return rankings


def top_k_ids(scores: np.ndarray, ids: Sequence[str], k: int) -> list:
    count = int(scores.shape[0])
    if count == 0:
        return []
    take = min(k, count)
    if take == count:
        order = np.argsort(scores)[::-1]
    else:
        partial = np.argpartition(scores, -take)[-take:]
        order = partial[np.argsort(scores[partial])[::-1]]
    return [ids[int(index)] for index in order]


def summarize_retrieval(rankings: Mapping, qrels: Mapping, k: int, corpus_ids: Sequence[str]) -> dict:
    """Mean Recall@k, MRR@k, and nDCG@k. Queries with no judged hits in the corpus are skipped."""
    available = set(corpus_ids)
    recalls = []
    reciprocal_ranks = []
    gains = []
    skipped = 0
    for query_id, ranked in rankings.items():
        relevance = {
            doc_id: score
            for doc_id, score in qrels.get(query_id, {}).items()
            if doc_id in available and score > 0
        }
        if not relevance:
            skipped += 1
            continue
        recalls.append(recall_at_k(ranked, relevance.keys(), k))
        reciprocal_ranks.append(mrr_at_k(ranked, relevance.keys(), k))
        gains.append(ndcg_at_k(ranked, relevance, k))
    if not recalls:
        raise ValueError("No queries had relevant documents inside the embedded corpus")
    return {
        "k": k,
        "queries_scored": len(recalls),
        "queries_skipped": skipped,
        "recall_at_k": mean_metric(recalls),
        "mrr_at_k": mean_metric(reciprocal_ranks),
        "ndcg_at_k": mean_metric(gains),
    }


def evaluate_model(
    client: EmbeddingClient,
    model_name: str,
    dataset_dir: Path,
    batch_size: int = 32,
    k: int = 10,
    max_docs: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> dict:
    """Embed SciFact through the API and score retrieval at ``k``."""
    root = ensure_scifact(dataset_dir)
    qrels = load_qrels(root / "qrels" / "test.tsv")
    queries = select_queries(load_jsonl(root / "queries.jsonl"), qrels, max_queries)
    selected_qrels = {str(row["_id"]): qrels[str(row["_id"])] for row in queries}
    corpus_rows = select_corpus(load_jsonl(root / "corpus.jsonl"), selected_qrels, max_docs)
    if not corpus_rows or not queries:
        raise ValueError("SciFact subset is empty")

    _log(f"Retrieval eval for {model_name}: {len(corpus_rows)} documents, {len(queries)} queries")
    corpus_vectors = embed_texts(
        client,
        [(str(row["_id"]), document_text(row)) for row in corpus_rows],
        model_name,
        batch_size,
    )
    query_vectors = embed_texts(
        client,
        [(str(row["_id"]), str(row["text"])) for row in queries],
        model_name,
        batch_size,
    )
    corpus_ids = list(corpus_vectors)
    corpus_matrix = _normalized_matrix([corpus_vectors[doc_id] for doc_id in corpus_ids])
    query_matrix = {
        query_id: _normalized_vector(vector) for query_id, vector in query_vectors.items()
    }
    rankings = rank_queries(query_matrix, corpus_ids, corpus_matrix, k)
    summary = summarize_retrieval(rankings, qrels, k, corpus_ids)
    summary.update({
        "dataset": "beir/scifact",
        "model_name": model_name,
        "corpus_size": len(corpus_ids),
        "query_count": len(queries),
        "subset": max_docs is not None or max_queries is not None,
    })
    return summary


def render_retrieval_markdown(rows: Sequence[dict]) -> str:
    """Markdown table of retrieval scores. One row per model."""
    k = rows[0]["k"] if rows else 10
    lines = [
        "# Retrieval evaluation",
        "",
        "Dataset: BEIR SciFact. Ranking is a dot product over normalized embeddings from `POST /embeddings`.",
        "",
        markdown_table(
            [
                "Model",
                "Corpus",
                "Queries scored",
                f"Recall@{k}",
                f"MRR@{k}",
                f"nDCG@{k}",
                "Subset",
            ],
            [
                [
                    row["model_name"],
                    row["corpus_size"],
                    row["queries_scored"],
                    fmt_score(row["recall_at_k"]),
                    fmt_score(row["mrr_at_k"]),
                    fmt_score(row["ndcg_at_k"]),
                    "yes" if row.get("subset") else "no",
                ]
                for row in rows
            ],
        ),
    ]
    return "\n".join(lines)


def _normalized_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalized_vector(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm


def _dataset_ready(root: Path) -> bool:
    return (
        (root / "corpus.jsonl").is_file()
        and (root / "queries.jsonl").is_file()
        and (root / "qrels" / "test.tsv").is_file()
    )


def _safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive.infolist():
        target = (dest / member.filename).resolve()
        if target != dest and not target.is_relative_to(dest):
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")
    archive.extractall(dest)


def _log(message: str) -> None:
    print(message, file=sys.stderr)
