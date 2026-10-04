"""Command line for the benchmark, retrieval evaluation, load test, and model comparison.

Run these from the repository root against an API that is already up:

    python -m benchmarks benchmark --base-url http://localhost:8000
    python -m benchmarks retrieval --base-url http://localhost:8000
    python -m benchmarks load --base-url http://localhost:8000
    python -m benchmarks compare --base-url http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

import httpx

from app.measurement import evict_model_cache
from benchmarks import DEFAULT_MODELS
from benchmarks.client import DEFAULT_EMBED_TIMEOUT_S, EmbeddingClient
from benchmarks.harness import (
    DEFAULT_BATCH_SIZES,
    ResourceMonitor,
    benchmark_model,
    render_benchmark_markdown,
)
from benchmarks.loadtest import DEFAULT_LEVELS, render_load_markdown, run_sweep
from benchmarks.report import write_named, write_result
from benchmarks.retrieval import DEFAULT_DATASET_DIR, evaluate_model, render_retrieval_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default="http://localhost:8000")
    common.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_EMBED_TIMEOUT_S,
        help="Seconds to wait for one embeddings response, including a model download.",
    )

    benchmark = subparsers.add_parser(
        "benchmark", parents=[common], help="Cold/warm latency and batch sweep"
    )
    _add_model_args(benchmark, multiple=True)
    _add_benchmark_args(benchmark)

    retrieval = subparsers.add_parser(
        "retrieval", parents=[common], help="SciFact Recall@k, MRR, and nDCG"
    )
    _add_model_args(retrieval, multiple=True)
    _add_retrieval_args(retrieval)

    load = subparsers.add_parser("load", parents=[common], help="Warm-path concurrency sweep")
    _add_model_args(load, multiple=False)
    load.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    load.add_argument("--duration", type=float, default=15.0, help="Seconds at each concurrency")
    load.add_argument("--batch-size", type=int, default=8)

    compare = subparsers.add_parser(
        "compare",
        parents=[common],
        help="Latency, resources, and retrieval for the default models",
    )
    _add_model_args(compare, multiple=True, default_all=True)
    _add_benchmark_args(compare)
    _add_retrieval_args(compare)
    compare.add_argument("--skip-retrieval", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "retrieval":
            return _retrieval(args)
        if args.command == "load":
            return _load(args)
        if args.command == "compare":
            return _compare(args)
    except httpx.ConnectError:
        print(f"Could not connect to {args.base_url}. Start the service, then retry.", file=sys.stderr)
        return 1
    except httpx.TimeoutException:
        print(
            "The embeddings request timed out while the API was still working. "
            f"A first load of a large model can exceed {args.timeout:.0f}s. "
            "Re-run with a larger --timeout. Completed models are in results/benchmark-partial.md.",
            file=sys.stderr,
        )
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"API returned {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    return 1


def _benchmark(args) -> int:
    results = _run_benchmarks(args)
    markdown = render_benchmark_markdown(results, args.base_url)
    path = write_result("benchmark", {"base_url": args.base_url, "models": results}, markdown)
    print(markdown)
    print(f"\nWrote {path}", file=sys.stderr)
    return 0


def _retrieval(args) -> int:
    client = EmbeddingClient(args.base_url, timeout_s=args.timeout)
    try:
        client.health()
        rows = [_evaluate(client, args, model) for model in _models(args)]
    finally:
        client.close()
    markdown = render_retrieval_markdown(rows)
    path = write_result("retrieval", {"base_url": args.base_url, "models": rows}, markdown)
    print(markdown)
    print(f"\nWrote {path}", file=sys.stderr)
    return 0


def _load(args) -> int:
    if args.duration <= 0:
        print("--duration must be positive", file=sys.stderr)
        return 1
    result = run_sweep(
        args.base_url,
        _models(args)[0],
        args.levels,
        args.duration,
        args.batch_size,
        timeout_s=args.timeout,
    )
    markdown = render_load_markdown(result)
    path = write_result("load", result, markdown)
    print(markdown)
    print(f"\nWrote {path}", file=sys.stderr)
    return 0


def _compare(args) -> int:
    results = _run_benchmarks(args)
    retrieval_rows = []
    if not args.skip_retrieval:
        client = EmbeddingClient(args.base_url, timeout_s=args.timeout)
        try:
            retrieval_rows = [_evaluate(client, args, model) for model in _models(args)]
        finally:
            client.close()
    benchmark_md = render_benchmark_markdown(results, args.base_url)
    parts = [benchmark_md]
    if retrieval_rows:
        parts.append("")
        parts.append(render_retrieval_markdown(retrieval_rows))
    markdown = "\n".join(parts)
    path = write_result(
        "comparison",
        {"base_url": args.base_url, "models": results, "retrieval": retrieval_rows},
        markdown,
    )
    print(markdown)
    print(f"\nWrote {path}", file=sys.stderr)
    return 0


def _run_benchmarks(args) -> list:
    if args.requests < 1:
        raise ValueError("--requests must be at least 1")
    client = EmbeddingClient(args.base_url, timeout_s=args.timeout)
    results = []
    try:
        client.health()
        with ResourceMonitor(args.base_url) as monitor:
            for model in _models(args):
                if args.measure_cache_miss:
                    if not args.cache_dir:
                        raise ValueError("--cache-dir is required with --measure-cache-miss")
                    removed = evict_model_cache(args.cache_dir, model)
                    print(f"Evicted cache for {model}: {removed}", file=sys.stderr)
                results.append(
                    benchmark_model(
                        client,
                        model,
                        args.batch_sizes,
                        args.requests,
                        args.cache_dir,
                        monitor,
                    )
                )
                partial = render_benchmark_markdown(results, args.base_url)
                path = write_named(
                    "results/benchmark-partial",
                    {"base_url": args.base_url, "models": results},
                    partial,
                )
                print(f"Saved {path}", file=sys.stderr)
    finally:
        client.close()
    return results


def _evaluate(client, args, model: str) -> dict:
    return evaluate_model(
        client,
        model,
        dataset_dir=args.dataset_dir,
        batch_size=args.retrieval_batch_size,
        k=args.k,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
    )


def _models(args) -> list:
    if getattr(args, "models", None):
        return list(args.models)
    return [args.model]


def _add_model_args(parser, multiple: bool, default_all: bool = False) -> None:
    if multiple and default_all:
        parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    elif multiple:
        parser.add_argument("--models", nargs="+", default=[DEFAULT_MODELS[0]])
    else:
        parser.add_argument("--model", default=DEFAULT_MODELS[0])


def _add_benchmark_args(parser) -> None:
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--requests", type=int, default=30, help="Steady-state requests per batch size")
    parser.add_argument(
        "--cache-dir",
        default="models",
        help="Host path of the model cache volume. Used to tell a disk cache miss from a cold load.",
    )
    parser.add_argument(
        "--measure-cache-miss",
        action="store_true",
        help="Delete the cached model first so the next load includes the download.",
    )


def _add_retrieval_args(parser) -> None:
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-docs", type=int, default=None, help="Subset the corpus. Default is the full SciFact corpus.")
    parser.add_argument("--max-queries", type=int, default=None, help="Subset the test queries. Default is all judged queries.")


if __name__ == "__main__":
    raise SystemExit(main())
