#!/bin/bash
# Create the local virtualenv used to run the benchmark CLI.
# The directory .venv is gitignored. Re-run this script on a new machine.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
  else
    PYTHON=""
    for candidate in python3.13 python3.12 python3.11 python3.10; do
      if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
      fi
    done
    if [[ -z "$PYTHON" ]]; then
      echo "Install Python 3.10+ or uv, then re-run this script." >&2
      exit 1
    fi
    "$PYTHON" -m venv .venv
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements-benchmark.txt
else
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-benchmark.txt
fi

echo ""
echo "Benchmark CLI is ready. From the repository root:"
echo "  .venv/bin/python -m benchmarks benchmark --base-url http://localhost:8000"
