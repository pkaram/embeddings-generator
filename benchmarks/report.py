"""Markdown and JSON writers for measurement results."""

import json
import time
from pathlib import Path
from typing import Optional, Sequence


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavored markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def write_named(stem: str, payload: dict, markdown: str) -> Path:
    """Overwrite a stable JSON and markdown pair. Used for crash checkpoints."""
    folder = Path(stem).parent
    folder.mkdir(parents=True, exist_ok=True)
    json_path = Path(stem + ".json")
    markdown_path = Path(stem + ".md")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    markdown_path.write_text(markdown + "\n")
    return markdown_path


def write_result(name: str, payload: dict, markdown: str, directory: str = "results") -> Path:
    """Write a JSON document and a markdown report. Returns the markdown path."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = folder / f"{name}-{stamp}.json"
    markdown_path = folder / f"{name}-{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    markdown_path.write_text(markdown + "\n")
    return markdown_path


def fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def fmt_mib(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024 * 1024):.1f}"


def fmt_percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _cell(value: object) -> str:
    if value is None:
        return "n/a"
    return str(value)
