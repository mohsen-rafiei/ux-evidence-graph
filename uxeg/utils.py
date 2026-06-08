"""Small shared helpers used across the project.

Kept intentionally tiny and dependency-light so other modules can import from
here without circular imports.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

# A single shared console so all modules print with consistent styling.
console = Console()


def info(message: str) -> None:
    console.print(f"[cyan]ℹ[/cyan] {message}")


def success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")


def error(message: str) -> None:
    console.print(f"[red]✗[/red] {message}")


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_label(label: str) -> str:
    """Produce a canonical form of a label used for de-duplication.

    Lowercases, strips punctuation/extra whitespace. This is deliberately
    conservative; deeper merging is handled in ``normalization.py``.
    """

    text = label.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_json_block(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object from a model response.

    Local models sometimes wrap JSON in markdown fences or add prose. This
    pulls out the most likely JSON object so it can be parsed.
    """

    if not text:
        return None

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    # Otherwise grab from the first '{' to the last '}'.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def safe_json_loads(text: str) -> Optional[Any]:
    """Parse JSON, returning None instead of raising on failure."""

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
