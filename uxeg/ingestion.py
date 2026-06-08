"""Read source files from a folder and turn them into Document records.

Supported formats: .txt, .md, .csv, and optionally .pdf (via pypdf).

For CSV files we auto-detect likely free-text columns and combine them, while
keeping the remaining columns as per-row metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .schemas import Document
from .utils import now_iso, warn

# Columns that commonly hold free-text in UX datasets.
LIKELY_TEXT_COLUMNS = [
    "text",
    "comment",
    "comments",
    "response",
    "responses",
    "transcript",
    "note",
    "notes",
    "ticket",
    "review",
    "reviews",
    "feedback",
    "answer",
    "message",
]

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf"}


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pypdf is required to read PDF files. Install it with 'pip install pypdf'.") from exc

    reader = PdfReader(str(path))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts)


def _detect_text_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of columns that look like free-text columns."""

    matches = []
    lowered = {c.lower().strip(): c for c in df.columns}
    for candidate in LIKELY_TEXT_COLUMNS:
        if candidate in lowered:
            matches.append(lowered[candidate])
    return matches


def _read_csv(path: Path) -> List[Document]:
    """Read a CSV into one Document per row, combining detected text columns.

    Metadata columns (everything that is not a text column) are preserved in
    ``metadata_json`` so nothing is lost.
    """

    df = pd.read_csv(path)
    if df.empty:
        warn(f"CSV file '{path.name}' has no rows; skipping.")
        return []

    text_cols = _detect_text_columns(df)
    if not text_cols:
        # Fall back to the single column with the longest average string length.
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if not str_cols:
            warn(f"CSV file '{path.name}' has no obvious text columns; skipping.")
            return []
        avg_len = {c: df[c].astype(str).str.len().mean() for c in str_cols}
        text_cols = [max(avg_len, key=avg_len.get)]

    meta_cols = [c for c in df.columns if c not in text_cols]
    documents: List[Document] = []

    for idx, row in df.iterrows():
        text_parts = []
        for col in text_cols:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                text_parts.append(str(value).strip())
        combined = "\n\n".join(text_parts)
        if not combined.strip():
            continue

        metadata = {col: (None if pd.isna(row[col]) else row[col]) for col in meta_cols}
        metadata["row_index"] = int(idx)
        metadata["text_columns"] = text_cols

        documents.append(
            Document(
                file_name=f"{path.name}#row{idx}",
                file_path=str(path),
                file_type="csv",
                raw_text=combined,
                created_at=now_iso(),
                metadata_json=json.dumps(metadata, default=str),
            )
        )

    return documents


def read_file(path: Path) -> List[Document]:
    """Read a single file into one or more Document records."""

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        warn(f"Unsupported file type '{ext}' for '{path.name}'; skipping.")
        return []

    if ext == ".csv":
        return _read_csv(path)

    if ext == ".pdf":
        text = _read_pdf(path)
        file_type = "pdf"
    else:  # .txt or .md
        text = _read_txt(path)
        file_type = ext.lstrip(".")

    if not text or not text.strip():
        warn(f"File '{path.name}' is empty; skipping.")
        return []

    return [
        Document(
            file_name=path.name,
            file_path=str(path),
            file_type=file_type,
            raw_text=text,
            created_at=now_iso(),
            metadata_json="{}",
        )
    ]


def ingest_folder(folder: str | Path, recursive: bool = True) -> List[Document]:
    """Read every supported file in ``folder`` and return Document records."""

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    pattern = "**/*" if recursive else "*"
    documents: List[Document] = []
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(read_file(path))

    return documents
