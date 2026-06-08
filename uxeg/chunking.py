"""Split long documents into overlapping word-based chunks.

Chunks are the unit of work for extraction and embeddings. Overlap keeps
context from being cut awkwardly at chunk boundaries.
"""

from __future__ import annotations

from typing import List

from .schemas import Chunk


def chunk_text(
    text: str,
    document_id: str,
    file_name: str,
    chunk_size_words: int = 900,
    chunk_overlap_words: int = 120,
) -> List[Chunk]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size_words``.

    Returns a list of :class:`~uxeg.schemas.Chunk`. An empty or whitespace-only
    document yields an empty list.
    """

    if not text or not text.strip():
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words must not be negative")
    if chunk_overlap_words >= chunk_size_words:
        # Guard against an infinite loop; overlap must be smaller than the size.
        chunk_overlap_words = chunk_size_words // 4

    words = text.split()
    total = len(words)
    chunks: List[Chunk] = []

    # Step forward by (size - overlap) each time.
    step = chunk_size_words - chunk_overlap_words
    start = 0
    while start < total:
        end = min(start + chunk_size_words, total)
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append(
            Chunk(
                document_id=document_id,
                file_name=file_name,
                chunk_text=chunk_str,
                start_word=start,
                end_word=end,
            )
        )
        if end >= total:
            break
        start += step

    return chunks
