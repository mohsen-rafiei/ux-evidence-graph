"""Tests for word-based chunking."""

from uxeg.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", "doc1", "f.txt") == []
    assert chunk_text("   \n  ", "doc1", "f.txt") == []


def test_short_text_single_chunk():
    text = "one two three four five"
    chunks = chunk_text(text, "doc1", "f.txt", chunk_size_words=900, chunk_overlap_words=120)
    assert len(chunks) == 1
    assert chunks[0].chunk_text == text
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 5
    assert chunks[0].document_id == "doc1"
    assert chunks[0].file_name == "f.txt"


def test_chunking_splits_with_overlap():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, "doc1", "f.txt", chunk_size_words=40, chunk_overlap_words=10)
    # step = 30, so starts at 0 (0-40), 30 (30-70), 60 (60-100 -> stop)
    assert len(chunks) == 3
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 40
    assert chunks[1].start_word == 30
    # Overlap means chunk 1 repeats the last 10 words of chunk 0.
    assert chunks[0].chunk_text.split()[-10:] == chunks[1].chunk_text.split()[:10]
    # Last chunk ends at total length.
    assert chunks[-1].end_word == 100


def test_overlap_larger_than_size_is_corrected():
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    # overlap >= size would loop forever; chunker should self-correct.
    chunks = chunk_text(text, "doc1", "f.txt", chunk_size_words=20, chunk_overlap_words=25)
    assert len(chunks) >= 1
    assert chunks[-1].end_word == 50
