"""Chunker tests."""
from __future__ import annotations

import pytest

from medqa_rag.core.types import Chunk
from medqa_rag.data.chunking.recursive import RecursiveChunker


@pytest.mark.unit
def test_recursive_chunker_splits_long_text():
    long = "Sentence. " * 200
    doc = Chunk(id="x", text=long, source="testbook")
    chunks = RecursiveChunker(chunk_size=120, chunk_overlap=20).split([doc])
    assert len(chunks) > 1
    assert all(c.metadata["parent_id"] == "x" for c in chunks)
    assert all(c.id.startswith("x::chunk_") for c in chunks)


@pytest.mark.unit
def test_recursive_chunker_skips_empty():
    doc = Chunk(id="empty", text="", source="testbook")
    chunks = RecursiveChunker().split([doc])
    assert chunks == []
