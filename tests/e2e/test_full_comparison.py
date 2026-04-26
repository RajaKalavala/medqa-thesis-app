"""End-to-end smoke against the real Groq API. Skipped unless GROQ_API_KEY set."""
from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.e2e


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping e2e",
)
@pytest.mark.asyncio
async def test_naive_rag_answers_a_real_question(sample_question, monkeypatch):
    """Run a real Naive RAG call; relies on FAISS index being pre-built."""
    from medqa_rag.core.types import Architecture
    from medqa_rag.rags.factory import build_rag

    pipe = build_rag(Architecture.NAIVE, load_indices=True)
    out = await pipe.answer(sample_question)
    assert out.predicted_letter in {"A", "B", "C", "D"} or out.predicted_letter is None
    assert out.latency_ms > 0
