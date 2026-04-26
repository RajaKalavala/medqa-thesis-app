"""Unit tests for the Naive RAG pipeline (mocked LLM + retriever)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from medqa_rag.core.types import Chunk, Question, RetrievedDoc
from medqa_rag.llm.groq_client import LLMResponse
from medqa_rag.rags.naive_rag.pipeline import NaiveRAGPipeline


@pytest.fixture
def question() -> Question:
    return Question(
        id="q1",
        stem="A 45-year-old male presents with chest pain.",
        options={"A": "MI", "B": "PE", "C": "Aortic dissection", "D": "Pneumonia"},
        correct_index=1,
        subject="cardiology",
    )


@pytest.fixture
def mock_faiss():
    m = MagicMock()
    m.retrieve.return_value = [
        RetrievedDoc(
            chunk=Chunk(id="c1", text="PE causes chest pain.", source="harrison"),
            score=0.95,
            rank=0,
            retriever="dense",
        )
    ]
    return m


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.complete = AsyncMock(
        return_value=LLMResponse(
            text="Reasoning: based on evidence.\nAnswer: B",
            model="llama-3.3-70b-versatile",
            usage={"total_tokens": 42},
        )
    )
    return m


@pytest.mark.unit
@pytest.mark.asyncio
async def test_naive_pipeline_returns_letter(question, mock_faiss, mock_llm):
    pipe = NaiveRAGPipeline(llm=mock_llm, embedder=MagicMock(), faiss=mock_faiss)
    out = await pipe.answer(question)
    assert out.predicted_letter == "B"
    assert out.predicted_index == 1
    assert len(out.retrieved_docs) == 1
    assert out.token_usage["total_tokens"] == 42
    mock_faiss.retrieve.assert_called_once()
    mock_llm.complete.assert_awaited_once()
