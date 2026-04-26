"""Self-RAG unit tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from medqa_rag.core.types import Question
from medqa_rag.llm.groq_client import LLMResponse
from medqa_rag.rags.self_rag.config import SelfRAGConfig
from medqa_rag.rags.self_rag.pipeline import SelfRAGPipeline


@pytest.fixture
def question() -> Question:
    return Question(
        id="q1",
        stem="What is the first-line treatment for hypertension?",
        options={"A": "ACE inhibitor", "B": "Aspirin", "C": "Statin", "D": "Beta-blocker"},
        correct_index=0,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_high_confidence_skips_retrieval(question):
    """When confidence ≥ threshold, FAISS must not be queried."""
    faiss = MagicMock()
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(text="Confidence: 0.92", model="judge"),
            LLMResponse(text="Reasoning: known.\nAnswer: A", model="main"),
        ]
    )
    pipe = SelfRAGPipeline(
        llm=llm,
        embedder=MagicMock(),
        faiss=faiss,
        config=SelfRAGConfig(confidence_threshold=0.65),
    )
    out = await pipe.answer(question)

    assert out.retrieval_used is False
    assert out.predicted_letter == "A"
    faiss.retrieve.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_low_confidence_triggers_retrieval(question):
    faiss = MagicMock()
    faiss.retrieve.return_value = []
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(text="Confidence: 0.30", model="judge"),
            LLMResponse(text="Reasoning: per evidence.\nAnswer: A", model="main"),
        ]
    )
    pipe = SelfRAGPipeline(llm=llm, embedder=MagicMock(), faiss=faiss)
    out = await pipe.answer(question)

    assert out.retrieval_used is True
    faiss.retrieve.assert_called_once()
