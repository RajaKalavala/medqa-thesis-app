"""API response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from medqa_rag.core.types import Architecture


class RetrievedPassage(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float
    rank: int
    retriever: str


class QAResponse(BaseModel):
    question_id: str
    architecture: Architecture
    predicted_letter: str | None
    predicted_index: int | None
    correct: bool | None = Field(
        default=None, description="None if no gold answer was provided."
    )
    generated_answer: str
    retrieved: list[RetrievedPassage]
    latency_ms: float
    token_usage: dict[str, int] = {}
    extras: dict[str, Any] = {}


class EvaluateResponse(BaseModel):
    architecture: Architecture
    n: int
    accuracy: float
    ragas: dict[str, float]
    latency: dict[str, float]
    hallucination_rate: float


class ExplainResponse(BaseModel):
    question_id: str
    architecture: Architecture
    method: str
    passage_attributions: list[dict[str, Any]]
    explanation_target: str
