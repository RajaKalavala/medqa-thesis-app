"""Domain types shared across all modules.

Every RAG, evaluator, and API endpoint speaks in these types so
comparison results are uniform and serializable.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Architecture(StrEnum):
    NAIVE = "naive"
    SELF = "self"
    HYBRID = "hybrid"
    MULTIHOP = "multihop"


class Question(BaseModel):
    """A single MedQA-style multiple-choice question."""

    model_config = ConfigDict(frozen=True)

    id: str
    stem: str
    options: dict[str, str] = Field(..., description="Mapping of 'A'..'D' to option text")
    correct_index: int = Field(..., ge=0, le=3)
    subject: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def correct_letter(self) -> str:
        return ["A", "B", "C", "D"][self.correct_index]

    @property
    def correct_text(self) -> str:
        return self.options[self.correct_letter]


class Chunk(BaseModel):
    """A retrievable piece of textbook content."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDoc(BaseModel):
    """A chunk paired with the score that retrieved it."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str = Field(..., description="dense | sparse | hybrid | hop_<n>")


class RAGOutput(BaseModel):
    """Uniform output of every RAG architecture — comparable across systems."""

    question_id: str
    architecture: Architecture
    retrieved_docs: list[RetrievedDoc]
    generated_answer: str
    predicted_letter: str | None = None
    predicted_index: int | None = None
    latency_ms: float
    token_usage: dict[str, int] = Field(default_factory=dict)
    # Architecture-specific metadata
    retrieval_used: bool = True            # Self-RAG: false if model skipped retrieval
    hop_count: int = 1                     # Multi-Hop: number of retrieval rounds
    sub_queries: list[str] = Field(default_factory=list)  # Multi-Hop decompositions
    extras: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Per-question evaluation record."""

    question_id: str
    architecture: Architecture
    correct: bool
    ragas: dict[str, float] = Field(default_factory=dict)
    latency_ms: float
    token_usage: dict[str, int] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict)
