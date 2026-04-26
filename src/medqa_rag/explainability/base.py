"""Explainer protocol + Attribution type."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from medqa_rag.core.types import Question, RAGOutput


class Attribution(BaseModel):
    """Per-passage attribution scores for one (question, RAGOutput) pair."""

    question_id: str
    architecture: str
    method: str  # "lime" | "shap"
    passage_scores: list[float]  # same length as RAGOutput.retrieved_docs
    explanation_target: str  # the predicted answer text being explained
    extras: dict[str, float] = {}


@runtime_checkable
class Explainer(Protocol):
    name: str

    async def explain(self, question: Question, output: RAGOutput) -> Attribution: ...
