"""API request schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field

from medqa_rag.core.types import Architecture


class QARequest(BaseModel):
    """Ask one MCQ to a chosen RAG architecture."""

    question_id: str = Field(default="ad_hoc", examples=["usmle_001"])
    stem: str = Field(..., examples=["A 45-year-old male presents with sudden chest pain ..."])
    options: dict[str, str] = Field(
        ...,
        examples=[{"A": "MI", "B": "PE", "C": "Aortic dissection", "D": "Pneumonia"}],
    )
    correct_index: int | None = Field(default=None, ge=0, le=3, examples=[1])
    subject: str | None = Field(default=None, examples=["cardiology"])


class EvaluateRequest(BaseModel):
    """Trigger an evaluation run on the canned MedQA test set."""

    architecture: Architecture
    n_questions: int | None = Field(
        default=None,
        ge=1,
        description="Limit the number of test questions; null = full set.",
    )
    metrics: list[str] | None = Field(
        default=None,
        description="Optional override of the RAGAS metric list.",
    )


class ExplainRequest(BaseModel):
    """Run LIME or SHAP for a previously-answered question."""

    architecture: Architecture
    question: QARequest
    method: str = Field(default="lime", pattern="^(lime|shap)$")
