"""Explainability endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from medqa_rag.api.dependencies import get_rag
from medqa_rag.api.schemas import ExplainRequest, ExplainResponse
from medqa_rag.core.types import Question
from medqa_rag.explainability.lime_explainer import LimeExplainer
from medqa_rag.explainability.shap_explainer import ShapExplainer

router = APIRouter(prefix="/v1/explain", tags=["explainability"])


@router.post(
    "",
    response_model=ExplainResponse,
    summary="Run LIME or SHAP for a single (question, architecture)",
)
async def explain(payload: ExplainRequest) -> ExplainResponse:
    pipe = get_rag(payload.architecture)
    q = Question(
        id=payload.question.question_id,
        stem=payload.question.stem,
        options=payload.question.options,
        correct_index=payload.question.correct_index or 0,
        subject=payload.question.subject,
    )
    output = await pipe.answer(q)

    if payload.method == "lime":
        explainer = LimeExplainer(pipe)
    elif payload.method == "shap":
        explainer = ShapExplainer(pipe)
    else:
        raise HTTPException(400, f"Unknown method: {payload.method}")

    attribution = await explainer.explain(q, output)

    return ExplainResponse(
        question_id=q.id,
        architecture=payload.architecture,
        method=payload.method,
        passage_attributions=[
            {
                "chunk_id": d.chunk.id,
                "source": d.chunk.source,
                "score": s,
                "rank": d.rank,
            }
            for d, s in zip(output.retrieved_docs, attribution.passage_scores, strict=True)
        ],
        explanation_target=attribution.explanation_target,
    )
