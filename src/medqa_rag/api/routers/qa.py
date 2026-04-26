"""Single-question QA across the four architectures."""
from __future__ import annotations

from fastapi import APIRouter, Path

from medqa_rag.api.dependencies import get_rag
from medqa_rag.api.schemas import QARequest, QAResponse, RetrievedPassage
from medqa_rag.core.types import Architecture, Question

router = APIRouter(prefix="/v1/qa", tags=["qa"])


@router.post(
    "/{architecture}",
    response_model=QAResponse,
    summary="Answer one MCQ with the chosen RAG architecture",
)
async def ask(
    architecture: Architecture = Path(..., description="One of naive | self | hybrid | multihop"),
    payload: QARequest = ...,  # noqa: B008
) -> QAResponse:
    pipe = get_rag(architecture)
    q = Question(
        id=payload.question_id,
        stem=payload.stem,
        options=payload.options,
        correct_index=payload.correct_index if payload.correct_index is not None else 0,
        subject=payload.subject,
    )
    out = await pipe.answer(q)

    correct: bool | None = None
    if payload.correct_index is not None and out.predicted_index is not None:
        correct = payload.correct_index == out.predicted_index

    return QAResponse(
        question_id=out.question_id,
        architecture=out.architecture,
        predicted_letter=out.predicted_letter,
        predicted_index=out.predicted_index,
        correct=correct,
        generated_answer=out.generated_answer,
        retrieved=[
            RetrievedPassage(
                chunk_id=d.chunk.id,
                source=d.chunk.source,
                text=d.chunk.text,
                score=d.score,
                rank=d.rank,
                retriever=d.retriever,
            )
            for d in out.retrieved_docs
        ],
        latency_ms=out.latency_ms,
        token_usage=out.token_usage,
        extras={
            "retrieval_used": out.retrieval_used,
            "hop_count": out.hop_count,
            "sub_queries": out.sub_queries,
            **out.extras,
        },
    )
