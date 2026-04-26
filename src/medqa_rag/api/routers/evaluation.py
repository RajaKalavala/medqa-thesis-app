"""Trigger an evaluation run from the API (small N, intended for smoke testing)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from medqa_rag.api.dependencies import get_rag
from medqa_rag.api.schemas import EvaluateRequest, EvaluateResponse
from medqa_rag.core.config import get_settings
from medqa_rag.data.loaders.medqa_loader import load_medqa_dir
from medqa_rag.evaluation.hallucination_detector import HallucinationDetector
from medqa_rag.evaluation.non_llm_metrics import accuracy, latency_summary
from medqa_rag.evaluation.ragas_evaluator import RagasEvaluator

router = APIRouter(prefix="/v1/evaluate", tags=["evaluation"])


@router.post(
    "",
    response_model=EvaluateResponse,
    summary="Evaluate one architecture on the canned MedQA test set",
)
async def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    settings = get_settings()
    questions = list(load_medqa_dir(settings.paths.raw_medqa))
    if not questions:
        raise HTTPException(404, "No MedQA questions found in raw directory")

    if payload.n_questions is not None:
        questions = questions[: payload.n_questions]

    pipe = get_rag(payload.architecture)
    outputs = []
    for q in questions:
        outputs.append(await pipe.answer(q))

    gold = {q.id: q for q in questions}
    acc = accuracy(outputs, gold)
    lat = latency_summary(outputs)

    ragas_scores: dict[str, float] = {}
    try:
        evaluator = RagasEvaluator(metrics=payload.metrics)
        ragas_scores = evaluator.evaluate(outputs, gold)
    except Exception:  # noqa: BLE001 — RAGAS is best-effort here
        ragas_scores = {}

    detector = HallucinationDetector()
    flags = detector.evaluate_batch(outputs)
    halluc_rate = sum(1 for f in flags.values() if f.is_flagged) / max(1, len(outputs))

    return EvaluateResponse(
        architecture=payload.architecture,
        n=len(outputs),
        accuracy=acc,
        ragas=ragas_scores,
        latency=lat,
        hallucination_rate=halluc_rate,
    )
