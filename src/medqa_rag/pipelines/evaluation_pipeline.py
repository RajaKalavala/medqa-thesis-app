"""Run one architecture over the test set; emit per-question outputs."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from medqa_rag.core.config import get_settings
from medqa_rag.core.types import Architecture, Question, RAGOutput
from medqa_rag.data.loaders.medqa_loader import load_medqa_dir
from medqa_rag.evaluation.hallucination_detector import HallucinationDetector
from medqa_rag.evaluation.non_llm_metrics import accuracy, latency_summary, token_summary
from medqa_rag.evaluation.ragas_evaluator import RagasEvaluator
from medqa_rag.observability.logger import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
)
from medqa_rag.observability.mlflow_tracker import (
    log_metrics,
    log_params,
    mlflow_run,
)
from medqa_rag.rags.factory import build_rag

logger = get_logger(__name__)


async def run_architecture(
    arch: Architecture,
    *,
    n_questions: int | None = None,
    use_ragas: bool = True,
) -> Path:
    """Run one architecture and write per-question outputs to disk."""
    configure_logging()
    settings = get_settings()
    bind_context(architecture=str(arch))

    logger.info("eval_start", arch=str(arch))
    questions: list[Question] = list(load_medqa_dir(settings.paths.raw_medqa))
    if n_questions is not None:
        questions = questions[:n_questions]

    pipe = build_rag(arch, load_indices=True)

    outputs: list[RAGOutput] = []
    for i, q in enumerate(questions):
        try:
            outputs.append(await pipe.answer(q))
        except Exception:
            logger.exception("question_failed", qid=q.id)
        if (i + 1) % 25 == 0:
            logger.info("progress", done=i + 1, total=len(questions))

    gold = {q.id: q for q in questions}
    acc = accuracy(outputs, gold)
    lat = latency_summary(outputs)
    tok = token_summary(outputs)

    ragas_scores: dict[str, float] = {}
    if use_ragas:
        try:
            evaluator = RagasEvaluator()
            ragas_scores = evaluator.evaluate(outputs, gold)
        except Exception:
            logger.exception("ragas_failed")

    detector = HallucinationDetector()
    flags = detector.evaluate_batch(outputs)
    halluc_rate = sum(1 for f in flags.values() if f.is_flagged) / max(1, len(outputs))

    # ---- Persist outputs ----
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(settings.paths.results_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arch.value}_{timestamp}.json"

    correctness = {
        o.question_id: bool(
            o.predicted_index is not None
            and gold[o.question_id].correct_index == o.predicted_index
        )
        for o in outputs
    }

    record = {
        "architecture": str(arch),
        "timestamp": timestamp,
        "n": len(outputs),
        "accuracy": acc,
        "ragas": ragas_scores,
        "latency": lat,
        "tokens": tok,
        "hallucination_rate": halluc_rate,
        "correctness": correctness,
        "flags": {qid: f.to_dict() for qid, f in flags.items()},
        "outputs": [o.model_dump(mode="json") for o in outputs],
    }
    out_path.write_text(json.dumps(record, indent=2, default=str))
    logger.info("eval_written", path=str(out_path))

    # ---- MLflow ----
    try:
        with mlflow_run(run_name=f"eval-{arch.value}-{timestamp}", tags={"arch": str(arch)}):
            log_params({"architecture": str(arch), "n_questions": len(outputs)})
            metrics_to_log: dict[str, float] = {
                "accuracy": acc,
                "hallucination_rate": halluc_rate,
                **{f"latency_{k}": v for k, v in lat.items() if isinstance(v, (int, float))},
                **{f"ragas_{k}": v for k, v in ragas_scores.items()},
            }
            log_metrics(metrics_to_log)
    except Exception:
        logger.warning("mlflow_logging_skipped", exc_info=True)

    clear_context()
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=[a.value for a in Architecture])
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--no-ragas", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        run_architecture(
            Architecture(args.arch),
            n_questions=args.n,
            use_ragas=not args.no_ragas,
        )
    )


if __name__ == "__main__":
    main()
