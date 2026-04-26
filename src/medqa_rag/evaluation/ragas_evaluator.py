"""RAGAS evaluator wrapper.

Uses Groq's small judge model so per-question evaluation cost stays bounded.
Lazy imports of ``ragas`` keep import-time cost off the hot path.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from medqa_rag.core.config import get_settings
from medqa_rag.core.exceptions import EvaluationError
from medqa_rag.core.types import Question, RAGOutput
from medqa_rag.observability.logger import get_logger

logger = get_logger(__name__)

_METRIC_MAP_NAMES = (
    "faithfulness",
    "answer_correctness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
)


class RagasEvaluator:
    """Wraps the RAGAS evaluation suite around our RAGOutput schema."""

    def __init__(self, metrics: list[str] | None = None, judge_model: str | None = None) -> None:
        cfg = get_settings()
        self.metric_names = metrics or cfg.evaluation.ragas_metrics
        self.judge_model = judge_model or cfg.llm.judge_model

    # ------------------------------------------------------------------
    def _load_metrics(self) -> list[Any]:
        try:
            from ragas import metrics as rm  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EvaluationError("ragas is not installed") from exc

        registry = {
            "faithfulness": rm.faithfulness,
            "answer_correctness": rm.answer_correctness,
            "context_precision": rm.context_precision,
            "context_recall": rm.context_recall,
            "answer_relevancy": rm.answer_relevancy,
        }
        chosen: list[Any] = []
        for name in self.metric_names:
            if name not in registry:
                raise EvaluationError(f"Unknown RAGAS metric: {name}")
            chosen.append(registry[name])
        return chosen

    # ------------------------------------------------------------------
    @staticmethod
    def _to_rows(outputs: list[RAGOutput], gold: dict[str, Question]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for o in outputs:
            q = gold[o.question_id]
            rows.append(
                {
                    "question": q.stem,
                    "answer": o.generated_answer,
                    "contexts": [d.chunk.text for d in o.retrieved_docs],
                    "ground_truth": q.correct_text,
                }
            )
        return rows

    # ------------------------------------------------------------------
    def evaluate(self, outputs: list[RAGOutput], gold: dict[str, Question]) -> dict[str, float]:
        """Return a dict of mean metric values across the set."""
        if not outputs:
            return {m: 0.0 for m in self.metric_names}

        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EvaluationError("ragas / datasets not installed") from exc

        rows = self._to_rows(outputs, gold)
        ds = Dataset.from_list(rows)
        metrics = self._load_metrics()

        logger.info("ragas_eval_start", n=len(rows), metrics=self.metric_names)
        result = ragas_evaluate(ds, metrics=metrics)
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        # ragas returns metric_name -> mean
        cleaned = {k: float(v) for k, v in scores.items()}
        logger.info("ragas_eval_done", scores=cleaned)
        return cleaned

    # ------------------------------------------------------------------
    def evaluate_per_question(
        self, outputs: list[RAGOutput], gold: dict[str, Question]
    ) -> list[dict[str, Any]]:
        """Return one row per question with all metric scores."""
        if not outputs:
            return []
        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EvaluationError("ragas not installed") from exc

        rows = self._to_rows(outputs, gold)
        ds = Dataset.from_list(rows)
        metrics = self._load_metrics()
        result = ragas_evaluate(ds, metrics=metrics)
        df = result.to_pandas()
        df["question_id"] = [o.question_id for o in outputs]
        return df.to_dict(orient="records")
