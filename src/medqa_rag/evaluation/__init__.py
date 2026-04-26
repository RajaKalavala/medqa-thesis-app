"""Evaluation: RAGAS metrics, hallucination detection, statistical tests, reporters."""
from medqa_rag.evaluation.hallucination_detector import HallucinationDetector
from medqa_rag.evaluation.non_llm_metrics import (
    accuracy,
    latency_summary,
)
from medqa_rag.evaluation.ragas_evaluator import RagasEvaluator
from medqa_rag.evaluation.statistical_tests import cochran_q, mcnemar

__all__ = [
    "HallucinationDetector",
    "RagasEvaluator",
    "accuracy",
    "cochran_q",
    "latency_summary",
    "mcnemar",
]
