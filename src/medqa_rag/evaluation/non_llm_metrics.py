"""Non-LLM evaluation: accuracy, latency, token-cost summaries."""
from __future__ import annotations

import statistics
from collections.abc import Iterable

from medqa_rag.core.types import Question, RAGOutput


def accuracy(outputs: list[RAGOutput], gold: dict[str, Question]) -> float:
    """Mean accuracy of predicted index vs. gold correct index."""
    if not outputs:
        return 0.0
    n_correct = sum(
        1
        for o in outputs
        if o.predicted_index is not None
        and gold[o.question_id].correct_index == o.predicted_index
    )
    return n_correct / len(outputs)


def latency_summary(outputs: Iterable[RAGOutput]) -> dict[str, float]:
    """Return p50, p95, p99, mean latency."""
    lats = sorted(o.latency_ms for o in outputs)
    if not lats:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    def pct(p: float) -> float:
        i = max(0, min(len(lats) - 1, int(round(p * (len(lats) - 1)))))
        return lats[i]

    return {
        "n": len(lats),
        "mean_ms": statistics.mean(lats),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
    }


def token_summary(outputs: Iterable[RAGOutput]) -> dict[str, int]:
    total_prompt = 0
    total_completion = 0
    total = 0
    for o in outputs:
        total_prompt += o.token_usage.get("prompt_tokens", 0)
        total_completion += o.token_usage.get("completion_tokens", 0)
        total += o.token_usage.get("total_tokens", 0)
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total,
    }
