"""Stratified sampler tests."""
from __future__ import annotations

import pytest

from medqa_rag.core.types import Question
from medqa_rag.explainability.sampler import stratified_sample


def _q(i: int, subject: str) -> Question:
    return Question(
        id=f"q{i}",
        stem=f"stem {i}",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        correct_index=0,
        subject=subject,
    )


@pytest.mark.unit
def test_stratified_returns_n():
    qs = [_q(i, "cardio") for i in range(60)] + [_q(i, "neuro") for i in range(60, 100)]
    out = stratified_sample(qs, n=20, by="subject", seed=0)
    assert len(out) == 20


@pytest.mark.unit
def test_stratified_keeps_balance():
    qs = [_q(i, "cardio") for i in range(80)] + [_q(i, "neuro") for i in range(80, 100)]
    out = stratified_sample(qs, n=20, by="subject", seed=0)
    counts = {"cardio": 0, "neuro": 0}
    for q in out:
        counts[q.subject] += 1
    # cardio is 80% of pool → should be ≥ 12 of 20
    assert counts["cardio"] >= 12
