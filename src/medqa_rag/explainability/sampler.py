"""Stratified sampler: sub-sample questions for the costly XAI pass."""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from medqa_rag.core.types import Question


def stratified_sample(
    questions: Iterable[Question],
    *,
    n: int,
    by: str = "subject",
    seed: int = 42,
) -> list[Question]:
    """Stratify by ``q.<by>`` (defaults to subject) and sample proportionally."""
    rng = random.Random(seed)
    qs = list(questions)
    if n >= len(qs):
        return qs

    buckets: dict[str, list[Question]] = defaultdict(list)
    for q in qs:
        key = getattr(q, by, None) or "_unknown"
        buckets[key].append(q)

    total = len(qs)
    out: list[Question] = []
    for key, group in buckets.items():
        share = max(1, round(n * len(group) / total))
        out.extend(rng.sample(group, k=min(share, len(group))))

    rng.shuffle(out)
    return out[:n]
