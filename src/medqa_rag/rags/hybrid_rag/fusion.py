"""Reciprocal Rank Fusion."""
from __future__ import annotations

from medqa_rag.core.types import RetrievedDoc


def reciprocal_rank_fusion(
    runs: list[list[RetrievedDoc]],
    *,
    k: int = 60,
    top_n: int = 5,
) -> list[RetrievedDoc]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    score(d) = Σ_run 1 / (k + rank_run(d))
    """
    score_by_id: dict[str, float] = {}
    canonical: dict[str, RetrievedDoc] = {}
    for run in runs:
        for doc in run:
            cid = doc.chunk.id
            score_by_id[cid] = score_by_id.get(cid, 0.0) + 1.0 / (k + doc.rank + 1)
            # keep first observation as canonical doc reference
            canonical.setdefault(cid, doc)

    ranked = sorted(score_by_id.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        RetrievedDoc(
            chunk=canonical[cid].chunk,
            score=score,
            rank=i,
            retriever="hybrid",
        )
        for i, (cid, score) in enumerate(ranked)
    ]
