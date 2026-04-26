"""Merge per-hop retrievals into a deduplicated, ranked evidence chain."""
from __future__ import annotations

from medqa_rag.core.types import RetrievedDoc


def aggregate_chain(runs: list[list[RetrievedDoc]], *, top_n: int) -> list[RetrievedDoc]:
    """Round-robin interleave + dedupe by chunk id, preserving the originating hop."""
    seen: set[str] = set()
    out: list[RetrievedDoc] = []
    max_len = max((len(r) for r in runs), default=0)
    for i in range(max_len):
        for run in runs:
            if i >= len(run):
                continue
            d = run[i]
            if d.chunk.id in seen:
                continue
            seen.add(d.chunk.id)
            out.append(d.model_copy(update={"rank": len(out)}))
            if len(out) >= top_n:
                return out
    return out
