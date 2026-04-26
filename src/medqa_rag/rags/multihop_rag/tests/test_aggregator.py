"""Aggregator tests."""
from __future__ import annotations

import pytest

from medqa_rag.core.types import Chunk, RetrievedDoc
from medqa_rag.rags.multihop_rag.chain_aggregator import aggregate_chain


def _doc(cid: str, rank: int, hop: int) -> RetrievedDoc:
    return RetrievedDoc(
        chunk=Chunk(id=cid, text=f"text-{cid}", source="x"),
        score=1.0,
        rank=rank,
        retriever=f"hop_{hop}",
    )


@pytest.mark.unit
def test_dedup_across_hops():
    runs = [
        [_doc("a", 0, 0), _doc("b", 1, 0)],
        [_doc("a", 0, 1), _doc("c", 1, 1)],
    ]
    out = aggregate_chain(runs, top_n=10)
    ids = [d.chunk.id for d in out]
    assert ids == ["a", "b", "c"]
    assert all(d.rank == i for i, d in enumerate(out))


@pytest.mark.unit
def test_top_n_truncation():
    runs = [[_doc(str(i), i, 0) for i in range(20)]]
    out = aggregate_chain(runs, top_n=5)
    assert len(out) == 5
