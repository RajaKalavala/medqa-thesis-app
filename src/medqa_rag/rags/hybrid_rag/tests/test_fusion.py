"""RRF fusion unit tests."""
from __future__ import annotations

import pytest

from medqa_rag.core.types import Chunk, RetrievedDoc
from medqa_rag.rags.hybrid_rag.fusion import reciprocal_rank_fusion


def _doc(cid: str, rank: int, retriever: str) -> RetrievedDoc:
    return RetrievedDoc(
        chunk=Chunk(id=cid, text=f"text-{cid}", source="x"),
        score=1.0 / (rank + 1),
        rank=rank,
        retriever=retriever,
    )


@pytest.mark.unit
def test_rrf_promotes_doc_appearing_in_both_runs():
    dense = [_doc("a", 0, "dense"), _doc("b", 1, "dense"), _doc("c", 2, "dense")]
    sparse = [_doc("b", 0, "sparse"), _doc("d", 1, "sparse"), _doc("a", 2, "sparse")]

    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_n=4)
    ids = [d.chunk.id for d in fused]
    # 'b' appears at top of one and high in the other → should rank #1
    assert ids[0] == "b"
    # 'a' is in both → should rank above 'c' (only dense) and 'd' (only sparse) of equal rank
    assert "a" in ids[:3]


@pytest.mark.unit
def test_rrf_top_n_truncation():
    dense = [_doc(str(i), i, "dense") for i in range(10)]
    sparse = [_doc(str(i + 100), i, "sparse") for i in range(10)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_n=5)
    assert len(fused) == 5
