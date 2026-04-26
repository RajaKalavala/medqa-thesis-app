"""Hybrid retrieval = dense FAISS ∪ sparse BM25 → RRF fusion."""
from __future__ import annotations

from medqa_rag.core.types import RetrievedDoc
from medqa_rag.rags.hybrid_rag.fusion import reciprocal_rank_fusion
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever


def retrieve_hybrid(
    faiss: FaissRetriever,
    bm25: BM25Retriever,
    query: str,
    *,
    pool_k: int,
    top_k: int,
    rrf_k: int,
) -> list[RetrievedDoc]:
    dense_run = faiss.retrieve(query, pool_k)
    sparse_run = bm25.retrieve(query, pool_k)
    return reciprocal_rank_fusion([dense_run, sparse_run], k=rrf_k, top_n=top_k)
