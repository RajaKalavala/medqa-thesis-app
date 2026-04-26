"""Naive RAG retrieval: dense top-k only."""
from __future__ import annotations

from medqa_rag.core.types import RetrievedDoc
from medqa_rag.retrieval.dense_faiss import FaissRetriever


def retrieve_naive(faiss: FaissRetriever, query: str, top_k: int) -> list[RetrievedDoc]:
    return faiss.retrieve(query, top_k)
