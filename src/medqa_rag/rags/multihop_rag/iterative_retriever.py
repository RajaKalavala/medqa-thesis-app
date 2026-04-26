"""Iterative retrieval: one FAISS pass per hop."""
from __future__ import annotations

from medqa_rag.core.types import RetrievedDoc
from medqa_rag.retrieval.dense_faiss import FaissRetriever


def iterative_retrieve(
    faiss: FaissRetriever,
    queries: list[str],
    *,
    top_k: int,
) -> list[list[RetrievedDoc]]:
    """One retrieval list per query. Tagged with hop index in `retriever`."""
    runs: list[list[RetrievedDoc]] = []
    for hop, q in enumerate(queries):
        docs = faiss.retrieve(q, top_k)
        # Re-tag retriever to record hop index for downstream introspection
        runs.append(
            [
                d.model_copy(update={"retriever": f"hop_{hop}"})
                for d in docs
            ]
        )
    return runs
