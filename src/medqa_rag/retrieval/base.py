"""Retriever protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from medqa_rag.core.types import RetrievedDoc


@runtime_checkable
class Retriever(Protocol):
    """Retrievers map a query string to ranked documents."""

    name: str

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]: ...
