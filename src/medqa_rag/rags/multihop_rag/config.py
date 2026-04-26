"""Multi-Hop RAG configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field

from medqa_rag.core.config import get_settings
from medqa_rag.core.constants import MAX_HOPS, MAX_SUBQUERIES


class MultiHopRAGConfig(BaseModel):
    top_k_per_hop: int = Field(default_factory=lambda: get_settings().retrieval.top_k)
    max_hops: int = MAX_HOPS
    max_subqueries: int = MAX_SUBQUERIES
