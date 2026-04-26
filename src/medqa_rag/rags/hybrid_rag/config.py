"""Hybrid RAG configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field

from medqa_rag.core.config import get_settings
from medqa_rag.core.constants import RRF_K


class HybridRAGConfig(BaseModel):
    top_k: int = Field(default_factory=lambda: get_settings().retrieval.top_k)
    pool_k: int = 20  # how many to retrieve from each side before fusion
    rrf_k: int = RRF_K
