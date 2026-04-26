"""Naive RAG configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field

from medqa_rag.core.config import get_settings


class NaiveRAGConfig(BaseModel):
    top_k: int = Field(default_factory=lambda: get_settings().retrieval.top_k)
