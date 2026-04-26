"""Self-RAG configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field

from medqa_rag.core.config import get_settings
from medqa_rag.core.constants import CONFIDENCE_THRESHOLD


class SelfRAGConfig(BaseModel):
    top_k: int = Field(default_factory=lambda: get_settings().retrieval.top_k)
    confidence_threshold: float = CONFIDENCE_THRESHOLD
