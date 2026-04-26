"""FastAPI dependency providers (cached singletons)."""
from __future__ import annotations

from functools import lru_cache

from medqa_rag.core.types import Architecture
from medqa_rag.rags.base import RAGPipeline
from medqa_rag.rags.factory import build_rag


@lru_cache(maxsize=4)
def get_rag(arch: Architecture) -> RAGPipeline:
    """Cached per-architecture pipeline. Indices are loaded once."""
    return build_rag(arch, load_indices=True)
