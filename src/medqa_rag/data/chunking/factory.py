"""Chunker factory."""
from __future__ import annotations

from medqa_rag.core.config import get_settings
from medqa_rag.core.exceptions import ConfigError
from medqa_rag.data.chunking.base import Chunker
from medqa_rag.data.chunking.recursive import RecursiveChunker


def build_chunker() -> Chunker:
    cfg = get_settings().chunking
    if cfg.strategy == "recursive":
        return RecursiveChunker(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    raise ConfigError(f"Unknown chunking strategy: {cfg.strategy}")
