"""Embedder factory."""
from __future__ import annotations

from medqa_rag.core.config import get_settings
from medqa_rag.embeddings.base import Embedder
from medqa_rag.embeddings.huggingface_embedder import HuggingFaceEmbedder


def build_embedder() -> Embedder:
    settings = get_settings()
    cfg = settings.embedder
    cache_dir = settings.paths.embeddings_dir if cfg.cache_enabled else None
    return HuggingFaceEmbedder(
        model_name=cfg.model_name,
        device=cfg.device,
        batch_size=cfg.batch_size,
        normalize=cfg.normalize,
        cache_dir=cache_dir,
    )
