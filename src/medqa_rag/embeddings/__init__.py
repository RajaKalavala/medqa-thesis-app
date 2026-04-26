"""Embeddings module — local HuggingFace embedder with on-disk cache."""
from medqa_rag.embeddings.base import Embedder
from medqa_rag.embeddings.factory import build_embedder
from medqa_rag.embeddings.huggingface_embedder import HuggingFaceEmbedder

__all__ = ["Embedder", "HuggingFaceEmbedder", "build_embedder"]
