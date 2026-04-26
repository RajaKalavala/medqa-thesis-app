"""All four RAG architectures live here, each in its own self-contained folder."""
from medqa_rag.rags.base import RAGPipeline
from medqa_rag.rags.factory import build_rag

__all__ = ["RAGPipeline", "build_rag"]
