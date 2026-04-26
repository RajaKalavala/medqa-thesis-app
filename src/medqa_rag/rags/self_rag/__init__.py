"""Self-RAG — adaptive retrieval gated by model self-confidence."""
from medqa_rag.rags.self_rag.config import SelfRAGConfig
from medqa_rag.rags.self_rag.pipeline import SelfRAGPipeline

__all__ = ["SelfRAGConfig", "SelfRAGPipeline"]
