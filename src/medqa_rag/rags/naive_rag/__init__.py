"""Naive RAG — retrieve top-k FAISS, stuff into prompt, answer."""
from medqa_rag.rags.naive_rag.config import NaiveRAGConfig
from medqa_rag.rags.naive_rag.pipeline import NaiveRAGPipeline

__all__ = ["NaiveRAGConfig", "NaiveRAGPipeline"]
