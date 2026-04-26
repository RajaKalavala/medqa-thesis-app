"""Factory: build a RAG pipeline by name."""
from __future__ import annotations

from medqa_rag.core.config import get_settings
from medqa_rag.core.exceptions import ConfigError
from medqa_rag.core.types import Architecture
from medqa_rag.embeddings.factory import build_embedder
from medqa_rag.llm.groq_client import GroqClient
from medqa_rag.rags.base import RAGPipeline
from medqa_rag.rags.hybrid_rag.pipeline import HybridRAGPipeline
from medqa_rag.rags.multihop_rag.pipeline import MultiHopRAGPipeline
from medqa_rag.rags.naive_rag.pipeline import NaiveRAGPipeline
from medqa_rag.rags.self_rag.pipeline import SelfRAGPipeline
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever


def build_rag(arch: Architecture | str, *, load_indices: bool = True) -> RAGPipeline:
    """Construct a fully-wired RAG pipeline.

    All shared dependencies (LLM, embedder, indices) are loaded once here
    so a comparison run instantiates them only once per architecture.
    """
    arch = Architecture(arch) if isinstance(arch, str) else arch
    settings = get_settings()

    embedder = build_embedder()
    llm = GroqClient()

    faiss: FaissRetriever | None = None
    bm25: BM25Retriever | None = None

    if arch in (Architecture.NAIVE, Architecture.SELF, Architecture.HYBRID, Architecture.MULTIHOP):
        faiss = FaissRetriever(embedder)
        if load_indices:
            faiss.load(settings.paths.faiss_dir)

    if arch in (Architecture.HYBRID,):
        bm25 = BM25Retriever()
        if load_indices:
            bm25.load(settings.paths.bm25_dir)

    if arch is Architecture.NAIVE:
        return NaiveRAGPipeline(llm=llm, embedder=embedder, faiss=faiss)
    if arch is Architecture.SELF:
        return SelfRAGPipeline(llm=llm, embedder=embedder, faiss=faiss)
    if arch is Architecture.HYBRID:
        return HybridRAGPipeline(llm=llm, embedder=embedder, faiss=faiss, bm25=bm25)
    if arch is Architecture.MULTIHOP:
        return MultiHopRAGPipeline(llm=llm, embedder=embedder, faiss=faiss)

    raise ConfigError(f"Unknown architecture: {arch}")
