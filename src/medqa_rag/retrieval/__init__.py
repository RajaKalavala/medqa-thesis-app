"""Low-level retrievers shared by all four RAG architectures."""
from medqa_rag.retrieval.base import Retriever
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever

__all__ = ["BM25Retriever", "FaissRetriever", "Retriever"]
