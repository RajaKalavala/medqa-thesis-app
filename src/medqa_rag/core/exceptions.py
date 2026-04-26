"""Domain exceptions used throughout the framework."""
from __future__ import annotations


class MedQARAGError(Exception):
    """Base exception for all medqa-rag errors."""


class ConfigError(MedQARAGError):
    """Raised on invalid or missing configuration."""


class DataError(MedQARAGError):
    """Raised on dataset loading / preprocessing failures."""


class EmbeddingError(MedQARAGError):
    """Raised on embedding model failures."""


class RetrievalError(MedQARAGError):
    """Raised on retrieval (FAISS / BM25) failures."""


class LLMError(MedQARAGError):
    """Raised on LLM provider failures (Groq)."""


class RateLimitError(LLMError):
    """Raised when the LLM provider rate limit is hit after retries."""


class EvaluationError(MedQARAGError):
    """Raised on evaluation failures (RAGAS, statistical tests)."""


class ExplainabilityError(MedQARAGError):
    """Raised on LIME / SHAP failures."""
