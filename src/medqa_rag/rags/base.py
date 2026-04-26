"""Shared abstract base for all RAG architectures.

Every concrete RAG must implement ``answer(question)`` returning a
``RAGOutput``.  The base class provides:

* a Jinja prompt loader scoped to the subclass's package
* an MCQ letter parser
* uniform latency measurement
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from importlib.resources import files
from pathlib import Path

from jinja2 import Template

from medqa_rag.core.constants import LETTERS, LETTER_TO_INDEX
from medqa_rag.core.types import (
    Architecture,
    Question,
    RAGOutput,
    RetrievedDoc,
)
from medqa_rag.embeddings.base import Embedder
from medqa_rag.llm.groq_client import GroqClient
from medqa_rag.observability.logger import get_logger
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever

logger = get_logger(__name__)

_LETTER_RE = re.compile(r"\b([A-D])\b")


class RAGPipeline(ABC):
    """Abstract pipeline shared by all four RAG architectures."""

    architecture: Architecture
    package: str  # subclass overrides, e.g. "medqa_rag.rags.naive_rag"

    def __init__(
        self,
        llm: GroqClient,
        embedder: Embedder,
        faiss: FaissRetriever | None = None,
        bm25: BM25Retriever | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.faiss = faiss
        self.bm25 = bm25

    # ------------------------------------------------------------------
    @abstractmethod
    async def answer(self, question: Question) -> RAGOutput:
        """Run the architecture end-to-end on one question."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def load_prompt(self, name: str) -> Template:
        """Load a Jinja2 prompt template from this RAG's ``prompts/`` folder."""
        try:
            text = (files(self.package) / "prompts" / name).read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback for source layouts during development
            here = Path(__file__).parent
            candidate = here / self.package.split(".")[-1] / "prompts" / name
            text = candidate.read_text(encoding="utf-8")
        return Template(text)

    # ------------------------------------------------------------------
    @staticmethod
    def format_question(q: Question) -> str:
        opts = "\n".join(f"{l}. {q.options[l]}" for l in LETTERS)
        return f"Question:\n{q.stem}\n\nOptions:\n{opts}"

    # ------------------------------------------------------------------
    @staticmethod
    def format_context(docs: list[RetrievedDoc]) -> str:
        if not docs:
            return "(no retrieved evidence)"
        lines = []
        for i, d in enumerate(docs, 1):
            lines.append(f"[{i}] (source: {d.chunk.source})\n{d.chunk.text.strip()}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def parse_letter(text: str) -> str | None:
        """Extract the first A/B/C/D the model emits.

        Looks for explicit ``Answer: X`` first, then any standalone letter.
        """
        m = re.search(r"Answer\s*[:\-]\s*([A-D])", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m2 = _LETTER_RE.search(text)
        return m2.group(1).upper() if m2 else None

    @classmethod
    def letter_to_index(cls, letter: str | None) -> int | None:
        if letter is None:
            return None
        return LETTER_TO_INDEX.get(letter.upper())
