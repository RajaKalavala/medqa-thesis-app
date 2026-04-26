"""Recursive character-based chunker (LangChain-compatible)."""
from __future__ import annotations

from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from medqa_rag.core.types import Chunk
from medqa_rag.observability.logger import get_logger

logger = get_logger(__name__)


class RecursiveChunker:
    """Recursive character text splitter wrapper.

    Uses LangChain's separator hierarchy (paragraph -> sentence -> word -> char)
    so we don't break in the middle of medical terms when avoidable.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, docs: Iterable[Chunk]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            pieces = self._splitter.split_text(doc.text)
            for i, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                out.append(
                    Chunk(
                        id=f"{doc.id}::chunk_{i:05d}",
                        text=piece,
                        source=doc.source,
                        metadata={
                            **doc.metadata,
                            "parent_id": doc.id,
                            "chunk_index": i,
                            "chunk_size": self.chunk_size,
                        },
                    )
                )
        logger.info("chunking_done", n_input_docs=sum(1 for _ in docs) if isinstance(docs, list) else None, n_chunks=len(out))
        return out
