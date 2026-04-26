"""BM25 sparse retriever (rank-bm25)."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from medqa_rag.core.exceptions import RetrievalError
from medqa_rag.core.types import Chunk, RetrievedDoc
from medqa_rag.observability.logger import get_logger
from medqa_rag.utils.io import ensure_dir

logger = get_logger(__name__)

_STORE_FILE = "bm25.pkl"

_TOKENIZER = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKENIZER.findall(text)]


class BM25Retriever:
    name = "sparse"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise RetrievalError("Cannot build BM25 index from zero chunks")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise RetrievalError("rank-bm25 not installed") from exc

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self._chunks = list(chunks)
        logger.info("bm25_built", n=len(chunks))

    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        if self._bm25 is None:
            raise RetrievalError("BM25 not built")
        d = ensure_dir(directory)
        with (d / _STORE_FILE).open("wb") as fh:
            pickle.dump(
                {"bm25": self._bm25, "chunks": self._chunks, "k1": self.k1, "b": self.b}, fh
            )
        logger.info("bm25_saved", path=str(d), n=len(self._chunks))

    # ------------------------------------------------------------------
    def load(self, directory: str | Path) -> None:
        d = Path(directory)
        f = d / _STORE_FILE
        if not f.exists():
            raise RetrievalError(f"BM25 not found at {d}")
        with f.open("rb") as fh:
            blob = pickle.load(fh)  # noqa: S301
        self._bm25 = blob["bm25"]
        self._chunks = blob["chunks"]
        self.k1 = blob["k1"]
        self.b = blob["b"]
        logger.info("bm25_loaded", path=str(d), n=len(self._chunks))

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        if self._bm25 is None:
            raise RetrievalError("BM25 not built or loaded")
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedDoc(
                chunk=self._chunks[i],
                score=float(scores[i]),
                rank=rank,
                retriever=self.name,
            )
            for rank, i in enumerate(order)
        ]
