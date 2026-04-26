"""FAISS-backed dense retriever."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from medqa_rag.core.exceptions import RetrievalError
from medqa_rag.core.types import Chunk, RetrievedDoc
from medqa_rag.embeddings.base import Embedder
from medqa_rag.observability.logger import get_logger
from medqa_rag.utils.io import ensure_dir

logger = get_logger(__name__)

_INDEX_FILE = "index.faiss"
_DOCSTORE_FILE = "docstore.pkl"
_META_FILE = "meta.json"


class FaissRetriever:
    """In-memory FAISS index + parallel docstore.

    Uses inner-product similarity on L2-normalized vectors (== cosine).
    """

    name = "dense"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._index = None
        self._chunks: list[Chunk] = []
        self._dim: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def _faiss(cls):  # noqa: D401
        try:
            import faiss

            return faiss
        except ImportError as exc:  # pragma: no cover
            raise RetrievalError("faiss-cpu not installed") from exc

    # ------------------------------------------------------------------
    def build(self, chunks: list[Chunk]) -> None:
        """Build the index from scratch."""
        if not chunks:
            raise RetrievalError("Cannot build FAISS index from zero chunks")

        faiss = self._faiss()
        texts = [c.text for c in chunks]
        vecs = self.embedder.embed_documents(texts).astype(np.float32)
        self._dim = vecs.shape[1]

        index = faiss.IndexFlatIP(self._dim)
        index.add(vecs)
        self._index = index
        self._chunks = list(chunks)
        logger.info("faiss_built", n=len(chunks), dim=self._dim)

    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        if self._index is None:
            raise RetrievalError("Index not built")
        d = ensure_dir(directory)
        faiss = self._faiss()
        faiss.write_index(self._index, str(d / _INDEX_FILE))
        with (d / _DOCSTORE_FILE).open("wb") as fh:
            pickle.dump(self._chunks, fh)
        (d / _META_FILE).write_text(
            json.dumps({"dim": self._dim, "n": len(self._chunks)}, indent=2)
        )
        logger.info("faiss_saved", path=str(d), n=len(self._chunks))

    # ------------------------------------------------------------------
    def load(self, directory: str | Path) -> None:
        d = Path(directory)
        meta_path = d / _META_FILE
        if not meta_path.exists():
            raise RetrievalError(f"FAISS index not found at {d}")

        faiss = self._faiss()
        meta = json.loads(meta_path.read_text())
        self._index = faiss.read_index(str(d / _INDEX_FILE))
        with (d / _DOCSTORE_FILE).open("rb") as fh:
            self._chunks = pickle.load(fh)  # noqa: S301
        self._dim = meta["dim"]
        logger.info("faiss_loaded", path=str(d), n=len(self._chunks))

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        if self._index is None:
            raise RetrievalError("Index not built or loaded")

        qvec = self.embedder.embed_query(query).astype(np.float32).reshape(1, -1)
        scores, ids = self._index.search(qvec, top_k)
        results: list[RetrievedDoc] = []
        for rank, (i, s) in enumerate(zip(ids[0].tolist(), scores[0].tolist(), strict=True)):
            if i == -1 or i >= len(self._chunks):
                continue
            results.append(
                RetrievedDoc(
                    chunk=self._chunks[i],
                    score=float(s),
                    rank=rank,
                    retriever=self.name,
                )
            )
        return results
