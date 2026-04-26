"""HuggingFace / sentence-transformers embedder with optional caching."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from medqa_rag.core.exceptions import EmbeddingError
from medqa_rag.embeddings.cache import EmbeddingCache
from medqa_rag.observability.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceEmbedder:
    """Wraps a ``sentence-transformers`` model. Defaults to PubMedBERT.

    Lazy-loads the model on first use so that import-time cost is zero.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None  # lazy
        self._cache = EmbeddingCache(cache_dir, model_name) if cache_dir else None

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "sentence-transformers / torch missing. Install [.dev] extras."
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("embedder_loading", model=self.model_name, device=device)
        self._model = SentenceTransformer(self.model_name, device=device)

    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        self._load()
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        self._load()
        cache_hits: dict[int, np.ndarray] = {}
        to_compute_idx: list[int] = []
        to_compute_text: list[str] = []

        if self._cache is not None:
            for i, t in enumerate(texts):
                v = self._cache.get(t)
                if v is not None:
                    cache_hits[i] = v
                else:
                    to_compute_idx.append(i)
                    to_compute_text.append(t)
        else:
            to_compute_idx = list(range(len(texts)))
            to_compute_text = list(texts)

        if to_compute_text:
            new_vecs = self._model.encode(  # type: ignore[union-attr]
                to_compute_text,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            ).astype(np.float32)
        else:
            new_vecs = np.empty((0, self.dim), dtype=np.float32)

        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, vec in cache_hits.items():
            out[i] = vec
        for j, i in enumerate(to_compute_idx):
            out[i] = new_vecs[j]
            if self._cache is not None:
                self._cache.set(texts[i], new_vecs[j])

        return out

    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
