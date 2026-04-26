"""Disk-backed embedding cache keyed by sha256(text + model)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from medqa_rag.utils.io import ensure_dir


class EmbeddingCache:
    """Tiny on-disk cache: one ``.npy`` file per text hash.

    For very large corpora, swap for an LMDB / SQLite-backed store.
    """

    def __init__(self, root: str | Path, model_name: str) -> None:
        self.root = ensure_dir(Path(root) / model_name.replace("/", "__"))

    def _key(self, text: str) -> Path:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.root / f"{h}.npy"

    def get(self, text: str) -> np.ndarray | None:
        p = self._key(text)
        if p.exists():
            return np.load(p)
        return None

    def set(self, text: str, vec: np.ndarray) -> None:
        np.save(self._key(text), vec.astype(np.float32))
