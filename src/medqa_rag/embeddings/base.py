"""Embedder protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """An embedder maps strings to dense float32 vectors."""

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return shape ``(len(texts), dim)``."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return shape ``(dim,)``."""
