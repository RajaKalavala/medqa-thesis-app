"""Project-wide constants."""
from __future__ import annotations

from typing import Final

LETTERS: Final[tuple[str, ...]] = ("A", "B", "C", "D")
LETTER_TO_INDEX: Final[dict[str, int]] = {l: i for i, l in enumerate(LETTERS)}
INDEX_TO_LETTER: Final[dict[int, str]] = {i: l for i, l in enumerate(LETTERS)}

DEFAULT_TOP_K: Final[int] = 5
DEFAULT_CHUNK_SIZE: Final[int] = 512
DEFAULT_CHUNK_OVERLAP: Final[int] = 64

# Multi-hop guardrails
MAX_HOPS: Final[int] = 3
MAX_SUBQUERIES: Final[int] = 3

# Self-RAG confidence gate
CONFIDENCE_THRESHOLD: Final[float] = 0.65

# Hybrid RAG
RRF_K: Final[int] = 60
