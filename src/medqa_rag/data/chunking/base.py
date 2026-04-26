"""Chunker protocol."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from medqa_rag.core.types import Chunk


@runtime_checkable
class Chunker(Protocol):
    """A chunker takes whole documents and yields smaller, retrievable Chunks."""

    def split(self, docs: Iterable[Chunk]) -> list[Chunk]: ...
