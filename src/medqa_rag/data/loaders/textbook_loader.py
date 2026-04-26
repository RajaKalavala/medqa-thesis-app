"""Load the medical-textbook corpus.

Each textbook can be a ``.txt`` or ``.md`` file under
``settings.paths.raw_textbooks``.  Returns one ``Document`` per file
which is later split by the chunker.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from medqa_rag.core.exceptions import DataError
from medqa_rag.core.types import Chunk
from medqa_rag.observability.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_textbooks(directory: str | Path) -> Iterator[Chunk]:
    """Yield one whole-file Chunk per textbook (pre-chunking).

    The chunker is applied downstream.
    """
    d = Path(directory)
    if not d.exists():
        raise DataError(f"Textbook directory does not exist: {d}")

    files = [f for f in sorted(d.rglob("*")) if f.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        logger.warning("no_textbooks_found", directory=str(d))
        return

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="latin-1")

        if not text.strip():
            continue

        yield Chunk(
            id=f"book::{f.stem}",
            text=text,
            source=str(f.relative_to(d)),
            metadata={"file": f.name, "size": len(text)},
        )

    logger.info("textbooks_loaded", n=len(files), directory=str(d))
