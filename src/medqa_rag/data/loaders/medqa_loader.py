"""Load MedQA USMLE multiple-choice questions from JSONL.

Expected schema (one of two common formats):
    A) {"question": "...", "options": {"A": "...", ...}, "answer_idx": "B", "meta_info": "..."}
    B) {"id": "...", "stem": "...", "opa": "...", "opb": "...", "opc": "...",
        "opd": "...", "correct": 1, "subject": "..."}
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from medqa_rag.core.constants import LETTER_TO_INDEX, LETTERS
from medqa_rag.core.exceptions import DataError
from medqa_rag.core.types import Question
from medqa_rag.observability.logger import get_logger
from medqa_rag.utils.io import read_jsonl

logger = get_logger(__name__)


def _normalise(row: dict[str, Any], idx: int) -> Question:
    """Map a raw row (either schema A or B) to a Question."""
    qid = str(row.get("id") or row.get("qid") or f"medqa_{idx:06d}")

    # Stem
    stem = row.get("stem") or row.get("question") or row.get("query")
    if not stem:
        raise DataError(f"Row {idx}: missing question stem")

    # Options
    options: dict[str, str]
    if "options" in row and isinstance(row["options"], dict):
        options = {k.upper(): str(v) for k, v in row["options"].items()}
    elif all(k in row for k in ("opa", "opb", "opc", "opd")):
        options = {
            "A": str(row["opa"]),
            "B": str(row["opb"]),
            "C": str(row["opc"]),
            "D": str(row["opd"]),
        }
    else:
        raise DataError(f"Row {idx}: cannot parse options")

    if not all(letter in options for letter in LETTERS):
        raise DataError(f"Row {idx}: missing one of A/B/C/D options")

    # Correct index
    correct: int
    if "correct" in row:
        correct = int(row["correct"])
    elif "answer_idx" in row:
        correct = LETTER_TO_INDEX[str(row["answer_idx"]).upper()]
    elif "answer" in row:
        ans = str(row["answer"]).strip().upper()
        if ans in LETTER_TO_INDEX:
            correct = LETTER_TO_INDEX[ans]
        else:
            raise DataError(f"Row {idx}: cannot parse answer field {ans!r}")
    else:
        raise DataError(f"Row {idx}: no correct-answer field")

    if not 0 <= correct <= 3:
        raise DataError(f"Row {idx}: correct index {correct} out of range")

    subject = row.get("subject") or row.get("meta_info")
    return Question(
        id=qid,
        stem=stem,
        options=options,
        correct_index=correct,
        subject=str(subject) if subject else None,
        metadata={k: v for k, v in row.items() if k not in {"id", "qid", "question", "stem", "opa", "opb", "opc", "opd", "options", "correct", "answer", "answer_idx", "subject", "meta_info"}},
    )


def load_medqa(path: str | Path, limit: int | None = None) -> list[Question]:
    """Load MedQA from a single JSONL file."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"MedQA file not found: {p}")

    questions: list[Question] = []
    seen_ids: set[str] = set()

    for i, row in enumerate(read_jsonl(p)):
        try:
            q = _normalise(row, i)
        except DataError as exc:
            logger.warning("medqa_row_skipped", row=i, error=str(exc))
            continue

        if q.id in seen_ids:
            logger.warning("medqa_duplicate_id_skipped", id=q.id)
            continue
        seen_ids.add(q.id)
        questions.append(q)

        if limit is not None and len(questions) >= limit:
            break

    logger.info("medqa_loaded", n=len(questions), path=str(p))
    return questions


def stream_medqa(path: str | Path) -> Iterator[Question]:
    """Streaming loader for very large files."""
    p = Path(path)
    for i, row in enumerate(read_jsonl(p)):
        try:
            yield _normalise(row, i)
        except DataError:
            continue


def load_medqa_dir(directory: str | Path) -> Iterable[Question]:
    """Load and concatenate every ``*.jsonl`` in a directory."""
    d = Path(directory)
    if not d.is_dir():
        raise DataError(f"Not a directory: {d}")
    for f in sorted(d.glob("*.jsonl")):
        yield from load_medqa(f)
