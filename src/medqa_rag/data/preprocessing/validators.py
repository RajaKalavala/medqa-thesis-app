"""Validators for loaded MedQA questions."""
from __future__ import annotations

from medqa_rag.core.types import Question


def validate_question(q: Question) -> list[str]:
    """Return a list of validation errors (empty == valid)."""
    errors: list[str] = []
    if not q.stem.strip():
        errors.append("empty stem")
    if len(q.options) != 4:
        errors.append(f"expected 4 options, got {len(q.options)}")
    for letter in ("A", "B", "C", "D"):
        if letter not in q.options or not q.options[letter].strip():
            errors.append(f"option {letter} missing or empty")
    if not 0 <= q.correct_index <= 3:
        errors.append(f"correct_index {q.correct_index} out of range")
    return errors
