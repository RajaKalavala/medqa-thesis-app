"""MedQA loader tests covering both common dataset schemas."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from medqa_rag.data.loaders.medqa_loader import load_medqa


@pytest.mark.unit
def test_loader_handles_schema_a(tmp_path: Path):
    rows = [
        {
            "question": "Question 1?",
            "options": {"A": "Apple", "B": "Banana", "C": "Cherry", "D": "Date"},
            "answer_idx": "B",
        }
    ]
    f = tmp_path / "a.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = load_medqa(f)
    assert len(out) == 1
    assert out[0].correct_index == 1
    assert out[0].correct_letter == "B"


@pytest.mark.unit
def test_loader_handles_schema_b(tmp_path: Path):
    rows = [
        {
            "id": "qx",
            "stem": "Question 2?",
            "opa": "A",
            "opb": "B",
            "opc": "C",
            "opd": "D",
            "correct": 2,
        }
    ]
    f = tmp_path / "b.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = load_medqa(f)
    assert out[0].id == "qx"
    assert out[0].correct_index == 2


@pytest.mark.unit
def test_loader_dedups_ids(tmp_path: Path):
    rows = [
        {"id": "dup", "stem": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "correct": 0},
        {"id": "dup", "stem": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "correct": 0},
    ]
    f = tmp_path / "c.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = load_medqa(f)
    assert len(out) == 1
