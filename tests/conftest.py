"""Shared test fixtures."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from medqa_rag.core.types import Chunk, Question


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_question() -> Question:
    return Question(
        id="q1",
        stem="A 45-year-old male presents with sudden-onset chest pain and dyspnea.",
        options={
            "A": "Myocardial infarction",
            "B": "Pulmonary embolism",
            "C": "Aortic dissection",
            "D": "Pneumonia",
        },
        correct_index=1,
        subject="cardiology",
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="c1",
            text="Pulmonary embolism causes sudden-onset chest pain and dyspnea.",
            source="harrison",
        ),
        Chunk(
            id="c2",
            text="Myocardial infarction presents with crushing substernal chest pain.",
            source="cecil",
        ),
        Chunk(
            id="c3",
            text="Aortic dissection often presents with tearing chest pain radiating to the back.",
            source="harrison",
        ),
    ]


@pytest.fixture
def env_with_groq(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide a fake Groq API key for tests that touch the GroqClient layer."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    yield
