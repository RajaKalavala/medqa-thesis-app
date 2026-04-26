"""Hallucination detector tests."""
from __future__ import annotations

import pytest

from medqa_rag.core.types import Architecture, Chunk, RAGOutput, RetrievedDoc
from medqa_rag.evaluation.hallucination_detector import HallucinationDetector


def _output_with(text: str, n_docs: int) -> RAGOutput:
    docs = [
        RetrievedDoc(
            chunk=Chunk(id=f"c{i}", text="x", source="s"),
            score=1.0,
            rank=i,
            retriever="dense",
        )
        for i in range(n_docs)
    ]
    return RAGOutput(
        question_id="q",
        architecture=Architecture.NAIVE,
        retrieved_docs=docs,
        generated_answer=text,
        predicted_letter="A",
        predicted_index=0,
        latency_ms=1.0,
    )


@pytest.mark.unit
def test_invalid_citation_flagged():
    out = _output_with("As stated in [1] and [4], the answer is A.", n_docs=2)
    flags = HallucinationDetector().evaluate(out)
    assert flags.invalid_citations == 1
    assert flags.is_flagged is True


@pytest.mark.unit
def test_high_certainty_no_evidence_flagged():
    out = _output_with("The answer is definitely A.", n_docs=0)
    flags = HallucinationDetector().evaluate(out)
    assert flags.high_certainty_no_evidence is True
    assert flags.is_flagged is True


@pytest.mark.unit
def test_low_faithfulness_flagged():
    out = _output_with("Answer: A", n_docs=2)
    flags = HallucinationDetector(faithfulness_threshold=0.7).evaluate(out, faithfulness=0.4)
    assert flags.faithfulness_below == 0.4
    assert flags.is_flagged is True


@pytest.mark.unit
def test_clean_answer_not_flagged():
    out = _output_with("Reasoning grounded in [1].\nAnswer: A", n_docs=2)
    flags = HallucinationDetector().evaluate(out, faithfulness=0.95)
    assert flags.is_flagged is False
