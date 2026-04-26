"""Three-layer hallucination control.

Layer 1: prompt-level grounding (handled inside each RAG's prompt template).
Layer 2: faithfulness scoring (RAGAS).
Layer 3: post-hoc rule-based flagging — answers that cite passages absent
         from the retrieved context, or claim certainty without evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from medqa_rag.core.types import RAGOutput

_CITATION_RE = re.compile(r"\[(\d+)\]")
_CERTAINTY_RE = re.compile(
    r"\b(definitely|certainly|always|never|clearly|obviously)\b", flags=re.IGNORECASE
)


@dataclass
class HallucinationFlags:
    """Per-question hallucination signals."""

    invalid_citations: int     # citation indices outside [1..n_docs]
    high_certainty_no_evidence: bool  # certainty words but zero retrieved docs
    faithfulness_below: float | None  # if RAGAS faithfulness was provided
    is_flagged: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "invalid_citations": self.invalid_citations,
            "high_certainty_no_evidence": self.high_certainty_no_evidence,
            "faithfulness_below": self.faithfulness_below,
            "is_flagged": self.is_flagged,
        }


class HallucinationDetector:
    def __init__(self, faithfulness_threshold: float = 0.7) -> None:
        self.faithfulness_threshold = faithfulness_threshold

    def evaluate(
        self,
        output: RAGOutput,
        faithfulness: float | None = None,
    ) -> HallucinationFlags:
        n_docs = len(output.retrieved_docs)
        text = output.generated_answer

        cited = [int(m) for m in _CITATION_RE.findall(text)]
        invalid = sum(1 for c in cited if c < 1 or c > n_docs)

        certainty_no_evidence = bool(_CERTAINTY_RE.search(text)) and n_docs == 0

        below = (
            faithfulness if (faithfulness is not None and faithfulness < self.faithfulness_threshold) else None
        )

        flagged = invalid > 0 or certainty_no_evidence or below is not None
        return HallucinationFlags(
            invalid_citations=invalid,
            high_certainty_no_evidence=certainty_no_evidence,
            faithfulness_below=below,
            is_flagged=flagged,
        )

    def evaluate_batch(
        self,
        outputs: list[RAGOutput],
        faithfulness_per_q: dict[str, float] | None = None,
    ) -> dict[str, HallucinationFlags]:
        faithfulness_per_q = faithfulness_per_q or {}
        return {
            o.question_id: self.evaluate(o, faithfulness_per_q.get(o.question_id))
            for o in outputs
        }
