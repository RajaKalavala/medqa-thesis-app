"""LIME-style passage attribution.

We treat each retrieved passage as a binary feature (kept / dropped) and
perturb its inclusion in the prompt, then re-call the LLM. The local linear
surrogate's coefficients become the per-passage attribution.
"""
from __future__ import annotations

import asyncio
from itertools import compress

import numpy as np

from medqa_rag.core.config import get_settings
from medqa_rag.core.types import Question, RAGOutput
from medqa_rag.explainability.base import Attribution
from medqa_rag.llm.groq_client import GroqClient
from medqa_rag.observability.logger import get_logger
from medqa_rag.rags.base import RAGPipeline

logger = get_logger(__name__)


class LimeExplainer:
    name = "lime"

    def __init__(self, pipeline: RAGPipeline, llm: GroqClient | None = None) -> None:
        self.pipeline = pipeline
        self.llm = llm or pipeline.llm
        self.num_samples = get_settings().explainability.lime_num_samples

    # ------------------------------------------------------------------
    async def _score_with_mask(
        self, question: Question, output: RAGOutput, mask: list[bool]
    ) -> int:
        """Re-run the generator with only mask-kept passages → binary score
        (1 if the predicted letter matches the original output).
        """
        kept = list(compress(output.retrieved_docs, mask))
        context = self.pipeline.format_context(kept)

        # Reconstruct a generic QA prompt (architecture-agnostic).
        prompt = (
            "Use only the evidence below to answer the question.\n\n"
            f"Evidence ({len(kept)} passages):\n---\n{context}\n---\n\n"
            f"{self.pipeline.format_question(question)}\n\n"
            "Reply with one of A, B, C, D after 'Answer:'."
        )
        response = await self.llm.complete(
            system="You are a careful medical QA assistant.",
            user=prompt,
            max_tokens=64,
        )
        letter = self.pipeline.parse_letter(response.text)
        return int(letter == output.predicted_letter)

    # ------------------------------------------------------------------
    async def explain(self, question: Question, output: RAGOutput) -> Attribution:
        n = len(output.retrieved_docs)
        if n == 0:
            return Attribution(
                question_id=question.id,
                architecture=str(output.architecture),
                method=self.name,
                passage_scores=[],
                explanation_target=output.predicted_letter or "",
            )

        rng = np.random.default_rng(seed=42)
        masks = rng.integers(0, 2, size=(self.num_samples, n)).astype(bool)
        # Always include "all on" and "all off"
        masks[0] = True
        masks[1] = False

        coros = [self._score_with_mask(question, output, m.tolist()) for m in masks]
        labels = await asyncio.gather(*coros)

        try:
            from sklearn.linear_model import LogisticRegression

            X = masks.astype(int)
            y = np.asarray(labels, dtype=int)
            if len(set(y)) < 2:
                # No variation — uniform attribution
                weights = np.zeros(n)
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                weights = model.coef_[0]
        except ImportError:  # pragma: no cover
            weights = np.mean(masks * np.asarray(labels)[:, None], axis=0)

        return Attribution(
            question_id=question.id,
            architecture=str(output.architecture),
            method=self.name,
            passage_scores=weights.tolist(),
            explanation_target=output.predicted_letter or "",
        )
