"""SHAP-style passage attribution via Kernel-SHAP-like Monte-Carlo Shapley.

For ``n`` passages we draw ``num_samples`` random subsets and estimate marginal
contribution of each passage (1 if including it changes the model's predicted
answer back to the original).  The mean marginal across samples approximates
Shapley values.
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


class ShapExplainer:
    name = "shap"

    def __init__(self, pipeline: RAGPipeline, llm: GroqClient | None = None) -> None:
        self.pipeline = pipeline
        self.llm = llm or pipeline.llm
        self.num_samples = get_settings().explainability.shap_num_samples

    # ------------------------------------------------------------------
    async def _score(self, question: Question, output: RAGOutput, mask: np.ndarray) -> int:
        kept = list(compress(output.retrieved_docs, mask.tolist()))
        context = self.pipeline.format_context(kept)
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

        rng = np.random.default_rng(seed=43)
        contributions = np.zeros(n)
        counts = np.zeros(n)

        for _ in range(self.num_samples):
            order = rng.permutation(n)
            mask = np.zeros(n, dtype=bool)
            prev_score = await self._score(question, output, mask)
            for i in order:
                mask[i] = True
                cur = await self._score(question, output, mask)
                contributions[i] += (cur - prev_score)
                counts[i] += 1
                prev_score = cur

        shapley = contributions / np.maximum(counts, 1)
        return Attribution(
            question_id=question.id,
            architecture=str(output.architecture),
            method=self.name,
            passage_scores=shapley.tolist(),
            explanation_target=output.predicted_letter or "",
        )
