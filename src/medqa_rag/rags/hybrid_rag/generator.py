"""Hybrid RAG generator (same shape as Naive)."""
from __future__ import annotations

from jinja2 import Template

from medqa_rag.core.types import RetrievedDoc
from medqa_rag.llm.groq_client import GroqClient, LLMResponse


async def generate_hybrid_answer(
    llm: GroqClient,
    template: Template,
    question_block: str,
    docs: list[RetrievedDoc],
    context_text: str,
) -> LLMResponse:
    user = template.render(question_block=question_block, context=context_text, n_docs=len(docs))
    return await llm.complete(
        system="You are a careful medical QA assistant. Use only the provided evidence.",
        user=user,
    )
