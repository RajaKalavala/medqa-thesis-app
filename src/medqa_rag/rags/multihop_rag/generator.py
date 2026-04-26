"""Multi-hop final-answer generator."""
from __future__ import annotations

from jinja2 import Template

from medqa_rag.core.types import RetrievedDoc
from medqa_rag.llm.groq_client import GroqClient, LLMResponse


async def generate_multihop_answer(
    llm: GroqClient,
    template: Template,
    question_block: str,
    sub_queries: list[str],
    docs: list[RetrievedDoc],
    context_text: str,
) -> LLMResponse:
    user = template.render(
        question_block=question_block,
        sub_queries=sub_queries,
        context=context_text,
        n_docs=len(docs),
    )
    return await llm.complete(
        system=(
            "You are a careful medical QA assistant. Reason step by step using the "
            "evidence chain below; cite passage indices like [1], [2] in your reasoning."
        ),
        user=user,
    )
