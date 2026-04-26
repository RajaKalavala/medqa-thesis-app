"""Self-RAG generator (one of two prompts: with or without retrieved evidence)."""
from __future__ import annotations

from jinja2 import Template

from medqa_rag.core.types import RetrievedDoc
from medqa_rag.llm.groq_client import GroqClient, LLMResponse


async def generate_self_rag_answer(
    llm: GroqClient,
    template: Template,
    question_block: str,
    docs: list[RetrievedDoc] | None,
    context_text: str | None,
) -> LLMResponse:
    user = template.render(
        question_block=question_block,
        context=context_text,
        n_docs=0 if docs is None else len(docs),
        retrieval_used=docs is not None,
    )
    return await llm.complete(
        system="You are a careful medical QA assistant.",
        user=user,
    )
