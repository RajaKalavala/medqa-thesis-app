"""Decompose a clinical question into 1..N focused sub-queries."""
from __future__ import annotations

import re

from jinja2 import Template

from medqa_rag.core.config import get_settings
from medqa_rag.llm.groq_client import GroqClient


_LINE_RE = re.compile(r"^\s*(?:\d+\.\s*|\-\s*|\*\s*)?(.+?)\s*$")


async def decompose_question(
    llm: GroqClient,
    template: Template,
    question_block: str,
    max_subqueries: int,
) -> list[str]:
    """Ask the small judge model for a list of sub-queries."""
    user = template.render(question_block=question_block, n=max_subqueries)
    response = await llm.complete(
        system="You decompose complex medical questions into focused sub-queries.",
        user=user,
        model=get_settings().llm.judge_model,
        max_tokens=256,
    )
    out: list[str] = []
    for line in response.text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        text = m.group(1).strip().strip('"').strip("'")
        if not text or text.lower().startswith(("answer", "reasoning", "note")):
            continue
        out.append(text)
        if len(out) >= max_subqueries:
            break
    return out
