"""Confidence gate: ask the model how confident it is BEFORE retrieving."""
from __future__ import annotations

import re

from jinja2 import Template

from medqa_rag.llm.groq_client import GroqClient


_CONF_RE = re.compile(r"confidence\s*[:=]\s*([0-1]?\.\d+|0|1)", flags=re.IGNORECASE)


async def estimate_confidence(
    llm: GroqClient,
    template: Template,
    question_block: str,
    judge_model: str,
) -> tuple[float, str]:
    """Run a cheap judge call. Returns (confidence in [0,1], raw text)."""
    user = template.render(question_block=question_block)
    response = await llm.complete(
        system="You self-assess your medical knowledge. Output a single number 0..1.",
        user=user,
        model=judge_model,
        max_tokens=64,
    )
    m = _CONF_RE.search(response.text)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val)), response.text
        except ValueError:
            pass
    return 0.5, response.text  # safe default → triggers retrieval
