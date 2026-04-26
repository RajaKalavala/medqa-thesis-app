"""Text cleaning utilities for medical corpora."""
from __future__ import annotations

import re

# Common artifacts in textbook OCR / extraction
_MULTI_SPACE = re.compile(r"\s+")
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$", flags=re.MULTILINE)
_REPEATED_PUNCT = re.compile(r"([.\-_])\1{3,}")
_FORM_FEED = re.compile(r"[\f\v]")


def clean_medical_text(text: str) -> str:
    """Basic textbook-corpus cleaning.

    - Collapse whitespace
    - Strip page-number lines
    - Normalize repeated punctuation runs
    - Normalize control chars
    """
    if not text:
        return ""
    text = _FORM_FEED.sub("\n", text)
    text = _PAGE_NUMBER.sub("", text)
    text = _REPEATED_PUNCT.sub(r"\1\1\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()
