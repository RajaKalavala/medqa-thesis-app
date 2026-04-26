"""Medical-terminology normalization (lightweight)."""
from __future__ import annotations

# A small map of common abbreviations -> canonical form.
# Extend as needed; for full coverage plug in UMLS/MetaMap.
_ABBREVIATIONS: dict[str, str] = {
    "MI": "myocardial infarction",
    "CHF": "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "DM": "diabetes mellitus",
    "HTN": "hypertension",
    "CABG": "coronary artery bypass graft",
    "PE": "pulmonary embolism",
    "CKD": "chronic kidney disease",
}


def normalize_terminology(text: str) -> str:
    """Expand a small set of common medical abbreviations.

    Conservative — only expands tokens that match exactly so we don't
    accidentally replace inside larger words.
    """
    if not text:
        return ""

    tokens = text.split()
    out = [_ABBREVIATIONS.get(t.strip(",.;:"), t) for t in tokens]
    return " ".join(out)
