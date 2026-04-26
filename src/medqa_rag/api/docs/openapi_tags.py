"""OpenAPI tags rendered in Swagger UI groups."""
from __future__ import annotations

OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "health", "description": "Liveness and readiness probes."},
    {"name": "qa", "description": "Single-question answering against a chosen RAG architecture."},
    {"name": "evaluation", "description": "Bulk evaluation runs (RAGAS + accuracy + latency)."},
    {"name": "explainability", "description": "LIME / SHAP passage-level attribution."},
]
