"""Passage-level explainability via LIME and SHAP."""
from medqa_rag.explainability.base import Attribution, Explainer
from medqa_rag.explainability.lime_explainer import LimeExplainer
from medqa_rag.explainability.sampler import stratified_sample
from medqa_rag.explainability.shap_explainer import ShapExplainer

__all__ = [
    "Attribution",
    "Explainer",
    "LimeExplainer",
    "ShapExplainer",
    "stratified_sample",
]
