"""Settings loading tests."""
from __future__ import annotations

import pytest

from medqa_rag.core.config import reload_settings


@pytest.mark.unit
def test_settings_load_yaml_defaults():
    s = reload_settings()
    assert s.llm.model.startswith("llama-")
    assert s.embedder.model_name
    assert s.retrieval.top_k > 0


@pytest.mark.unit
def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("MEDQA_RETRIEVAL__TOP_K", "11")
    s = reload_settings()
    assert s.retrieval.top_k == 11
