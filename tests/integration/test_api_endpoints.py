"""FastAPI endpoint smoke tests (with mocked RAG dependency)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_healthz_ok():
    from medqa_rag.api.main import create_app

    client = TestClient(create_app())
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.integration
def test_readyz_ok():
    from medqa_rag.api.main import create_app

    client = TestClient(create_app())
    r = client.get("/readyz")
    assert r.status_code == 200


@pytest.mark.integration
def test_openapi_schema_includes_qa_route():
    from medqa_rag.api.main import create_app

    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    assert any(p.startswith("/v1/qa/") for p in schema["paths"])
