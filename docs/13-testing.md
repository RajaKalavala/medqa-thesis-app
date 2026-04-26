# 13 — Testing Strategy

A three-tier pyramid: unit tests on every PR, integration nightly, e2e on demand.

| Tier | Marker | Scope | External calls | When |
|---|---|---|---|---|
| Unit | `unit` | One module, one behavior | None — everything mocked | Every PR / every commit |
| Integration | `integration` | Two or more modules wired together | Mocked Groq + fixture indices | Nightly / pre-merge |
| E2E | `e2e` | Full pipeline against real dependencies | **Real Groq calls** | On demand; gated by `GROQ_API_KEY` env |

Markers are declared in [pyproject.toml](../pyproject.toml) and enforced with `--strict-markers`.

---

## Layout

```
tests/
├── conftest.py                    # shared fixtures
├── unit/
│   ├── test_config.py             # YAML + env override
│   ├── test_chunking.py           # recursive chunker
│   ├── test_medqa_loader.py       # both schemas + dedupe
│   ├── test_statistical_tests.py  # McNemar + Cochran's Q
│   ├── test_hallucination_detector.py
│   └── test_sampler.py
├── integration/
│   └── test_api_endpoints.py      # FastAPI TestClient
├── e2e/
│   └── test_full_comparison.py    # real Groq call
└── fixtures/
    └── sample_questions.jsonl     # 3 hand-curated MedQA-style rows
```

Per-RAG tests live next to the code they cover:

```
src/medqa_rag/rags/<name>/tests/
    └── test_<aspect>.py
```

These are picked up because `pyproject.toml` declares:

```toml
testpaths = ["tests", "src/medqa_rag/rags"]
```

---

## Running tests

```bash
make test            # everything (with coverage report)
make test-unit       # unit only — fast, ~ a second
make test-integration
make test-e2e        # requires GROQ_API_KEY
```

Direct pytest invocations:

```bash
pytest -m unit
pytest -m "unit and not slow"
pytest -m "integration or e2e"
pytest src/medqa_rag/rags/hybrid_rag/tests/   # one folder
pytest -k mcnemar                              # by keyword
pytest -x                                      # stop on first failure
pytest --lf                                    # last failed only
```

---

## Coverage

Configured in [pyproject.toml](../pyproject.toml):

```toml
addopts = "--cov=src/medqa_rag --cov-report=term-missing --cov-report=xml"
```

Branch coverage on; `coverage.xml` generated for CI integration.

**Target**: ≥ 80 % on `src/medqa_rag/`. Lower is acceptable for the
explainability module specifically (XAI logic is partially data-dependent and
expensive to fully cover; e2e + manual inspection complements it).

---

## What each tier should test

### Unit
- Single-function purity: same input → same output.
- Mock external collaborators (FAISS, Groq, embedder).
- Cover branches (e.g. Self-RAG: `confidence ≥ threshold` vs `<`).
- Edge cases: empty input, malformed input, large input.

### Integration
- HTTP API endpoints up to (but not including) live Groq.
- Pipelines wired with fixture indices and a fake LLM client.
- JSON schema round-trips.

### E2E
- Real Groq call on 1–5 fixture questions.
- Exists primarily to detect drift in the Groq SDK / model behavior.
- Skipped automatically if `GROQ_API_KEY` is missing — CI never accidentally pays.

---

## Fixtures

`tests/conftest.py` provides:

| Fixture | Returns |
|---|---|
| `fixtures_dir` | `Path` to `tests/fixtures/` |
| `sample_question` | a single `Question` for fast tests |
| `sample_chunks` | three `Chunk`s simulating retrieval candidates |
| `env_with_groq` | sets a fake `GROQ_API_KEY` for unit tests that touch GroqClient init |

Per-RAG fixtures are declared inline in their own test files (no cross-pollution).

---

## Mocking patterns we use

### Mock the LLM

```python
from unittest.mock import AsyncMock, MagicMock
from medqa_rag.llm.groq_client import LLMResponse

llm = MagicMock()
llm.complete = AsyncMock(return_value=LLMResponse(text="Answer: B", model="m"))
```

### Mock the retriever

```python
faiss = MagicMock()
faiss.retrieve.return_value = [RetrievedDoc(...), ...]
```

### Side-effects for multi-call paths

Self-RAG calls Groq twice (gate, then answer); use `side_effect=[...]`:

```python
llm.complete = AsyncMock(side_effect=[
    LLMResponse(text="Confidence: 0.92", model="judge"),
    LLMResponse(text="Answer: A", model="main"),
])
```

---

## CI integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs:

1. `ruff check`
2. `mypy --strict`
3. `pytest -m unit`

E2E jobs are not in CI by default. To add them, gate on a `GROQ_API_KEY`
secret and a separate workflow trigger (manual dispatch or scheduled).

---

## Test-writing rules of thumb

- One assertion per concept; multiple `assert` calls in one test are fine when they verify the same behavior.
- Test names start with `test_` and read like English: `test_low_confidence_triggers_retrieval`.
- `pytest.mark.unit` (or whichever) on every test — `--strict-markers` will fail otherwise.
- Don't import from `conftest.py` directly — pytest injects it.
- Use `tmp_path` for any FS state; never write into the actual `data/` or `results/`.

---

## Adding tests when you add features

| You added… | Add a test in… |
|---|---|
| A new chunker | `tests/unit/test_chunking.py` |
| A new RAG | `src/medqa_rag/rags/<name>/tests/` |
| A new metric | `tests/unit/test_<metric>.py` |
| A new API endpoint | `tests/integration/test_api_endpoints.py` |
| A new exception | usually no test — caught by static + integration |
| A new prompt template | render-only test under the RAG's tests folder if behavior depends on Jinja branching |
