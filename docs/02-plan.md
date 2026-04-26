# 02 — End-to-End Plan

Eight phases, ~9–10 weeks of focused work. Aligns with the LJMU MSc thesis
window (March 2026 → August 2026).

## Phase map

| # | Phase | Duration | Deliverable |
|---|---|---|---|
| 0 | Project setup | 2 d | Tooling, CI, configs |
| 1 | Data ingestion | 4 d | Cleaned chunks ready for indexing |
| 2 | Shared infrastructure | 4 d | Embedder, Groq client, base retrievers |
| 3 | Four RAG modules | 12 d | All four pipelines passing unit tests |
| 4 | Evaluation framework | 7 d | RAGAS scores + statistical tests |
| 5 | Explainability | 7 d | LIME + SHAP attributions on stratified sample |
| 6 | API + Swagger | 4 d | FastAPI service, Docker image |
| 7 | Testing (continuous) | 3 d (net) | ≥ 80 % coverage on `src/medqa_rag/` |
| 8 | Reporting | 5 d | LaTeX-ready tables, comparison report |

---

## Phase 0 — Setup

**Goal:** A buildable, lintable, testable repo with one config and one CI.

- `pyproject.toml` (PEP 621) → ruff, black, mypy `--strict`, pytest, coverage
- `config/settings.yaml` + `.env.example`
- `core/config.py` — Pydantic Settings with YAML + env override
- `observability/logger.py` — structlog JSON
- `Makefile` → `lint`, `format`, `type`, `test`, `api`, `index`, `run-*`
- `.github/workflows/ci.yml` — ruff + mypy + unit tests on every PR

**Done when:** `make lint && make type && make test-unit` is green.

---

## Phase 1 — Data ingestion

**Goal:** Raw inputs → cleaned, validated, ready-to-chunk content.

- `data/loaders/medqa_loader.py` — parse JSONL, normalise both common schemas
- `data/loaders/textbook_loader.py` — read 18 textbooks
- `data/preprocessing/` — clean, normalise medical terms, validate
- `data/chunking/recursive.py` — recursive 512/64 (start point; ablate later)
- `scripts/build_index.py` — emits FAISS + BM25 to `data/indices/`

**Done when:** `make index` produces non-empty FAISS + BM25 stores.

---

## Phase 2 — Shared infrastructure

**Goal:** All RAGs depend on the same swappable pieces.

- `embeddings/huggingface_embedder.py` — PubMedBERT 768-d, on-disk cache
- `llm/groq_client.py` — async client, token-bucket rate limit, exp-backoff retry, request-keyed cache
- `retrieval/dense_faiss.py` + `retrieval/sparse_bm25.py` — both implement `Retriever` protocol

**Done when:** A query through each retriever returns ranked `RetrievedDoc` objects.

---

## Phase 3 — Four RAG modules

**Goal:** Each architecture passes its own unit tests; the comparison can run end-to-end.

Per-RAG checklist (≈ 3 days each):

- `config.py` — Pydantic config dataclass
- `pipeline.py` — `answer(question) -> RAGOutput`
- `retriever.py` — architecture-specific retrieval logic
- `generator.py` — prompt assembly + LLM call
- `prompts/*.jinja2` — externalised templates
- `tests/test_pipeline.py` — covers happy path + at least one edge case
- `README.md` — explains the design choice

**Done when:** `pytest src/medqa_rag/rags` is fully green.

---

## Phase 4 — Evaluation

**Goal:** Reduce raw outputs to comparable scores + significance tests.

- `evaluation/ragas_evaluator.py` — Faithfulness, Context Precision/Recall, Answer Correctness/Relevancy
- `evaluation/non_llm_metrics.py` — accuracy, F1, p50/p95/p99 latency
- `evaluation/hallucination_detector.py` — three-layer flagging
- `evaluation/statistical_tests.py` — McNemar (pairwise) + Cochran's Q (across four)
- `evaluation/reporters/` — Markdown + LaTeX
- `pipelines/evaluation_pipeline.py` — orchestrates one architecture end-to-end
- MLflow tracking per run

**Done when:** A single architecture produces a JSON metrics record + an MLflow run.

---

## Phase 5 — Explainability

**Goal:** Per-passage attribution for the predicted answer.

- **Stratified sample** ≈ 400 questions across subjects (NOT the full 12,723 — see roadmap)
- `explainability/lime_explainer.py` — passage-mask perturbation + logistic surrogate
- `explainability/shap_explainer.py` — Monte-Carlo Shapley
- `pipelines/comparison_pipeline.py` — joins metrics + XAI attributions

**Done when:** Each architecture produces an `Attribution` record per sampled question.

---

## Phase 6 — API + Swagger

**Goal:** A REST surface that an external reviewer can poke without reading any code.

- FastAPI: `/v1/qa/{architecture}`, `/v1/evaluate`, `/v1/explain`, `/healthz`, `/readyz`
- Pydantic v2 schemas with examples → rich OpenAPI 3.1 docs
- Swagger UI at `/docs`, ReDoc at `/redoc`
- `RequestLoggingMiddleware`, `register_exception_handlers`

**Done when:** `make api` + a curl POST returns a valid `QAResponse`.

---

## Phase 7 — Testing (continuous)

Three tiers, marker-gated:

| Tier | Marker | Runs in CI | Hits Groq |
|---|---|---|---|
| Unit | `unit` | Every PR | No (mocked) |
| Integration | `integration` | Nightly | No (mocked or fixtures) |
| E2E | `e2e` | On demand | Yes |

Coverage gate ≥ 80 % on `src/medqa_rag/`.

---

## Phase 8 — Reporting

- `scripts/generate_thesis_tables.py` — emits LaTeX from latest comparison report
- Plots: per-architecture RAGAS bars, latency CDFs, hallucination rates, p-value matrix
- Final comparison framework table → straight into thesis Chapter 5

---

## Critical-path dependencies

```
Phase 0 ─► Phase 1 ─► Phase 2 ─► Phase 3 ─┬─► Phase 4 ─► Phase 8
                                          └─► Phase 5
Phase 7 (testing) runs alongside 3, 4, 5
Phase 6 (API) can run in parallel with 4-5 once Phase 3 is done
```

## Risk mitigations baked into the plan

| Risk | Mitigation |
|---|---|
| Groq rate limits / cost spikes | Token-bucket limiter + request-keyed disk cache (Phase 2) |
| RAGAS judge cost on 12,723 q × 4 archs | Use cheap `llama-3.1-8b-instant` as judge; cache responses |
| LIME/SHAP cost explosion | Stratified-sample 400 questions instead of full set (Phase 5) |
| Non-determinism | `set_global_seed`, FAISS deterministic, model-snapshot pin |
| Long-running runs lost | Periodic JSON dumps + MLflow checkpoints |
