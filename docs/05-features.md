# 05 — Feature Catalog

What's actually built.

## Cross-cutting features

| Feature | Where | Notes |
|---|---|---|
| Single-YAML config | [config/settings.yaml](../config/settings.yaml) + [core/config.py](../src/medqa_rag/core/config.py) | Env override via `MEDQA_<SECTION>__<KEY>` |
| Structured JSON logging | [observability/logger.py](../src/medqa_rag/observability/logger.py) | structlog + contextvars; bound `request_id` / `architecture` / `experiment_id` |
| MLflow experiment tracking | [observability/mlflow_tracker.py](../src/medqa_rag/observability/mlflow_tracker.py) | One run per evaluation |
| Async Groq client | [llm/groq_client.py](../src/medqa_rag/llm/groq_client.py) | Token-bucket RPM limiter + tenacity retry + sha256-keyed disk cache |
| Embedding cache | [embeddings/cache.py](../src/medqa_rag/embeddings/cache.py) | One `.npy` per text-hash; survives restarts |
| Deterministic seeds | [utils/seeds.py](../src/medqa_rag/utils/seeds.py) | Python / NumPy / Torch |
| Bounded async fan-out | [utils/async_utils.py](../src/medqa_rag/utils/async_utils.py) | `gather_with_concurrency(n, ...)` |
| Per-question timing | [utils/timing.py](../src/medqa_rag/utils/timing.py) | `Timer` cm + `@timed` decorator |

## RAG feature matrix

|  | Naive | Self-RAG | Hybrid | Multi-Hop |
|---|---|---|---|---|
| Dense FAISS retrieval | ✅ | ✅ (gated) | ✅ | ✅ |
| Sparse BM25 retrieval | — | — | ✅ | — |
| RRF fusion | — | — | ✅ | — |
| Confidence gate (cheap judge call) | — | ✅ | — | — |
| Query decomposition | — | — | — | ✅ |
| Iterative retrieval | — | — | — | ✅ |
| Evidence-chain aggregation | — | — | — | ✅ |
| Citation in prompt (`[1]`, `[2]`) | — | — | — | ✅ |
| Self-contained folder | ✅ | ✅ | ✅ | ✅ |
| Per-RAG unit tests | ✅ | ✅ | ✅ | ✅ |
| Per-RAG README | ✅ | ✅ | ✅ | ✅ |

All four expose the same `RAGPipeline.answer(Question) -> RAGOutput` contract.

## Data layer

| Feature | Where |
|---|---|
| MedQA loader (handles both common JSONL schemas) | [data/loaders/medqa_loader.py](../src/medqa_rag/data/loaders/medqa_loader.py) |
| Streaming JSONL loader | same — `stream_medqa()` |
| Duplicate-id deduplication | same — `seen_ids` |
| 18-textbook directory loader | [data/loaders/textbook_loader.py](../src/medqa_rag/data/loaders/textbook_loader.py) |
| Whitespace + page-number cleaner | [data/preprocessing/cleaners.py](../src/medqa_rag/data/preprocessing/cleaners.py) |
| Medical abbreviation normalizer | [data/preprocessing/normalizers.py](../src/medqa_rag/data/preprocessing/normalizers.py) |
| Schema validator for `Question` | [data/preprocessing/validators.py](../src/medqa_rag/data/preprocessing/validators.py) |
| Recursive character chunker | [data/chunking/recursive.py](../src/medqa_rag/data/chunking/recursive.py) |

## Retrieval

| Feature | Where |
|---|---|
| FAISS IndexFlatIP (cosine on normalised vectors) | [retrieval/dense_faiss.py](../src/medqa_rag/retrieval/dense_faiss.py) |
| FAISS save / load with metadata + parallel docstore | same |
| BM25Okapi with configurable k1 / b | [retrieval/sparse_bm25.py](../src/medqa_rag/retrieval/sparse_bm25.py) |
| Reciprocal Rank Fusion | [rags/hybrid_rag/fusion.py](../src/medqa_rag/rags/hybrid_rag/fusion.py) |
| Round-robin dedupe aggregator | [rags/multihop_rag/chain_aggregator.py](../src/medqa_rag/rags/multihop_rag/chain_aggregator.py) |

## Evaluation

| Feature | Where |
|---|---|
| 5 RAGAS metrics | [evaluation/ragas_evaluator.py](../src/medqa_rag/evaluation/ragas_evaluator.py) |
| Per-question RAGAS | same — `evaluate_per_question()` |
| Accuracy / F1 / token cost | [evaluation/non_llm_metrics.py](../src/medqa_rag/evaluation/non_llm_metrics.py) |
| Latency p50/p95/p99 | same — `latency_summary()` |
| 3-layer hallucination detector | [evaluation/hallucination_detector.py](../src/medqa_rag/evaluation/hallucination_detector.py) |
| McNemar (paired pairwise) | [evaluation/statistical_tests.py](../src/medqa_rag/evaluation/statistical_tests.py) |
| Cochran's Q (across all four) | same |
| Markdown report renderer | [evaluation/reporters/markdown_reporter.py](../src/medqa_rag/evaluation/reporters/markdown_reporter.py) |
| LaTeX-table renderer (thesis-ready) | [evaluation/reporters/latex_reporter.py](../src/medqa_rag/evaluation/reporters/latex_reporter.py) |

## Explainability

| Feature | Where |
|---|---|
| LIME via passage-mask perturbation + logistic surrogate | [explainability/lime_explainer.py](../src/medqa_rag/explainability/lime_explainer.py) |
| Monte-Carlo Shapley over passage subsets | [explainability/shap_explainer.py](../src/medqa_rag/explainability/shap_explainer.py) |
| Stratified sampler (subject) | [explainability/sampler.py](../src/medqa_rag/explainability/sampler.py) |
| `Attribution` Pydantic schema | [explainability/base.py](../src/medqa_rag/explainability/base.py) |

## API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/healthz` | GET | Liveness |
| `/readyz` | GET | Readiness |
| `/v1/qa/{architecture}` | POST | Answer one MCQ with chosen RAG |
| `/v1/evaluate` | POST | Run an architecture on the canned test set |
| `/v1/explain` | POST | LIME / SHAP for one (question, architecture) |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |
| `/openapi.json` | GET | Raw OpenAPI 3.1 schema |

All endpoints emit `X-Request-ID` and accept incoming `X-Request-ID`.
All errors map to a uniform [ErrorResponse](../src/medqa_rag/api/schemas/errors.py).

## Orchestration

| Feature | Where |
|---|---|
| Build FAISS + BM25 from textbooks | [pipelines/ingestion_pipeline.py](../src/medqa_rag/pipelines/ingestion_pipeline.py) → `make index` |
| Run one architecture, dump per-Q outputs | [pipelines/evaluation_pipeline.py](../src/medqa_rag/pipelines/evaluation_pipeline.py) → `make run-naive` etc. |
| Run all four + paired stats + report | [pipelines/comparison_pipeline.py](../src/medqa_rag/pipelines/comparison_pipeline.py) → `make run-all` |
| Re-render reports from saved metrics | [scripts/evaluate_results.py](../scripts/evaluate_results.py) |
| Emit thesis-ready LaTeX | [scripts/generate_thesis_tables.py](../scripts/generate_thesis_tables.py) |

## Quality features

| Feature | Where |
|---|---|
| PEP 8 + import-order linting | ruff in [pyproject.toml](../pyproject.toml) |
| Auto-format | black + ruff-format |
| Static typing | mypy `--strict` |
| Pre-commit hooks | [.pre-commit-config.yaml](../.pre-commit-config.yaml) |
| Test markers (`unit` / `integration` / `e2e`) | pyproject |
| Coverage reporting | pytest-cov, term + xml |
| CI gate | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
| Containerised runtime | [deployment/Dockerfile](../deployment/Dockerfile) |
| Compose stack | [deployment/docker-compose.yml](../deployment/docker-compose.yml) |
