# 04 — Folder Structure

Annotated tree. Anything not annotated is a stock convention (e.g. `__init__.py`).

```
medqa-rag-comparison/
│
├── .github/workflows/
│   └── ci.yml                              # Ruff + mypy + unit tests on PR
│
├── .pre-commit-config.yaml                 # black, ruff, mypy hooks
├── .env.example                            # GROQ_API_KEY + path overrides
├── .gitignore
├── pyproject.toml                          # PEP 621 — single source of truth
├── Makefile                                # make install/lint/test/api/index/run-*
├── README.md                               # quickstart
│
├── config/
│   └── settings.yaml                       # The ONE config file
│
├── docs/                                   # ← you are here
│   ├── README.md                           # Index of docs
│   ├── 01-overview.md                      # …
│   ├── architecture/
│   │   └── adr/                            # Architecture Decision Records
│   └── experiments/                        # Run logs, notes
│
├── data/                                   # gitignored except .gitkeep
│   ├── raw/
│   │   ├── medqa/                          # MedQA *.jsonl drops here
│   │   └── textbooks/                      # 18 textbook .txt files
│   ├── interim/
│   ├── processed/
│   ├── embeddings/                         # cached vectors (.npy per text)
│   └── indices/
│       ├── faiss/                          # built by scripts/build_index.py
│       └── bm25/
│
├── notebooks/                              # exploration only — not prod
│
├── src/medqa_rag/
│   │
│   ├── __version__.py                      # Single string literal
│   │
│   ├── core/                               # Cross-cutting; no I/O
│   │   ├── config.py                       # Pydantic Settings (loads YAML+env)
│   │   ├── types.py                        # Question, Chunk, RAGOutput, …
│   │   ├── exceptions.py                   # MedQARAGError + subclasses
│   │   └── constants.py                    # LETTERS, MAX_HOPS, RRF_K, …
│   │
│   ├── observability/                      # Imported by everything
│   │   ├── logger.py                       # structlog JSON config
│   │   └── mlflow_tracker.py               # mlflow_run() context manager
│   │
│   ├── utils/
│   │   ├── timing.py                       # @timed decorator + Timer cm
│   │   ├── seeds.py                        # set_global_seed()
│   │   ├── io.py                           # read_jsonl / write_jsonl / ensure_dir
│   │   └── async_utils.py                  # gather_with_concurrency
│   │
│   ├── data/                               # SHARED across all 4 RAGs
│   │   ├── loaders/
│   │   │   ├── medqa_loader.py             # parses both common JSONL schemas
│   │   │   └── textbook_loader.py
│   │   ├── preprocessing/
│   │   │   ├── cleaners.py                 # whitespace, page-numbers
│   │   │   ├── normalizers.py              # MI → myocardial infarction, …
│   │   │   └── validators.py
│   │   └── chunking/
│   │       ├── base.py                     # Chunker protocol
│   │       ├── recursive.py                # langchain RecursiveCharacterTextSplitter
│   │       └── factory.py
│   │
│   ├── embeddings/                         # SHARED
│   │   ├── base.py                         # Embedder protocol
│   │   ├── huggingface_embedder.py         # PubMedBERT, lazy-loaded
│   │   ├── cache.py                        # sha256-keyed disk cache
│   │   └── factory.py
│   │
│   ├── llm/                                # Groq only — every LLM call goes here
│   │   ├── groq_client.py                  # Async, retried, rate-limited, cached
│   │   ├── rate_limiter.py                 # Token bucket
│   │   └── cache.py                        # Request-keyed JSON cache
│   │
│   ├── retrieval/                          # SHARED low-level retrievers
│   │   ├── base.py                         # Retriever protocol
│   │   ├── dense_faiss.py                  # IndexFlatIP + docstore
│   │   └── sparse_bm25.py                  # rank-bm25
│   │
│   ├── rags/                               # ★ The four self-contained RAG modules ★
│   │   ├── base.py                         # RAGPipeline ABC + helpers
│   │   ├── factory.py                      # build_rag(arch) → RAGPipeline
│   │   │
│   │   ├── naive_rag/
│   │   │   ├── config.py                   # NaiveRAGConfig
│   │   │   ├── pipeline.py                 # NaiveRAGPipeline.answer()
│   │   │   ├── retriever.py                # dense FAISS only
│   │   │   ├── generator.py
│   │   │   ├── prompts/naive_qa.jinja2
│   │   │   ├── README.md
│   │   │   └── tests/test_pipeline.py
│   │   │
│   │   ├── self_rag/
│   │   │   ├── config.py
│   │   │   ├── pipeline.py
│   │   │   ├── confidence_gate.py          # cheap judge call
│   │   │   ├── retriever.py
│   │   │   ├── generator.py
│   │   │   ├── prompts/
│   │   │   │   ├── confidence_check.jinja2
│   │   │   │   └── self_rag_qa.jinja2
│   │   │   ├── README.md
│   │   │   └── tests/test_pipeline.py
│   │   │
│   │   ├── hybrid_rag/
│   │   │   ├── config.py
│   │   │   ├── pipeline.py
│   │   │   ├── retriever.py                # dense + sparse fused
│   │   │   ├── fusion.py                   # Reciprocal Rank Fusion
│   │   │   ├── generator.py
│   │   │   ├── prompts/hybrid_qa.jinja2
│   │   │   ├── README.md
│   │   │   └── tests/test_fusion.py
│   │   │
│   │   └── multihop_rag/
│   │       ├── config.py
│   │       ├── pipeline.py
│   │       ├── decomposer.py               # judge → sub-queries
│   │       ├── iterative_retriever.py      # one FAISS pass per hop
│   │       ├── chain_aggregator.py         # round-robin dedupe
│   │       ├── generator.py
│   │       ├── prompts/
│   │       │   ├── decompose.jinja2
│   │       │   └── final_answer.jinja2
│   │       ├── README.md
│   │       └── tests/test_aggregator.py
│   │
│   ├── evaluation/
│   │   ├── ragas_evaluator.py              # 5 RAGAS metrics
│   │   ├── non_llm_metrics.py              # accuracy, latency, tokens
│   │   ├── hallucination_detector.py       # 3-layer flagging
│   │   ├── statistical_tests.py            # McNemar + Cochran's Q
│   │   └── reporters/
│   │       ├── markdown_reporter.py
│   │       └── latex_reporter.py
│   │
│   ├── explainability/
│   │   ├── base.py                         # Explainer protocol + Attribution
│   │   ├── lime_explainer.py               # passage-mask perturbation
│   │   ├── shap_explainer.py               # Monte-Carlo Shapley
│   │   └── sampler.py                      # stratified sub-sampler
│   │
│   ├── api/
│   │   ├── main.py                         # FastAPI app factory
│   │   ├── lifespan.py
│   │   ├── dependencies.py                 # cached get_rag()
│   │   ├── middleware/
│   │   │   ├── logging.py                  # request_id binding
│   │   │   └── error_handler.py            # domain → HTTP envelope
│   │   ├── routers/
│   │   │   ├── health.py                   # /healthz, /readyz
│   │   │   ├── qa.py                       # POST /v1/qa/{architecture}
│   │   │   ├── evaluation.py               # POST /v1/evaluate
│   │   │   └── explainability.py           # POST /v1/explain
│   │   ├── schemas/
│   │   │   ├── request.py
│   │   │   ├── response.py
│   │   │   └── errors.py
│   │   └── docs/
│   │       └── openapi_tags.py
│   │
│   └── pipelines/                          # Top-level orchestration
│       ├── ingestion_pipeline.py           # textbooks → indices
│       ├── evaluation_pipeline.py          # one architecture, full set
│       └── comparison_pipeline.py          # all four + statistical tests + report
│
├── tests/
│   ├── conftest.py                         # Shared fixtures
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_chunking.py
│   │   ├── test_medqa_loader.py
│   │   ├── test_statistical_tests.py
│   │   ├── test_hallucination_detector.py
│   │   └── test_sampler.py
│   ├── integration/
│   │   └── test_api_endpoints.py
│   ├── e2e/
│   │   └── test_full_comparison.py         # only runs if GROQ_API_KEY is set
│   └── fixtures/
│       └── sample_questions.jsonl
│
├── scripts/
│   ├── build_index.py                      # CLI → ingestion_pipeline
│   ├── run_one_rag.py                      # CLI → evaluation_pipeline
│   ├── run_all_experiments.py              # CLI → comparison_pipeline
│   ├── evaluate_results.py                 # Re-render reports from saved metrics
│   └── generate_thesis_tables.py           # LaTeX from latest comparison
│
├── deployment/
│   ├── Dockerfile                          # python:3.11-slim + uvicorn
│   ├── docker-compose.yml                  # API + bind-mounted data/logs/results
│   └── k8s/                                # (placeholder — to be added)
│
├── logs/                                   # gitignored — structlog JSON
└── results/                                # gitignored
    ├── metrics/                            # one JSON per evaluation run
    ├── reports/                            # comparison_*.{json,md,tex}
    └── mlruns/                             # MLflow file store
```

## Folder responsibility rules

| If you're adding… | Put it in… |
|---|---|
| A new domain type | `core/types.py` |
| A new exception class | `core/exceptions.py` |
| A new chunking strategy | `data/chunking/` and register in `factory.py` |
| A new embedder backend | `embeddings/` implementing `Embedder` protocol |
| A new LLM provider | New module under `llm/`; mirror `groq_client.py` API |
| A 5th RAG | New folder under `rags/<name>/` mirroring an existing one |
| A new metric | `evaluation/non_llm_metrics.py` or wrap in `ragas_evaluator.py` |
| A new XAI method | `explainability/<name>_explainer.py` implementing `Explainer` protocol |
| A new API endpoint | `api/routers/<name>.py` + register in `api/main.py` |
| A new orchestration script | `scripts/` (thin wrapper over `pipelines/`) |
