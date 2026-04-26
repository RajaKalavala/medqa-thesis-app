# 03 — System Architecture

## Design principles

1. **Single varying factor.** Across all four architectures, the only thing that
   changes is the retrieval strategy. Same LLM, same embedder, same indices,
   same prompt shape. Enforced *in code* via the [`RAGPipeline`](../src/medqa_rag/rags/base.py) abstract base + dependency injection — not by convention.

2. **Per-RAG self-containment.** Everything specific to a RAG (retrieval logic,
   prompts, hyperparameters, tests) lives in one folder under
   [src/medqa_rag/rags/](../src/medqa_rag/rags/). Anything reused (FAISS, BM25,
   embedder, Groq client, evaluator) lives at the top level.

3. **Uniform output schema.** Every RAG returns a `RAGOutput` (see [types.py](../src/medqa_rag/core/types.py)). Evaluators, explainers, and the API all consume the same shape.

4. **Boundary validation only.** Pydantic at the edges (config, API, dataset
   loaders); plain dataclasses internally. No defensive validation between
   trusted internal modules.

5. **Cost & rate-limit are first-class.** Every Groq call goes through one
   client wrapper that enforces an RPM token-bucket and a request-keyed disk
   cache. There is no "raw" path.

6. **Reproducibility by construction.** YAML + env settings, fixed seeds,
   pinned model snapshots, MLflow per run, deterministic FAISS.

7. **One config, one secret store.** A single `config/settings.yaml` and a
   single `.env`. No per-RAG YAML files.

---

## Layer diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                          API surface                             │
│  FastAPI + Swagger UI  •  /v1/qa  /v1/evaluate  /v1/explain      │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ Pydantic v2
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Orchestration                              │
│  pipelines/  •  scripts/  •  CLIs                                │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
┌────────────────────┐ ┌────────────────────┐ ┌─────────────────────┐
│      RAGs          │ │    Evaluation       │ │   Explainability   │
│  Naive | Self      │ │ RAGAS · stats ·     │ │  LIME · SHAP ·     │
│  Hybrid | MultiHop │ │ hallucination       │ │  stratified sample │
└─────────┬──────────┘ └────────┬────────────┘ └──────────┬─────────┘
          │                     │                         │
          └──────────┬──────────┴────────────┬────────────┘
                     ▼                       ▼
        ┌────────────────────────┐  ┌────────────────────┐
        │  Retrieval (shared)    │  │      LLM           │
        │  FAISS dense  · BM25   │  │  Groq async client │
        │                        │  │  + rate limit + RR │
        └─────────┬──────────────┘  │  + cache           │
                  │                 └─────────┬──────────┘
                  ▼                           │
        ┌────────────────────────┐            │
        │   Embeddings (shared)  │            │
        │   PubMedBERT + cache   │            │
        └─────────┬──────────────┘            │
                  ▼                           ▼
        ┌─────────────────────────────────────────────┐
        │   Data layer  ·  Core (config, types, log)  │
        │   loaders · preprocessing · chunking        │
        │   structlog · MLflow · Pydantic Settings    │
        └─────────────────────────────────────────────┘
```

---

## Module responsibilities

| Module | Responsibility |
|---|---|
| [core/](../src/medqa_rag/core/) | Pydantic Settings, domain types (`Question`, `RAGOutput`), exceptions, constants. No I/O. |
| [observability/](../src/medqa_rag/observability/) | structlog config, MLflow wrapper. Imported by everything. |
| [utils/](../src/medqa_rag/utils/) | Tiny generic helpers — Timer, seed, JSONL, async semaphore. |
| [data/](../src/medqa_rag/data/) | Load MedQA + textbooks, clean, normalize, chunk. **Stops at `Chunk`.** |
| [embeddings/](../src/medqa_rag/embeddings/) | HuggingFace embedder + on-disk cache. Implements `Embedder` protocol. |
| [llm/](../src/medqa_rag/llm/) | Groq async client + token-bucket + LLM cache. **All LLM calls flow through here.** |
| [retrieval/](../src/medqa_rag/retrieval/) | Generic FAISS + BM25 retrievers. Implement `Retriever` protocol. Build / save / load / retrieve. |
| [rags/](../src/medqa_rag/rags/) | Four self-contained RAG modules + a shared `RAGPipeline` base + a `factory.build_rag()` |
| [evaluation/](../src/medqa_rag/evaluation/) | RAGAS, non-LLM metrics, hallucination detector, statistical tests, reporters. |
| [explainability/](../src/medqa_rag/explainability/) | LIME / SHAP wrappers around any RAGPipeline; stratified sampler. |
| [api/](../src/medqa_rag/api/) | FastAPI app, routers, schemas, middleware, OpenAPI tags & examples. |
| [pipelines/](../src/medqa_rag/pipelines/) | Top-level orchestration (ingestion, single-arch eval, full comparison). |

---

## Data flow

### Ingestion path (offline; run once)

```
data/raw/textbooks/*.txt
        │
        ▼
load_textbooks()     →    Chunk (whole-document)
        │
        ▼
clean_medical_text()
        │
        ▼
RecursiveChunker     →    list[Chunk] (≈ 512 chars / 64 overlap)
        │
   ┌────┴────┐
   ▼         ▼
embed_documents   tokenize_for_bm25
   │              │
   ▼              ▼
FaissRetriever  BM25Retriever
.build → .save .build → .save
   │              │
   ▼              ▼
data/indices/faiss/   data/indices/bm25/
```

### Query path (online; per request)

```
Question
   │
   ▼
RAGPipeline.answer(q)             ── only this varies
   │
   │   Naive:        FAISS.retrieve(stem, k)
   │   Self-RAG:     judge.confidence(stem) ?
   │                   skip retrieval : FAISS.retrieve(stem, k)
   │   Hybrid:       FAISS(20) ∪ BM25(20) → RRF → top-k
   │   Multi-Hop:    judge.decompose(stem) → for q in [stem, *subs]:
   │                   FAISS.retrieve(q, k) → aggregate
   │
   ▼
list[RetrievedDoc]
   │
   ▼
generator.format_prompt(jinja2)
   │
   ▼
GroqClient.chat(...)        ─►  rate_limit  ─►  cache_lookup  ─►  HTTP
   │
   ▼
parse_letter() → predicted_index
   │
   ▼
RAGOutput   (uniform across all four architectures)
```

### Evaluation path (offline; periodic)

```
list[Question]  ─►  RAGPipeline.answer(...)  ─►  list[RAGOutput]
                                                       │
                                                       ▼
                            ┌──── RagasEvaluator (Groq judge) ────┐
                            │   HallucinationDetector             │
                            │   accuracy / latency / token cost   │
                            └──────────────────┬──────────────────┘
                                               ▼
                                          metrics.json + MLflow run
                                               │
                                               ▼
                              statistical_tests (Cochran's Q, McNemar)
                                               │
                                               ▼
                          comparison_*.json + .md + .tex (thesis-ready)
```

---

## Concurrency model

- **API requests** — async FastAPI; each request awaits Groq with bounded concurrency.
- **Bulk evaluation** — sequential per-question by default; `gather_with_concurrency` available for parallelism within rate limits.
- **Rate limiter** — process-global token bucket; safe under asyncio.

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Modern typing (`Protocol`, `StrEnum`, `ParamSpec`) |
| Validation | Pydantic v2 | Settings + API + domain types |
| API | FastAPI 0.110+ | Async-native, Swagger built-in |
| LLM | Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) | Open-weight + low latency + free-tier friendly |
| Embedder | PubMedBERT (`NeuML/pubmedbert-base-embeddings`) via `sentence-transformers` | Domain-tuned medical embeddings, local |
| Dense retrieval | FAISS-CPU | Standard, deterministic, easy to ship |
| Sparse retrieval | `rank-bm25` | Pure Python, no Java dependency |
| Prompts | Jinja2 templates per RAG | Diffable, versionable |
| Reasoning chains | LangChain (text splitter only — kept minimal) | Battle-tested splitter |
| Evaluation | RAGAS ≥ 0.2 | 5 standard metrics |
| Stats | scipy + statsmodels (optional) | Cochran's Q + McNemar |
| Explainability | scikit-learn surrogate + custom Monte-Carlo Shapley | Avoids LIME/SHAP-tabular tooling mismatches |
| Logging | structlog | JSON, contextvars-aware |
| Tracking | MLflow | Local file store, zero infra |
| Tests | pytest + pytest-asyncio + pytest-cov | Standard |
| Lint / type | ruff + black + mypy `--strict` | Strict by default |
| Container | Python 3.11-slim + uvicorn | Small, ubiquitous |

---

## Non-goals (deliberately out of scope)

- Fine-tuning the LLM.
- Streaming responses (synchronous answers only — the API is for evaluation, not chat).
- Multi-tenancy / authn-z (research tool).
- A persistent metadata DB (filesystem JSON + MLflow are sufficient).
