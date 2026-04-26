# medqa-rag — Comparative Study of Four RAG Architectures for Medical QA

Systematic side-by-side comparison of **Naive RAG**, **Self-RAG**, **Hybrid RAG**, and **Multi-Hop Explainable RAG** on the **MedQA USMLE** benchmark, under controlled conditions where only the retrieval strategy varies.

> Thesis: *Systematic Comparison of Multiple Retrieval-Augmented Generative AI Architectures for Evidence-Based Medical Question Answering with Explainability and Hallucination Control* — LJMU MSc AI/ML, 2026.

---

## Quickstart

```bash
# 1. Install
make dev-install

# 2. Configure
cp .env.example .env
# Edit .env and set GROQ_API_KEY

# 3. Build retrieval indices (FAISS + BM25)
make index

# 4. Run a single architecture over the test set
make run-naive

# 5. Or run all four
make run-all

# 6. Boot the API (Swagger at http://localhost:8000/docs)
make api
```

## Layout

| Path | Purpose |
|---|---|
| `src/medqa_rag/core/`       | Cross-cutting: config, types, exceptions |
| `src/medqa_rag/data/`       | Loaders, preprocessing, chunking |
| `src/medqa_rag/embeddings/` | HuggingFace embedder + cache |
| `src/medqa_rag/llm/`        | Groq async client (rate-limited, cached) |
| `src/medqa_rag/retrieval/`  | FAISS dense + BM25 sparse base retrievers |
| `src/medqa_rag/rags/<name>/`| Self-contained RAG modules — one folder per architecture |
| `src/medqa_rag/evaluation/` | RAGAS, hallucination detection, statistical tests |
| `src/medqa_rag/explainability/` | LIME / SHAP passage attribution |
| `src/medqa_rag/api/`        | FastAPI app (Swagger UI at `/docs`) |
| `src/medqa_rag/pipelines/`  | Ingestion, evaluation, comparison orchestration |
| `tests/`                    | unit / integration / e2e |

Each RAG folder under `src/medqa_rag/rags/` contains its own `pipeline.py`, `retriever.py`, `generator.py`, prompt templates, config dataclass, and tests.

## Configuration

- **One YAML**: [config/settings.yaml](config/settings.yaml) — all runtime knobs.
- **One env file**: `.env` — secrets only (Groq API key).
- Override any YAML value with `MEDQA_<SECTION>__<KEY>` env vars.

## Development

```bash
make lint         # ruff
make format       # black + ruff --fix
make type         # mypy --strict
make test         # full pytest with coverage
make test-unit    # fast unit tests only
```

## License

MIT (research use).
