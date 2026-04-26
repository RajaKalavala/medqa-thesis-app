# medqa-rag — Documentation Index

This folder is the single source of truth for the project's design, plan,
operating instructions, and decisions. Read top-to-bottom for first-time
onboarding; jump to a specific doc otherwise.

## Read order

| # | Document | Purpose |
|---|---|---|
| 01 | [Overview](01-overview.md) | What this is, why it exists, problem & vision |
| 02 | [End-to-End Plan](02-plan.md) | Phases, milestones, effort estimate |
| 03 | [System Architecture](03-architecture.md) | Layers, data flow, design principles |
| 04 | [Folder Structure](04-folder-structure.md) | Annotated tree |
| 05 | [Feature Catalog](05-features.md) | What's built, per-RAG feature matrix |
| 06 | [Data & Schemas](06-data-and-schemas.md) | Domain types, API schemas, file formats |
| 07 | [RAG Architectures](07-rag-architectures.md) | Naive / Self / Hybrid / Multi-hop in depth |
| 08 | [Evaluation Framework](08-evaluation.md) | RAGAS, hallucination, statistical tests |
| 09 | [Explainability](09-explainability.md) | LIME / SHAP passage attribution |
| 10 | [API Reference](10-api-reference.md) | Endpoints, request / response, Swagger |
| 11 | [Configuration](11-configuration.md) | settings.yaml + env overrides |
| 12 | [Observability](12-observability.md) | structlog, MLflow, request tracing |
| 13 | [Testing Strategy](13-testing.md) | Unit / integration / e2e, coverage |
| 14 | [Setup & Run](14-setup-and-run.md) | Install, indices, run, API |
| 15 | [Deployment Plan](15-deployment.md) | Docker, compose, k8s sketch |
| 16 | [Methodology Mapping](16-methodology-mapping.md) | Thesis section → code module |
| 17 | [Roadmap & Limitations](17-roadmap.md) | What's not built; future work |

## Architecture decisions

ADRs live under [`architecture/adr/`](architecture/adr/):

- [ADR 0001 — LLM provider: Groq](architecture/adr/0001-llm-provider-groq.md)
- [ADR 0002 — Vector DB: FAISS + BM25](architecture/adr/0002-vector-db-faiss-bm25.md)
- [ADR 0003 — Per-RAG folder isolation](architecture/adr/0003-per-rag-folder-isolation.md)
- [ADR 0004 — Single-YAML configuration](architecture/adr/0004-single-yaml-config.md)

## Convention

- Each doc is self-contained; don't repeat content across docs — link instead.
- Every code reference uses `[file.py](path)` markdown links so they're clickable in editors.
- ADRs use the [Nygard format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions): Status / Context / Decision / Consequences.
