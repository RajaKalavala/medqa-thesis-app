# ADR 0002 — Vector store: FAISS-CPU + rank-bm25

- **Status:** Accepted
- **Date:** 2026-04
- **Supersedes:** —
- **Superseded by:** —

## Context

We need two retrievers:

1. **Dense** semantic — for paraphrases ("chest pain" → "thoracic discomfort").
2. **Sparse** lexical — for exact tokens (drug names, lab values, dosage strings).

Constraints:

- Single-host, single-process for the thesis. Kubernetes is a future option, not a current requirement.
- Indices must be built once and served from disk.
- Must not require external services (Postgres, Qdrant, Elasticsearch).
- Must be deterministic enough to support reproducible experiments.

## Decision

- **Dense**: [`faiss-cpu`](https://github.com/facebookresearch/faiss) with `IndexFlatIP` over L2-normalized embeddings (== cosine similarity).
- **Sparse**: [`rank-bm25`](https://github.com/dorianbrown/rank_bm25) (`BM25Okapi`) on whitespace-tokenised text.

Both retrievers implement the same `Retriever` protocol from
[`retrieval/base.py`](../../../src/medqa_rag/retrieval/base.py). Index files
land under `data/indices/{faiss,bm25}/`.

## Consequences

**Positive**

- Both libraries are pure-pip installable, deterministic, exact (no ANN approximation), and require no daemons.
- `IndexFlatIP` is exact and small enough at thesis scale (FAISS ~1–5 GB for the 18-textbook corpus).
- BM25 in pure Python — no Java / Anserini dependency, no Docker image required for indexing.
- Easy to ship in a slim Python image.

**Negative**

- **Memory-bound**: both indices live in RAM after `load()`. Won't scale to multi-million chunks. *Mitigated*: thesis corpus is small.
- **Single-process**: no concurrent index updates. Build is offline (`make index`).
- **No metadata filtering** (e.g. filter by textbook). *Mitigated*: filter `RetrievedDoc` post-hoc on `chunk.source` if needed.
- **BM25 quality limited** by whitespace tokenisation — a SciSpacy / UMLS pretokenizer would improve recall on rare drug names. Listed in [17-roadmap.md](../../17-roadmap.md).

## Migration path

If the corpus outgrows in-memory FAISS:

1. Add `retrieval/qdrant_retriever.py` (or HNSW + persistent variant) implementing the same `Retriever` protocol.
2. Toggle via `settings.retrieval.backend` (currently absent — add when needed).
3. Existing RAG modules require **no change** — they only import `FaissRetriever` from `factory.py`.

## Notes on index format

```
data/indices/faiss/
├── index.faiss      # faiss.write_index
├── docstore.pkl     # parallel list[Chunk] (pickled)
└── meta.json        # {"dim": 768, "n": <count>}

data/indices/bm25/
└── bm25.pkl         # {"bm25": BM25Okapi, "chunks": [...], "k1": 1.5, "b": 0.75}
```

Both stores are version-tagged by their pickle layout. If you rebuild with a
different chunker or embedder, **delete the old indices first** (`rm -rf data/indices/`); `build()` will recreate them.
