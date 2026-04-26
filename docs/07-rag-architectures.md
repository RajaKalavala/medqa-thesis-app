# 07 — RAG Architectures in Depth

All four implement the same contract — [`RAGPipeline`](../src/medqa_rag/rags/base.py):

```python
async def answer(self, question: Question) -> RAGOutput
```

Differences live entirely in the body of `answer()` and the prompts under
`prompts/`. This document goes architecture by architecture.

---

## 1. Naive RAG — [`rags/naive_rag/`](../src/medqa_rag/rags/naive_rag/)

### Why
Lower-bound baseline. Anything more complex must beat this to justify the cost.

### How it works
```
question.stem
    │
    ▼
embed_query
    │
    ▼
faiss.retrieve(top_k=5)
    │
    ▼
stuff into Jinja prompt → Groq
    │
    ▼
parse "Answer: X" → RAGOutput
```

### Prompt — [`prompts/naive_qa.jinja2`](../src/medqa_rag/rags/naive_rag/prompts/naive_qa.jinja2)
> "Use only the evidence below … if insufficient, say so. Reasoning + Answer."

### Knobs (`NaiveRAGConfig`)

| Field | Default | What it controls |
|---|---|---|
| `top_k` | 5 (from `settings.retrieval.top_k`) | passages stuffed |

### Cost per question
1 Groq call (the LLM) + 1 retrieval.

### Failure modes
- Retrieval misses → model hallucinates from priors.
- Long stems blow past chunk locality → relevant evidence might not be top-5.

---

## 2. Self-RAG — [`rags/self_rag/`](../src/medqa_rag/rags/self_rag/)

### Why
Skip retrieval when the model already knows. Trades a cheap judge call for the
cost of dragging in irrelevant context.

### How it works
```
question.stem
    │
    ▼
confidence_gate(judge llama-3.1-8b-instant) → 0..1
    │
    ├── ≥ threshold (0.65) → answer WITHOUT retrieval
    │
    └── <  threshold       → faiss.retrieve(top_k) → answer WITH evidence
```

The same final QA prompt is used in both branches; a Jinja `if retrieval_used`
block conditionally renders the evidence section.

### Prompts
- [`confidence_check.jinja2`](../src/medqa_rag/rags/self_rag/prompts/confidence_check.jinja2) — outputs `Confidence: <0..1>`
- [`self_rag_qa.jinja2`](../src/medqa_rag/rags/self_rag/prompts/self_rag_qa.jinja2) — branched on `retrieval_used`

### Knobs (`SelfRAGConfig`)

| Field | Default | What it controls |
|---|---|---|
| `top_k` | 5 | when retrieving |
| `confidence_threshold` | 0.65 | gate cut-off |

### Cost per question
1 cheap judge call + (sometimes) 1 retrieval + 1 main LLM call.
Lower-bound cost equals Naive only when **every** question triggers retrieval.

### Failure modes
- The judge can be over-confident, skipping retrieval and hallucinating.
- The judge model itself can return malformed confidence — defaults to `0.5` which forces retrieval (safe).

### Naming note
The original thesis used the label "Sparse RAG" for this idea (Asai et al. 2023).
Renamed to "Self-RAG" / "Adaptive" to avoid confusion with sparse retrieval (BM25).
See [ADR 0003](architecture/adr/0003-per-rag-folder-isolation.md).

---

## 3. Hybrid RAG — [`rags/hybrid_rag/`](../src/medqa_rag/rags/hybrid_rag/)

### Why
Dense embeddings find paraphrases; BM25 finds drug names, lab values, exact
acronyms. Either alone misses cases the other catches.

### How it works
```
question.stem
    │
    ├──► faiss.retrieve(pool_k=20)   ── dense, semantic
    │
    └──► bm25.retrieve(pool_k=20)    ── sparse, lexical
                  │
                  ▼
           Reciprocal Rank Fusion (k=60)
                  │
                  ▼
           top_k=5 fused passages
                  │
                  ▼
            Groq → Answer
```

RRF score for a doc: `sum_runs( 1 / (60 + rank_in_run) )`. Implementation in
[`fusion.py`](../src/medqa_rag/rags/hybrid_rag/fusion.py); rank-tagged and re-emitted
as `RetrievedDoc(retriever="hybrid")`.

### Prompt — [`hybrid_qa.jinja2`](../src/medqa_rag/rags/hybrid_rag/prompts/hybrid_qa.jinja2)
Identical structure to Naive — the change is upstream in retrieval.

### Knobs (`HybridRAGConfig`)

| Field | Default | What it controls |
|---|---|---|
| `top_k` | 5 | final fused passages shown to LLM |
| `pool_k` | 20 | candidates from each side |
| `rrf_k` | 60 | RRF damping; lower = more weight to top ranks |

### Cost per question
2 retrievals (dense + sparse) + 1 LLM call. No extra Groq spend over Naive.

### Failure modes
- BM25 over-rewards rare acronyms → off-topic chunks fuse high.
- If both retrievers return disjoint sets, RRF picks safe-but-mediocre passages.

---

## 4. Multi-Hop Explainable RAG — [`rags/multihop_rag/`](../src/medqa_rag/rags/multihop_rag/)

### Why
Some questions require chaining: "If a patient has X and labs show Y, what is
the next step?" — needs evidence about *both* X and Y. Single-shot retrieval
often misses the second leg.

### How it works
```
question.stem
    │
    ▼
decomposer(judge) → ≤ max_subqueries focused sub-questions
    │
    ▼
queries = [stem, *sub_queries][:max_hops]
    │
    ▼
for q in queries:
    faiss.retrieve(q, top_k_per_hop)
        │ tagged "hop_<i>"
    runs.append(...)
    │
    ▼
chain_aggregator → round-robin dedupe → evidence chain
    │
    ▼
final_answer prompt forces citations [1], [2], …
    │
    ▼
Groq → Answer with chain-of-thought
```

### Prompts
- [`decompose.jinja2`](../src/medqa_rag/rags/multihop_rag/prompts/decompose.jinja2) — outputs sub-queries one per line, no numbering
- [`final_answer.jinja2`](../src/medqa_rag/rags/multihop_rag/prompts/final_answer.jinja2) — lists sub-queries + concatenated evidence; reasoning must cite `[N]`

### Knobs (`MultiHopRAGConfig`)

| Field | Default | What it controls |
|---|---|---|
| `top_k_per_hop` | 5 | per-retrieval breadth |
| `max_hops` | 3 | hard cap on retrieval rounds |
| `max_subqueries` | 3 | hard cap on decomposer output |

### Cost per question
1 cheap judge call (decomposition) + ≤ `max_hops` retrievals + 1 main LLM call.
Roughly 2–3× Naive in retrieval cost; ~1.5× in LLM cost.

### Why "Explainable"?
The cited passages `[1]…[N]` create a chain-of-evidence trace that
post-hoc XAI (LIME / SHAP) can verify. The chain itself is the artifact.

### Failure modes
- Decomposer drifts (asks questions adjacent to the stem) → noisy evidence.
- Aggregator's round-robin can starve a high-value hop if its results overlap heavily with hop 0.

---

## How "only retrieval varies" is enforced in code

| Property | Enforced by |
|---|---|
| Same LLM model | `GroqClient` reads `settings.llm.model` once |
| Same embedder | `build_embedder()` is shared across the four factories |
| Same FAISS index | `factory.build_rag()` loads it once and passes it in |
| Same BM25 index | same |
| Same prompt shape (Reasoning + Answer) | every Jinja template ends with the same answer-format directive |
| Same `RAGOutput` schema | enforced by `RAGPipeline` ABC |
| Same parsing | `RAGPipeline.parse_letter()` is shared |

If you ever need to vary something per-RAG that *isn't* retrieval — extend
`RAGOutput.extras`, never break the schema.
