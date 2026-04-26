# 17 — Roadmap & Limitations

What's deliberately out of scope, what's known to be limited, and what would
be a natural next step.

## Known limitations

### 1. LIME / SHAP are perturbation-based and Groq-bound

Both XAI methods re-call the LLM many times per question. Even after
stratified sampling (400 q × 50 perturbations × 4 archs ≈ 80 K Groq calls),
this is the costliest part of any run. Failure modes:

- Groq nondeterminism at `temperature=0` introduces small noise — single-perturbation differences should not be over-interpreted.
- Self-RAG runs that skip retrieval produce empty `Attribution.passage_scores` — handle in plots.

### 2. Multi-hop "Explainable" overlap with XAI

Multi-Hop already emits passage citations (`[1]`, `[2]`) inside the answer
text. LIME/SHAP attributions on top *corroborate* those citations rather
than discover them. Two notes for the thesis:

- Treat citations + attributions as a **two-source check** rather than independent evidence.
- A more principled approach would be to extract the citation graph and use *that* as the attribution; left for future work.

### 3. RAGAS ↔ MCQ semantics

RAGAS metrics were designed for free-form QA. We bridge by generating a
free-form rationale and parsing the letter; this works in practice but means:

- RAGAS scores can disagree with raw accuracy (e.g. correct letter, weak grounding → high accuracy / low faithfulness — exactly the case we *want* the metric to surface).
- Don't average RAGAS and accuracy into one composite score — they measure different things.

### 4. Sparse retrieval is bag-of-words

`rank-bm25` over whitespace tokens; no medical-aware tokenizer (no SciSpacy,
no UMLS). Likely loses recall on rare drug names and dosage strings.
Future work: Pyserini with an Anserini index using a SciSpacy pretokenizer.

### 5. FAISS is in-memory and single-process

`IndexFlatIP` is exact and trivial to ship but doesn't scale beyond a few
million chunks. For multi-corpus extensions:

- Switch to `IndexHNSWFlat` (still in-memory, faster ANN).
- Or back with a persistent vector DB (Qdrant / Weaviate / pgvector). The
  retriever protocol remains the same.

### 6. Single-process rate limit

The token-bucket lives in process memory. If you run multiple workers, each
worker has its own bucket and you can exceed the Groq quota. For multi-process:

- Move the limiter behind Redis (`aioredis` + Lua script).
- Or just keep `replicas=1` ([15-deployment.md](15-deployment.md)).

### 7. No human-expert evaluation

Out of scope per the proposal. If a clinical reviewer wants to score the
top-K rationales by hand, the per-question outputs are already saved under
`results/metrics/<arch>_<ts>.json` — easy to surface in a notebook.

### 8. No streaming

The API returns full responses synchronously. Adding `text/event-stream`
would mean refactoring `GroqClient.chat` to expose async iterators.
Defer until the API actually faces end-users.

---

## Out-of-scope features (per the proposal)

These are explicitly *not* attempted and any future addition would expand the
scope of the thesis rather than refine it:

- LLM fine-tuning on medical data
- Knowledge-graph retrieval / agentic RAG
- Non-MedQA datasets
- Real-time clinical deployment
- Human expert validation panels
- Comparison with non-RAG baselines (prompt-only, supervised fine-tune)

---

## Future work (sequenced by effort)

### Cheap wins (≤ 1 week)

1. **Holm-Bonferroni adjustment** on pairwise McNemar p-values — drop into [`statistical_tests.py`](../src/medqa_rag/evaluation/statistical_tests.py).
2. **Rate-limit-aware progress bar** in `evaluation_pipeline` (currently emits log every 25 q).
3. **`/v1/qa/{architecture}` streaming variant** for demo UX.
4. **Notebook in [`notebooks/`](../notebooks/)** that loads `results/metrics/*.json` and renders comparison plots (matplotlib / seaborn).
5. **Holm-adjusted p-value column** in the LaTeX reporter.

### Medium (1–4 weeks)

6. **Fixed-size + semantic chunking ablation** — implement a `SemanticChunker` and toggle via `settings.chunking.strategy`.
7. **Multi-corpus retrieval** — multi-index FAISS routing (e.g. one per textbook), pick top-N per index → fuse.
8. **Graph-of-thoughts variant of Multi-Hop** — instead of round-robin aggregation, model evidence as a DAG and retrieve along edges.
9. **Persistent vector DB backend** — implement `QdrantRetriever` behind the same `Retriever` protocol; keep `dense_faiss` for ablation.
10. **Scheduled comparison job** — k8s `CronJob` running `make run-all` weekly with new question samples.

### Large / out-of-thesis-scope

11. **Active retrieval** — re-query mid-generation, not just up-front (FLARE-style).
12. **Tool-augmented RAG** — let the LLM decide whether to retrieve, browse, or compute, and log the tool trace.
13. **Bayesian decision-theoretic comparison** — rather than null-hypothesis tests, posterior over architecture quality (Stan / PyMC).
14. **Public deployment with abuse safeguards** — rate-limited public demo for the supervisor / external reviewers.

---

## Decision log discipline

When you tackle any of the above:

1. Write an ADR under [`docs/architecture/adr/`](architecture/adr/).
2. Cite the existing ADRs you're superseding (if any).
3. Add the new module to [04-folder-structure.md](04-folder-structure.md).
4. Add a feature row to [05-features.md](05-features.md).

This keeps the docs honest as the code evolves.
