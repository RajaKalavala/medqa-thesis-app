# Multi-Hop Explainable RAG

Pipeline:

1. **Decompose**: cheap judge call breaks the question into ≤ `max_subqueries` focused sub-queries.
2. **Iterate retrieval**: one FAISS retrieval per (original stem + each sub-query), capped at `max_hops`.
3. **Aggregate**: round-robin interleave + dedupe → a single ranked evidence chain.
4. **Generate**: final-answer prompt forces the model to cite passage indices `[1], [2]` —
   that citation pattern is the source of "Explainable" in the architecture name.

Cost: ~3× more LLM calls and retrieval calls than Naive — pay for it only if multi-step
clinical reasoning is required.
