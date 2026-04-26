# Self-RAG (adaptive retrieval)

Two-step pipeline:

1. **Confidence gate**: cheap call to the small judge model (`llama-3.1-8b-instant`)
   to estimate whether the LLM already knows the answer.
2. If confidence ≥ `CONFIDENCE_THRESHOLD` → answer **without** retrieval.
   Otherwise → retrieve top-k from FAISS and answer with evidence.

Trades a small extra LLM call for fewer retrieve-and-answer cycles overall.
Originally proposed in *Self-RAG* (Asai et al., 2023).
