# Naive RAG

Single-pass pipeline:

1. Embed the question stem.
2. Retrieve top-k passages from FAISS (dense).
3. Stuff every passage into the prompt.
4. Generate (Groq) — return `Answer: X`.

No query rewriting, no fusion, no iteration. Used as the lower-bound baseline.
