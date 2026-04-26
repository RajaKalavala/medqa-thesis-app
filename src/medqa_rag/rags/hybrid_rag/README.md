# Hybrid RAG

Pipeline:

1. Retrieve `pool_k` passages with **FAISS** (dense / semantic).
2. Retrieve `pool_k` passages with **BM25** (sparse / lexical).
3. Fuse via **Reciprocal Rank Fusion** (`k=60`).
4. Take the top `top_k` fused passages → generate.

Captures both literal keyword matches (drug names, lab values) and semantic
similarity (paraphrases, related concepts).
