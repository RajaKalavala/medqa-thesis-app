# 06 — Data & Schemas

This is the contract everything else upholds. If you change a field here,
expect work in evaluation, API, and tests.

## Domain types — [`src/medqa_rag/core/types.py`](../src/medqa_rag/core/types.py)

### `Architecture` (StrEnum)

```python
class Architecture(StrEnum):
    NAIVE = "naive"
    SELF = "self"
    HYBRID = "hybrid"
    MULTIHOP = "multihop"
```

### `Question`

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | unique within run |
| `stem` | `str` | clinical scenario / question text |
| `options` | `dict[str, str]` | keys must be `A`, `B`, `C`, `D` |
| `correct_index` | `int` `0..3` | gold label |
| `subject` | `str?` | optional ("cardiology", "neurology", …) |
| `metadata` | `dict[str, Any]` | passthrough from raw row |

Convenience props: `correct_letter`, `correct_text`. Pydantic `frozen=True`.

### `Chunk`

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | `book::<stem>::chunk_NNNNN` |
| `text` | `str` | the actual content |
| `source` | `str` | relative path within textbook root |
| `metadata` | `dict[str, Any]` | `parent_id`, `chunk_index`, `chunk_size`, file size |

### `RetrievedDoc`

| Field | Type | Notes |
|---|---|---|
| `chunk` | `Chunk` | |
| `score` | `float` | retriever-specific (cosine for FAISS, BM25 raw, RRF score for hybrid) |
| `rank` | `int` | 0-indexed within the result list |
| `retriever` | `str` | `"dense" \| "sparse" \| "hybrid" \| "hop_<n>"` |

### `RAGOutput` (uniform across all four architectures)

| Field | Type | Notes |
|---|---|---|
| `question_id` | `str` | |
| `architecture` | `Architecture` | |
| `retrieved_docs` | `list[RetrievedDoc]` | the evidence shown to the LLM |
| `generated_answer` | `str` | full text from the LLM |
| `predicted_letter` | `str?` | parsed `A` / `B` / `C` / `D` |
| `predicted_index` | `int?` | mirrors `predicted_letter` |
| `latency_ms` | `float` | wall-clock for the full `answer()` call |
| `token_usage` | `dict[str,int]` | prompt / completion / total |
| `retrieval_used` | `bool` | False only in Self-RAG when gate skipped retrieval |
| `hop_count` | `int` | ≥ 1; Multi-Hop reports >1 |
| `sub_queries` | `list[str]` | populated by Multi-Hop |
| `extras` | `dict[str, Any]` | architecture-specific (confidence, RRF k, …) |

### `EvaluationResult`

Per-question evaluation row used by reporters.

| Field | Type |
|---|---|
| `question_id` | `str` |
| `architecture` | `Architecture` |
| `correct` | `bool` |
| `ragas` | `dict[str, float]` |
| `latency_ms` | `float` |
| `token_usage` | `dict[str, int]` |
| `extras` | `dict[str, Any]` |

---

## API schemas — [`src/medqa_rag/api/schemas/`](../src/medqa_rag/api/schemas/)

### `QARequest`

```jsonc
{
  "question_id": "usmle_001",
  "stem": "A 45-year-old male presents with sudden-onset chest pain ...",
  "options": {"A": "MI", "B": "PE", "C": "Aortic dissection", "D": "Pneumonia"},
  "correct_index": 1,            // optional — set null to disable correctness flag
  "subject": "cardiology"
}
```

### `QAResponse`

```jsonc
{
  "question_id": "usmle_001",
  "architecture": "naive",
  "predicted_letter": "B",
  "predicted_index": 1,
  "correct": true,
  "generated_answer": "Reasoning: ...\nAnswer: B",
  "retrieved": [
    {
      "chunk_id": "book::harrison::chunk_00012",
      "source": "harrison.txt",
      "text": "Pulmonary embolism causes sudden ...",
      "score": 0.812,
      "rank": 0,
      "retriever": "dense"
    }
  ],
  "latency_ms": 1432.7,
  "token_usage": { "prompt_tokens": 980, "completion_tokens": 36, "total_tokens": 1016 },
  "extras": { "retrieval_used": true, "hop_count": 1, "sub_queries": [] }
}
```

### `EvaluateRequest` / `EvaluateResponse`

```jsonc
// request
{ "architecture": "hybrid", "n_questions": 100, "metrics": null }

// response
{
  "architecture": "hybrid",
  "n": 100,
  "accuracy": 0.74,
  "ragas": { "faithfulness": 0.81, "answer_correctness": 0.69, ... },
  "latency": { "n": 100, "mean_ms": 1820.5, "p50_ms": 1610.2, "p95_ms": 3104.8, "p99_ms": 4920.1 },
  "hallucination_rate": 0.06
}
```

### `ExplainRequest` / `ExplainResponse`

```jsonc
// request
{
  "architecture": "multihop",
  "question": { "stem": "...", "options": {"A":..,"B":..,"C":..,"D":..}, "correct_index": 0 },
  "method": "lime"
}

// response
{
  "question_id": "ad_hoc",
  "architecture": "multihop",
  "method": "lime",
  "explanation_target": "A",
  "passage_attributions": [
    { "chunk_id": "...", "source": "...", "score": 0.42, "rank": 0 },
    { "chunk_id": "...", "source": "...", "score": -0.08, "rank": 1 },
    ...
  ]
}
```

### `ErrorResponse`

```jsonc
{
  "error": "RetrievalError",
  "detail": "FAISS index not found at ./data/indices/faiss",
  "request_id": "8f3c..."
}
```

Status mapping:

| Exception class | HTTP code |
|---|---|
| `RateLimitError` | 429 |
| `LLMError` | 502 |
| `DataError` | 400 |
| `ConfigError` / `RetrievalError` / `EmbeddingError` / `EvaluationError` / `ExplainabilityError` | 500 |
| Any other `Exception` | 500 |

---

## File formats

### MedQA JSONL — supported schemas

The loader normalises **both**:

```jsonc
// Schema A
{"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
 "answer_idx": "B", "meta_info": "cardiology"}

// Schema B
{"id": "qx", "stem": "...",
 "opa": "...", "opb": "...", "opc": "...", "opd": "...",
 "correct": 1, "subject": "cardiology"}
```

Drop one or more `.jsonl` files into [data/raw/medqa/](../data/raw/medqa/);
all are concatenated, IDs deduplicated.

### Textbook corpus

`.txt` or `.md` files anywhere under [data/raw/textbooks/](../data/raw/textbooks/).
The loader walks the tree, decodes UTF-8 (falls back to latin-1), strips empty
files. Each file becomes one whole-document `Chunk` before chunking.

### FAISS index — [data/indices/faiss/](../data/indices/faiss/)

```
data/indices/faiss/
├── index.faiss      # written by faiss.write_index
├── docstore.pkl     # parallel list[Chunk]
└── meta.json        # { "dim": 768, "n": <chunk_count> }
```

### BM25 index — [data/indices/bm25/](../data/indices/bm25/)

```
data/indices/bm25/
└── bm25.pkl         # { "bm25": BM25Okapi, "chunks": [...], "k1": 1.5, "b": 0.75 }
```

### Embedding cache — [data/embeddings/](../data/embeddings/)

```
data/embeddings/
└── <model_name_safe>/<sha256(text)>.npy
```

### LLM response cache — `data/processed/llm_cache/`

One JSON file per `sha256(payload)` where payload is
`{model, temperature, max_tokens, messages}`. **Cache is bypassed when
`temperature > 0`** to keep nondeterministic runs honest.

### Evaluation outputs — [results/metrics/](../results/metrics/)

```
results/metrics/
└── <architecture>_<UTC-timestamp>.json
```

```jsonc
{
  "architecture": "Architecture.HYBRID",
  "timestamp": "20260801T120000Z",
  "n": 12723,
  "accuracy": 0.74,
  "ragas": { "faithfulness": 0.81, ... },
  "latency": { "n": 12723, "mean_ms": 1820.5, "p50_ms": 1610.2, ... },
  "tokens": { "prompt_tokens": 19200000, ... },
  "hallucination_rate": 0.06,
  "correctness": { "<question_id>": true|false, ... },   // for paired stats
  "flags": { "<question_id>": { "is_flagged": false, ... }, ... },
  "outputs": [ { ...full RAGOutput JSON... } ]
}
```

### Comparison reports — [results/reports/](../results/reports/)

For each comparison run, three files share a timestamp suffix:

```
results/reports/
├── comparison_<ts>.json    # raw metric/stats blob
├── comparison_<ts>.md      # human-readable
└── comparison_<ts>.tex     # thesis-ready LaTeX table
```
