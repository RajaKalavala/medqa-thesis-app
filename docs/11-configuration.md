# 11 — Configuration

Two files. That's the whole story.

| File | Purpose | Committed? |
|---|---|---|
| [`config/settings.yaml`](../config/settings.yaml) | All non-secret runtime settings | ✅ yes |
| `.env` | `GROQ_API_KEY` and rare path overrides | ❌ never (in `.gitignore`) |

Per-RAG knobs live as Pydantic models inside each RAG folder
(`<rag>/config.py`) — they default to values pulled from `settings.yaml`
and can be overridden in code if you ever need to ablate.

## How loading works

[`src/medqa_rag/core/config.py`](../src/medqa_rag/core/config.py):

1. `get_settings()` (cached singleton) walks parents of the source file looking for `config/settings.yaml`.
2. Loads it as a dict.
3. Hands the dict to `Settings(**data)` — a `pydantic_settings.BaseSettings` subclass.
4. Pydantic Settings then **overrides any field** with environment variables.

Env override convention:

```
MEDQA_<SECTION>__<KEY>
e.g.   MEDQA_LLM__MODEL=llama-3.1-8b-instant
       MEDQA_RETRIEVAL__TOP_K=10
       MEDQA_API__PORT=9000
```

`__` (double underscore) is the nesting delimiter. Single `_` is part of the
key name.

To force a reload (for example in tests):

```python
from medqa_rag.core.config import reload_settings
settings = reload_settings()
```

---

## `settings.yaml` reference

```yaml
env: development                     # development | staging | production

paths:
  data_dir:        ./data
  raw_medqa:       ./data/raw/medqa
  raw_textbooks:   ./data/raw/textbooks
  processed_dir:   ./data/processed
  embeddings_dir:  ./data/embeddings
  index_dir:       ./data/indices
  faiss_dir:       ./data/indices/faiss
  bm25_dir:        ./data/indices/bm25
  log_dir:         ./logs
  results_dir:     ./results

logging:
  level: INFO       # DEBUG | INFO | WARNING | ERROR
  json: true        # JSON renderer; false → human dev console
  console: true     # mirror to stdout in addition to file

llm:
  provider: groq
  model:        llama-3.3-70b-versatile      # main reasoning model
  judge_model:  llama-3.1-8b-instant         # cheap judge for RAGAS / Self-RAG / Multi-Hop decompose
  temperature:        0.0                    # cache only honoured at 0.0
  max_tokens:         1024
  timeout_seconds:    60
  max_retries:        5
  rate_limit_rpm:     30                     # token-bucket request-per-minute cap
  cache_enabled:      true
  cache_dir:          ./data/processed/llm_cache

embedder:
  model_name: NeuML/pubmedbert-base-embeddings
  device:     auto                # auto | cpu | cuda
  batch_size: 32
  normalize:  true
  cache_enabled: true

chunking:
  strategy:      recursive        # recursive | (extension point: fixed/semantic)
  chunk_size:    512
  chunk_overlap: 64

retrieval:
  top_k:        5
  faiss_metric: cosine            # IndexFlatIP on normalised vectors
  bm25_k1:      1.5
  bm25_b:       0.75

evaluation:
  ragas_metrics:
    - faithfulness
    - answer_correctness
    - context_precision
    - context_recall
    - answer_relevancy
  judge_temperature: 0.0
  test_set_size:     null         # null = full set; int for subset
  random_seed:       42

explainability:
  sample_size:      400           # stratified sub-sample for LIME/SHAP
  stratify_by:      subject
  lime_num_samples: 50
  shap_num_samples: 50

api:
  host: 0.0.0.0
  port: 8000
  cors_origins: ["*"]             # tighten in production
  request_timeout: 120

mlflow:
  tracking_uri:    ./results/mlruns
  experiment_name: medqa-rag-comparison
```

---

## `.env` (local, never committed)

Copy from [`.env.example`](../.env.example):

```
GROQ_API_KEY=...                        # required
MEDQA_ENV=development                   # rare overrides
MEDQA_LLM__MODEL=llama-3.3-70b-versatile
MEDQA_LOG_LEVEL=DEBUG
```

Anything here takes precedence over the YAML.

---

## Per-RAG configs

Each RAG has a `config.py` with a Pydantic model. Defaults pull from
`get_settings()`; you only override when you want to ablate.

| File | Notable knobs |
|---|---|
| [`naive_rag/config.py`](../src/medqa_rag/rags/naive_rag/config.py) | `top_k` |
| [`self_rag/config.py`](../src/medqa_rag/rags/self_rag/config.py) | `top_k`, `confidence_threshold` (0.65) |
| [`hybrid_rag/config.py`](../src/medqa_rag/rags/hybrid_rag/config.py) | `top_k`, `pool_k` (20), `rrf_k` (60) |
| [`multihop_rag/config.py`](../src/medqa_rag/rags/multihop_rag/config.py) | `top_k_per_hop`, `max_hops` (3), `max_subqueries` (3) |

Constants like `MAX_HOPS`, `RRF_K`, `CONFIDENCE_THRESHOLD` live in
[`core/constants.py`](../src/medqa_rag/core/constants.py) so they're discoverable from one place.

---

## Configuration patterns

### Smoke run

```bash
MEDQA_EVALUATION__TEST_SET_SIZE=100 make run-naive
```

### Switch to a smaller LLM (cost-cap)

```bash
MEDQA_LLM__MODEL=llama-3.1-8b-instant make run-all
```

### Run on CUDA

```bash
MEDQA_EMBEDDER__DEVICE=cuda make index
```

### Bump rate limit (paid Groq tier)

```bash
MEDQA_LLM__RATE_LIMIT_RPM=120 make run-all
```

### Tighten CORS for production

Edit `settings.yaml`:
```yaml
api:
  cors_origins: ["https://your-research-portal.example.com"]
```

---

## What's deliberately **not** configurable

- The `RAGOutput` schema (changes break evaluation).
- The `Architecture` enum members (changes break the API path / factory).
- The set of RAGAS metrics emitted **per** call (you can override which are *evaluated*, but not invent new ones — register them in `RagasEvaluator._load_metrics`).

Adding a new variant (e.g. fixed-size chunking) is a code change, not a config change. See [04-folder-structure.md](04-folder-structure.md) — "If you're adding…" table.
