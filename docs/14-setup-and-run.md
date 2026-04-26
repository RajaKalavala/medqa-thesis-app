# 14 — Setup & Run

From a fresh clone to a comparison report in eight steps.

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | 3.11+ | Modern typing features used throughout |
| pip | recent | `pyproject.toml` PEP 621 needs `pip>=23` |
| make | any | task runner |
| Docker (optional) | 24+ | for containerised runs and `docker-compose` |
| Groq API key | free tier OK | sign up at https://console.groq.com |

Hardware:
- **CPU only** is fine for the embedder + FAISS at this scale (~1M tokens of textbooks).
- **CUDA** speeds up index build by ~5×; set `MEDQA_EMBEDDER__DEVICE=cuda`.
- Disk: ~2 GB for indices + caches.

---

## Step 1 — Install

```bash
make dev-install
# expands to:
#   pip install -e ".[dev]"
#   pre-commit install
```

Verify:

```bash
make lint
make type
make test-unit
```

All three should be green.

---

## Step 2 — Configure secrets

```bash
cp .env.example .env
# Open .env, paste your GROQ_API_KEY
```

Optional overrides at the same time (env vars beat YAML):

```
MEDQA_LLM__MODEL=llama-3.3-70b-versatile
MEDQA_LLM__JUDGE_MODEL=llama-3.1-8b-instant
MEDQA_LOG_LEVEL=INFO
```

See [11-configuration.md](11-configuration.md) for the full env-var grammar.

---

## Step 3 — Drop in the data

### MedQA

Place one or more JSONL files under [`data/raw/medqa/`](../data/raw/medqa/).

Either schema is accepted (mix and match if you like):

```jsonc
// Schema A
{"question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"answer_idx":"B"}

// Schema B
{"id":"qx","stem":"...","opa":"...","opb":"...","opc":"...","opd":"...","correct":1,"subject":"cardiology"}
```

The English MedQA USMLE dump is **12,723** questions — the loader dedupes by id.

### Textbooks

Drop the 18 medical textbooks under [`data/raw/textbooks/`](../data/raw/textbooks/) as
`.txt` or `.md` files. Subdirectories are fine; the loader walks them all.

For a smoke test, even one or two textbook files works.

---

## Step 4 — Build the indices

```bash
make index
```

This runs [`pipelines/ingestion_pipeline.py`](../src/medqa_rag/pipelines/ingestion_pipeline.py):

```
load_textbooks()
    → clean_medical_text()
    → RecursiveChunker.split()
    → FaissRetriever.build() + .save()       → data/indices/faiss/
    → BM25Retriever.build()  + .save()       → data/indices/bm25/
```

Expect 5–30 minutes depending on corpus size and hardware. Embedding cache is
reused on subsequent runs — second `make index` is much faster.

---

## Step 5 — Smoke test one architecture

Start small. 100 questions, no RAGAS:

```bash
MEDQA_EVALUATION__TEST_SET_SIZE=100 \
  python scripts/run_one_rag.py --rag naive --n 100 --no-ragas
```

Output:

```
results/metrics/naive_<UTC-timestamp>.json
results/mlruns/<experiment>/<run_id>/...
```

Inspect:

```bash
jq '{accuracy, n, hallucination_rate, latency}' results/metrics/naive_*.json | tail
```

If accuracy looks plausible (> random's 0.25), proceed.

---

## Step 6 — Run all four

```bash
make run-all              # full set
# or
python scripts/run_all_experiments.py --n 500
```

This drives [`pipelines/comparison_pipeline.py`](../src/medqa_rag/pipelines/comparison_pipeline.py):

```
for arch in [NAIVE, SELF, HYBRID, MULTIHOP]:
    run_architecture(arch, n_questions)
join correctness vectors → cochran_q + pairwise mcnemar
render comparison_<ts>.json/md/tex
```

Outputs:

| File | Purpose |
|---|---|
| `results/metrics/<arch>_<ts>.json` | per-arch detailed records |
| `results/reports/comparison_<ts>.json` | aggregated metrics + stats |
| `results/reports/comparison_<ts>.md` | human-readable summary |
| `results/reports/comparison_<ts>.tex` | thesis-ready LaTeX |

---

## Step 7 — Boot the API

```bash
make api
```

Open Swagger:
- http://localhost:8000/docs

Try a request:

```bash
curl -X POST http://localhost:8000/v1/qa/hybrid \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/sample_qa_request.json   # or paste a body
```

See [10-api-reference.md](10-api-reference.md) for full endpoint details.

---

## Step 8 — Generate thesis tables

```bash
python scripts/generate_thesis_tables.py
# → results/reports/thesis_tables.tex
```

Drop that LaTeX snippet straight into your thesis document.

---

## Common Make targets (full list)

```
make help              # show all targets
make install           # pip install -e .
make dev-install       # + dev extras + pre-commit
make lint              # ruff
make format            # black + ruff --fix
make type              # mypy
make test              # full pytest with coverage
make test-unit         # fast unit tests only
make test-integration  # mocked-Groq integration
make test-e2e          # real Groq calls
make api               # uvicorn dev server
make index             # build FAISS + BM25
make run-naive
make run-self
make run-hybrid
make run-multihop
make run-all           # all four + comparison report
make clean             # caches + build artifacts
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `LLMError: GROQ_API_KEY is not set` | `.env` missing or unread | `source .env` or set as environment variable |
| `RetrievalError: FAISS index not found` | indices not built | run `make index` first |
| RAGAS evaluation hangs | rate limit hit + retry storms | bump `MEDQA_LLM__RATE_LIMIT_RPM` for paid tier, or `--n 100` for smoke |
| `ImportError: faiss` | pip install issue on macOS | `pip install --no-cache-dir faiss-cpu` |
| Out-of-memory on `make index` | full embedding done in one pass | drop `embedder.batch_size` in `settings.yaml` |
| Tests fail on import | dev extras not installed | `make dev-install` |
| `mypy` complains about Pydantic v2 | wrong types-* installed | `pip install -e ".[dev]"` reinstalls everything pinned |

---

## What success looks like

After Step 6 you should have:

- A `comparison_<ts>.md` table showing **accuracy / RAGAS / latency / hallucination rate** for all four architectures.
- A non-trivial Cochran's Q p-value under `stats.cochran_q.pvalue`.
- An MLflow UI showing four runs you can compare side-by-side.
- A LaTeX file you can paste into your thesis.

That's the end-to-end pipeline working. From here it's parameter sweeps,
sub-sampled XAI runs, and writing.
