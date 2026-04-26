# 10 — API Reference

FastAPI app at [`src/medqa_rag/api/main.py`](../src/medqa_rag/api/main.py).
Documented under OpenAPI 3.1 — Swagger UI is the canonical reference at runtime:

| URL | What it serves |
|---|---|
| `http://localhost:8000/docs` | **Swagger UI** (interactive) |
| `http://localhost:8000/redoc` | ReDoc (read-only) |
| `http://localhost:8000/openapi.json` | Raw OpenAPI 3.1 schema |

This document mirrors what Swagger shows, so you can reason about contracts
without booting the app.

## Run

```bash
make api
# or
uvicorn medqa_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Tags

| Tag | Purpose |
|---|---|
| `health` | liveness / readiness |
| `qa` | single-question answering |
| `evaluation` | bulk evaluation runs |
| `explainability` | LIME / SHAP attribution |

---

## `GET /healthz`  — liveness

```bash
curl http://localhost:8000/healthz
```
```json
{"status": "ok", "version": "0.1.0"}
```

## `GET /readyz` — readiness

```bash
curl http://localhost:8000/readyz
```
```json
{"status": "ready"}
```

---

## `POST /v1/qa/{architecture}` — answer one MCQ

`architecture` ∈ `naive | self | hybrid | multihop`.

### Request body

See [`schemas/request.py — QARequest`](../src/medqa_rag/api/schemas/request.py).

```json
{
  "question_id": "usmle_001",
  "stem": "A 45-year-old male presents with sudden-onset chest pain and dyspnea after a long flight. What is the most likely diagnosis?",
  "options": {
    "A": "Myocardial infarction",
    "B": "Pulmonary embolism",
    "C": "Aortic dissection",
    "D": "Pneumonia"
  },
  "correct_index": 1,
  "subject": "cardiology"
}
```

### Response — `QAResponse`

```json
{
  "question_id": "usmle_001",
  "architecture": "hybrid",
  "predicted_letter": "B",
  "predicted_index": 1,
  "correct": true,
  "generated_answer": "Reasoning: Sudden-onset chest pain after immobility is classic for PE.\nAnswer: B",
  "retrieved": [
    {
      "chunk_id": "book::harrison::chunk_00012",
      "source": "harrison.txt",
      "text": "Pulmonary embolism causes sudden onset of pleuritic chest pain ...",
      "score": 0.812,
      "rank": 0,
      "retriever": "hybrid"
    }
  ],
  "latency_ms": 1432.7,
  "token_usage": { "prompt_tokens": 980, "completion_tokens": 36, "total_tokens": 1016 },
  "extras": { "retrieval_used": true, "hop_count": 1, "sub_queries": [], "pool_k": 20, "rrf_k": 60 }
}
```

### Curl example

```bash
curl -X POST http://localhost:8000/v1/qa/hybrid \
  -H 'Content-Type: application/json' \
  -d '{
    "stem":"A 45-year-old male presents with sudden chest pain after a long flight.",
    "options":{"A":"MI","B":"PE","C":"Aortic dissection","D":"Pneumonia"},
    "correct_index":1
  }'
```

---

## `POST /v1/evaluate` — bulk evaluation run

Runs the chosen architecture across the canned MedQA test set under
[data/raw/medqa/](../data/raw/medqa/). Best for **smoke runs** (≤ ~200 q);
for full 12,723-question evaluations use the CLI (`make run-naive`, etc.) so
the request doesn't time out.

### Request — `EvaluateRequest`

```json
{
  "architecture": "naive",
  "n_questions": 100,
  "metrics": ["faithfulness", "answer_correctness"]
}
```

### Response — `EvaluateResponse`

```json
{
  "architecture": "naive",
  "n": 100,
  "accuracy": 0.71,
  "ragas": { "faithfulness": 0.82, "answer_correctness": 0.69 },
  "latency": { "n": 100, "mean_ms": 1820.5, "p50_ms": 1610.2, "p95_ms": 3104.8, "p99_ms": 4920.1 },
  "hallucination_rate": 0.08
}
```

---

## `POST /v1/explain` — passage attribution

### Request — `ExplainRequest`

```json
{
  "architecture": "multihop",
  "method": "lime",
  "question": {
    "stem": "...",
    "options": {"A":"...","B":"...","C":"...","D":"..."},
    "correct_index": 0
  }
}
```

`method` ∈ `lime | shap`.

### Response — `ExplainResponse`

```json
{
  "question_id": "ad_hoc",
  "architecture": "multihop",
  "method": "lime",
  "explanation_target": "A",
  "passage_attributions": [
    { "chunk_id": "book::cecil::chunk_00007", "source": "cecil.txt", "score":  0.42, "rank": 0 },
    { "chunk_id": "book::harrison::chunk_00181", "source": "harrison.txt", "score": -0.08, "rank": 1 }
  ]
}
```

A **positive** score means the passage drove the model toward the predicted
answer; **negative** means including it pushed the model away.

---

## Errors — `ErrorResponse`

Every error returns the same envelope:

```json
{ "error": "RetrievalError", "detail": "FAISS index not found at ./data/indices/faiss", "request_id": "8f3c..." }
```

| Exception | HTTP code |
|---|---|
| `RateLimitError` | 429 |
| `LLMError` | 502 |
| `DataError` | 400 |
| `ConfigError` / `RetrievalError` / `EmbeddingError` / `EvaluationError` / `ExplainabilityError` | 500 |
| Any other | 500 |

Mapping is in [`api/middleware/error_handler.py`](../src/medqa_rag/api/middleware/error_handler.py).

---

## Headers

| Header | Direction | Purpose |
|---|---|---|
| `X-Request-ID` | client → server (optional) | re-use a caller-supplied id for log correlation |
| `X-Request-ID` | server → client | always echoed; equals incoming or a fresh UUID |

Logged automatically via `RequestLoggingMiddleware`.

---

## Auth & rate limits

The research API is intentionally **un-authenticated**. Don't expose it on the
open internet. Put it behind your campus VPN, add a reverse-proxy basic-auth,
or wrap it with FastAPI's `Depends(auth)` if you need to share access.

The Groq-side rate limit is enforced inside `GroqClient`; the API itself does
not throttle.

---

## OpenAPI customisation

- Tags + descriptions: [`api/docs/openapi_tags.py`](../src/medqa_rag/api/docs/openapi_tags.py)
- Examples on every request schema (Swagger renders them as defaults)
- Response models declared on every router → generated `examples` are accurate
