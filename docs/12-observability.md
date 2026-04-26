# 12 — Observability

You cannot debug what you cannot see. Three pillars: **structured logs**,
**experiment tracking**, **request tracing**.

## Pillar 1 — Structured logs (structlog)

Configured in [`observability/logger.py`](../src/medqa_rag/observability/logger.py).

- **JSON renderer** in non-development environments. One line = one event = one JSON object.
- Mirrored to stdout *and* to `logs/medqa_rag.log` (path from `settings.paths.log_dir`).
- Built on stdlib `logging` so third-party libraries (FAISS, sentence-transformers, ragas) stay in the same stream.

### Get a logger

```python
from medqa_rag.observability.logger import get_logger
logger = get_logger(__name__)
logger.info("naive_rag_done", qid="usmle_001", n_docs=5, latency_ms=1432.7)
```

### Bind context for the duration of a task

```python
from medqa_rag.observability.logger import bind_context, clear_context

bind_context(experiment_id="run42", architecture="hybrid")
... do work, every log line will carry these fields ...
clear_context()
```

### Conventions

| Field | When |
|---|---|
| `request_id` | Bound by API middleware on every HTTP request |
| `architecture` | Bound by `evaluation_pipeline` for the duration of a run |
| `qid` | Per-question events |
| `n_docs`, `latency_ms`, `predicted` | Per-question result payloads |

Aggregate analysis: pipe `logs/medqa_rag.log` into `jq` or your favorite log
processor — every event has a structured `event` key:

```bash
jq -c 'select(.event=="naive_rag_done") | {qid, predicted, latency_ms}' logs/medqa_rag.log | head
```

### Log level

`MEDQA_LOGGING__LEVEL=DEBUG` for development; `INFO` for runs (default).

---

## Pillar 2 — Experiment tracking (MLflow)

Wrapper at [`observability/mlflow_tracker.py`](../src/medqa_rag/observability/mlflow_tracker.py).
Default tracking URI is a local file store at `./results/mlruns/`.

### Auto-tracked per evaluation run

The evaluation pipeline opens an MLflow run with the tag `arch=<architecture>`
and logs:

| Param | Source |
|---|---|
| `architecture` | run argument |
| `n_questions` | dataset size |

| Metric | Source |
|---|---|
| `accuracy` | non_llm_metrics |
| `hallucination_rate` | hallucination_detector batch |
| `latency_mean_ms`, `_p50_ms`, `_p95_ms`, `_p99_ms` | latency_summary |
| `ragas_<metric>` | one per RAGAS metric returned |

### View the dashboard

```bash
mlflow ui --backend-store-uri ./results/mlruns
# → http://localhost:5000
```

### Custom logging

```python
from medqa_rag.observability.mlflow_tracker import mlflow_run, log_params, log_metrics, log_artifact

with mlflow_run(run_name="ablation-rrf-k", tags={"arch":"hybrid"}):
    log_params({"rrf_k": 30})
    log_metrics({"accuracy": 0.71})
    log_artifact("results/metrics/hybrid_xxx.json")
```

---

## Pillar 3 — Request tracing (FastAPI middleware)

[`api/middleware/logging.py`](../src/medqa_rag/api/middleware/logging.py) attaches a
`request_id` (UUID4 unless the client supplies `X-Request-ID`) to every
request, binds it into structlog's contextvars, and emits two events:

```jsonc
{"event":"request_received","method":"POST","path":"/v1/qa/hybrid","request_id":"8f3c..."}
{"event":"request_completed","status":200,"duration_ms":1432.7,"request_id":"8f3c..."}
```

The same `request_id` is echoed back in the response header so callers can
correlate.

### Errors

`api/middleware/error_handler.py` catches every domain exception and logs
`request_failed` with stack trace, then maps it to the appropriate HTTP code
(see [10-api-reference.md](10-api-reference.md)).

---

## Recommended log retention

For a thesis-scale project, log files grow slowly. For longer runs:

```yaml
# settings.yaml
logging:
  level: INFO
  json: true
```

…and rotate weekly with logrotate (or just delete `logs/medqa_rag.log` between
runs — the structured records you care about are also persisted under
`results/metrics/<arch>_<ts>.json`).

---

## What's deliberately *not* there (and how to add it)

| Want | Add |
|---|---|
| OpenTelemetry traces | `pip install opentelemetry-instrumentation-fastapi` and wire `FastAPIInstrumentor.instrument_app(app)` in `api/main.py` |
| Prometheus `/metrics` | `pip install prometheus-fastapi-instrumentator` and call `Instrumentator().instrument(app).expose(app)` |
| Centralised log shipping | Loki/ELK pick up the JSON file untouched; just point your daemon at `logs/*.log` |
| Sentry for errors | Add `sentry_sdk.init(...)` in `api/lifespan.py` |

These are deferred because they're operational concerns; the research workflow
doesn't need them yet.

---

## Diagnostic recipes

| Symptom | Where to look |
|---|---|
| "Why was retrieval skipped?" | `event=self_rag_done`, fields `confidence`, `retrieval_used` |
| "Which questions were hallucination-flagged?" | `metrics/<arch>_<ts>.json` → `flags.<qid>.is_flagged` |
| "Why was the response slow?" | `event=request_completed`, sort by `duration_ms` |
| "Did Groq cache hit?" | `event=llm_cache_hit` (DEBUG level) |
| "Did we hit the rate limit?" | look for tenacity retry attempts via `event=request_failed` upstream of `RateLimitError` |
