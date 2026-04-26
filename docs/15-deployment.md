# 15 — Deployment Plan

Three deployment surfaces, in increasing order of formality:

1. **Local dev** — your laptop / lab machine for daily work.
2. **Container (Docker / Compose)** — reproducible single-host run.
3. **Kubernetes** — the production-shaped option, sketched but not built.

This is a research artefact — option 1 is enough for the thesis. Options 2-3
are documented so they exist when needed.

---

## 1. Local development

Already covered in [14-setup-and-run.md](14-setup-and-run.md). In one line:

```bash
make dev-install && cp .env.example .env && make api
```

When to stop here: thesis runs, ablations, supervisor demos.

---

## 2. Containerised run

### Docker image — [`deployment/Dockerfile`](../deployment/Dockerfile)

```dockerfile
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --upgrade pip && pip install -e .
EXPOSE 8000
CMD ["uvicorn", "medqa_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Image is built on `python:3.11-slim` (~150 MB base) plus the package and its
dependencies (~3 GB once `torch` + `sentence-transformers` are in). The
build-time `build-essential` is needed for FAISS wheels on some architectures.

### Build & run

```bash
docker build -f deployment/Dockerfile -t medqa-rag:0.1.0 .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v "$PWD/data:/app/data" \
  -v "$PWD/logs:/app/logs" \
  -v "$PWD/results:/app/results" \
  medqa-rag:0.1.0
```

Volume-mounting `data/`, `logs/`, `results/` keeps indices and outputs on the
host so you don't lose them on container restart.

### docker-compose — [`deployment/docker-compose.yml`](../deployment/docker-compose.yml)

```bash
cd deployment
docker compose up --build
```

Same idea, just declarative:

- Builds from the project root.
- Mounts `../data`, `../logs`, `../results`.
- Reads `../.env` for `GROQ_API_KEY`.
- Exposes port 8000.
- Restart policy: `unless-stopped`.

### Slimming the image

If image size becomes a problem:

- Switch to `python:3.11-slim-bookworm` — already used.
- Multi-stage build: install heavy deps in a builder stage, copy site-packages
  into a clean runtime stage. (Not needed yet.)
- Use `faiss-cpu` (already pinned) — `faiss-gpu` is enormous.
- Skip `mlflow` if you don't need the UI inside the container — make it a
  dev-only extra in `pyproject.toml`.

### Health probes

| URL | Use as |
|---|---|
| `GET /healthz` | Docker `HEALTHCHECK`, k8s `livenessProbe` |
| `GET /readyz`  | k8s `readinessProbe`, load-balancer drain signal |

Add to compose if you want:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
  interval: 30s
  timeout: 5s
  retries: 3
```

---

## 3. Kubernetes (sketch — not implemented)

A future option for sharing the API with peers / supervisors via a permanent URL.

### Topology

```
            ┌──────── ingress ────────┐
            │   (TLS, basic-auth)     │
            └────────────┬────────────┘
                         ▼
                ┌─── medqa-rag-api ───┐    Deployment, replicas=1
                │  uvicorn :8000      │
                └────┬────────────┬───┘
                     │            │
         volumes ────┘            └──── persistent volume claim
        (data/, indices/)              (results/, mlruns/)
```

### Why replicas=1

The framework caches FAISS, BM25, and embeddings *in memory* via
`@lru_cache(get_rag)`. Multiple replicas double the memory cost without
benefit since the bottleneck is Groq, not local compute. Scale up only if
you actually need throughput beyond Groq's tier.

### Secrets

```bash
kubectl create secret generic medqa-rag-secrets \
  --from-literal=GROQ_API_KEY=<value>
```

### Manifests (to add under `deployment/k8s/`)

A minimal trio:

- `deployment.yaml` — pod spec, env from secret, mounts the PVC at `/app/data`.
- `service.yaml` — ClusterIP on port 8000.
- `configmap.yaml` — copies `config/settings.yaml` into the pod (mount at `/app/config/`).

Resource ask:

```yaml
resources:
  requests: { cpu: 1, memory: 4Gi }
  limits:   { cpu: 4, memory: 12Gi }
```

CPU is bursty during retrieval; memory is dominated by FAISS + chunked corpus
+ the PubMedBERT weights.

### Build & push

```bash
docker build -t ghcr.io/<you>/medqa-rag:0.1.0 .
docker push   ghcr.io/<you>/medqa-rag:0.1.0
```

### When to bother

Only if: you need a permanent demo URL, you want to run scheduled comparison
jobs as `CronJob`s, or your evaluation runs exceed the local laptop's
patience. Otherwise, Compose on a beefy VM is more than enough.

---

## CI / CD recommendations

| Stage | Trigger | Action |
|---|---|---|
| Lint + unit | every PR | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
| Build image | tag push (`v*`) | `docker build` + push to GHCR |
| Deploy compose | manual | SSH + `docker compose pull && up -d` |
| Smoke run | nightly | run `make run-naive --n 50` against staging API, report metrics |

Building image on every commit is unnecessary and slow — bind to releases.

---

## Pre-flight checklist for any deployment

- [ ] `GROQ_API_KEY` set in target environment, **not committed**
- [ ] FAISS + BM25 indices present (or build step in deployment)
- [ ] `data/`, `logs/`, `results/` are persistent volumes (not container-local)
- [ ] CORS origins in `settings.yaml` tightened for non-dev env
- [ ] Log level: `INFO` (`json: true`) for any non-dev env
- [ ] If exposed publicly: reverse-proxy basic-auth or VPN — there's no built-in auth
- [ ] Rate limit (`llm.rate_limit_rpm`) matches Groq tier
- [ ] MLflow tracking URI: file store local for dev, remote (Postgres + S3) if shared

---

## Costs to plan for

| Resource | Approx ask |
|---|---|
| Disk | ~2 GB indices + embedding cache + 1–5 GB results over the thesis |
| RAM | 4–12 GB during a run (FAISS + embedder + LLM context) |
| Compute | CPU-only is fine for queries; CUDA only helps index build |
| Groq spend | dominated by RAGAS + LIME/SHAP — bound with `--n` and the stratified sampler |

Most of the project's *real* cost is Groq tokens, not infrastructure.
