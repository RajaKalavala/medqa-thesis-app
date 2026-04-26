# ADR 0004 — Single YAML + env override (no per-component config files)

- **Status:** Accepted
- **Date:** 2026-04
- **Supersedes:** an earlier draft using one YAML per RAG + per-environment files
- **Superseded by:** —

## Context

The first scaffold attempt used:

```
config/
├── base.yaml
├── development.yaml
├── staging.yaml
├── production.yaml
├── logging.yaml
└── rag/
    ├── naive.yaml
    ├── self_rag.yaml
    ├── hybrid.yaml
    └── multihop.yaml
```

Nine YAML files for a research artefact. The user pushed back: "too many
yaml files." Fair point. The cost of fragmentation:

- Duplicated keys (Groq model name appeared in three places).
- Per-RAG YAMLs essentially restated `top_k` and a per-RAG knob — two lines per file.
- Environment-tier YAMLs (dev/staging/prod) were aspirational; this is a
  research artefact, not a multi-tier production app.
- New RAG = new YAML + edit factory.py + edit settings — three places to keep in sync.

## Decision

**One YAML, one `.env`.** Per-RAG knobs become Pydantic models inside each
RAG folder, with defaults resolved from `get_settings()`.

```
config/
└── settings.yaml          # single source of runtime config

.env                       # secrets only (GROQ_API_KEY)

src/medqa_rag/rags/<rag>/config.py    # per-RAG Pydantic model
```

Environment overrides are **environment variables**, not separate files.
Pydantic Settings supports nested overrides:

```
MEDQA_LLM__MODEL=...           # overrides settings.llm.model
MEDQA_RETRIEVAL__TOP_K=...     # overrides settings.retrieval.top_k
```

For prod-style deployment, set those env vars in the orchestrator
(docker-compose `environment:`, k8s ConfigMap, GitHub secret).

## Consequences

**Positive**

- One file to read; one file to grep.
- Env-var overrides are 12-factor-style, friendly to Docker / k8s / CI.
- Per-RAG knobs colocated with the code they affect — no jumping
  between `rags/` and `config/` to understand a parameter.
- Adding a new RAG no longer requires editing config — just adding a
  `config.py` to the new folder.
- Defaults still flow from the central YAML so the "shared baseline" claim
  holds.

**Negative**

- Less obvious that per-RAG hyperparameters exist (mitigated: each RAG's
  README lists its knobs).
- Switching environments at runtime requires shell exports rather than
  picking a YAML file (mitigated: any orchestrator does this naturally).
- Loading at startup is slightly more magical (YAML + env merge in
  Pydantic's `BaseSettings`) — implementation in [`core/config.py`](../../../src/medqa_rag/core/config.py) is short and tested.

## When this would break down

If we ever need:

- Multiple long-lived deployment tiers with persistent, version-controlled
  config differences (e.g. `staging` vs `production` schemas in production
  forever) — at that point, add per-environment YAMLs *layered* on top of
  the base, à la Hydra. But not before.
- Per-tenant configs — irrelevant to the thesis.

## Implementation summary

- [`core/config.py`](../../../src/medqa_rag/core/config.py): walks parents, finds `config/settings.yaml`, hands the dict to `pydantic_settings.BaseSettings(**data)`.
- Env prefix `MEDQA_` and nested delimiter `__`.
- `@lru_cache(1)` on `get_settings()` so repeated calls are free; `reload_settings()` for tests.
- Per-RAG configs use `Field(default_factory=lambda: get_settings().retrieval.top_k)` so they pick up YAML changes without re-import.
