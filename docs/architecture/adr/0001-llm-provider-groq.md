# ADR 0001 — LLM provider: Groq (open-weight LLaMA models)

- **Status:** Accepted
- **Date:** 2026-04
- **Supersedes:** —
- **Superseded by:** —

## Context

The thesis requires:

- An open-weight model (so the proposal isn't tied to a closed API).
- Strong reasoning capability on medical board-style questions.
- A small "judge" model for cheap auxiliary calls (Self-RAG confidence gate, Multi-Hop decomposition, RAGAS judge).
- Free or low-cost API access for thesis-scale evaluation (~50 K–250 K calls).
- Reasonable latency to keep evaluation feasible (12,723 questions × 4 architectures).

Three viable shapes:

1. **Self-host** open-weight LLaMA on a GPU box.
2. **OpenAI / Anthropic** APIs (closed-weight).
3. **Groq** API (open-weight LLaMA via fast inference hardware).

## Decision

Use **Groq** with two model snapshots:

- `llama-3.3-70b-versatile` — main reasoning model.
- `llama-3.1-8b-instant` — judge model (decomposition, confidence, RAGAS).

Calls are made through one wrapper at [`llm/groq_client.py`](../../../src/medqa_rag/llm/groq_client.py) that:

- Enforces a token-bucket RPM limiter.
- Retries on rate-limit / transient errors with exponential backoff (tenacity).
- Caches responses on disk, keyed by `sha256(model, temperature, max_tokens, messages)`.
- Bypasses cache when `temperature > 0` so nondeterministic runs stay honest.

## Consequences

**Positive**

- Open-weight: thesis claims of reproducibility hold.
- Latency: Groq is ~10× faster than self-hosted at this scale.
- Free tier / low spend: viable for the thesis budget.
- Single SDK, fully async, integrates cleanly with FastAPI.

**Negative**

- Vendor coupling at the API call site (mitigated: wrapper exposes a generic `complete()` method, not Groq-specific signatures).
- Rate limits are tighter than OpenAI's paid tiers — necessitates the cache + limiter built into the wrapper.
- Groq doesn't host embedding models — embedding is local (PubMedBERT via `sentence-transformers`).
- Model snapshots may shift; pin both names in `settings.yaml` and record the values used in each MLflow run.

## Implementation notes

- All callers (RAGs, RAGAS, XAI) go through `GroqClient`. There is no "raw" path.
- `llm.cache_enabled = false` if you want to force fresh calls during debugging.
- If migrating provider in future: implement a new client at `llm/<provider>_client.py` with the same `chat()` / `complete()` signatures, then change the import in `groq_client.py` callers (or add a factory).
