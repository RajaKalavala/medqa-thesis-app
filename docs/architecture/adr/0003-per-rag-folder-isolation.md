# ADR 0003 — Per-RAG folder isolation + shared `RAGPipeline` ABC

- **Status:** Accepted
- **Date:** 2026-04
- **Supersedes:** —
- **Superseded by:** —

## Context

The thesis claim is "only the retrieval strategy varies". For that claim to
hold against examiner scrutiny, it must be enforced **structurally** — not
by code review or convention.

Two starting designs were considered:

1. **One-file-per-RAG** under `src/medqa_rag/rags/`. Compact, but hides
   architecture-specific helpers (decomposer, fusion, gate) and tangles
   cross-RAG imports.
2. **One-folder-per-RAG**, each with its own pipeline / retriever / generator
   / prompts / config / tests, all conforming to a shared abstract base.

## Decision

Adopt option 2:

```
src/medqa_rag/rags/
├── base.py              # RAGPipeline ABC + format helpers + parse_letter()
├── factory.py           # build_rag(arch) → RAGPipeline
├── naive_rag/           ┐
├── self_rag/            │  one folder per architecture
├── hybrid_rag/          │  each contains pipeline.py + retriever.py +
└── multihop_rag/        ┘  generator.py + prompts/ + config.py + tests/
```

Each folder is **self-contained**: anything specific to that architecture
lives there. Anything shared (Groq client, FAISS index, embedder) is
injected through `__init__` so the four pipelines can never accidentally
diverge on those.

## Consequences

**Positive**

- The ABC enforces the invariant: every architecture exposes
  `answer(Question) -> RAGOutput` and uses `format_question`,
  `format_context`, `parse_letter` from the same base.
- Per-folder tests reach 100 % isolation; failures point unambiguously at one
  RAG.
- A new RAG = copy any folder, rename, change the body of `answer()`.
- Per-folder READMEs document design rationale next to the code that
  implements it (not in a separate /docs page that drifts).
- Prompts are externalized as Jinja templates inside each folder — diffable,
  versionable, no string-concat surprises.

**Negative**

- Slight import verbosity (`medqa_rag.rags.hybrid_rag.fusion` vs.
  `medqa_rag.rag_helpers.fusion`). Worth it for the discoverability.
- Some duplication across `generator.py` files (all four are very similar).
  Conscious choice: the ~5 lines saved aren't worth coupling all four to a
  shared "generator" module that would gradually accumulate per-RAG flags.

## How invariants are upheld

- `RAGPipeline.__init__` accepts `llm: GroqClient`, `embedder: Embedder`,
  `faiss: FaissRetriever?`, `bm25: BM25Retriever?`. The `factory.build_rag`
  wires these from singletons → identical across architectures.
- `RAGOutput` is the single return shape — evaluators / API / explainers can
  assume the same fields regardless of architecture.
- The base class provides `format_question`, `format_context`, `parse_letter`
  → no per-RAG drift in MCQ parsing.

## Adding a fifth RAG

```bash
cp -r src/medqa_rag/rags/naive_rag src/medqa_rag/rags/<new>_rag
# edit pipeline.py, retriever.py, generator.py, prompts/*.jinja2
# add config.py + tests
# register in factory.py
# add Architecture.<NEW> to core/types.py
```

That's the full diff. No other module changes.
