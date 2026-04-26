# 01 — Overview

## Context

Large Language Models are now embedded in clinical workflows: physician AI-tool
adoption rose from **38 % (2023) to 66 % (2024)** per the AMA. But the same models
hallucinate — recent oncology studies report ~40 % of model-generated medical
responses contain incorrect or unsupported claims. In medicine, a hallucinated
drug dosage or fabricated diagnosis is not an inconvenience; it is a patient-safety
incident.

**Retrieval-Augmented Generation (RAG)** mitigates hallucination by grounding
generation in retrieved evidence. But "RAG" is not one architecture — it's a
family. Existing literature evaluates RAG variants in isolation, on different
datasets, with different LLMs, making head-to-head comparison impossible.

## Problem statement

> _Which RAG architecture produces the most accurate, least hallucinated, and most
> explainable answers on a fixed medical benchmark, when only the retrieval
> strategy varies?_

Four candidate architectures are compared:

1. **Naive RAG** — single-pass dense retrieval + stuff-the-context.
2. **Self-RAG** — adaptive: retrieve only when model self-confidence is low.
3. **Hybrid RAG** — dense (FAISS) + sparse (BM25) fused via Reciprocal Rank Fusion.
4. **Multi-Hop Explainable RAG** — decompose the question into sub-queries, iterate retrieval, aggregate evidence, cite passages.

## Why this project exists

| Concern | What's missing today | What this project delivers |
|---|---|---|
| **Patient safety** | No comparative evidence on which RAG is safest | Per-architecture hallucination flag rate + RAGAS faithfulness |
| **Reproducibility** | Most studies evaluate one RAG, on one dataset, with one model | Single LLM + single embedder + single dataset across all four RAGs |
| **Explainability** | Most RAG evaluations stop at accuracy | LIME + SHAP passage attribution layered on every architecture |
| **Statistical rigor** | Few RAG comparisons report significance | Cochran's Q across the four + pairwise McNemar |
| **Practitioner guidance** | Selecting a RAG for a clinical use case is folklore | Comparison framework that maps clinical priority → architecture |

## What is built

A **production-grade Python framework** that:

- Ingests the MedQA USMLE corpus + 18 medical textbooks.
- Builds a shared FAISS dense index + BM25 sparse index.
- Routes every question through any of the four RAG pipelines, with **only the retrieval strategy varying**.
- Evaluates outputs with the five RAGAS metrics, paired statistical tests, and a three-layer hallucination detector.
- Explains predictions with LIME and SHAP at the passage level on a stratified sub-sample.
- Exposes everything behind a FastAPI + Swagger surface.
- Logs every call as structured JSON, tracks every run in MLflow.
- Ships with unit / integration / e2e tests and a Dockerfile.

## How: experimental control

```
                ┌────── question ──────┐
                ▼                       ▼
         shared LLM (Groq)      shared embedder (PubMedBERT)
                ▲                       ▲
                └───────── shared FAISS + BM25 ──────────┐
                          │           │           │      │
                       Naive       Self-RAG    Hybrid   Multi-Hop
                          │           │           │      │
                          └─────── different retrieval ──┘
                                       │
                                       ▼
                            uniform RAGOutput schema
                                       │
                                       ▼
                          RAGAS + stats + LIME/SHAP
```

The only variable across runs is `RAGPipeline.answer()`. Everything else
(model, embedder, prompts shape, indices) is held constant — so any difference
in accuracy / hallucination / latency is attributable to retrieval design.

## Out of scope

- Fine-tuning or retraining LLMs on medical data.
- Non-MedQA datasets (clinical notes, EHR data).
- Real-time clinical deployment / patient-facing testing.
- Knowledge-graph / agentic RAG variants.
- Human-expert evaluation panels.
- Comparison against non-RAG baselines (prompt-only, fine-tuned).

These are all reasonable future-work directions but are explicitly excluded
to keep the experiment controlled and the timeline realistic.

## Audiences

- **Thesis examiners** — read [01](01-overview.md) → [03](03-architecture.md) → [16](16-methodology-mapping.md).
- **Engineers onboarding** — read [03](03-architecture.md) → [04](04-folder-structure.md) → [14](14-setup-and-run.md).
- **Reviewers / replicators** — read [02](02-plan.md), [11](11-configuration.md), [13](13-testing.md), [15](15-deployment.md).
