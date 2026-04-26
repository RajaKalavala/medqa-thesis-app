# 16 — Methodology Mapping (Thesis ↔ Code)

This document is for thesis examiners and supervisors: it shows where each
section of the proposal is realised in the codebase. Examiners should be able
to read a methodology section and locate the file that implements it.

## Research questions → modules

| RQ | Question | Where it's answered |
|---|---|---|
| **RQ1** | Relative impact of retrieval architecture on accuracy + hallucination | [`pipelines/comparison_pipeline.py`](../src/medqa_rag/pipelines/comparison_pipeline.py) → `accuracy` + `hallucination_rate` per arch + Cochran's Q |
| **RQ2** | Best faithfulness / context-relevance scores via RAGAS | [`evaluation/ragas_evaluator.py`](../src/medqa_rag/evaluation/ragas_evaluator.py) → 5 metrics in the comparison report |
| **RQ3** | Can LIME / SHAP attribute answers to specific retrieved passages? | [`explainability/lime_explainer.py`](../src/medqa_rag/explainability/lime_explainer.py) + [`shap_explainer.py`](../src/medqa_rag/explainability/shap_explainer.py) → `Attribution.passage_scores` |
| **RQ4** | Does multi-hop have statistically lower hallucination? | [`evaluation/statistical_tests.py`](../src/medqa_rag/evaluation/statistical_tests.py) → pairwise McNemar on `is_flagged` vectors |

## Aim & objectives → deliverables

| Objective | Code deliverable |
|---|---|
| Implement & validate four RAGs with shared LLM/embedder/indices | [`rags/factory.build_rag()`](../src/medqa_rag/rags/factory.py) wires the same singletons into all four pipelines |
| Evaluate factual accuracy across 12,723 questions | [`evaluation/non_llm_metrics.accuracy()`](../src/medqa_rag/evaluation/non_llm_metrics.py) |
| Measure hallucination via 5 RAGAS metrics | [`RagasEvaluator`](../src/medqa_rag/evaluation/ragas_evaluator.py) configured in [`settings.yaml`](../config/settings.yaml) |
| Analyze explainability with LIME + SHAP at passage level | `LimeExplainer` / `ShapExplainer` over stratified sample |
| Produce a comparison framework | [`reporters/markdown_reporter.py`](../src/medqa_rag/evaluation/reporters/markdown_reporter.py) + [`latex_reporter.py`](../src/medqa_rag/evaluation/reporters/latex_reporter.py) |

---

## Methodology section ↔ implementation

References below use the proposal's section numbers (see thesis Chapter 7).

| Proposal § | What it says | Implemented in |
|---|---|---|
| 7.2.3 | Experimental + comparative + design-science methodology | The factory-injected pipeline pattern; per-RAG isolation |
| 7.3 | MedQA dataset, 12,723 English questions | [`data/loaders/medqa_loader.py`](../src/medqa_rag/data/loaders/medqa_loader.py) |
| 7.3.2 | Two-track preprocessing (questions vs. textbooks) | [`data/loaders/`](../src/medqa_rag/data/loaders/) + [`data/preprocessing/`](../src/medqa_rag/data/preprocessing/) |
| 7.4.1 | Recursive chunking (start point for ablation) | [`data/chunking/recursive.py`](../src/medqa_rag/data/chunking/recursive.py) |
| 7.4.2 | PubMedBERT as the shared embedder | [`embeddings/huggingface_embedder.py`](../src/medqa_rag/embeddings/huggingface_embedder.py); model name in [`settings.yaml`](../config/settings.yaml) |
| 7.4.3 | FAISS dense + BM25 sparse | [`retrieval/dense_faiss.py`](../src/medqa_rag/retrieval/dense_faiss.py), [`retrieval/sparse_bm25.py`](../src/medqa_rag/retrieval/sparse_bm25.py) |
| 7.5.1 | LLaMA 3.3 70B Versatile via Groq | [`llm/groq_client.py`](../src/medqa_rag/llm/groq_client.py); `settings.llm.model` |
| 7.5.2 | Same prompt template across all four | All four prompts share Reasoning/Answer format and grounding directive |
| 7.6.1 | Naive RAG: simple retrieve-then-generate | [`rags/naive_rag/`](../src/medqa_rag/rags/naive_rag/) |
| 7.6.2 | "Sparse" / Self-RAG: retrieve only when low confidence | [`rags/self_rag/`](../src/medqa_rag/rags/self_rag/) — *renamed from "Sparse" to avoid clashing with sparse-retrieval terminology* |
| 7.6.3 | Hybrid RAG: BM25 + FAISS via fusion | [`rags/hybrid_rag/`](../src/medqa_rag/rags/hybrid_rag/) — RRF in [`fusion.py`](../src/medqa_rag/rags/hybrid_rag/fusion.py) |
| 7.6.4 | Multi-Hop Explainable RAG | [`rags/multihop_rag/`](../src/medqa_rag/rags/multihop_rag/) — `decomposer.py` + `iterative_retriever.py` + `chain_aggregator.py` |
| 7.7.1 | Hallucination detection via RAGAS | [`evaluation/ragas_evaluator.py`](../src/medqa_rag/evaluation/ragas_evaluator.py) |
| 7.7.2 | Three-layer hallucination control | Layer 1: prompt grounding (every Jinja template). Layer 2: faithfulness scoring (RAGAS). Layer 3: rule-based flags in [`hallucination_detector.py`](../src/medqa_rag/evaluation/hallucination_detector.py) |
| 7.8.1 | LIME for passage attribution | [`explainability/lime_explainer.py`](../src/medqa_rag/explainability/lime_explainer.py) |
| 7.8.2 | SHAP for passage attribution | [`explainability/shap_explainer.py`](../src/medqa_rag/explainability/shap_explainer.py) |
| 7.9.1 | RAGAS evaluation metrics | `settings.evaluation.ragas_metrics` |
| 7.9.3 | Non-LLM metrics | [`evaluation/non_llm_metrics.py`](../src/medqa_rag/evaluation/non_llm_metrics.py) |
| 7.10 | Experimental workflow | [`pipelines/comparison_pipeline.py`](../src/medqa_rag/pipelines/comparison_pipeline.py) |

---

## Deviations from the proposal (state these explicitly in the thesis)

The implementation made three corrections to the proposal:

### 1. "Sparse RAG" → "Self-RAG" / "Adaptive RAG"

**Proposal §7.6.2** described Asai et al.'s adaptive retrieval but called it
"Sparse RAG". In the broader literature *sparse* refers to BM25-style lexical
retrieval — which is already used inside Hybrid RAG (§7.6.3), so the
terminology collided. The implementation uses **Self-RAG** consistently
(folder name [`rags/self_rag/`](../src/medqa_rag/rags/self_rag/), enum value
`Architecture.SELF`).

The thesis chapter should adopt the same renaming.

### 2. Statistical tests added (proposal didn't specify them)

The proposal claims the framework will identify performance differences but
does not name a statistical test. The implementation adds:

- **Cochran's Q** across the four systems (omnibus).
- **McNemar exact test** for every pairwise architecture comparison.

Both are paired-design tests appropriate for binary correctness on the same
test set ([§13.4 of the thesis appendix should cite them]).

### 3. Stratified sampling for explainability

The proposal implies LIME / SHAP would run on the full set. At
12,723 q × 4 archs × 50 perturbations that's > 2.5 M Groq calls. The
implementation replaces this with a **stratified sample of 400 questions**
(`settings.explainability.sample_size`), balanced by `subject`. The thesis
should document this trade-off explicitly in the limitations section
([§7.11.2](../docs/17-roadmap.md)).

---

## Tables in the thesis ↔ generators

| Thesis table | Generated by |
|---|---|
| Table 7.1 (Research design summary) | hand-written, mirrors `core/types.Architecture` + `settings.evaluation.ragas_metrics` |
| Table 7.2 (Dataset structure) | mirrors `core/types.Question` schema in [06-data-and-schemas.md](06-data-and-schemas.md) |
| Table 7.5 (Vector DB comparison) | hand-written; selected DB recorded in ADR 0002 |
| Table 7.7 (Hallucination layers) | hand-written; matches three flags in `HallucinationDetector` |
| Table 7.9 (RAGAS metrics) | mirrors `settings.evaluation.ragas_metrics` |
| Comparison results tables (Chapter 5) | [`scripts/generate_thesis_tables.py`](../scripts/generate_thesis_tables.py) → `results/reports/thesis_tables.tex` |

---

## How to cite the artifact in the thesis

> "All four architectures, evaluation, and explainability components were
> implemented in a single Python package (`medqa_rag`, version 0.1.0) under
> a unified `RAGPipeline` interface, ensuring that the only variable across
> experiments was the retrieval strategy. Source code, configuration, and
> reproduction scripts accompany this thesis as a supplementary artifact."

Mention version (`__version__`), commit hash, and the pinned model snapshots
(`settings.llm.model`, `settings.embedder.model_name`).
