# 08 — Evaluation Framework

Three concentric layers, each addressing one of the thesis research questions:

1. **Did it answer correctly?** — accuracy, F1, latency, token cost
2. **Did it stay grounded?** — RAGAS faithfulness + 3-layer hallucination detector
3. **Did the difference matter?** — paired statistical tests across architectures

---

## Layer 1 — Non-LLM metrics — [`evaluation/non_llm_metrics.py`](../src/medqa_rag/evaluation/non_llm_metrics.py)

Cheap, deterministic, no Groq calls.

| Function | Returns |
|---|---|
| `accuracy(outputs, gold)` | mean correctness on `predicted_index == correct_index` |
| `latency_summary(outputs)` | `n`, `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms` |
| `token_summary(outputs)` | summed `prompt_tokens`, `completion_tokens`, `total_tokens` |

Latency reporting is **mandatory** for the thesis: a system that's twice as
accurate but five times slower may be unusable in clinical practice.

---

## Layer 2 — RAGAS — [`evaluation/ragas_evaluator.py`](../src/medqa_rag/evaluation/ragas_evaluator.py)

Five metrics, all evaluated with **`llama-3.1-8b-instant` as the judge model**
(configurable via `settings.llm.judge_model`).

| Metric | What it measures | Closer to 1 means |
|---|---|---|
| **Faithfulness** | claims in answer supported by retrieved context | answer is grounded |
| **Answer Correctness** | semantic match to gold answer | accurate |
| **Context Precision** | retrieved-context signal-to-noise | retriever is selective |
| **Context Recall** | retrieved-context coverage of gold answer | retriever captures the right facts |
| **Answer Relevancy** | answer addresses the question | not off-topic |

`RagasEvaluator.evaluate(outputs, gold)` returns mean values across the set;
`evaluate_per_question` returns one row per question for stratified analysis.

### Cost control

- Judge is the cheap 8B model.
- All judge calls flow through the same cached, rate-limited [`GroqClient`](../src/medqa_rag/llm/groq_client.py).
- The full 12,723-question MedQA test set × 5 RAGAS metrics × 4 architectures will hit Groq hard — start with `--n 200` for smoke runs.

### MCQ ↔ free-form bridge

RAGAS expects free-form answers. The pipelines emit free-form generation
(`Reasoning: …\nAnswer: A`) and **also** a parsed `predicted_letter`. RAGAS
sees the full text; accuracy uses the parsed letter. This satisfies both
worlds with one inference per question.

---

## Layer 3 — Hallucination detector — [`evaluation/hallucination_detector.py`](../src/medqa_rag/evaluation/hallucination_detector.py)

Three independent flags; if **any** trips, the answer is `is_flagged = True`.

| Flag | Trigger |
|---|---|
| `invalid_citations` | model emits `[N]` where `N` is outside `1..n_retrieved_docs` |
| `high_certainty_no_evidence` | model uses absolutist language ("definitely", "always") **and** `len(retrieved_docs) == 0` |
| `faithfulness_below` | RAGAS faithfulness `< faithfulness_threshold` (default `0.7`) |

```python
detector = HallucinationDetector(faithfulness_threshold=0.7)
flags = detector.evaluate(output, faithfulness=0.62)
# flags.is_flagged → True
```

Per-architecture **hallucination flag rate** = `count(is_flagged) / N`.

### Layered defense

Layer 1 — **Prompt-level grounding**: every RAG prompt explicitly says
"Use only the provided evidence."

Layer 2 — **Faithfulness scoring**: RAGAS catches semantic drift even when
the prompt rule was nominally followed.

Layer 3 — **Post-hoc rule-based**: catches structural issues (bad citations,
unbacked certainty) that RAGAS misses.

---

## Statistical tests — [`evaluation/statistical_tests.py`](../src/medqa_rag/evaluation/statistical_tests.py)

The thesis claims architectures differ. The reviewer will ask "by how much,
and is it significant?" The answer is paired tests on the same questions.

### `cochran_q(correctness_matrix)` — across all four

Tests the null hypothesis that all `K` systems have the same probability of
being correct, given they were all evaluated on the same `N` paired questions.

- Returns `TestResult(statistic, pvalue, description)`
- Distribution: chi-squared with `df = K - 1`
- Use it **first** to decide whether any pair is worth post-hoc testing.

### `mcnemar(correct_a, correct_b)` — pairwise

Exact-binomial McNemar's test on discordant pairs. Reports both `b01`
(A wrong, B right) and `b10` (A right, B wrong); rejects the null when their
imbalance is improbable under p=0.5. Falls back to scipy's exact binomial
if `statsmodels` isn't installed.

### Workflow

```
1. Run all four architectures on the same questions.
2. For each architecture, build a correctness vector aligned by question_id.
3. cochran_q(...)  → omnibus p-value
4. If p < 0.05:    → run mcnemar() for every (a,b) pair, report adjusted p-values
                     (Holm-Bonferroni recommended; not implemented — easy add).
5. Tabulate in the LaTeX reporter for the thesis.
```

The comparison pipeline does steps 1-4 automatically. Step 5 is rendered by
[`reporters/latex_reporter.py`](../src/medqa_rag/evaluation/reporters/latex_reporter.py).

---

## Reporters — [`evaluation/reporters/`](../src/medqa_rag/evaluation/reporters/)

| Renderer | Output |
|---|---|
| `markdown_reporter.render_markdown(report)` | one MD file with metric tables, latency summary, hallucination rates, embedded JSON for stats |
| `latex_reporter.render_latex(report)` | `\begin{table}…\end{table}` thesis-ready snippet |

Both consume the same `report` dict produced by `comparison_pipeline.run_all`.

---

## End-to-end evaluation flow

```
                                ┌─── make run-naive ─────────► metrics/naive_<ts>.json
                                ├─── make run-self  ─────────► metrics/self_<ts>.json
load_medqa_dir()  ──►  4 RAGs ──┤
                                ├─── make run-hybrid ────────► metrics/hybrid_<ts>.json
                                └─── make run-multihop ──────► metrics/multihop_<ts>.json
                                                                      │
                                                                      ▼
                                            scripts/evaluate_results.py  (or run-all)
                                                                      │
                                                                      ▼
                                                     Cochran's Q + pairwise McNemar
                                                                      │
                                                                      ▼
                                       results/reports/comparison_<ts>.{json, md, tex}
                                                                      │
                                                                      ▼
                                       scripts/generate_thesis_tables.py  → thesis appendix
```

## Practical tuning notes

| Knob | Where | When to change |
|---|---|---|
| `evaluation.test_set_size` | `settings.yaml` | Smoke runs on 100–500 q while iterating |
| `evaluation.judge_temperature` | `settings.yaml` | Keep 0 unless explicitly studying judge variance |
| `evaluation.random_seed` | `settings.yaml` | Pin for stratified sampling reproducibility |
| `HallucinationDetector(faithfulness_threshold=…)` | code | Tighten if you want a stricter clinical bar |
| `mcnemar` exact vs. asymptotic | `statistical_tests.py` | Prefer exact for `n < 25` discordant pairs |
