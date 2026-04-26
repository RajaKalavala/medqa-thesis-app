# 09 — Explainability

A correct medical answer with no traceable evidence path is not enough — the
clinician needs to know *which retrieved passage drove the model's choice*.
This module attributes the predicted answer back to specific retrieved
passages using two complementary methods.

## Why two methods?

| Method | What it asks | When it's strong |
|---|---|---|
| **LIME** | "Locally, how does dropping this passage change the answer?" | Fast, intuitive surrogate — useful when stakeholders want a single-question explanation |
| **SHAP** | "On average across all passage subsets, what's this passage's marginal contribution?" | Theoretically grounded (game-theoretic) — preferred for aggregate / cross-architecture comparisons |

Reporting both lets us cross-validate: passages that rank high under both
are unambiguously load-bearing.

---

## Architecture-agnostic wrapper

Both explainers work on **any** `RAGPipeline` because they treat it as a
black-box function:

```
input :  list[passage]  (subset of the original retrieval)
output:  predicted letter (recomputed via re-prompted Groq call)
target:  the architecture's original predicted_letter

attribution score per passage = effect on probability of returning target
```

This means the same explainer code attributes for Naive, Self, Hybrid, and
Multi-Hop without modification. See [`explainability/lime_explainer.py`](../src/medqa_rag/explainability/lime_explainer.py)
and [`explainability/shap_explainer.py`](../src/medqa_rag/explainability/shap_explainer.py).

---

## LIME details

```
n = number of retrieved passages
masks = random binary matrix (num_samples × n)
masks[0] = all-on    # the original output
masks[1] = all-off   # noise floor

for each mask:
    keep only passages where mask is True
    re-call Groq with that smaller context
    label = 1 if predicted_letter == original predicted_letter else 0

fit logistic regression: features = mask, target = label
coef[i] = LIME attribution for passage i
```

Logistic regression chosen over linear regression because the target is binary.
Falls back to a mean-marginal estimate if `scikit-learn` is unavailable.

---

## SHAP details (Monte-Carlo Shapley)

```
for sample in 1..num_samples:
    order = random permutation of passages
    mask = all-False
    prev_score = score(mask)
    for i in order:
        mask[i] = True
        cur_score = score(mask)
        contributions[i] += cur_score - prev_score
        prev_score = cur_score
        counts[i] += 1

shapley[i] = contributions[i] / counts[i]
```

This is Kernel-SHAP without the kernel weighting — appropriate when each
"feature" (passage) has a binary inclusion state and we have a small `n`.

---

## Cost: why we sample

The XAI cost equation:

```
total_groq_calls = num_questions × num_samples × architectures
```

For LIME with `num_samples=50`, full MedQA, four architectures:

```
12,723 × 50 × 4 = 2,544,600 Groq calls
```

That blows past any reasonable Groq tier and any reasonable timeline. Hence
**stratified sampling**.

### Stratified sampler — [`explainability/sampler.py`](../src/medqa_rag/explainability/sampler.py)

```python
sample = stratified_sample(questions, n=400, by="subject", seed=42)
```

- Stratifies by `subject` (cardiology, neurology, …) to keep XAI conclusions
  generalisable rather than dominated by one specialty.
- Returns shuffled subset; deterministic given the seed.
- Default `n=400` per architecture (`settings.explainability.sample_size`).

With `n=400`, `num_samples=50`, four archs → `80,000` calls. Manageable.

---

## Attribution output — [`explainability/base.py`](../src/medqa_rag/explainability/base.py)

```python
class Attribution(BaseModel):
    question_id: str
    architecture: str
    method: str                  # "lime" | "shap"
    passage_scores: list[float]  # aligned with RAGOutput.retrieved_docs
    explanation_target: str      # the predicted letter being explained
    extras: dict[str, float] = {}
```

The list `passage_scores[i]` is the attribution for `output.retrieved_docs[i]`
— same indexing → directly joinable with retrieval metadata in reports.

---

## Interpretation guide for the thesis

Per architecture, summarize:

1. **Top-k attribution concentration** — does the model rely on 1 passage or
   spread weight across all retrieved? High concentration = retrieval is
   doing real work; flat = LLM is mostly using priors.
2. **Sign of attributions** — positive = passage supports the prediction;
   negative = passage *hurt* the prediction (the model would have answered
   the same way more often without it).
3. **Cross-architecture passage overlap** — when two RAGs answer correctly on
   the same question, do they cite the same passages? If yes, that passage
   is *the* evidence for that question.
4. **Hallucination correlation** — questions flagged by the hallucination
   detector should show low or scattered attribution → evidence the model
   wasn't grounded.

These four lines of analysis go directly into thesis Chapter 5.

---

## Limitations (state these in the thesis)

- **LIME / SHAP-as-perturbation** asks the LLM to re-answer many times. Groq
  has nondeterminism even at `temperature=0` (small batch effects), so very
  small attribution differences should be treated as noise.
- **Multi-hop attribution** is more expensive and more interpretable since
  the chain already contains citations — the LIME/SHAP scores corroborate or
  refute those citations rather than discover them.
- **Self-RAG with retrieval skipped** has no passages to attribute — the
  module returns an empty `Attribution` and we report this as a feature, not
  a bug: it's the architecture saying "I didn't need evidence."
- **Token cost** of explainability dwarfs evaluation cost; budget accordingly.
