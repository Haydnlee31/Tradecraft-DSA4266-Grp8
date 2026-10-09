# Preparing reliable explanations for the official 39 feature models

The saved models and explanation samples are now verified and frozen. A small
SHAP numerical pilot has also completed twice with identical results. It does
**not yet pass the checks needed for feature interpretation**: some individual
class scores are reconstructed too inaccurately by the approximate attributions.
Keep the cloud stopped. The next step is a bounded numerical convergence check
on the same cases, not retraining or changing the architecture.

No test split was opened, no model was trained or promoted, and no security
policy or feature ranking was generated. The original 46-feature explanation
pipeline and all training/decision defaults remain unchanged.

## The models and samples are fixed

The seven models are the final step-20 centralized-light and IID-light
checkpoints for seeds 7, 17 and 27, plus controlled non-IID seed 7. All were
trained on the 2M cohort and use the same 39-feature scaler and 5,096-parameter
LayerNorm architecture. These are `last.pt` endpoints, not validation-best
checkpoints or newly selected seeds. The archive, checkpoint, environment,
feature order, class order, source code and reported training work are checked.
Checkpoints use restricted weights-only loading after their hashes are verified.

The preparation uses the existing disk-backed arrays. It copies feature batches
of at most 513 rows during validation, matching the historical singleton-tail
rule; it does not load the full feature matrix into memory. Label and prediction
vectors are retained in memory, so total memory is not constant in dataset size.

| Sample | Fixed size | Purpose |
|---|---:|---|
| Training background | 128 unique training rows | Shared empirical reference for all seven models; no validation or test rows enter it. |
| Balanced validation core | 64 rows, eight per class | Common examples for subsequent class-specific comparisons, not population-weighted importance. |
| Error supplement | 155 additional rows | Union of one seeded example per nonempty model/class/error group. Keep separate from core averages. |
| Complete frozen selection | 219 unique validation rows | Prepared for later explanation, not fully attributed in this stage. |
| Numerical pilot | Eight shared core rows, one per class | Check numerical behavior before interpreting features. |

The error groups distinguish correct classifications, benign false alerts,
attacks predicted as Benign, and wrong attack categories. Every model sees the
same union of rows. Selection is reproducible and independent of model iteration
order. Empty groups remain explicit: controlled non-IID has no correct Web-based
or Brute Force predictions, and IID seeds 7 and 27 have no correct Web-based
predictions. The procedure does not invent successful detections to fill a chart.

The uniform training-background draw contains 12 Benign, 65 DDoS, 22 DoS, eight
Recon, six Spoofing and 15 Mirai rows, with no Web-based or Brute Force rows.
That is the observed finite draw, not the exact training proportions. It is
shared across models and was not changed after inspection. Background-sampling
sensitivity remains a separate unresolved question; repeating Monte Carlo
integration with the same background does not answer it.

## Why the CPU reference is separate from the cloud results

The first protocol required exact historical GPU confusion counts. It stopped
at centralized seed 7 before selecting samples or running SHAP. A read-only
audit of all seven models then found four exact matches and three one-count
confusion shifts, all between DDoS and DoS:

| Model | CPU minus archived GPU count change | CPU macro-F1 minus archived macro-F1 |
|---|---|---:|
| Centralized seed 7 | True DDoS: one fewer DDoS prediction, one more DoS prediction | −0.0000001541 |
| IID seed 7 | True DoS: one more DDoS prediction, one fewer DoS prediction | −0.0000002548 |
| IID seed 27 | True DDoS: one more DDoS prediction, one fewer DoS prediction | +0.0000001566 |
| Other four models | No confusion-count change | 0 |

All comparisons contain 2,059,284 validation flows. Benign and rare-class counts
are unchanged. The same checkpoint bytes, inputs and inference source passed
verification. The observed differences are consistent with numerical inference
differences between the cloud and local environments, but the precise cause was
not isolated. Archived GPU per-row predictions are unavailable: a one-count
confusion shift does not prove exactly one differing flow or exclude cancelling
differences elsewhere.

The original [strict protocol](explanation-preparation-plan.json) is retained.
The explicit [CPU protocol revision](explanation-cpu-plan.json) freezes a separate
[CPU inference reference](explanation-cpu-reference.json) and requires an exact
repeat of both its full confusion counts and its complete per-row prediction
hash. All seven models passed that repeat. This changes the stated explanation
reference, not the weights, data, classification rule or historical results.
The [experiment ledger](decision-summary.md) continues to report the original
cloud measurements unchanged. No GPU-parity tolerance was silently widened.

## What the numerical pilot measures

The pilot uses SHAP's GradientExplainer expected-gradients method on all eight
raw class scores, called logits. It approximates how each feature contributes
to the difference between a case's score and the mean score of the training
background. The approximation uses sampled backgrounds and interpolation
points; finite-sample completeness is not guaranteed. See the
[official GradientExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html).

Each model explains the same eight pilot cases twice, with Monte Carlo seeds
71 and 72 and 512 draws per repeat. Both repeats use the same 128-row background.
No smoothing, calibration, feature selection or threshold adjustment is applied.
Every model's state is checked for changes after explanation.

The reconstruction residual asks whether the feature contributions add up:

```text
residual = case logit − mean background logit − sum of feature contributions
```

Smaller residuals are better. Repeat disagreement asks whether two separate
Monte Carlo integrations give similar feature contributions. Agreement between
repeats alone is insufficient if both reconstruct the score poorly.

The thresholds were fixed before this pilot: relative mean absolute residual
at most 10%, relative mean absolute repeat difference at most 25%, and every
case/class residual within `0.25 + 0.10 × abs(case logit − background logit)`.
There are 64 case/class outputs per model. These are conservative engineering
screens, not confidence intervals, formal statistical tests or deployment gates.

| Model | Relative mean residual | Relative repeat difference | Outputs passing local residual check | Maximum absolute residual in logits |
|---|---:|---:|---:|---:|
| Centralized seed 7 | 6.17% | 14.84% | 53 / 64 | 2.0541 |
| Centralized seed 17 | 4.68% | 13.58% | 55 / 64 | 1.2130 |
| Centralized seed 27 | 10.47% | 14.90% | 41 / 64 | 3.7008 |
| IID seed 7 | 5.24% | 14.58% | 60 / 64 | 1.2665 |
| IID seed 17 | 5.03% | 12.67% | 59 / 64 | 1.1078 |
| IID seed 27 | 4.61% | 13.09% | 60 / 64 | 1.2555 |
| Controlled non-IID seed 7 | 5.45% | 15.00% | 62 / 64 | 0.7312 |

All seven pass the repeat-disagreement screen. Six pass the aggregate residual
screen. None passes the requirement for all individual outputs, so the recorded
status is `numerical_review_required`. This is a limitation of this bounded
approximation, not another training failure and not proof that the models cannot
be explained. Publishing a polished feature ranking now would conceal the
unresolved local reconstruction error.

## The next bounded local check

Before a full 219-case attribution run, freeze a convergence comparison using
the same eight cases, background, checkpoints and Monte Carlo seeds. Compare
the existing 512-draw result with one declared larger budget, such as 2,048
draws per repeat. Report all models and outputs, not just those that improve.
More samples may reduce approximation error; they are not guaranteed to make
every output pass.

Also check predicted-versus-true score differences for the error cases, since
those are the quantities a later explanation of a mistake should address.
Keep the existing all-output warnings rather than retroactively replacing the
failed criterion with a more favourable one. If reconstruction remains poor,
review the attribution method or integration scheme before further expansion.
Do not retrain the networks to make an explanation method look better.

Only after that numerical review should the shared core and error supplement
receive their full attribution budget. Class rankings must stay separate from
case explanations, and centralized/IID feature-pattern stability must be checked
across saved training seeds. Non-IID has only one seed. Correlated and discrete
features, interpolation outside observed traffic combinations, the small
background and reused within-collection validation all limit interpretation.
SHAP cannot establish attack causation, identifiable device vulnerabilities or
automatically safe firewall rules. The held-out test stays sealed.

## Evidence and reproduction

Implementation: `src/explain/official.py`. The new tests cover deterministic
sampling and absent groups, analytic linear attributions, state preservation,
hash/endpoint guards, fail-before-SHAP behavior, exact CPU prediction replay,
rejection of silent protocol changes and numerical warnings that cannot become
a ready status. The old explanation code and its comments are untouched.

The [published preparation result](explanation-preparation-results.json) includes
the model identities, CPU reference hashes, numerical diagnostics and all
class/error-group counts. Raw features, sample row indices and attribution arrays
remain in ignored outputs. The canonical receipt is
`outputs/official39-explanation-preparation-v2/receipt.json`, with SHA256
`0e504d2cd7bdcffffe6efa11266348a93b1c27c037e0d27631677c1cd085a976`.
An independent repeat matched the receipt, sampling and every saved array file
exactly, including the warnings. Repeatability is not the same as numerical adequacy.

Reproduce locally using the existing environment, downloaded archives and packed
2M data, with a fresh output directory. There is no reason to start RONIN:

```bash
.venv/bin/python -m src.explain.official \
  --data-root outputs/official39-packed-2m-v1 \
  --archive-root /Users/haydn/Downloads \
  --plan-path reports/full_data_extension/explanation-cpu-plan.json \
  --cpu-reference-path reports/full_data_extension/explanation-cpu-reference.json \
  --output outputs/official39-explanation-preparation-recheck
```

The CPU reference pins the local PyTorch and NumPy versions. Another environment
must not overwrite it or claim an exact replay without passing the checks.
Fixture tests and publication-integrity tests need no downloaded dataset:

```bash
python -m unittest tests.test_explain_official -q
```
