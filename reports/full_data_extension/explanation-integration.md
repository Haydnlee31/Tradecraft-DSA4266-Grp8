# Deterministic integration check for the official 39 feature explanations

The deterministic method check passes for all seven frozen models. At both
64 and 128 integration points per background-to-case path, **448 of 448 class
scores and 29 of 29 error margins** pass the unchanged reconstruction checks.
Feature contributions also agree between the two resolutions within the
predeclared limits. This gives us a tested calculation method to propose for
the full 219-case analysis, without retraining or restarting RONIN.

The eight-case pilot is not the full explanation study. No feature rankings,
causal claims or security policies were produced. The models, classification
results and decision rules are unchanged; the final test remains unopened.

## What the calculation does

The previous [Monte Carlo convergence check](explanation-convergence.md) randomly
sampled background examples and positions along interpolation paths. Increasing
its budget to 2,048 draws per repeat helped, but left five output checks and one
predicted-versus-true margin check failing. Those warnings remain recorded.

This cross-check uses every one of the same 128 training-background rows with
equal weight. For each case and background row, it integrates the model's input
gradients along the straight path between them, multiplies by their feature
differences, then averages over backgrounds. The approach builds on
[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html).
It targets the same finite-background path-gradient expectation approximated by
expected gradients, but is reported as **background-averaged integrated
gradients**, not exact conditional Shapley values or an automatically equivalent
SHAP-library result.

The [frozen plan](explanation-integration-plan.json) uses 32-point Gauss-Legendre
quadrature on two or four equal path segments, for 64 or 128 points in total.
Using composite rules avoids a single high-degree rule above the tested range
noted in [NumPy's quadrature documentation](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html).
Neither rule is guaranteed exact through neural-network nonlinearities and ReLU
kinks. The two resolutions explicitly test that remaining numerical sensitivity.

The model and gradients stay in float32; weighted contributions are accumulated
in float64. Inputs are streamed in batches of at most 128 path points.
[PyTorch input gradients](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html)
are computed without accumulating parameter gradients. Model state and existing
parameter gradients are checked for changes after each calculation. No residual
is redistributed among features to force the contributions to add up.

## Checks fixed before the run

All seven models use the same eight cases and original training background.
Archive, checkpoint, source, package and input hashes are verified before the
first integration. Pilot logits and mean background logits must match the saved
CPU reference exactly. The prior 2,048-draw diagnostics are independently
recomputed from their saved arrays. The full dataset is not reopened: the runner
uses the pinned small snapshots and packed-data manifest only.

The original reconstruction limits are unchanged: relative mean absolute
residual at most 10%, and each output residual within
`0.25 + 0.10 × abs(case logit − mean background logit)`. The same completeness
checks apply to predicted-minus-true margins for misclassified cases. Correct
cases are excluded from margin counts rather than counted as trivial passes.

This deterministic calculation has no Monte Carlo repeat-disagreement test.
Instead, a separately predeclared resolution check compares the 64-point and
128-point contributions feature by feature: global relative mean absolute
difference at most 10%, and relative L1 difference at most 25% for every
case/output. The denominator is the fine-grid attribution magnitude. The same
limits apply to error-margin contributions. This catches changes hidden by
positive and negative feature errors cancelling in the total.

Both resolutions must pass all reconstruction checks, and their contributions
must pass the resolution checks, for a model to pass this method screen.
These are engineering tolerances, not statistical confidence or deployment gates.

## Results for all seven models

Every model passes all 64 score checks at both resolutions. The table shows the
remaining error and featurewise differences; percentages here describe numerical
approximation, **not classification accuracy**.

| Model | Relative mean residual at 64 points | At 128 points | Featurewise difference between resolutions | Largest per-output relative L1 difference |
|---|---:|---:|---:|---:|
| Centralized seed 7 | 0.411% | 0.185% | 0.399% | 2.502% |
| Centralized seed 17 | 0.231% | 0.129% | 0.360% | 0.745% |
| Centralized seed 27 | 0.475% | 0.201% | 0.415% | 0.815% |
| IID seed 7 | 0.176% | 0.058% | 0.261% | 0.840% |
| IID seed 17 | 0.177% | 0.058% | 0.262% | 0.557% |
| IID seed 27 | 0.156% | 0.050% | 0.258% | 0.516% |
| Controlled non-IID seed 7 | 0.221% | 0.113% | 0.358% | 0.968% |

All 29 misclassified model/case margins pass at both resolutions, including the
centralized seed-17 DoS-to-DDoS case that failed the Monte Carlo margin check.
Their featurewise resolution comparisons also pass. The fine-grid relative mean
margin residual ranges from 0.083% to 0.751% across models.

The fine-grid contributions differ from the mean of the two saved 2,048-draw
Monte Carlo runs by about 3.72–4.10% in global relative mean absolute difference.
This is a descriptive cross-method comparison, not an uncertainty interval or
proof that either array is ground truth. Unlike a reconstruction check alone,
the small differences between deterministic resolutions directly examine the
individual feature contributions.

An independent replay matched the complete receipt and all seven attribution
files byte-for-byte. Deterministic replay demonstrates reproducibility, not
independent uncertainty estimation.

## What improved and what did not

The earlier 2,048-draw Monte Carlo estimates had relative mean reconstruction
errors of 2.17–2.99%; the fine deterministic grid gives 0.050–0.201%. Together with
the resolution agreement, this supports using the tested integration recipe for
the next explanation stage. It does not show that the networks became better
classifiers or that their feature reasoning is causally correct.

This is **not an equal-compute method comparison**. The previous two Monte Carlo
repeats used 4,096 sampled path points per case/output in total. The deterministic
64-point grid uses 8,192 points across the 128 backgrounds; the 128-point grid
uses 16,384. Running both uses six times the previous combined path-point budget.
Across seven models and eight cases, that is 1,376,256 model forward path rows
and 11,010,048 output-gradient rows, excluding endpoint verification. No speed
or cost-efficiency superiority is claimed.

Background sensitivity remains unresolved. The frozen 128-row background has no
Web-based or Brute Force samples; using all its rows removes random subsampling
within that background, not uncertainty about its representativeness. Correlated
features, unrealistic interpolated traffic combinations, eight pilot cases and
reused within-collection validation still limit conclusions. Non-IID has only
one training seed. Passing this pilot does not guarantee that the remaining 211
selected cases will pass.

## Next stage

The next step can now move from numerical troubleshooting to a bounded full
explanation analysis of the **already frozen 219 cases**. Before running it,
fix the following rules in a separate protocol: retain both integration
resolutions for per-case QA; preserve the 64-row class-balanced core separately
from the 155-row error supplement; report failed cases explicitly without
silently excluding them from an average; and compare centralized/IID patterns
across their three saved seeds. Non-IID findings remain single-seed diagnostics.

Use the numerical checks to decide which case explanations are supportable,
then identify consistent protocol/flow-statistic patterns and formulate cautious
security-policy hypotheses. Do not translate an attribution into an automatic
blocking rule, identifiable-device vulnerability or proof of causation. The
full-run protocol and interpretation remain a subsequent stage, not an action
authorized by this pilot's `complete` status. No new cloud training is needed.

## Evidence and reproduction

The [published numerical receipt](explanation-integration-results.json) includes
all score and margin residuals, resolution differences and provenance. Raw feature
values and attribution arrays remain in ignored output folders. The canonical
receipt `outputs/official39-explanation-integration-v1/receipt.json` has SHA256
`93e4702e45c66d05e0e03e0fc1702833528edec9558577d16012ca8783631371`.

```bash
.venv/bin/python -m src.explain.official_integration \
  --preparation-root outputs/official39-explanation-preparation-v2 \
  --convergence-root outputs/official39-explanation-convergence-v1 \
  --archive-root /Users/haydn/Downloads \
  --manifest-path outputs/official39-packed-2m-v1/manifest.json \
  --preparation-plan-path reports/full_data_extension/explanation-cpu-plan.json \
  --convergence-plan-path reports/full_data_extension/explanation-convergence-plan.json \
  --plan-path reports/full_data_extension/explanation-integration-plan.json \
  --output outputs/official39-explanation-integration-recheck
```

The tests include analytic linear, quadratic, interaction and ReLU examples;
quadrature moments; batching; unchanged model/gradient state; no random draws;
feature-error cancellation; passing/failing QA; provenance guards; and published
result integrity. No dependencies, old comments or training defaults were changed.
