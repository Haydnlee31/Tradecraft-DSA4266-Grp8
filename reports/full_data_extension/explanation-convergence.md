# Numerical convergence of the official 39 feature explanations

Increasing the SHAP integration budget from 512 to 2,048 samples substantially
improved score reconstruction on the eight fixed pilot cases. Passing individual
output checks rose from **390 of 448 to 443 of 448**. Four of seven models now
pass every original numerical screen, compared with none before. However, five
individual outputs and one error-margin check still fail. The pilot therefore
remains `numerical_review_required`; no feature ranking or security policy is
ready to publish from it.

This is progress in the explanation calculation, not an improvement in model
accuracy. The weights, predictions, training results and decision rules did not
change. Keep RONIN stopped. The next useful step is a small integration-method
cross-check on these same cases, not more training or an open-ended sample sweep.

## What stayed fixed

The [frozen comparison plan](explanation-convergence-plan.json) specifies exactly
one larger budget. It retains the same seven final step-20 models, eight
validation cases, 128 training-background rows, Monte Carlo seeds 71 and 72,
logit outputs, batch size, CPU environment and numerical limits from the
[preparation stage](explanation-preparation.md). Both repeats receive 2,048
samples. The original 512-sample arrays remain unchanged.

Before attribution, all archive and checkpoint identities, input hashes,
baseline numerical checks and small-pilot logits were verified. The logits
and mean background logits match the saved pilot exactly for every model.
The runner reads the frozen input snapshots and packed-data manifest; it does
not reopen the full dataset or rerun full validation. The earlier exact CPU
validation reference remains separately pinned. Historical GPU metrics are
unchanged, including the previously disclosed tiny CPU/GPU count differences.

Every model's state is checked after attribution. No model was trained or
promoted, no threshold was tuned, and the final test was not opened. The full
219-case attribution run has not started. The original 46-feature study remains
separate.

## More samples improved the numerical checks

Expected gradients estimates feature contributions using sampled background
rows and positions along the path to a case. Increasing `nsamples` changes the
integration budget, not the trained model. The
[official GradientExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html)
describes this approximation and its seeded sampling interface.

The reconstruction residual is the case's logit minus the mean background logit,
minus the sum of estimated feature contributions. Ideally it is zero. The
unchanged screens require relative mean absolute residual at most 10%, relative
repeat disagreement at most 25%, and every output residual within
`0.25 + 0.10 × abs(case logit − background logit)`. These are engineering
tolerances, not confidence levels or model-performance thresholds.

| Model | Relative mean residual at 512 → 2,048 | Passing output checks at 512 → 2,048 | Repeat disagreement at 2,048 | All original screens at 2,048 |
|---|---:|---:|---:|---|
| Centralized seed 7 | 6.17% → 2.54% | 53 → 64 of 64 | 7.10% | Pass |
| Centralized seed 17 | 4.68% → 2.30% | 55 → 62 of 64 | 8.40% | Review |
| Centralized seed 27 | 10.47% → 2.72% | 41 → 62 of 64 | 9.23% | Review |
| IID seed 7 | 5.24% → 2.99% | 60 → 63 of 64 | 7.85% | Review |
| IID seed 17 | 5.03% → 2.60% | 59 → 64 of 64 | 7.90% | Pass |
| IID seed 27 | 4.61% → 2.35% | 60 → 64 of 64 | 7.33% | Pass |
| Controlled non-IID seed 7 | 5.45% → 2.17% | 62 → 64 of 64 | 8.58% | Pass |

All seven now pass the aggregate residual and repeat-disagreement screens.
Fifty-three formerly failing output checks now pass; no formerly passing output
check crosses into failure. Nonetheless, absolute residuals improve for only
348 outputs and worsen for 100. More samples helped overall, not monotonically
for every output. The same random seeds do not imply nested draws for every case
when the sample budget changes. Two budgets and two repeats do not establish an
asymptotic convergence rate.

The five remaining failures are shown below. The case label and the score being
explained are different columns: a model emits all eight class scores even when
the case is classified correctly.

| Model | True case class | Score being explained | Absolute residual | Allowed residual |
|---|---|---|---:|---:|
| Centralized seed 17 | Mirai | Benign | 0.2629 | 0.2556 |
| Centralized seed 17 | Mirai | DoS | 0.6640 | 0.4465 |
| Centralized seed 27 | Spoofing | Benign | 0.7809 | 0.4074 |
| Centralized seed 27 | Spoofing | Web-based | 0.5122 | 0.4266 |
| IID seed 7 | Mirai | DoS | 0.3749 | 0.2764 |

Some exceed the limit narrowly; others do not. The limits were not widened after
seeing those values. Passing on the eight pilot cases is also not a certificate
for the remaining 211 selected cases.

## Checking explanations of mistakes

For a misclassification, a useful question is why the model preferred its
predicted class over the true class. The additional diagnostic subtracts the
true-class contributions from the predicted-class contributions, then checks
whether they reconstruct the corresponding score difference relative to the
same background. It uses the existing class attributions, not a second model or
a changed classification rule. Correct cases are excluded because subtracting a
class from itself would give a meaningless perfect check.

There are 29 model/case error margins: four per centralized or IID model, and
five for the non-IID model. At both sample budgets, 28 of 29 pass the individual
margin residual check, but they are not the same 28. The former IID seed-27
failure resolves; a centralized seed-17 case moves into failure. For that DoS
case predicted as DDoS, the residual is **0.5152**, slightly above the unchanged
limit of **0.4988**. This remains a warning even though both individual class
outputs pass their own checks.

Six of seven models pass all the supplemental margin screens at 2,048 samples;
centralized seed 17 does not. The margin check adds information and does not
replace the original all-output checks. Neither a passing sum nor agreement
between two repeats proves that individual feature contributions are correct.

## Interpretation and next step

The broad improvement supports finite-sampling error as an important contributor
to the earlier warnings; it does not isolate every remaining source of error.
It also shows why a SHAP numerical warning should not trigger retraining a
classifier whose saved predictions have not changed. Numerical reliability and
model detection quality are separate questions.

The next local check should compare a deterministic path-integration calculation
at two predeclared resolutions, averaging over the **same 128 background rows**,
on the **same eight cases and seven models**. This would separate interpolation
integration from random background subsampling without changing the reference
distribution. It must report all outputs and error margins, compare feature
contributions as well as their sums, preserve the current warnings, and remain
labelled a method diagnostic rather than an automatically equivalent SHAP result.
No such method comparison has been launched in this stage.

Do not start the 219-case interpretation merely because 98.9% of these output
checks pass, and do not automatically launch a still-larger Monte Carlo budget.
If an independently checked calculation is reliable, define the full-run budget
and per-case reporting rules before expansion. Any subsequent feature rankings
must distinguish the balanced core from error-selected examples and account for
training-seed variability. Background sensitivity remains unresolved: the fixed
128-row draw contains no Web-based or Brute Force examples. Correlated/discrete
features, interpolated traffic combinations and the within-collection split
continue to limit causal or operational claims.

## Evidence and reproduction

The [published numerical results](explanation-convergence-results.json) preserve
all seven models, both budgets, every output residual, error-margin details and
input/checkpoint fingerprints. Raw features and attribution arrays remain in
ignored output folders. The canonical receipt is
`outputs/official39-explanation-convergence-v1/receipt.json`, SHA256
`7d596f610784e823883541ae0d14abf01b02571b039f4d581a1bce0c1adb8ac7`.
An independent rerun matched that receipt and all seven attribution files
byte-for-byte. The original preparation inputs and 512-sample results are
unchanged. This verifies reproducibility, not adequacy of the remaining estimates.

Run locally with the existing environment and a fresh output directory:

```bash
.venv/bin/python -m src.explain.official_convergence \
  --preparation-root outputs/official39-explanation-preparation-v2 \
  --archive-root /Users/haydn/Downloads \
  --manifest-path outputs/official39-packed-2m-v1/manifest.json \
  --preparation-plan-path reports/full_data_extension/explanation-cpu-plan.json \
  --plan-path reports/full_data_extension/explanation-convergence-plan.json \
  --output outputs/official39-explanation-convergence-recheck
```

Tests cover the budget and scope guards, changed inputs, verification of every
baseline before attribution, exact pilot logits, contrast arithmetic, explicit
exclusion of correct self-contrasts, both passing and failing numerical outcomes,
no dataset reopening, and published-evidence consistency. No dependencies or
training defaults were added or changed.
