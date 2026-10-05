# Pre-cloud checkpoint — 2026-10-05

This is the final local preparation checkpoint before a **small cloud pilot**.
It is not full-scale training clearance or operational security approval. No cloud
resources were launched, and this study did not evaluate the held-out test split.
See [CLOUD_HANDOFF.md](CLOUD_HANDOFF.md) for the remaining cloud-specific checks.

## Consolidated findings

The [12-run screen](../screening_2026_10_05/README.md) is complete. The frozen
[option-1 confirmation](../confirmation_2026_10_05/README.md) added eight runs
(model seeds 1 and 2) to four reused seed-0 screening runs. All figures below are
validation results, mean ± sample standard deviation over those three model seeds.

| Configured lane | Macro-F1 | Benign false-positive rate | Web recall | Brute Force recall |
| --- | ---: | ---: | ---: | ---: |
| Centralized heavy, BatchNorm | 0.6747 ± 0.0031 | 21.97% | 31.05% | 25.51% |
| Centralized light, BatchNorm | 0.6583 ± 0.0029 | 23.57% | 29.78% | 16.49% |
| Federated IID light, LayerNorm | 0.6328 ± 0.0024 | 24.23% | 12.79% | 14.80% |
| Federated non-IID light, LayerNorm | 0.5557 ± 0.0128 | 4.30% | 1.43% | 14.80% |

These are comparisons of selected configurations, not isolated estimates of the
effect of federation or model capacity: normalization and dropout differ. Seed 0
was selected during screening; the independent new-seed view is reported separately.
All runs reuse the same validation data, and federated runs use one fixed partition
seed. Standard deviations are not confidence intervals or partition-robustness evidence.

The heavy model has about 36 times the parameters of the light model but gains only
1.63 percentage points of mean macro-F1. This does **not** justify simply enlarging
it. A prior single-seed centralized XGBoost screen reached 0.8528 macro-F1, a useful
reference but not a matched, fully tuned comparison. No federated XGBoost was added.

LayerNorm improved the non-IID configuration during screening, but the new seeds
both have zero Web recall. Its low false-positive rate is not sufficient: Web flows
were accepted as benign in 57.01% and 70.54% of cases, and Brute Force flows in
66.06% and 67.87%. Most DoS errors instead confuse DoS with DDoS; they should not
be described as benign acceptance. Some best checkpoints occur at the 60-step cap,
so convergence is not established for every lane.

The [SHAP audit](../explanations_2026_10_05/README.md) reproduced all 12 validation
confusion matrices before explanation. It identifies recurring feature-level
hypotheses, not causal or device-specific vulnerabilities. Its small background has
no Web/Brute Force examples, average relative additivity residuals are about 5–9%,
and some individual residuals exceed the classification margin. Background
sensitivity and stronger numerical checks are needed before precise local claims.

The [decision integration](../decision_2026_10_05/README.md) selects only heavy under
the unchanged research gates. Requiring non-IID federation yields no eligible model.
Those gates have no false-positive ceiling and average over seeds; heavy still has
substantial false positives and one seed's Web recall below 20%. The result is a
research comparison, not authorization for automated blocking or deployment.

## Final local verification

[Recorded checks](checks/summary.json) and [test log](checks/tests.log) establish:

- All **40 tests passed**; `pip check` found no broken requirements.
- Fresh two-step actual-data CPU pilots for heavy BatchNorm and non-IID LayerNorm
  match uninterrupted runs exactly after pause/resume: final model tensors and
  validation metrics. This is a recovery check, not another tuning trial.
- CUDA was unavailable; no GPU training or cloud recovery was tested here.
- The private transfer inventory records SHA-256 and sizes for train/validation
  samples, selected run artifacts and raw explanation artifacts: **87,577,713 bytes**.
  It intentionally excludes test data. These files remain outside Git.

Re-run locally with fresh output directories:

```bash
python reports/pre_cloud_2026_10_05/preflight.py \
  --output reports/pre_cloud_recheck/checks --runs outputs/pre-cloud-recheck
```

This script requires the existing local study artifacts. It is a CPU preflight,
not a portable cloud bootstrap or GPU benchmark. Installed versions and source
hashes are captured in the summary; cross-platform CI is configured separately,
but this local check does not prove Linux, Windows or CUDA execution.

## Integration boundary

This checkpoint preserves the completed study rather than changing its training
code. A remote fetch found newer `origin/centralized-heavy` commit `7dd1459`
(dropout, batch-size and learning-rate interaction tuning). It is **not merged**
here. Review its evidence and reconcile its training changes in a separately
versioned experiment before using it; it was not part of these measured results.

## Recommended research direction after the pilot

Freeze a new bounded plan before spending on further experiments; the original
12-screen allowance is exhausted. Prioritize rare-class errors and convergence,
not parameter count: controlled learning-rate/schedule trials, rare-class loss or
sampling ablations, and multiple non-IID partition seeds. Include matched
normalization controls when claiming a federation effect. Preserve a budget for
independent confirmation and explanation-background sensitivity. Choose the final
configuration on validation, then perform the held-out test evaluation once under
a recorded protocol. Hardware feasibility remains projected unless measured on
the actual hardware being discussed.
