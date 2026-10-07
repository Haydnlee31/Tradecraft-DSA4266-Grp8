# Local distribution and SHAP sensitivity checks

## Decision

No new cloud training is justified by these diagnostics alone. The suspicious median jumps do not establish a local preprocessing bug. Higher SHAP integration budgets improve the numerical approximation overall, but changing the background changes the question and some feature rankings. Keep precise feature thresholds and automated blocking recommendations provisional.

## The median warning is a mixture effect

Earlier analysis flagged huge train/validation IAT median changes. The new streaming raw-CSV check shows two widely separated regimes, with approximately half the rows in each for these classes. Moving a few percent of rows across the 50% point can move the median by orders of magnitude. This is not a corresponding orders-of-magnitude change in every row.

| Split | Source | Class | Rows | IAT below 1 | IAT at least 100 million |
|---|---|---|---:|---:|---:|
| train | raw | Benign | 129538 | 49.79% | 50.21% |
| train | raw | Web-based | 2821 | 49.98% | 49.98% |
| train | raw | Brute Force | 1541 | 47.31% | 52.69% |
| train | sampled | Benign | 20000 | 49.60% | 50.40% |
| train | sampled | Web-based | 2821 | 49.98% | 49.98% |
| train | sampled | Brute Force | 1541 | 47.31% | 52.69% |
| val | raw | Benign | 27519 | 50.00% | 50.00% |
| val | raw | Web-based | 657 | 48.25% | 51.75% |
| val | raw | Brute Force | 290 | 52.41% | 47.59% |
| val | sampled | Benign | 19859 | 50.12% | 49.88% |
| val | sampled | Web-based | 628 | 48.41% | 51.59% |
| val | sampled | Brute Force | 277 | 52.71% | 47.29% |

Bins use recorded feature units for description only, not time-unit claims or security thresholds. For sampled Brute Force, the low-IAT proportion changes from 47.31% to 52.71%; the enormous median jump should not be described as proof of a severe domain shift. Smaller mixture differences may still affect predictions; no causal effect was estimated.

## Raw data and preprocessing checks

- Streamed only train and validation raw CSVs in bounded record batches. No test data was opened and no new training was performed.
- Raw label counts match the legacy sampling manifest. Rare training classes retain all raw rows; their three-feature value multisets match at 1e-12 relative/absolute tolerance.
- The legacy manifest is pre-cleaning: validation has 2,882 fewer rows than its recorded sample count. This is flagged, not silently rewritten.
- For all seven rare raw labels, a read-only reconstruction of the existing train-overlap anti-join reproduces retained validation counts and the IAT/Number/Weight multisets. This checks the rare labels, not the full historical sampling process.
- Saved scaler means and variances match sampled training values for all three inspected features. Current split hashes match the cloud model manifests.
- The legacy manifest lacks a sampling-seed field. Current sampler code records seeds, but that does not retroactively prove the historical sampling seed. Preserve the hashed sampled files.

The scope is three inspected features, not a universal guarantee against all data issues. No data was regenerated, repaired, pooled or re-split.

## What the original dataset documentation supports

The [dataset paper, Table 4](https://mdpi-res.com/d_attachment/sensors/sensors-23-05941/article_deploy/sensors-23-05941.pdf) describes IAT as the gap from the previous packet, Number as a packet count, and Weight as the product of incoming and outgoing packet counts. The [official UNB statistics](https://www.unb.ca/cic/datasets/iotdataset-2023.html) already report large IAT values and fractional Number/Weight summaries (including medians 9.5 and 141.55). Thus fractional values and large IAT magnitudes are not, by themselves, evidence of a local pipeline bug. The precise units and aggregation implementation were not independently reconstructed from the original packet extraction code. Do not invent a unit conversion.

## Frozen sensitivity screen

Four seed-7 checkpoints (heavy, central light, IID, non-IID); 16 balanced core rows plus 16 fixed error examples shared by every model and variant. These rows are a subset of the previous frozen selection, not chosen after seeing these results.

- Original natural-prevalence 128-row background at 256 and 1,024 integration samples.
- An independent seeded natural-prevalence 128-row background at 1,024 samples.
- A 128-row balanced background (16 training rows per class) at 1,024 samples.
- Two independent Monte Carlo repeats per condition. No architecture, weights, loss, validation rows or predictions were changed.

The balanced background changes the reference population. It is a sensitivity analysis, not a more accurate substitute for natural prevalence.

| Model | Variant | Relative mean absolute residual | Mean repeat top-5 overlap | Error residual at least margin |
|---|---|---:|---:|---:|
| heavy | original_256 | 7.13% | 92.5% | 5 / 18 |
| heavy | original_1024 | 3.89% | 100.0% | 3 / 18 |
| heavy | independent_1024 | 4.00% | 97.5% | 2 / 18 |
| heavy | balanced_1024 | 6.47% | 92.5% | 3 / 18 |
| light | original_256 | 7.01% | 85.0% | 7 / 19 |
| light | original_1024 | 5.01% | 82.5% | 3 / 19 |
| light | independent_1024 | 4.43% | 85.0% | 5 / 19 |
| light | balanced_1024 | 5.73% | 92.5% | 7 / 19 |
| iid | original_256 | 7.07% | 87.5% | 3 / 20 |
| iid | original_1024 | 3.82% | 97.5% | 2 / 20 |
| iid | independent_1024 | 3.46% | 97.5% | 2 / 20 |
| iid | balanced_1024 | 4.63% | 100.0% | 3 / 20 |
| dirichlet | original_256 | 7.65% | 92.5% | 1 / 23 |
| dirichlet | original_1024 | 3.66% | 95.0% | 1 / 23 |
| dirichlet | independent_1024 | 3.37% | 92.5% | 0 / 23 |
| dirichlet | balanced_1024 | 5.16% | 95.0% | 0 / 23 |

Residual means the error in reconstructing logit differences from attributions, not classification error. The reported ratio is aggregated; inspect individual margins before using a signed explanation. Tiny class cohorts (two core rows per class) make rankings exploratory.

## Background dependence of rare class features

| Model | Comparison | Web top-5 overlap | Brute Force overlap | DoS overlap |
|---|---|---:|---:|---:|
| heavy | original_256_vs_original_1024 | 100% | 100% | 100% |
| heavy | original_1024_vs_independent_1024 | 100% | 100% | 80% |
| heavy | original_1024_vs_balanced_1024 | 100% | 100% | 60% |
| light | original_256_vs_original_1024 | 80% | 100% | 80% |
| light | original_1024_vs_independent_1024 | 80% | 80% | 60% |
| light | original_1024_vs_balanced_1024 | 80% | 80% | 20% |
| iid | original_256_vs_original_1024 | 100% | 100% | 100% |
| iid | original_1024_vs_independent_1024 | 80% | 100% | 80% |
| iid | original_1024_vs_balanced_1024 | 80% | 80% | 60% |
| dirichlet | original_256_vs_original_1024 | 100% | 100% | 100% |
| dirichlet | original_1024_vs_independent_1024 | 100% | 100% | 100% |
| dirichlet | original_1024_vs_balanced_1024 | 60% | 80% | 80% |

## Next decision

Retain the fixed-budget cloud comparison and the failed FedProx screen. Use only qualified, feature-level explanation statements with background and approximation caveats. Do not convert feature importance into packet thresholds or claims about device vulnerabilities. A future capacity or feature-ablation experiment would need its own frozen hypothesis and fair centralized/federated comparison; it is not launched or approved by this report.

## Reproduce

Run from the repository root with the existing private cloud-analysis checkpoint staging directory:

```bash
python -m reports.local_diagnostics_2026_10_07.distributions --output outputs/diagnostic-recheck/distributions
python -m reports.local_diagnostics_2026_10_07.sensitivity --output outputs/diagnostic-recheck/sensitivity
python -m reports.local_diagnostics_2026_10_07.summarize --input outputs/diagnostic-recheck --output outputs/diagnostic-recheck/report
```

Source and package hashes, full distributions, per-class validation metrics and all attribution quality statistics are in [evidence.json](evidence.json). Row selections and raw attribution tensors remain in ignored outputs/. No new dependencies were added. Historical reports remain dated records; this report refines their median warning.
