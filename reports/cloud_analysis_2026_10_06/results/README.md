# Cloud diagnostic analysis — 2026-10-06

## Decision

Keep the cloud stopped. Neither FedProx candidate passes the frozen promotion gates. This report diagnoses the existing models; it does not select a deployable model or authorize another sweep.

The artifact audit reproduced the historical FedAvg bridge exactly: all round metrics/task losses and the selected best tensors. The local CPU explanation run independently reproduced the full validation confusion matrix for all 12 cloud checkpoints before computing attributions. No test data was opened.

## FedProx screen

| Configuration | Macro-F1 | F1 delta | False alerts | Gate A | Gate B | Training + validation seconds |
|---|---:|---:|---:|---|---|---:|
| fedavg_bridge | 0.55251 | +0.00000 | 3.56% | False | False | 523.2 |
| fedprox_mu001 | 0.55702 | +0.00451 | 4.38% | False | False | 596.3 |
| fedprox_mu01 | 0.57754 | +0.02503 | 12.55% | False | False | 599.2 |

Gate A: F1 gain ≥0.01, false-alert increase ≤0.02, every class recall loss ≤0.02. Gate B requires a 0.05 absolute false-alert reduction and is impossible from this 0.03555 baseline. The reference row is not a candidate. Criteria were not relaxed after seeing results.

## Matched light-model evidence

| Lane | Seeds | Macro-F1 mean ± sample SD | Mean false alerts |
|---|---|---:|---:|
| light | 7/8/9 | 0.68722 ± 0.01278 | 20.99% |
| iid | 7/8/9 | 0.63455 ± 0.00448 | 25.50% |
| dirichlet | 7/8/9 | 0.55589 ± 0.00632 | 4.27% |

Seed 7 informed earlier screening; seeds 8/9 are the new confirmation seeds. The heavy reference has only seed 7 here and different normalization/dropout. It cannot establish a pure model-capacity effect. See the earlier cloud study for all per-class means.

## Learning curves: budget limits, not convergence claims

| Run | Best step | Final step | Macro-F1 gain over last 10 steps |
|---|---:|---:|---:|
| cloud-reference/dirichlet-seed7 | 59 | 60 | +0.00433 |
| cloud-reference/heavy-seed7 | 33 | 60 | -0.02548 |
| cloud-tuning/dirichlet-120-seed7 | 111 | 120 | +0.00168 |
| cloud-tuning/heavy-lr0003-seed7 | 39 | 60 | -0.01276 |
| cloud-loss/dirichlet-weighted-ce-seed7 | 58 | 60 | +0.00452 |
| cloud-controls/iid-layernorm-seed7 | 59 | 60 | +0.01092 |
| cloud-controls/light-layernorm-seed7 | 50 | 60 | -0.01207 |
| cloud-confirmation/dirichlet-seed8 | 59 | 60 | +0.00610 |
| cloud-confirmation/dirichlet-seed9 | 56 | 60 | +0.00414 |
| cloud-confirmation/iid-seed8 | 59 | 60 | +0.00442 |
| cloud-confirmation/iid-seed9 | 60 | 60 | +0.00807 |
| cloud-confirmation/light-seed8 | 57 | 60 | +0.00606 |
| cloud-confirmation/light-seed9 | 46 | 60 | +0.00803 |
| cloud-partitions/dirichlet-seed7-partition1 | 58 | 60 | +0.00049 |
| cloud-partitions/dirichlet-seed7-partition2 | 60 | 60 | +0.01120 |
| fedavg_bridge | 59 | 60 | +0.00433 |
| fedprox_mu001 | 59 | 60 | +0.00132 |
| fedprox_mu01 | 60 | 60 | +0.00996 |

A positive tail gain is descriptive, not proof that more rounds will satisfy the safety gates. The existing 120-round FedAvg experiment already failed promotion. Do not restart it or expand FedProx coefficients.

## Expected-gradients numerical quality

All models share 64 balanced validation examples plus 59 error-enriched examples; background: 128 training rows. Two Monte Carlo repeats, 256 samples each. These are diagnostic samples, not population performance estimates.

| Model | Mean absolute residual / mean absolute output difference | Mean repeat top-5 overlap |
|---|---:|---:|
| heavy-seed-7 | 7.9% | 100.0% |
| light-seed-7 | 9.4% | 90.0% |
| light-seed-8 | 9.0% | 90.0% |
| light-seed-9 | 7.9% | 95.0% |
| iid-seed-7 | 8.7% | 95.0% |
| iid-seed-8 | 8.1% | 95.0% |
| iid-seed-9 | 8.5% | 97.5% |
| dirichlet-seed-7 | 8.4% | 95.0% |
| dirichlet-seed-8 | 8.6% | 97.5% |
| dirichlet-seed-9 | 8.2% | 97.5% |
| fedprox_mu001-seed-7 | 8.4% | 95.0% |
| fedprox_mu01-seed-7 | 8.4% | 95.0% |

High residual means the approximation does not closely reconstruct the explained logit difference. High repeat overlap alone is not sufficient: both repeats may share approximation bias. No precise SHAP-derived thresholds or automated blocking rules are justified.

## Rare-class errors on the full validation split

| Model | True class | Recall | Leading wrong destination (fraction of true class) |
|---|---|---:|---|
| heavy-seed-7 | Web-based | 37.58% | Recon: 35.83% |
| heavy-seed-7 | Brute Force | 23.83% | Recon: 42.96% |
| heavy-seed-7 | DoS | 78.70% | DDoS: 21.07% |
| light-seed-7 | Web-based | 36.78% | Spoofing: 34.55% |
| light-seed-7 | Brute Force | 21.30% | Recon: 41.52% |
| light-seed-7 | DoS | 82.01% | DDoS: 17.75% |
| light-seed-8 | Web-based | 29.94% | Recon: 38.22% |
| light-seed-8 | Brute Force | 19.13% | Recon: 47.65% |
| light-seed-8 | DoS | 92.02% | DDoS: 7.72% |
| light-seed-9 | Web-based | 25.96% | Recon: 40.76% |
| light-seed-9 | Brute Force | 18.41% | Recon: 48.38% |
| light-seed-9 | DoS | 77.04% | DDoS: 22.76% |
| iid-seed-7 | Web-based | 12.42% | Spoofing: 42.20% |
| iid-seed-7 | Brute Force | 15.52% | Recon: 41.52% |
| iid-seed-7 | DoS | 78.45% | DDoS: 21.34% |
| iid-seed-8 | Web-based | 14.97% | Recon: 47.77% |
| iid-seed-8 | Brute Force | 14.80% | Recon: 56.32% |
| iid-seed-8 | DoS | 78.25% | DDoS: 21.53% |
| iid-seed-9 | Web-based | 15.29% | Recon: 44.75% |
| iid-seed-9 | Brute Force | 14.80% | Recon: 45.49% |
| iid-seed-9 | DoS | 77.10% | DDoS: 22.65% |
| dirichlet-seed-7 | Web-based | 3.18% | Benign: 72.29% |
| dirichlet-seed-7 | Brute Force | 14.80% | Benign: 70.76% |
| dirichlet-seed-7 | DoS | 16.55% | DDoS: 83.30% |
| dirichlet-seed-8 | Web-based | 6.69% | Benign: 58.92% |
| dirichlet-seed-8 | Brute Force | 14.80% | Benign: 59.21% |
| dirichlet-seed-8 | DoS | 15.55% | DDoS: 84.31% |
| dirichlet-seed-9 | Web-based | 2.87% | Benign: 76.27% |
| dirichlet-seed-9 | Brute Force | 14.80% | Benign: 71.12% |
| dirichlet-seed-9 | DoS | 14.76% | DDoS: 85.13% |
| fedprox_mu001-seed-7 | Web-based | 3.03% | Benign: 64.81% |
| fedprox_mu001-seed-7 | Brute Force | 14.80% | Benign: 63.90% |
| fedprox_mu001-seed-7 | DoS | 17.09% | DDoS: 82.76% |
| fedprox_mu01-seed-7 | Web-based | 2.87% | Recon: 36.31% |
| fedprox_mu01-seed-7 | Brute Force | 14.80% | Benign: 40.07% |
| fedprox_mu01-seed-7 | DoS | 27.89% | DDoS: 71.92% |

## Rare-class feature hypotheses (shared balanced core)

| Model | True class | Top five absolute true-class-logit features | Repeat overlap |
|---|---|---|---:|
| heavy-seed-7 | Web-based | IAT, Header_Length, Weight, Number, rst_count | 100% |
| heavy-seed-7 | Brute Force | Number, IAT, Weight, syn_count, Variance | 100% |
| heavy-seed-7 | DoS | Magnitue, Weight, Number, rst_count, syn_count | 100% |
| light-seed-7 | Web-based | IAT, Weight, Header_Length, Magnitue, Tot sum | 100% |
| light-seed-7 | Brute Force | IAT, flow_duration, Weight, Number, syn_flag_number | 100% |
| light-seed-7 | DoS | IAT, Weight, Max, syn_flag_number, Magnitue | 60% |
| iid-seed-7 | Web-based | IAT, Weight, Number, Magnitue, AVG | 100% |
| iid-seed-7 | Brute Force | IAT, Number, Weight, syn_count, Variance | 80% |
| iid-seed-7 | DoS | Number, IAT, Weight, AVG, syn_count | 80% |
| dirichlet-seed-7 | Web-based | Number, IAT, Weight, Tot sum, AVG | 80% |
| dirichlet-seed-7 | Brute Force | Number, syn_count, Weight, IAT, ack_flag_number | 100% |
| dirichlet-seed-7 | DoS | IAT, Weight, Number, AVG, Protocol Type | 100% |
| fedprox_mu001-seed-7 | Web-based | Number, IAT, Weight, Tot sum, AVG | 100% |
| fedprox_mu001-seed-7 | Brute Force | Number, syn_count, IAT, ack_flag_number, Variance | 80% |
| fedprox_mu001-seed-7 | DoS | IAT, Number, Weight, AVG, Protocol Type | 100% |
| fedprox_mu01-seed-7 | Web-based | Weight, IAT, AVG, Number, Min | 100% |
| fedprox_mu01-seed-7 | Brute Force | IAT, Number, syn_count, ack_flag_number, Variance | 100% |
| fedprox_mu01-seed-7 | DoS | IAT, Number, Weight, AVG, Variance | 100% |

### Rare raw-label slices (still predicting eight classes)

| Raw label | Validation support | Central light seed 7 correct | FedAvg seed 7 correct | FedProx 0.1 correct |
|---|---:|---:|---:|---:|
| Backdoor_Malware | 88 | 46 | 4 | 5 |
| BrowserHijacking | 149 | 33 | 2 | 5 |
| CommandInjection | 124 | 66 | 1 | 4 |
| DictionaryBruteForce | 277 | 59 | 41 | 41 |
| SqlInjection | 144 | 42 | 10 | 3 |
| Uploading_Attack | 33 | 14 | 0 | 0 |
| XSS | 90 | 30 | 3 | 1 |

FedAvg and both FedProx candidates detect the same Brute Force rows: True. Counts and all model/raw-label prediction distributions are preserved in evidence.json. These validation slices are descriptive and small; they do not justify subclass-level performance guarantees.

## Distribution warning before any further scaling

| Class | Train median IAT | Validation median IAT | Train median Number / Weight | Validation median Number / Weight |
|---|---:|---:|---|---|
| Benign | 1.66516e+08 | 0.0585768 | 13.5 / 244.6 | 5.5 / 38.5 |
| Web-based | 23.9194 | 1.66482e+08 | 5.5 / 38.5 | 13.5 / 244.6 |
| Brute Force | 1.66577e+08 | 0.0511063 | 13.5 / 244.6 | 5.5 / 38.5 |

These features also rank highly in the explanation sample. Their very different medians are consistent with changing mixtures of feature regimes across the upstream splits, not proof of corruption, leakage or causality. IAT units/semantics have not been independently verified. Benign and Brute Force have overlapping Number/Weight quantiles. DoS and DDoS both have 10th/50th/90th quantiles of Number=9.5 and Weight=141.55. Such overlap does not establish indistinguishability in the complete feature space. Preserve the upstream splits; do not pool or resplit them.

## Explanation limits that need a local follow-up

The sampled natural-prevalence background contains no Web-based or Brute Force rows. A stratified-background sensitivity comparison would change the reference population and must be labelled as such, not silently substituted to obtain prettier explanations.

| Model | Error cases with absolute residual ≥ decision margin / sampled errors |
|---|---:|
| heavy-seed-7 | 10 / 58 |
| light-seed-7 | 20 / 62 |
| light-seed-8 | 17 / 54 |
| light-seed-9 | 18 / 64 |
| iid-seed-7 | 9 / 66 |
| iid-seed-8 | 11 / 66 |
| iid-seed-9 | 7 / 66 |
| dirichlet-seed-7 | 2 / 74 |
| dirichlet-seed-8 | 5 / 76 |
| dirichlet-seed-9 | 5 / 74 |
| fedprox_mu001-seed-7 | 2 / 75 |
| fedprox_mu01-seed-7 | 5 / 76 |

These cases are unsuitable for precise signed local-rule interpretation at this integration budget. Global repeat stability does not rescue an unreliable individual explanation.

| Lane | Web top-5 seed overlap | Brute Force overlap | DoS overlap |
|---|---:|---:|---:|
| light | 80.0% | 100.0% | 66.7% |
| iid | 66.7% | 66.7% | 66.7% |
| dirichlet | 60.0% | 80.0% | 53.3% |

Feature quantiles (10th/50th/90th) for the above feature union on train and validation are saved in evidence.json, stratified by Benign/Web/Brute Force/DoS/DDoS. They are descriptive overlap checks, not proof of distribution equality or a causal mechanism.

## Policy interpretation and next step

- Treat predictions as analyst-review signals, not automatic blocking decisions: no model is operationally approved.
- A non-IID benign prediction is not evidence that Web or Brute Force activity is safe. Independent application/authentication telemetry would be needed for such decisions; it is not supplied by these flow CSVs.
- DoS/DDoS confusion motivates reviewing flood-related flow features, not assigning device-specific vulnerabilities.
- Inspect residuals, seed stability and background sensitivity before making feature-specific recommendations. No physical edge latency/power claims are made.
- Preserve the failed screens. Credits remaining are not evidence that more parameters will solve these errors.

## Limits and provenance

Only seed 7 is available for heavy and FedProx here; their between-seed stability is null, not zero. The three light-model lanes use seeds 7/8/9. Attributions are on logits with a shared natural-prevalence training background; correlated flow statistics and off-manifold interpolations limit interpretation. A second background/sample sensitivity study has not been run. Historical test inspection remains disclosed in the original study; the test split cannot be called untouched across the entire project.

Aggregate evidence, source/package hashes, all class metrics and numerical diagnostics: [evidence.json](evidence.json). Private rows, scaler/checkpoint files and attribution arrays remain under ignored outputs/.
