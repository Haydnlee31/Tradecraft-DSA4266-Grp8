# Frozen validation-only research decision

**Research comparison only. No deployment is authorized.**

Gate-based research choice: **centralized-heavy**. Engine status: `ready`.

The frozen recipes are not re-tuned. Default non-IID selection and all existing metric, size and seed gates are preserved.

# Tradecraft decision summary

Status: **ready**

Recommendation: **centralized-heavy**

Configurations are selected by validation macro-F1. Test metrics are used only for descriptive comparison when available; N/A means not evaluated.

Illustrative research gates (not operational security requirements), applied on validation: loss=sqrt_weighted_ce, macro-F1 ≥ 0.600, every class recall ≥ 0.200, model ≤ 1.000 MiB, at least 3 seeds, and macro-F1 drop from the best ≤ 0.020.

| lane | seeds | val macro-F1 | test macro-F1 | worst val recall | model MiB | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| centralized-heavy | 3 | 0.6747 ± 0.0031 | N/A | 0.2551 | 0.7591 | yes |
| centralized-light | 3 | 0.6583 ± 0.0029 | N/A | 0.1649 | 0.0211 | no |
| federated-light | 3 | 0.5557 ± 0.0128 | N/A | 0.0143 | 0.0211 | no |

## Failed validation gates

- centralized-light: validation recall missing or below threshold for Brute Force
- federated-light: validation macro-F1 0.5557 < 0.6000; validation recall missing or below threshold for DoS, Web-based, Brute Force

Model MiB is derived from parameter tensors (float32 unless a report provides an exact byte count). It is not measured runtime RAM. No latency, power, or edge-hardware measurement is inferred here.


## New-seed-only sensitivity

Status: `incomplete`; provisional research choice: `centralized-heavy`. Only seeds 1/2 are new, so the unchanged requirement of three seeds is not met.

## All candidates, including IID diagnostic control

| Lane / partition | Seeds | Mean validation F1 | Worst mean class recall |
|---|---|---:|---:|
| centralized-heavy / central | [0, 1, 2] | 0.6747 | 0.2551 |
| centralized-light / central | [0, 1, 2] | 0.6583 | 0.1649 |
| federated-light / iid | [0, 1, 2] | 0.6328 | 0.1279 |
| federated-light / dirichlet | [0, 1, 2] | 0.5557 | 0.0143 |

## Every class: mean validation recall

| Class | Heavy | Light | IID control | Non-IID target |
|---|---:|---:|---:|---:|
| Benign | 78.03% | 76.43% | 75.77% | 95.70% |
| DDoS | 79.63% | 79.19% | 80.00% | 98.65% |
| DoS | 82.57% | 79.23% | 76.08% | 17.34% |
| Recon | 74.97% | 76.56% | 77.35% | 52.08% |
| Web-based | 31.05% | 29.78% | 12.79% | 1.43% |
| Brute Force | 25.51% | 16.49% | 14.80% | 14.80% |
| Spoofing | 81.25% | 77.36% | 70.41% | 48.48% |
| Mirai | 99.50% | 99.44% | 99.25% | 99.18% |

## Error types: full validation, each seed

Each attack row is partitioned into correct category, predicted benign, and wrong attack category. These supplemental counts do not replace the eight-class target or gate metrics.

| Model / seed | True class | Support | Correct category | Predicted benign | Wrong attack category |
|---|---|---:|---:|---:|---:|
| heavy-seed-0 | DDoS | 166191 | 133572 | 1 | 32618 |
| heavy-seed-0 | DoS | 61572 | 50138 | 1 | 11433 |
| heavy-seed-0 | Recon | 8551 | 6350 | 771 | 1430 |
| heavy-seed-0 | Web-based | 628 | 210 | 21 | 397 |
| heavy-seed-0 | Brute Force | 277 | 70 | 25 | 182 |
| heavy-seed-0 | Spoofing | 11900 | 9632 | 1076 | 1192 |
| heavy-seed-0 | Mirai | 58279 | 58010 | 0 | 269 |
| heavy-seed-1 | DDoS | 166191 | 133190 | 1 | 33000 |
| heavy-seed-1 | DoS | 61572 | 50513 | 1 | 11058 |
| heavy-seed-1 | Recon | 8551 | 6625 | 604 | 1322 |
| heavy-seed-1 | Web-based | 628 | 251 | 14 | 363 |
| heavy-seed-1 | Brute Force | 277 | 73 | 17 | 187 |
| heavy-seed-1 | Spoofing | 11900 | 9358 | 964 | 1578 |
| heavy-seed-1 | Mirai | 58279 | 57985 | 0 | 294 |
| heavy-seed-2 | DDoS | 166191 | 130251 | 1 | 35939 |
| heavy-seed-2 | DoS | 61572 | 51878 | 1 | 9693 |
| heavy-seed-2 | Recon | 8551 | 6256 | 755 | 1540 |
| heavy-seed-2 | Web-based | 628 | 124 | 18 | 486 |
| heavy-seed-2 | Brute Force | 277 | 69 | 28 | 180 |
| heavy-seed-2 | Spoofing | 11900 | 10015 | 1040 | 845 |
| heavy-seed-2 | Mirai | 58279 | 57964 | 0 | 315 |
| light-seed-0 | DDoS | 166191 | 133901 | 1 | 32289 |
| light-seed-0 | DoS | 61572 | 47506 | 2 | 14064 |
| light-seed-0 | Recon | 8551 | 6604 | 880 | 1067 |
| light-seed-0 | Web-based | 628 | 191 | 26 | 411 |
| light-seed-0 | Brute Force | 277 | 46 | 34 | 197 |
| light-seed-0 | Spoofing | 11900 | 9024 | 1122 | 1754 |
| light-seed-0 | Mirai | 58279 | 57946 | 0 | 333 |
| light-seed-1 | DDoS | 166191 | 132317 | 1 | 33873 |
| light-seed-1 | DoS | 61572 | 48389 | 1 | 13182 |
| light-seed-1 | Recon | 8551 | 6209 | 844 | 1498 |
| light-seed-1 | Web-based | 628 | 197 | 21 | 410 |
| light-seed-1 | Brute Force | 277 | 46 | 32 | 199 |
| light-seed-1 | Spoofing | 11900 | 9628 | 1024 | 1248 |
| light-seed-1 | Mirai | 58279 | 57954 | 0 | 325 |
| light-seed-2 | DDoS | 166191 | 128578 | 2 | 37611 |
| light-seed-2 | DoS | 61572 | 50460 | 1 | 11111 |
| light-seed-2 | Recon | 8551 | 6826 | 816 | 909 |
| light-seed-2 | Web-based | 628 | 173 | 20 | 435 |
| light-seed-2 | Brute Force | 277 | 45 | 26 | 206 |
| light-seed-2 | Spoofing | 11900 | 8966 | 1035 | 1899 |
| light-seed-2 | Mirai | 58279 | 57955 | 1 | 323 |
| iid-seed-0 | DDoS | 166191 | 133412 | 25 | 32754 |
| iid-seed-0 | DoS | 61572 | 46707 | 2 | 14863 |
| iid-seed-0 | Recon | 8551 | 6738 | 1023 | 790 |
| iid-seed-0 | Web-based | 628 | 77 | 35 | 516 |
| iid-seed-0 | Brute Force | 277 | 41 | 46 | 190 |
| iid-seed-0 | Spoofing | 11900 | 8196 | 1560 | 2144 |
| iid-seed-0 | Mirai | 58279 | 57838 | 4 | 437 |
| iid-seed-1 | DDoS | 166191 | 132444 | 19 | 33728 |
| iid-seed-1 | DoS | 61572 | 46978 | 1 | 14593 |
| iid-seed-1 | Recon | 8551 | 6536 | 940 | 1075 |
| iid-seed-1 | Web-based | 628 | 115 | 33 | 480 |
| iid-seed-1 | Brute Force | 277 | 41 | 39 | 197 |
| iid-seed-1 | Spoofing | 11900 | 8500 | 1409 | 1991 |
| iid-seed-1 | Mirai | 58279 | 57857 | 2 | 420 |
| iid-seed-2 | DDoS | 166191 | 132980 | 21 | 33190 |
| iid-seed-2 | DoS | 61572 | 46840 | 1 | 14731 |
| iid-seed-2 | Recon | 8551 | 6568 | 1073 | 910 |
| iid-seed-2 | Web-based | 628 | 49 | 47 | 532 |
| iid-seed-2 | Brute Force | 277 | 41 | 52 | 184 |
| iid-seed-2 | Spoofing | 11900 | 8440 | 1634 | 1826 |
| iid-seed-2 | Mirai | 58279 | 57839 | 2 | 438 |
| dirichlet-seed-0 | DDoS | 166191 | 161533 | 26 | 4632 |
| dirichlet-seed-0 | DoS | 61572 | 13862 | 4 | 47706 |
| dirichlet-seed-0 | Recon | 8551 | 4593 | 3308 | 650 |
| dirichlet-seed-0 | Web-based | 628 | 27 | 354 | 247 |
| dirichlet-seed-0 | Brute Force | 277 | 41 | 166 | 70 |
| dirichlet-seed-0 | Spoofing | 11900 | 5838 | 5538 | 524 |
| dirichlet-seed-0 | Mirai | 58279 | 57817 | 34 | 428 |
| dirichlet-seed-1 | DDoS | 166191 | 164891 | 34 | 1266 |
| dirichlet-seed-1 | DoS | 61572 | 9401 | 4 | 52167 |
| dirichlet-seed-1 | Recon | 8551 | 4364 | 3453 | 734 |
| dirichlet-seed-1 | Web-based | 628 | 0 | 358 | 270 |
| dirichlet-seed-1 | Brute Force | 277 | 41 | 183 | 53 |
| dirichlet-seed-1 | Spoofing | 11900 | 5866 | 5641 | 393 |
| dirichlet-seed-1 | Mirai | 58279 | 57790 | 40 | 449 |
| dirichlet-seed-2 | DDoS | 166191 | 165442 | 29 | 720 |
| dirichlet-seed-2 | DoS | 61572 | 8768 | 4 | 52800 |
| dirichlet-seed-2 | Recon | 8551 | 4402 | 3653 | 496 |
| dirichlet-seed-2 | Web-based | 628 | 0 | 443 | 185 |
| dirichlet-seed-2 | Brute Force | 277 | 41 | 188 | 48 |
| dirichlet-seed-2 | Spoofing | 11900 | 5603 | 5874 | 423 |
| dirichlet-seed-2 | Mirai | 58279 | 57801 | 2 | 476 |

## False alerts: each seed

| Model / seed | Benign support | False alerts | Rate |
|---|---:|---:|---:|
| heavy-seed-0 | 19859 | 4041 | 20.35% |
| heavy-seed-1 | 19859 | 4762 | 23.98% |
| heavy-seed-2 | 19859 | 4287 | 21.59% |
| light-seed-0 | 19859 | 4500 | 22.66% |
| light-seed-1 | 19859 | 4680 | 23.57% |
| light-seed-2 | 19859 | 4862 | 24.48% |
| iid-seed-0 | 19859 | 4742 | 23.88% |
| iid-seed-1 | 19859 | 5108 | 25.72% |
| iid-seed-2 | 19859 | 4583 | 23.08% |
| dirichlet-seed-0 | 19859 | 950 | 4.78% |
| dirichlet-seed-1 | 19859 | 874 | 4.40% |
| dirichlet-seed-2 | 19859 | 737 | 3.71% |

## Interpretation safeguards

- Seed 0 influenced screening and is not independent confirmation.
- All seeds reuse the same validation set; these are training-seed stability checks, not independent data confirmation.
- Fixed partition seed measures no variability across client partitions.
- Normalization differs between central-light and federated-light; dropout differs between heavy and light.
- No deployment or physical edge claim; no cloud resources.
- heavy-seed-2: individual-seed recall below 0.20 for Web-based. Existing gates use means, not per-seed minima.
- light-seed-0: individual-seed recall below 0.20 for Brute Force. Existing gates use means, not per-seed minima.
- light-seed-1: individual-seed recall below 0.20 for Brute Force. Existing gates use means, not per-seed minima.
- light-seed-2: individual-seed recall below 0.20 for Brute Force. Existing gates use means, not per-seed minima.
- iid-seed-0: individual-seed recall below 0.20 for Web-based, Brute Force. Existing gates use means, not per-seed minima.
- iid-seed-1: individual-seed recall below 0.20 for Web-based, Brute Force. Existing gates use means, not per-seed minima.
- iid-seed-2: individual-seed recall below 0.20 for Web-based, Brute Force. Existing gates use means, not per-seed minima.
- dirichlet-seed-0: individual-seed recall below 0.20 for Web-based, Brute Force. Existing gates use means, not per-seed minima.
- dirichlet-seed-1: individual-seed recall below 0.20 for DoS, Web-based, Brute Force. Existing gates use means, not per-seed minima.
- dirichlet-seed-2: individual-seed recall below 0.20 for DoS, Web-based, Brute Force. Existing gates use means, not per-seed minima.
- Decision gates are unchanged illustrative research preferences, not operational standards.
- No false-alert-rate ceiling exists in the inherited gates; high false alerts remain a warning, not a hidden new gate.
- All-seed eligibility includes reused screening seed 0; new-seed sensitivity retains the three-seed requirement and is incomplete.
- IID remains a diagnostic control and cannot replace the non-IID target.
- Model MiB describes saved float32 parameter tensors, not runtime RAM or edge feasibility.

## Human-review policy hypotheses

- Do not let a model benign output suppress independent web/application or authentication alerts.
- Review Recon/Spoofing alerts with service context rather than automatically blocking traffic.
- Preserve attack alerts while treating uncertain DoS/DDoS subtypes separately.
- SHAP feature rankings do not establish causal rules, thresholds, or device vulnerability.
- These are investigation proposals; no operational policy or decision threshold is changed.
