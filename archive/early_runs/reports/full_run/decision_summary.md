# Tradecraft decision summary

Status: **ready**

Recommendation: **centralized-heavy**

Configurations are selected by validation macro-F1. Test metrics are used only for the final comparison.

Illustrative research gates (not operational security requirements), applied on validation: loss=sqrt_weighted_ce, macro-F1 ≥ 0.600, every class recall ≥ 0.200, model ≤ 1.000 MiB, at least 3 seeds, and macro-F1 drop from the best ≤ 0.020.

| lane | seeds | val macro-F1 | test macro-F1 | worst recall | model MiB | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| centralized-heavy | 3 | 0.6558 ± 0.0119 | 0.6580 ± 0.0126 | 0.2542 | 0.7591 | yes |
| centralized-light | 3 | 0.6458 ± 0.0034 | 0.6466 ± 0.0036 | 0.2065 | 0.0211 | no |
| federated-light | 3 | 0.4409 ± 0.0267 | 0.4412 ± 0.0304 | 0.0062 | 0.0211 | no |

## Failed validation gates

- centralized-light: validation recall missing or below threshold for Brute Force
- federated-light: validation macro-F1 0.4409 < 0.6000; validation recall missing or below threshold for DoS, Web-based, Brute Force

## Warnings

- ignored non-training JSON: data_audit.json
- ignored non-training JSON: environment.json
- ignored non-training JSON: normalization_audit.json
- ignored non-training JSON: study_summary.json

Model MiB is derived from parameter tensors (float32 unless a report provides an exact byte count). It is not measured runtime RAM. No latency, power, or edge-hardware measurement is inferred here.
