# Tradecraft decision summary

Status: **incomplete**

Recommendation: none yet.

Configurations are selected by validation macro-F1. Test metrics are used only for the final comparison.

Current gates: loss=sqrt_weighted_ce, macro-F1 ≥ 0.600, every class recall ≥ 0.200, model ≤ 1.000 MiB, at least 3 seeds, and macro-F1 drop from the best ≤ 0.020.

| lane | seeds | val macro-F1 | test macro-F1 | worst recall | model MiB | eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| centralized-heavy | — | — | — | — | — | no report |
| centralized-light | 3 | 0.6458 ± 0.0016 | 0.6455 ± 0.0016 | 0.1970 | 0.0211 | no |
| federated-light | — | — | — | — | — | no report |

## Blocking issues

- missing centralized-heavy report with loss=sqrt_weighted_ce
- missing federated-light report with loss=sqrt_weighted_ce
- no candidate passes all metric and size constraints

## Warnings

- centralized-heavy has only these excluded losses: ['weighted_ce']

Model MiB is derived from parameter tensors (float32 unless a report provides an exact byte count). It is not measured runtime RAM. No latency, power, or edge-hardware measurement is inferred here.
