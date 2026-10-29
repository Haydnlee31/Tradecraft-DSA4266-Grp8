# Study scope

The thesis investigates the performance/communication trade-off between pooled
centralized training and simulated non-IID federated training. It asks whether
shared-gradient optimization closes the gap and whether blockwise compression
preserves that improvement at lower payload cost.

## Primary evidence

The [primary registry](../studies/primary/study.json) references the existing
frozen plan, evidence and renderer. It does not copy or alter the protocol.
All 11 configurations remain in the report, including unfavorable results:

| Role | Configurations |
| --- | --- |
| Main narrative | Central Adam, FedAvg-Adam, Shared Adam, compressed Shared Adam |
| Supporting alternatives | FedProx mu=0.01 and 0.1, SCAFFOLD-SGD, full MIME |
| Optimizer controls | Central SGD, FedAvg-SGD, centralized MIME base |

The main study uses 21.6M backward examples per run, three seeds and the already
evaluated confirmation cohort. It is a fixed-setting retrospective comparison.
Matching this budget does not equalize FLOPs, time, synchronization or tuning effort.

The separate [MimeLite supplement](../studies/mimelite-supplement/README.md) uses
23.1M backward examples and development-only scoring. Neither its endpoint
values nor the earlier unverified MimeLite claims belong in the primary table.

## Conclusions to emphasize

Shared Adam substantially narrows the observed centralized/FedAvg gap. Compression
preserves most of that gain while reducing message size; it retains frequent
synchronization. This is a measured engineering comparison in one data collection
and client partition, not a new optimizer theorem, universal superiority, privacy
guarantee or a solution to rare-class detection.

The original heavy/light comparison, cloud scaling, SHAP and classifier searches
are background or supporting material. No dashboard or further classifier search
is required. SHAP findings remain feature-level observations, not causal security
rules or device vulnerabilities. [Historical index](../studies/historical/README.md).
