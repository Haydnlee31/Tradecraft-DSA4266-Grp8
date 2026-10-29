# Results index

| Study | Endpoint population | Budget per run | Evidence |
| --- | --- | --- | --- |
| Primary matched methods | Development + retrospective confirmation | 21.6M backward examples | [Results](../reports/methodology_comparison/results.md) |
| Original shared-optimizer confirmation | Previously reserved confirmation, now exposed | 21.6M | [Findings](../reports/federated_shared_optimizer/confirmation-findings.md) |
| MimeLite supplement | Development audit only | 23.1M | [Results](../reports/mimelite_fair_comparison/results.md) |
| Earlier source/architecture studies | Different legacy/official39 populations | Historical protocols | [Index](../studies/historical/README.md) |

The first two share original control endpoints; do not count them as independent
confirmation. Each primary result includes three seeds, all class recalls and
actual false alerts. Round counts differ from logical barriers for MIME.
Communication counts application bytes, not physical-network traffic or latency.

Figures should be generated from the matching study's evidence using its renderer.
Do not mix metrics or curves across populations, budgets, architectures or hardware.
