# Primary study: centralized/federated quality and communication

The report's main question is whether shared-gradient optimization narrows the
centralized/FedAvg gap and whether compression retains its quality at lower
payload cost. The main four methods are Central Adam, FedAvg-Adam, Shared Adam
and compressed Shared Adam. All supporting methods and controls remain in the
11-configuration, three-seed table.

- [Methodology](../../docs/methodology.md)
- [Results and figures](../../reports/methodology_comparison/results.md)
- [Interpretation](../../reports/methodology_comparison/findings.md)
- [Reproduction commands](../../docs/reproduction.md)
- [Study registry](study.json): references immutable existing evidence, not copied settings.

This is the 21.6M-backward-example, 500k-fit-row, retrospective-confirmation
study. It does not represent the legacy heavy model or full-source training.
Use `python scripts/study.py --help` from the repository root.
