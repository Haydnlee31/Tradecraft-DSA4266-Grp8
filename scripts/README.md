# Supported commands

Use **`python scripts/study.py --help`** for the primary workflow: `setup`, `check`, `audit`, `train`, `evaluate`, `compare`, `report`, `reproduce`. See [reproduction](../docs/reproduction.md) for runtime, inputs and commands.

| Other entry point | Status |
| --- | --- |
| `reproduce_shared_adam.py` | Input bootstrap and separate compressed strict/portable reproduction |
| `methodology_report.py` | Primary auditor/renderer, called by `study.py` |
| `mimelite_report.py` | Supplement renderer |
| `archive_tools.py` | Verify bytes or restore historical paths for regression tests |
| `study_thesis.py`, `shared_optimizer_thesis_figures.py`, `render_thesis_pdf.py` | Historical integrated writing tools |
| `official39_*` | Frozen historical tooling; follow its locked protocol |

Retired classifier scripts and the superseded single-seed table audit live in `archive/`. Restore them with `archive_tools.py`; do not execute them directly from relocated paths. No script starts cloud infrastructure.
