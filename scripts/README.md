# Active analysis and writing commands

The main study uses `python -m src.experiment`, the research runner described in
[src/models/RESEARCH.md](../src/models/RESEARCH.md), and the official39 and shared
optimizer modules under `src/`. These scripts support audited evaluation and
writing rather than automatically launching a new experiment.

| Scripts | Purpose |
| --- | --- |
| `study_thesis.py`, `shared_optimizer_thesis_figures.py`, `render_thesis_pdf.py` | Rebuild the [thesis and figures](../reports/thesis/README.md) from saved evidence |
| `official39_study_synthesis.py`, `official39_final_report.py`, `official39_closeout.py` | Audit and summarize completed official39 results |
| `official39_ablation_confirmation.py`, `official39_final_evaluation.py`, `official39_test_panel.py` | Frozen historical evaluation tooling; read the locked protocol before use |
| `archive_tools.py` | Verify archived bytes, reconstruct historical paths and run retired regression tests |

The 21 classifier/routing search scripts have moved to
[archive/classifier_search/scripts](../archive/classifier_search/scripts).
They are not part of the current centralized-versus-federated workflow. Their
comments and associated tests are preserved; see the [archive guide](../archive/README.md).
