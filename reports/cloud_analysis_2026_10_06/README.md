# Local post-cloud diagnostics

Read the [results and interpretation](results/README.md). This closes the bounded
FedProx screen and performs an initial validation-only explanation audit, not final
policy validation. No new training or cloud spending was launched.

## Reproduction

Run from the repository root in the existing local environment, with the trusted
user archives and sampled train/validation files available. Use fresh output paths:

```bash
python -m reports.cloud_analysis_2026_10_06.prepare \
  --archives /path/to/downloaded/archives --output outputs/cloud-analysis-recheck
python -m src.explain.research \
  --confirmation outputs/cloud-analysis-recheck/study.json \
  --plan configs/cloud_fedprox_plan.json \
  --output outputs/cloud-analysis-recheck/explanations \
  --per-class 8 --background 128 --nsamples 256
python -m reports.cloud_analysis_2026_10_06.summarize \
  --input outputs/cloud-analysis-recheck \
  --output outputs/cloud-analysis-recheck/report
python -m unittest discover -v
```

The plan hash binds the completed FedProx training screen, not a new SHAP selection
protocol. The explanation settings are recorded separately in the output summary.
Archive preparation checks the three full FedProx-stage runs, historical FedAvg
parity, frozen settings, checkpoints, scaler/partition hashes and old archive hashes.
It reuses the previous 19-run audit for earlier experiments. GPU pilot checks were
previously reviewed; this preparation script does not rerun them on a GPU.

The explanation runner checks inference-source hashes, training/validation hashes,
checkpoint integrity and full-validation prediction parity before SHAP. It now
supports arbitrary recorded seed IDs and represents single-seed stability as null.
All original code comments remain. Training code and dependencies are unchanged.

## Git checkpoint and follow-up

Commit the report, aggregate evidence, preparation/summarization scripts and the
seed-handling correction together after review. Do not commit outputs/, downloaded
archives, datasets, checkpoint/scaler files or raw explanation arrays. The recorded
local run passed all 46 existing tests and completed all 24 SHAP repeats across 12
models. Remote CI has not yet been run for this uncommitted checkpoint.

Before another training experiment, inspect the observed feature-regime differences
across the fixed upstream splits and run a bounded local background/integration
sensitivity check. This is a diagnostic hypothesis, not permission to reshape the
validation distribution, tune against the test split or expand the failed FedProx
sweep. Final feature-specific policy recommendations remain provisional.
