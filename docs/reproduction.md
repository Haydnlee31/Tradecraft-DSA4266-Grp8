# Reproduce the primary report

## Environment

Exact primary replay uses **Python 3.13.9, PyTorch 2.14.0, NumPy 2.5.3, Flower
1.38.0 on macOS ARM CPU**, two intra-op threads and one inter-op thread. The
existing strict guards remain. No RONIN session is needed. Generic cross-platform
unit tests do not guarantee bitwise numerical equality of full training.

From a new clone of branch `cleanup/focused-study`, with Python 3.13.9 installed:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-methodology.txt
python -m pip check
```

Do not recreate or overwrite a working environment. On Linux/WSL, the separate
[compressed-model portable route](../reports/federated_shared_optimizer/reproduction-guide.md)
reports observed metric differences; the all-method primary wrapper does not
offer a silent portable fallback.

## Route 1: audit/rebuild evidence, no training

```bash
python scripts/study.py audit
python scripts/study.py report --output outputs/report-rebuild
```

This recalculates metrics from committed confusion matrices and renders tables,
per-seed CSV and figures. It needs the software environment, but no private data
or checkpoints. It verifies presentation of recorded results, not fresh training.

## Route 2: download inputs and reproduce all 33 runs

```bash
python scripts/study.py reproduce --output outputs/primary-reproduction
```

The command downloads the pinned public release asset, verifies it, checks the
runtime and source contracts, trains every arm/seed from initialization, calibrates
on development, scores the already exposed confirmation with frozen thresholds,
compares all 33 endpoints to the published primary results, then builds a report.
It does not rerun searches or select favorable checkpoints. The output directory
must not already exist. The bootstrap never installs into your chosen environment.

For an already downloaded archive:

```bash
python scripts/study.py reproduce --archive /path/to/tradecraft-shared-adam-inputs-v1.tgz --output outputs/primary-local-inputs
```

All stages can also run separately:

```bash
python scripts/study.py setup
python scripts/study.py check
python scripts/study.py train --output outputs/primary-run
python scripts/study.py evaluate --run outputs/primary-run
python scripts/study.py compare --run outputs/primary-run
python scripts/study.py report --run outputs/primary-run --output outputs/primary-report
```

`train` saves all development endpoints and thresholds before `evaluate` opens
confirmation for inference. `compare` checks weights, sampled-row hashes, all
development/confirmation metrics, logits, settings, work and recovery results
for all 33 runs. Wall times and serialization packaging are not equality targets.
The comparison receipt is `reproduction-checks.json`; the complete command also
writes `report/` under the run directory.

The historical primary trainer has final-two-step recovery tests but **no
user-facing interrupted-job resume**. Preserve a partial run and choose a new
directory to retrain. A completed training stage can proceed to evaluation without
retraining. If evaluation/report already exists, preserve it and use `compare`
or a new report directory. No command discards failures or relaxes strict checks.

## Route 3: source-data reconstruction

See [data-protocol.md](data-protocol.md). Raw acquisition and the full provenance
chain are optional advanced work, not required to retrain the released cohorts.
This cleanup does not assert a newly executed full raw-data reconstruction.

## Tests and other studies

```bash
python -m unittest discover -q
python scripts/archive_tools.py test
```

These use synthetic fixtures and committed receipts; full data training is a
separate explicit action. The [MimeLite guide](../studies/mimelite-supplement/README.md)
has a different budget/evaluation scope and its own resume support. The legacy
Kaggle/cloud workflow is [historical](../studies/historical/README.md).
