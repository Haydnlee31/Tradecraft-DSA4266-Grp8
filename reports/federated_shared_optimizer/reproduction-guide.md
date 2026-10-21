# Reproducing the compressed Shared Adam result

The target is **mean confirmation macro-F1 0.6325367323327683**, rounded to
**0.6325**, across training seeds 7, 17 and 27. It is not recall, not the result
of every seed, and not the default argmax score. Decisions use the frozen
development-calibrated 5% benign false-alert target. Actual confirmation
false-alert rates are approximately 5.36% on average.

The reproduction runner trains all three compressed models from random seeded
initialization on the exact prepared fit inputs. It does not rerun the earlier
model search, choose better checkpoints, change the partition, or tune against
confirmation labels. Historical checkpoints are comparison targets only.

## Two commands after cloning the release

Use the versioned `shared-adam-reproduction-v1` release rather than the generic
legacy training entry point. With Python **3.13.9** installed:

```bash
git clone --branch shared-adam-reproduction-v1 https://github.com/Haydnlee31/Tradecraft-DSA4266-Grp8.git Tradecraft-reproduce
cd Tradecraft-reproduce
python3.13 scripts/reproduce_shared_adam.py setup
python3.13 scripts/reproduce_shared_adam.py run --output outputs/my-reproduction-v1
```

Setup downloads the release's data-only archive, verifies its pinned SHA-256,
size and every member checksum, then creates `.venv-shared-adam`, installs the
pinned numerical dependencies and runs the input/runtime checks. It never
installs into the caller's environment. Training runs separately so setup does
not silently launch computation. No activation command is required.

On Windows, use `py -3.13` instead of `python3.13`. These setup commands are
portable; **exact numerical equality is only verified on the recorded macOS
ARM runtime**, not guaranteed across platforms. Python must report 3.13.9.

For a previously downloaded bundle, replace setup with:

```bash
python3.13 scripts/reproduce_shared_adam.py setup --archive /path/to/tradecraft-shared-adam-inputs-v1.tgz
```

For an already configured environment, pass its interpreter explicitly to both
commands (this does not install or upgrade any packages in that environment):

```bash
.venv/bin/python scripts/reproduce_shared_adam.py setup --archive outputs/tradecraft-shared-adam-inputs-v1.tgz --python .venv/bin/python
.venv/bin/python scripts/reproduce_shared_adam.py run --python .venv/bin/python --output outputs/my-reproduction-v1
```

If an existing input differs, setup stops without overwriting it. Identical inputs
are safely reused. The archive is never allowed to install code, symlinks or files
outside `outputs/`. To repeat training, always choose a new output directory.

## Data distribution

Code and the pinned asset manifest live in Git. The prepared arrays and original
checkpoints are distributed as a GitHub Release asset rather than stored in Git
history. "Local artifacts" does not mean confidential data. The repository and
approved release asset are public.

Git alone does **not** contain the prepared arrays or checkpoints. Setup fetches
`tradecraft-shared-adam-inputs-v1.tgz` automatically. This data-only archive supplies:

- `outputs/shared-optimizer-development-data-v1/`: manifest, fitted scaler,
  exact fit/development arrays, row IDs and client ownership; no staging cache.
- `outputs/shared-optimizer-confirmation-v1/data/`: the already evaluated
  confirmation arrays and their reconstruction receipt.
- The original `last.pt` checkpoint for each of the three compressed models,
  under the paths pinned by the published confirmation lock.

The archive does not contain the original project test split or the raw
46-million-row release. Training uses the same 500,000 fit rows and 20 simulated
non-IID clients. Confirmation contains 398,502 rows. Provenance and array hashes
are verified before training; this is a from-scratch **training** reproduction
from frozen prepared inputs, not a fresh CSV-to-split reconstruction.

## Runtime

The original runtime is Python **3.13.9**, PyTorch **2.14.0**, NumPy **2.5.3**,
CPU with two threads, on macOS arm64. The separate reproduction requirements
pin the local numerical stack without changing the project's general cross-platform
requirements. No GPU, Flower server, Ray cluster or RONIN session is needed.

Exact equality is verified on Haydn's local runtime. Matching versions alone
does not guarantee bitwise equality on different CPUs, operating systems or
PyTorch builds. The runner stops if runtime versions or results differ; do not
remove those checks and call the run an exact reproduction. The old RONIN
Python 3.12 / PyTorch 2.7 environment is not this frozen runtime.

## Manual alternative

The bootstrap above is preferred. To manage the environment manually instead,
clone the same release, download its data-only asset and use a fresh directory
so no existing project work or outputs are overwritten.

```bash
git clone --branch shared-adam-reproduction-v1 https://github.com/Haydnlee31/Tradecraft-DSA4266-Grp8.git Tradecraft-reproduce
cd Tradecraft-reproduce
```

Save the data-only archive to Downloads. Compare its SHA-256 with `archive.sha256`
in `reports/federated_shared_optimizer/reproduction-inputs-v1.json`, then extract
it. `-k` refuses to overwrite existing files. The earlier mixed code/data archive
`tradecraft-shared-adam-reproduction.tgz` is a different, legacy handoff and is
not accepted by the new bootstrap.

```bash
shasum -a 256 "$HOME/Downloads/tradecraft-shared-adam-inputs-v1.tgz"
tar -xzkf "$HOME/Downloads/tradecraft-shared-adam-inputs-v1.tgz"
```

On Linux, `sha256sum` can replace `shasum -a 256`. Do not extract an untrusted
archive. If extraction reports existing files, stop and use a fresh clone.

Confirm that `python3.13 --version` reports **3.13.9** before creating the
environment. If that interpreter or the pinned package builds are unavailable,
stop and ask Haydn for help; do not substitute the RONIN environment.

```bash
python3.13 --version
python3.13 -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r requirements-reproduce-shared-adam.txt
python -m pip check
python -m unittest tests.test_shared_optimizer_reproduce -q
python -m src.eval.shared_optimizer_reproduce check
```

The last command verifies the frozen source/evidence chain, runtime, prepared
inputs, confirmation receipt and reference checkpoint checksums without training.
Continue only if it prints `PASS`.

```bash
python -u -m src.eval.shared_optimizer_reproduce run --output outputs/shared-optimizer-my-reproduction-v1
```

The command prints progress every 180 of 2,160 synchronization steps per seed.
Each seed processes 21.6 million training examples, sampled with replacement;
all three together process 64.8 million. That is not 64.8 million distinct rows.
The command requires a **new output directory** and never silently resumes.
If interrupted, preserve that directory and use a different output name for a
new from-scratch attempt.

Read the final result:

```bash
python -c "import json; r=json.load(open('outputs/shared-optimizer-my-reproduction-v1/summary.json')); print('Status:', r['status']); print('Mean macro-F1:', r['mean_macro_f1']); print('Sample SD:', r['sample_sd']); print([(x['seed'], x['primary']['macro_f1']) for x in r['results']])"
```

Expected values:

| Seed | Confirmation macro-F1 |
| --- | ---: |
| 7 | 0.630267419693709 |
| 17 | 0.630400343579381 |
| 27 | 0.6369424337252149 |
| Mean | 0.6325367323327683 |
| Sample standard deviation | 0.0038160281392305927 |

Each per-seed `result.json` also contains all per-class precision, recall, F1,
support, confusion matrices, false-alert rates and secondary operating points.

## What a successful run proves

For every seed, the runner verifies exact equality of the entire final training
state: server and delivered model weights, Adam optimizer state, compression
feedback residuals, sampled-row digest, training work and serialized-byte counters.
It then reproduces development logits and development-only thresholds, followed
by confirmation logits and all metrics using those unchanged thresholds.

Checkpoint file bytes need not match because new run metadata and elapsed time
differ; the numerical state inside the checkpoints must match exactly. No
original validation/test split is opened, no confirmation threshold is fitted,
and no new model selection occurs. This is a reproducibility check of the closed
study, not additional independent evidence of generalization.

For the existing local checkout and `.venv`, Haydn can use:

```bash
.venv/bin/python -m src.eval.shared_optimizer_reproduce check
.venv/bin/python -u -m src.eval.shared_optimizer_reproduce run --output outputs/shared-optimizer-another-fresh-reproduction
```

Preserve all original results and use a new output name for every attempt.

## Rebuilding from the original dataset

The automated release reproduces training from prepared inputs. It does not
claim to rebuild the entire experiment from an arbitrary CICIoT2023 download.
The original official39 pipeline must first recreate the checksum-pinned raw
Parquet shards, nested two-million-row subset, historical training arrays/scaler
and client partition recorded in `performance-plan.json`. The partial Kaggle
mirror is not interchangeable with that source.

`python -m src.data.shared_optimizer_data --help` lists the existing builder's
six source inputs: `--parent`, `--subsets`, `--historical-y`, `--historical-x`,
`--historical-scaler` and `--partition`, plus a new `--output` directory. It streams
raw shards, verifies historical ordering, reselects rows by feature-vector hashes,
fits the scaler only on fitting rows and rebuilds client assignments. Confirmation
reconstruction lives in `src/data/shared_optimizer_confirmation.py` and separately
checks membership and cross-cohort input overlap.

Those earlier source artifacts are intentionally not in this small release.
A fresh CSV-to-final-metric rebuild remains **unverified by this handoff**. Do not
describe the prepared-input reproduction as having established that stronger claim.
