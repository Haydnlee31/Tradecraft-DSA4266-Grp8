# Tradecraft

IoT network intrusion classification on CICIoT2023, comparing centralized-heavy,
centralized-light, and federated-light training/deployment settings, with SHAP-based
explainability translated into security-policy recommendations.

See [CLAUDE.md](CLAUDE.md) for full project context, locked scope decisions, and the
link to the complete narrative plan.

## Supplementary classical baselines

The separate [baseline runner](src/baselines/README.md) provides centralized and
federated (IID/non-IID) logistic regression, plus centralized XGBoost. It preserves
the original MLP pipelines and reports. XGBoost is optional; test evaluation is
off by default, and no federated XGBoost is implemented.

For validation-only MLP tuning with explicit CPU/CUDA selection, consistent
metrics and crash recovery, use the [research runner](src/models/RESEARCH.md).
The bounded plan is printed with `python -m src.eval.tuning_plan`; it does not
automatically launch experiments or cloud resources.

The [local readiness results](reports/local_readiness_2026_10_03/README.md) record
the completed baseline screen, training-parity checks and CPU recovery pilots.

The [60-step convergence controls](reports/convergence_2026_10_03/README.md)
compare longer training across all four lanes, including class-recall guardrails.

The [loss-function comparison](reports/loss_2026_10_05/README.md) records the four
unweighted-CE screens and why weighted loss is retained for the next experiment.

The [normalization comparison](reports/normalization_2026_10_05/README.md) records
LayerNorm's federated improvements and the remaining class-recall trade-offs.

The [completed 12-run screening report](reports/screening_2026_10_05/README.md)
includes the final heavy-dropout comparison and the confirmation-design handoff.

The [option-1 confirmation report](reports/confirmation_2026_10_05/README.md)
compares the frozen configurations across seeds, separating reused screening
results from the eight new runs and flagging persistent rare-class failures.

The [failure-focused SHAP audit](reports/explanations_2026_10_05/README.md)
checks all 12 frozen checkpoints, distinguishes category errors from benign
acceptance, and records explanation uncertainty and cautious policy hypotheses.

The [validation-only decision integration](reports/decision_2026_10_05/README.md)
applies unchanged research gates to the frozen study. Use
`python -m src.eval.research_decision --output outputs/decision --strict`;
no test dataset is needed, and a research recommendation is not deployment approval.

## Setup

The [cloud study synthesis](reports/cloud_study_2026_10_06/README.md) audits the
completed GPU experiments. The [optional FedProx extension](src/models/FEDPROX.md)
preserves FedAvg by default and defines a bounded, separately verified next screen.

The [pre-cloud checkpoint and study synthesis](reports/pre_cloud_2026_10_05/README.md)
records the final local checks and links to the cloud-pilot handoff. This is readiness
for a bounded pilot, not clearance for full-scale training or security deployment.

Use the shared `requirements.txt` on **macOS, Linux or Windows**. It is not a
macOS lock file: pip resolves the appropriate platform wheels. Python 3.11–3.13
is the intended team baseline; the CI configuration checks Python 3.12 on all
three OS families. Availability still depends on your CPU architecture and the
upstream packages—this is not a guarantee for every OS/Python/hardware combination.

Create and activate a virtual environment first.

macOS / Linux (bash or zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then run the same commands on any OS:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -v
```

The shared install covers data preparation, centralized training, sequential
Flower/FedAvg experiments, evaluation and SHAP. For **Ray worker simulation**
(`flwr run` or `src.federated.runtime_smoke`), also run:

```text
python -m pip install -r requirements-simulation.txt
```

For editable package installs, the equivalent choices are `pip install -e .`
and `pip install -e ".[simulation]"`. Both use the same declared dependencies.
Native Windows Ray simulation is experimental; use WSL2 if workers cannot start,
or keep using the sequential backend. See the
[Flower simulation platform notes](https://flower.ai/docs/framework/how-to-run-simulations.html).

No GPU is required. For a particular CUDA/ROCm build, install the matching `torch`
package using the [official PyTorch selector](https://pytorch.org/get-started/locally/)
before installing these requirements. GPU drivers are not installed by pip.

Tested locally on macOS Apple Silicon with Python 3.13.9. For the exact tested
environment use `pip install -r requirements-macos-py313.lock` instead of the broad
requirements. The lock is platform-specific; it is not a promise of identical
CUDA/Windows support. Keep the venv active: Flower launches companion executables
by name, so invoking only `.venv/bin/flwr` is insufficient if its bin folder is
not on `PATH`. Do not use that historical Mac snapshot for teammates' Windows or
Linux environments. Windows/Linux CI jobs have been added, but have not been run
from this local session.

## Reproduce the complete study

With the three parquet splits already present:

```bash
python -m src.data.audit --output reports/full_run/data_audit.json
python -u -m src.experiment --output reports/full_run --threads 2
python -m src.eval.decision --reports-dir reports/full_run
python -m src.eval.study --reports-dir reports/full_run
python -m src.explain.analyze
```

This runs heavy, light, FL IID control and FL Dirichlet alpha=0.5, each for seeds
0/1/2, at most 30 epochs/rounds with validation early stopping. It uses all existing
sampled split rows, not the full official dataset. It resumes completed reports;
use a new output folder for changed settings. The output includes model checkpoints,
package/data provenance, per-seed metrics, a comparison table/figure and exploratory
validation SHAP audits. Checkpoints and data are intentionally not committed.
These Python commands work in PowerShell as well as bash; `--threads` configures
the thread environment internally. The convenience `.sh` scripts elsewhere in
this README require bash (on Windows, use WSL2 or the Python entry points).

The full-study FL backend is sequential, using the same client helpers and Flower
aggregation; it is a learning experiment, not a systems benchmark. A separate
`python -m src.federated.runtime_smoke` checks actual Ray worker messages, training,
server checkpointing and reports on tiny **synthetic** data. Its output must never
be included in study metrics. The API smoke path is deprecated upstream but works
with the pinned Flower version. The optional `--backend cli` path creates a second
dependency environment; its initial download may be slow. The CLI startup was not
validated successfully in this environment (see the full-run assessment).

## Getting the data

1. Create a Kaggle API token at kaggle.com/settings/api and save it to
   `~/.kaggle/access_token` (new token-based auth) or `~/.kaggle/kaggle.json`
   (legacy username+key) — either is accepted.
2. Download the raw CSVs:
   ```bash
   python -m src.data.download_ciciot
   ```
   This pulls the `himadri07/ciciot2023` Kaggle mirror, which — confirmed by
   inspection — ships pre-split into `train.csv`/`validation.csv`/`test.csv`
   (~7.84M rows total, a downsampled/merged version of the official
   ~46M-row release, not the original 169-file layout). All 34 raw labels
   (33 attacks + BenignTraffic) are present in every split and match
   `src/data/label_map.py` exactly.
3. Build a stratified sample per split (capped per raw label, seeded):
   ```bash
   python -m src.data.sample_dataset --per-class-cap 50000 --seed 0
   ```
   Since the mirror's train/val/test boundary is already fixed upstream,
   this samples independently within each split rather than re-splitting,
   and writes straight to `data/splits/{train,val,test}.parquet` plus
   `data/sampled/manifest.csv` for provenance.
4. Check and remove exact duplicate rows across splits:
   ```bash
   python -m src.data.check_leakage --fix
   ```

Raw and derived data are gitignored — `data/raw/`, `data/sampled/`, and `data/splits/`
are populated locally by the commands above, not committed.

## Running the centralized lanes

Both variants use the same split files, train-only fitted scaler, label order,
training loop, loss factory, and report schema. The only intended difference is model
capacity.

```bash
python -m src.models.train_centralized --variant heavy --loss sqrt_weighted_ce --seed 0
python -m src.models.train_centralized --variant light --loss sqrt_weighted_ce --seed 0
```

Repeat the final chosen configuration for seeds 0, 1, and 2. Reports are written to
`reports/centralized_{heavy|light}_<loss>_seed<N>.json`; checkpoints go to
`models/checkpoints/` and are ignored by Git.

## Running the federated lane

Federated-light uses the exact same light architecture, scaler, loss, class order,
validation split, and JSON metric schema as centralized-light. The clients are
simulated data shards—not the 105 real devices, whose identities are absent from the
public CSVs.

Build a reproducible non-IID assignment, then run Flower:

```bash
python -m src.federated.partition --alpha 0.5 --num-partitions 20 --seed 0
NUM_NODES=20 ./run_fed.sh "dirichlet-alpha=0.5 seed=0"
```

The launcher also accepts `SEED=1 PARTITIONER=dirichlet ALPHA=0.5` environment
settings; explicit run-config arguments override them. Both launchers default to
20 simulated clients, and the partition file must match that number.

FedAvg remains the baseline. Joel's adaptive alternative is explicitly opt-in:

```bash
NUM_NODES=20 ./run_fed.sh "strategy='fedadagrad' fedadagrad-eta=0.1 fedadagrad-tau=0.001"
```

The sequential full-study runner stays FedAvg-only. Worker reports now include
the strategy in the filename; historical reports are unchanged. Each worker run
also writes `run_info.json`, `history.json` and its checkpoints beneath
`outputs/federated/<date>/<time>_seed<N>/`. Optional client evaluation
(`fraction-evaluate=1.0`) reports min/median/max client macro-F1, not a replacement
for global validation macro-F1. These client validation slices remain approximately
IID, so the spread is not an evaluation on each non-IID training distribution.

W&B is disabled by default. To opt in, install `python -m pip install -e ".[wandb]"`
and pass `use-wandb=true`. With no login, it uses offline logging. Authenticated
online logging sends run configuration and metrics to W&B; local history needs no
account. The learning-rate schedule remains off by default; `lr-decay-every=5`
enables halving after every five completed rounds.

Windows note: Flower 1.38's simulation extra omits Ray on Windows/Python 3.13+.
Use Python 3.12 for a native worker install or WSL2; the sequential runner does
not require Ray. Do not disable Smart App Control to run the experiment.

`alpha=0.1` is strongly non-IID, `0.5` is moderately skewed, and larger values
approach IID. `./run_sweep.sh` runs alpha `{0.1, 0.5, 1.0}` plus an IID baseline for
three seeds. Full operating notes are in [src/federated/README.md](src/federated/README.md).

If Ray cannot run on the machine, this slower sequential smoke path exercises the
same client code and Flower aggregation helpers:

```bash
python -m src.federated.run_simulation --rounds 2 --clients 3 --alpha 0.5
```

This particular two-round in-process runner is for debugging, not the final
benchmark. The full-study runner above adds checkpoint selection, three-seed
repetition and held-out reporting. Wall-clock time from either simulation path is
not measured edge-hardware latency, memory, or power.

## Building the decision summary

Once reports exist, the decision layer groups repeated seeds, selects configurations
using validation macro-F1 (not test performance), checks rare-class recall and model
size constraints, then compares the three lanes:

```bash
python -m src.eval.decision --reports-dir reports
```

It writes `reports/decision_summary.json` and `.md`. It deliberately returns an
`incomplete` decision when a lane or the required seed count is missing instead of
silently recommending from a partial comparison.

Gates use validation metrics; test is descriptive only. The default FL target is
Dirichlet, so the easier IID control cannot silently replace it. Thresholds are
illustrative research choices, not a security-service specification or permission
to deploy an automatic blocker.

## Metrics and interpretation

The headline is macro-F1 together with all eight per-class recalls. Accuracy remains
secondary because a high-accuracy model can still collapse on Brute Force and
Web-based traffic. The server's full shared validation split is used for model
selection; test is read once for the validation-selected checkpoint. Client-shard
metrics describe local behavior and are not cross-lane benchmarks.
