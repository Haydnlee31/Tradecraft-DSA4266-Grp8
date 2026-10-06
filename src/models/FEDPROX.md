# Optional FedProx research extension

FedAvg remains the default. The old `src.experiment` and Flower app entry points
are unchanged. This extension is exposed only through `src.models.research`:

```bash
python -m src.models.research --lane dirichlet --federated-method fedprox \
  --proximal-mu 0.01 --normalization layer --output outputs/example-fedprox
```

This example is not the frozen cloud screen; use the plan below for that. No new
dependency, dataset or model architecture is required. Existing beginner comments
are retained.

## Definition and limits

For each local minibatch, optimize the existing classification task loss plus
`mu/2 * sum((local_parameter - round_start_global_parameter)**2)`. The sum covers
all trainable parameters, including biases and LayerNorm scales/offsets, not model
buffers. It is NOT averaged over the number of parameters. The anchor is detached,
cloned and fixed for the whole client round; it is reset from the current global
model each round. Client Adam still resets each round, one local epoch and full
participation remain fixed, and aggregation is still sample-count FedAvg.

This is an Adam-based FedProx-objective experiment, not a claim to reproduce the
original paper's optimizer/benchmarks. FedProx targets heterogeneity; it does not
guarantee rare-class coverage, privacy or deployment fitness. Reference:
[Li et al., Federated Optimization in Heterogeneous Networks, MLSys 2020](https://proceedings.mlsys.org/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf).

Adam weight decay remains separate and unchanged. Task loss continues to be logged
without the proximal term; `train_batch_mean_proximal_penalty` separately records
the row-weighted mean of pre-update minibatch proximal penalties. Validation stays
unweighted CE and checkpoint selection stays validation macro-F1. At mu=0 the
penalty computation and anchor allocation are skipped, preserving the FedAvg path.

Method and mu are recorded in settings/results. Resume rejects method, coefficient,
source, packages, data or CUBLAS configuration changes. Historical checkpoints from
the pre-cloud revision must NOT be resumed under this branch. New runs use new
directories; returning to old code/environment is required for historical recovery.

## Verification before cloud screening

On the existing cloud clone, inspect `git status --short` first. If tracked files
have changed, stop rather than overwriting them. Fetch and select the new branch;
do not reclone the data, reinstall CUDA, or upgrade the working Python environment:

```bash
git fetch origin
git switch --track origin/experiment/fedprox-controlled
source .venv-cloud/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

If the local branch already exists, use `git switch experiment/fedprox-controlled`
and inspect its commit before continuing. Keep the old `pre-cloud-deployment`
branch and all historical outputs. Run in tmux to survive SSH disconnections.

Local tests cover analytic penalty/gradient, detached/nonaliasing anchors, actual
update changes, unchanged validation loss semantics, invalid arguments, exact
mu=0 FedAvg equivalence, positive-mu CPU recovery, and resume-setting rejection.

```bash
python -m unittest discover -v
python -m pip check
```

The actual-data pilot can also run locally with `--device cpu`. On the cloud, after
transferring this branch and preserving existing cloud packages, run ONLY the pilot:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.eval.fedprox_pilot --device cuda \
  --output outputs/fedprox-gpu-pilot
```

This runs four two-round jobs (FedAvg, mu=0, mu=0.1 uninterrupted, mu=0.1 with a
pause/resume). It checks exact final weights and validation metrics, observes a
nonzero penalty and excludes test evaluation. Output must be a new directory.
Missing data/GPU are checked before any job; zero-mu mismatch stops the pilot
before positive-mu runs. Every round's metrics/losses are compared, not just final
validation metrics. A failed pilot must be investigated, not repeatedly relaunched.
Send `checks.json` and the complete output directory for review. CUDA behavior of
this new code is unverified until that pilot passes; CPU success is insufficient.

## Frozen cloud screen, not automatic launch

Print the three full-length commands (does not execute them):

```bash
python -m src.eval.tuning_plan --plan configs/cloud_fedprox_plan.json \
  --output-root outputs/cloud-fedprox
```

Run the **FedAvg bridge first**, after the GPU pilot passes. Compare its metrics,
history and selected checkpoint against the original cloud non-IID seed-7 reference.
If they differ, investigate before any candidate run. The bridge is an explicit
additional regression-control run, not a third candidate or a changed baseline.

Then run only mu=0.01 and mu=0.1 at model seed 7, partition 0, 60 rounds with all
other settings unchanged. These are preselected screening values, not proven ideal
coefficients. No larger grids or loss/architecture changes are included.

Promotion criteria are copied unchanged from the earlier plan (deltas are absolute
proportions): A requires F1 gain >=0.01, false-positive increase <=0.02 and no class
recall drop >0.02. B requires false-positive reduction >=0.05, F1 drop <=0.01 and
Web/Brute Force recall drop <=0.02. At the original 3.56% false-positive baseline,
B is mathematically unattainable, so A is the applicable route; do not relax B.
Inspect precision and attack-to-benign errors as well, even when A passes.

If neither passes, stop the extension and report the negative result. If one or
both pass, choose the eligible candidate with highest macro-F1 (exact tie: smaller
mu), then obtain approval for a separately frozen confirmation plan across model
and partition seeds. Never select the easiest partition. No test access is needed.

Budget: one bridge and two candidates, plus the two-round correctness pilot. About
half an hour of training/validation is a rough baseline-derived estimate, not a
runtime or spending guarantee; extra proximal work can change throughput. Monitor
within one hour, verify the actual provider rate, download/hash backups, and stop
the instance between stages. This code does not provision or shut down machines.
