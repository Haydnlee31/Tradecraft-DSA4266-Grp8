# Bounded local update experiment

The official-data runner now supports capped client training with exact aggregate
optimizer-step accounting. The default remains a full local epoch. This extends
the client diagnostic into a recoverable training policy; it does not establish
that the policy improves model quality. Keep the cloud machine stopped until a
reviewed code snapshot is ready to transfer.

## What changes

`--local-max-batches 5` lets each federated client train on at most five batches
from its freshly seeded local shuffle before aggregation. Original partition-size
weights are retained: capped processed-example counts must not silently change
aggregation toward uniform client weighting. All clients participate each round.
The cap is a prefix sample, not a cursor through each client's dataset; repeated
rounds do not guarantee that every training row has been seen.

`--total-update-budget` optionally sets an exact sum of optimizer steps over all
clients and rounds. The budget must divide by the number of clients, and every
client must have enough batches for the cap. A final partial round reduces the
cap equally for every client. Incompatible round counts or budgets fail rather
than silently overshooting, dropping clients or reporting approximate matching.

For example, 19,680 updates across 20 clients with a five-batch cap means 196
rounds of five updates per client, followed by one round of four. The `--epochs`
argument still means federation rounds in this lane, so its value must be 197.
These options are federated-only; the centralized runner is unchanged.

Each history entry records per-client assigned rows, actual examples processed,
optimizer steps and the active cap. The result includes final validation metrics
separately from validation-best metrics. Resume remains at completed round
boundaries with strict source/environment/settings/data checks. Old cloud runs
must not be resumed under this changed source snapshot; preserve them as evidence.

## Planned comparison

The [machine-readable plan](bounded-update-plan.json) fixes the 500k cohort,
LayerNorm, square-root-weighted loss, seed 7, partition seed 7, 20 clients,
alpha 0.5, batch size 512 and learning rate 0.001.

| Partition | Policy | Rounds | Total optimizer updates | Training examples processed |
|---|---|---:|---:|---:|
| IID | Full local epoch | 20 | 19,600 | 10,000,000 |
| IID | Five-batch cap | 196 | 19,600 | 10,035,200 |
| Non IID | Full local epoch | 20 | 19,680 | 10,000,000 |
| Non IID | Five-batch cap with final four-batch round | 197 | 19,680 | 10,076,160 |

These are planned counts for the verified partitions, not completed experiment
results. Full epochs have short final batches; capped batches are full-sized here,
so equal optimizer-update counts do not imply equal example counts.

The primary comparison uses final full-validation metrics at matching aggregate
updates within each partition lane. Round 20 from each history provides a
secondary fixed-round comparison, without separately retraining that prefix.
Validation-best scores are descriptive only: a 197-round run has more checkpoint
selection opportunities than a 20-round run. Neither run uses early stopping or
automatic extensions, and the test split stays sealed.

The capped policy also changes per-client exposure, aggregation frequency and
Adam-reset frequency. It is a combined policy experiment, not an isolated causal
test of client drift. Report per-class recall, false alerts, actual training work,
round counts and validation cost, not macro-F1 alone. Additional seeds are required
before promoting a configuration.

## Local verification and cloud gate

CPU tests cover exact schedules, rejection of invalid budgets, a nonbinding cap
that leaves full-epoch models unchanged, and exact interrupted/resumed models,
optimizer/RNG state and non-timing histories. A direct two-step fixture comparison
against the runner at commit `1fecacc` checks all four default lanes for unchanged
model/optimizer/RNG/best-checkpoint behavior.

The real 500k cohort also has a two-round CPU recovery receipt under
`outputs/official39-bounded-cpu-recovery-v1/checks.json`. This pilot uses two local
batches followed by one, tests both IID and non-IID, and is not model selection.
CPU exactness does not certify GPU recovery. The CUDA gate, after transferring a
reviewed snapshot and activating the existing cloud environment, is:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.eval.official_bounded_recovery \
  --data outputs/official39-packed-500k-v1 \
  --device cuda \
  --output outputs/official39-bounded-gpu-recovery-v1
```

Inspect `checks.json` before any larger run. After that, rerun the two full-epoch
reference lanes under the same new code and compare them with the archived
LayerNorm reference. Require matching model tensors and numerical histories;
new accounting fields are expected, timing differences are allowed. Stop on any
unexplained difference. Only then launch the capped runs in the fixed plan.

No cloud run, Git push or additional dependency is part of this local change.
