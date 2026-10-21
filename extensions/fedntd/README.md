# FedNTD extension

**Real-data screen completed:** see the
[audited findings](../../reports/extensions/fedntd/real-data-findings.md).
Training used the copy of this extension in `outputs/Tradecraft-reproduce/`,
on branch `angel/fedntd-extension`, after exact three-seed baseline reproduction.
Neither tested nonzero beta improved development macro-F1; all results were retained.

This opt-in experiment adds **not-true distillation** to the repository's
population-weighted FedAvg with reset client Adam. It tests whether retaining
the round-start global model's predictions for other classes helps local training
under non-IID client populations.

The original `src/`, Flower app configuration, training entry points, dependency
files and published results are unchanged. Nothing in the original pipeline
imports this directory. Run this extension explicitly from the repository root;
it is not included in the existing wheel's `src` package.

## Method and comparison

For each local batch, the loss is:

`fixed-training-mean sqrt-weighted CE + beta * T² * KL(teacher_not_true || student_not_true)`.

Remove the true-class logit from both networks **before** softmax/log-softmax.
The remaining seven classes form a normalized distribution. The teacher is a
detached, eval-mode copy of the global model at the start of each round. It stays
fixed throughout that round; the next round creates a fresh teacher. No teacher
is constructed or evaluated when beta is zero.

This is an adaptation of [Lee et al., FedNTD, NeurIPS 2022](https://arxiv.org/abs/2106.03097),
not a claim of an exact reproduction or a new algorithm. We preserve this
project's weighted supervised objective, population aggregation, Adam recipe,
and light MLP. We use a sequential CPU simulation matching the repository's
optimizer study, rather than registering a new Flower strategy.

The frozen initial screen in `plan.json` runs three paired arms: beta **0, 0.1,
1.0**, all at temperature **1**. These coefficients are experimental choices,
not established optimal values for CICIoT2023. Each arm uses:

- The same seed 7, initialization, client partition and sampled row sequence.
- The existing official39 fit/development bundle and fit-only class weights.
- The 5,096-parameter 39→64→32→8 LayerNorm MLP, dropout 0.
- Learning rate .003, Adam betas (.9, .99), epsilon .001, weight decay 0;
  fresh client Adam state each round, matching the existing FedAvg control.
- Twenty clients, twenty local steps, batch size 500, 108 full rounds:
  21.6 million backward examples **per arm** (64.8 million across the screen).

Nonzero-beta arms additionally perform 21.6 million teacher-forward examples
each. Their communication payload and inference architecture are unchanged;
their training cost is higher. Payload bytes are modeled tensor traffic, and
elapsed training seconds are measured on the simulation host, not edge hardware.

## Run

Only Python, NumPy and PyTorch are needed for this extension and its tests; it
imports the existing canonical architecture, label map, loss and numerical
optimizer helpers. It adds no dependencies or installation changes.

Run the audit tests first:

```bash
python3 -m unittest extensions.fedntd.test_fedntd -v
```

Run a real development screen using the **already prepared optimizer study
bundle**, not a Kaggle CSV directory or the original official39 test arrays:

```bash
python3 -m extensions.fedntd.run \
  --data /absolute/path/to/prepared-optimizer-data \
  --output extensions/fedntd/runs/seed7-screen
```

The supplied data directory must contain `manifest.json`, `fit_x.npy`,
`fit_y.npy`, `development_x.npy`, `development_y.npy`, `fit_row_ids.npy`,
`development_row_ids.npy`, `clients.npz`, and `scaler.json`, produced under
`reports/federated_shared_optimizer/performance-plan.json`. The loader verifies
the parent-plan identity, per-file checksums, shapes, class counts and exact
client ownership. It memory-maps feature arrays and checks them in chunks.
It does not refit a scaler or repartition data. The manifest's feature order and
scaler identity are retained in run provenance. If this private prepared bundle
is unavailable, restore it from the original experiment storage first; the
extension does not download data or substitute another dataset silently.

An alternate extension plan can be supplied with `--plan path/to/plan.json`.
Use a new output directory for any changed plan, data, code or runtime. The
default is a substantial experiment, not a quick smoke test. The tests run tiny
synthetic versions of all three arms, including the actual disk-backed runner.
Synthetic scores are software checks, not evidence of an IDS improvement.

## Outputs and recovery

All outputs are restricted to a subdirectory of `extensions/fedntd/runs/`, which
is already ignored by the repository's `runs/` rule. Existing study reports and
checkpoints cannot be selected as an output destination through this CLI.

Each run saves a source/data/plan/runtime contract, one checkpoint and result per
beta, and a final `summary.json`. Metrics include macro-F1, all eight classes'
precision/recall/F1/support, the confusion matrix and benign false-alert rate.
The summary reports paired differences from beta zero. Decisions use ordinary
argmax; these numbers must not be compared as if they used the previous study's
calibrated 5% alarm threshold. No best-round or threshold search occurs here.

Re-run the **same command** after interruption to resume at the last completed
round. Optimizer state need not persist across rounds because this is reset-Adam
FedAvg. Teacher state is reconstructed from the restored round-start model;
sample seeds are derived from seed/round/client/step. The checkpoint stores
round, model, work counters and a hash chain of sampled fit-row indices.
Together with the hashed row-ID mapping and partition, this identifies the
sampled original rows reproducibly. Completed runs are protected from overwrite.

`RUNNING.lock` prevents concurrent writers. A hard process kill can leave a stale
lock; remove that one lock only after checking its recorded PID is no longer
running. Do not remove a live run's lock.

## Interpretation and boundaries

This is exploratory development evaluation on historical training-derived
data. It never opens original validation/test arrays or confirmation arrays,
including during manifest verification. It is not new independent evidence of
cross-capture generalization. Previously exposed test/confirmation scores must
not be used to select beta. Check rare-class recall and false alerts alongside
macro-F1; an aggregate improvement alone is insufficient.

The core intentionally requires CPU float32/float64, buffer-free models, as in
the existing numerical control. The supplied runner fixes LayerNorm and zero
dropout. It does not add GPU, BatchNorm buffer aggregation, partial participation,
personalized heads or live network deployment.

The audit tests execute the unchanged original FedAvg numerical function bodies
directly from their AST, avoiding unrelated optional app imports. They verify
bit-exact beta-zero trajectories and sample chains over multiple rounds,
KL direction/temperature/true-class removal, teacher detachment and immutability,
positive-beta behavior, work/payload accounting, real checkpoint interruption
and recovery, data tampering/overlap rejection, output restrictions, and metric
agreement with the original confusion-matrix implementation. See `AUDIT.md`
for the checks actually run during implementation.
