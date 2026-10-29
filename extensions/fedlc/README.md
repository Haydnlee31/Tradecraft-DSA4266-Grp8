# FedLC extension

**Status: integrated for validation.** The plan was frozen before local training
on branch `study/bounded-extensions`. The original teammate implementation and
comments are retained. See the [bounded extension guide](../../studies/bounded-extensions/README.md)
for the current run commands and consolidated results. CI includes these tests
through `tests/test_neural_fedlc.py`.

This opt-in experiment adds **client-specific logit calibration (FedLC)** to two
existing federated methods: population-weighted FedAvg with reset client Adam,
and the final blockwise four-bit shared-gradient Adam (`block128`). It asks
whether changing the *training* loss makes the global model keep minority
attack classes in play, instead of letting each client's dominant class
take over.

The original `src/`, Flower app, entry points, dependency files and published
results are unchanged, and nothing in them imports this directory. Run it
explicitly from the repository root.

## Method

On client `c`, each class `j` gets a margin from that client's fit-population
count `n_cj`:

`margin_cj = tau * max(n_cj, count_floor) ** (-1/4)`

The client trains on calibrated logits `z_j - margin_cj`. Rare local classes get
the largest margin, so the model must raise their raw logits to fit them.
Inference uses ordinary, unadjusted logits. This is an adaptation of
[Zhang et al., FedLC, ICML 2022](https://proceedings.mlr.press/v162/zhang22p.html),
not an exact reproduction:

- Counts are each client's whole fit population in raw rows, fixed for the run,
  not the current minibatch.
- A class the client never sees uses `count_floor` (1) instead of zero. The
  inverse root is undefined at zero, and the authors' code effectively removes
  such classes from the softmax.
- `tau = 1` follows the authors' default. It is not tuned here.
- The calibrated logits feed the project's frozen fixed-training-mean
  square-root-weighted CE (`base_loss` in `plan.json`).

**On stacking.** Applying FedLC on top of class weights may correct imbalance
twice. It is the default because the control arms are then *exactly* the
published runs, so the comparison is anchored to existing evidence, as FedNTD's
was. To test FedLC on ordinary cross-entropy instead, copy the plan, set
`"base_loss": "ce"`, and use a new output directory. Its controls then use
ordinary CE too, and there is no published bridge.

## Design

The original training loops are reused unchanged. They draw every training batch
from one client's rows and call `data.batch(rows)` just before computing that
batch's loss. FedLC arms pass them two small wrappers from `core.py`:

- `ClientRoutedData` tells the loss which client owns each batch. It does not
  change rows, order or sampling.
- `FedLCCriterion` applies that client's margins. A loss computed without a
  selected client raises an error.

Control arms use the published code path with no wrapper at all.

The frozen screen in `plan.json`:

| | |
| --- | --- |
| Seeds | 7, 17, 27 (the published training seeds) |
| Methods | `fedavg_adam` and `block128`, both at learning rate 0.003 |
| Arms | control and FedLC per method and seed: **12 runs** |
| Work | 21.6M backward examples per run: 108 FedAvg rounds or 2,160 block128 steps |
| Model | the 5,096-parameter 39→64→32→8 LayerNorm MLP, dropout 0 |
| Data | the existing 500,000 fit rows, 20-client partition and development rows |

Within a seed, both arms share initialization, client partition, sampled rows,
learning rate and work. Only the loss changes. The runner checks that each
pair's sample chains and initial models are identical.

## Evaluation and decision rule

Every endpoint is scored on development data with the **published 5% alarm
rule**, reusing `src/eval/shared_optimizer_alarms.py` unchanged:

- The score is the highest attack logit minus the benign logit.
- The threshold is fitted on benign rows in the published calibration half.
- Metrics come from the other (audit) half only.

The runner checks that its calibration split has the published checksum. All
four targets (1%, 5%, 10%, 20%) are reported, and the assessment uses 5%.

FedLC is supported for a method only if, at the 5% target:

- in **every seed**:
  - macro-F1 stays within 1 point of its control;
  - actual false alerts stay within 0.5 points of the control;
  - no non-target class loses more than 5 recall points;
- and, **averaged over seeds**:
  - mean recall of Web-based, Brute Force and Spoofing rises by at least 2 points;
  - their mean F1 does not fall.

The F1 condition stops a method from "winning" through false positives alone.
These are new proposed targets, not existing project gates. Change them only
before running.

## Run

1. **Get the data.** You need the prepared bundle
   `shared-optimizer-development-data-v1` from whoever ran the shared-Adam
   reproduction. It must contain `manifest.json`, `fit_x.npy`, `fit_y.npy`,
   `development_x.npy`, `development_y.npy`, `fit_row_ids.npy`,
   `development_row_ids.npy`, `clients.npz` and `scaler.json`. The loader
   verifies the published manifest checksum, every file checksum, shapes, class
   counts and exact client ownership. It never opens confirmation, original
   validation or test arrays.
2. **Use the frozen environment.** The runner imports the original training
   modules, so it needs the full project environment (Python 3.13 and the
   pinned packages). On Windows with Smart App Control, use WSL2 or another
   machine: PyTorch cannot load natively there.
3. **Run the tests.**

   ```bash
   python -m unittest extensions.fedlc.test_fedlc -v
   ```

   The loss and routing tests need only NumPy and PyTorch. The original-loop and
   runner tests need the full environment. **They are skipped, with the import
   error shown, if it is missing.** Check that none were skipped.
4. **Run the screen.**

   ```bash
   python -m extensions.fedlc.run \
     --data /absolute/path/to/shared-optimizer-development-data-v1 \
     --output extensions/fedlc/runs/screen-v1
   ```

   Each run should take about a minute of training on a CPU; the published
   FedAvg and block128 runs took about 59 s and 58 s. Allow roughly
   **20–30 minutes** for all 12, including evaluation. That figure is an
   estimate, not a measurement.

## Outputs, bridge and recovery

Outputs are restricted to a subdirectory of `extensions/fedlc/runs/`, which the
repository's `runs/` rule already ignores. Each run saves:

- `contract.json`: plan, data, source and runtime identities;
- `calibration_mask.npy`;
- one folder per arm, holding `last.pt` and `result.json`;
- `summary.json`.

Each result records:

- the model and logits hashes;
- the sample chain;
- the work counters;
- all eight classes' precision, recall and F1;
- confusion matrices at every target;
- the client margins, for FedLC arms.

The summary holds the per-seed FedLC-minus-control differences at every target
and the assessment.

**Published bridge.** On the real bundle, each control arm is compared exactly
with the published same-seed endpoint in
`reports/federated_shared_optimizer/block-replication-results.json`: model hash,
sample chain, work, default metrics, all four operating points, and wire bytes
for block128. A mismatch prints a `WARNING` and is recorded. Check that
`summary.json` has `"published_bridge_all_exact": true`.

A mismatch does not invalidate the paired comparison, since both arms ran on the
same machine. It does mean the controls differ numerically from the published
numbers, so don't quote them side by side. Set `"require_published_bridge":
true` to stop on any mismatch instead.

**Recovery.** Re-run the same command after an interruption to resume from the
last checkpoint. Completed arms are verified by checkpoint checksum and reused,
and a completed run is protected from overwrite. `RUNNING.lock` prevents
concurrent writers. Remove a stale lock only after checking that its recorded
process has ended.

## Interpretation and boundaries

- This is exploratory development evaluation on previously examined data, with
  one fixed non-IID partition and three training seeds.
- It is not new evidence of generalization to other capture sessions.
- Confirmation and test data must not be used to choose `tau` or any other
  setting.
- FedLC cannot create information about attacks that no client has useful
  examples of.
- The historical risk is more rare-class predictions with poor precision, which
  is why the decision rule requires F1 as well as recall.
- Report every class's recall and the actual false-alert rate alongside
  macro-F1.
- Payload is unchanged: margins are computed locally from each client's own
  counts and never sent. Under `base_loss` sqrt-weighted CE, the class weights
  still come from pooled fit counts, as in the controls.
