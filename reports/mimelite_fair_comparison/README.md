# Full-anchor MimeLite and matched baselines

This experiment implements both missing MimeLite configurations and compares
them with Central Adam, Shared Adam, full MIME and FedAvg-Adam. A centralized
MIME-base control is included because MIME's lagged-moment updater is not
ordinary bias-corrected Adam. The old untraceable MimeLite scores are not targets
and do not count as results of this pipeline.

## Completed evidence

- [Results and figures](results.md): all seven configurations, three seeds each,
  equal backward-example budgets, actual false alerts and per-class recall.
- [Interpretation](findings.md): observed trade-offs and why the earlier proposed
  MimeLite round counts would imply substantially more computation.
- [Numerical evidence](results.json) and [per-seed CSV](per-seed-metrics.csv).
- [Verification](verification.json): 659 working-checkout tests, 21 exact
  serialized-recovery checks, 15 exact historical checkpoints and 126 independently
  recalculated metric endpoints. Completed-job resume reused all 21 outputs with
  an unchanged summary hash.

The primary 5% calibration target gives 35/36 paired actual-FAR differences
within 0.5 percentage points. The seed-7 MimeLite-5/FedAvg exception is retained
and explained in the interpretation; it is not silently treated as matched.

## Executable implementation inventory

All seven configurations run through the same
[comparison entry point](../../src/eval/mimelite_comparison.py).

| Method | Training implementation |
| --- | --- |
| Central Adam | Existing pooled updater in [shared_optimizer_performance.py](../../src/eval/shared_optimizer_performance.py) |
| Shared Adam | Existing [shared_adam.py](../../src/federated/shared_adam.py) kernel and performance runner |
| MimeLite, 5 or 20 local steps | New [mime_lite.py](../../src/federated/mime_lite.py), parameterized by local-step count |
| Full MIME | Existing [mime_adam.py](../../src/federated/mime_adam.py), with an explicitly budgeted short final block |
| FedAvg-Adam | Existing reset-client-Adam performance runner, with an explicitly budgeted short final block |
| Central MIME base | Existing pooled MIME-base updater in the performance runner |

This is a single-machine federated simulation with explicit client ownership
and aggregation. It does not measure a physically distributed Flower deployment.

## Protocol fixed before training

The [plan](plan.json) specifies seven configurations and seeds 7, 17 and 27.
The training implementation and plan were committed as `b188cb3` before these
new runs began.
Every run processes exactly **23,100,000 backward examples**, including all
full-anchor gradients and MIME's two gradients per local correction. No
anchor-only padding or unused gradients are added to manufacture equal totals.
The 21 runs process 485,100,000 training backward examples; recovery checks are
counted separately.

| Configuration | Complete schedule | Logical barriers |
| --- | --- | ---: |
| Central Adam / central MIME base | 46,200 pooled updates | 0 |
| Shared Adam | 2,310 shared-gradient updates | 2,310 |
| MimeLite, 5 local steps | 42 rounds of 5 steps | 42 |
| MimeLite, 20 local steps | 33 rounds of 20 steps | 33 |
| Full MIME | 25 rounds of 20 steps, then 1 round of 5 | 52 |
| FedAvg-Adam | 115 rounds of 20 steps, then 1 round of 10 | 116 |

23.1M allows complete rounds for both MimeLite variants. The smaller final
blocks of full MIME and FedAvg consume their exact remaining budgets and are
declared before results. They are not changes prompted by model performance.
Historical 21.6M rows must not be placed in the same table as these new endpoints.

All methods use the same 500,000 fit rows, 39-feature LayerNorm MLP, fit-only
scaler, fixed-denominator square-root weighted CE, minibatch size 500, fixed
20-client ownership, full participation and population weights. Initializations
are paired by seed. Uniform replacement sampling uses deterministic client,
round and step identifiers; different block lengths and anchor costs imply
different numbers of local samples and different sample ordering.

Both MimeLite variants use learning rate 0.003, inherited from the existing
MIME/central-base setting; the existing methods retain their prior rates.
This is a fixed-setting comparison, not an equally exhaustive hyperparameter
search or evidence of each method's optimal performance.

## Algorithm and accounting

The new [MimeLite kernel](../../src/federated/mime_lite.py) follows Algorithm 1
and the Adam-like base in Table 5 of the
[MIME/MimeLite paper](https://arxiv.org/html/2008.03606): local gradients are
uncorrected, old server moments remain frozen during local steps, and moments
are updated once from full client gradients at the round-start model. Each
client's full fitting subset is streamed, including the final short batch.

Our population-weighted, full-participation simulation counts five FP32
model-sized vectors per client/round for MimeLite: model and two moments down;
local model and anchor gradient up. Full MIME additionally broadcasts a common
anchor gradient and uses two barriers. These are application-payload estimates,
not measured physical-network traffic; no transport overhead or privacy
guarantee is claimed.

Equal backward examples do not equalize FLOPs, unique rows, wall time, optimizer
steps, communication volume or synchronization. The report must retain both
quality and cost rather than claiming universally superior efficiency.

## Evaluation and validation

The 399,552 development rows retain the original grouped partition: 200,022
calibration and 199,530 audit rows. Only benign calibration scores set thresholds
at targets of 1%, 5%, 10% and 20%; 5% is primary. Report actual audit false-alert
rates, macro-F1, every class recall, and three-seed means/sample SD. Pairs with
more than 0.5 percentage points difference in realized FAR are flagged.

These development rows were already used in the study. This is not an untouched
test, and variation over three training seeds is not a confidence interval over
independent datasets or client partitions. Confirmation and original validation/
test arrays are not scored by this runner.

Every existing arm must reproduce its historical 21.6M endpoint exactly before
continuing: model, logits, thresholds/metrics, sampled rows and work counters.
Every run also replays its last two units from serialized state, including any
shortened final block. Scalar-reference tests verify MimeLite's local update and
full-anchor moment update independently. Additional tests check message/work
counts, invalid budgets, old-kernel equivalence and exact recovery.

## Commands

Use the prepared inputs and strict local CPU environment described in the
[methodology reproduction guide](../methodology_comparison/README.md).
`requirements-methodology.txt` remains sufficient; no new training dependency
was added. This run uses macOS ARM CPU, not a cloud GPU.

For a new compatible Mac checkout, create a new environment using Python
3.13.9, install `requirements-methodology.txt`, then fetch the approved,
checksum-pinned prepared inputs using the existing bootstrap:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-methodology.txt
.venv/bin/python scripts/reproduce_shared_adam.py setup --python .venv/bin/python
```

Only create `.venv` if it does not already exist; preserve a working environment.
The [input reproduction guide](../federated_shared_optimizer/reproduction-guide.md)
describes the release archive and local-file alternative. Stay on this comparison
branch: switching to an older release tag would remove the new MimeLite code.
The bootstrap's final suggested compressed-model training command is for the
older experiment; use the commands below for this seven-configuration comparison.

```bash
.venv/bin/python -m unittest tests.test_mimelite_comparison tests.test_mimelite_report tests.test_mime_adam tests.test_methodology_comparison -q
.venv/bin/python -m src.eval.mimelite_comparison check
.venv/bin/python -u -m src.eval.mimelite_comparison train --output outputs/my-mimelite-comparison
.venv/bin/python scripts/mimelite_report.py --run outputs/my-mimelite-comparison --output outputs/my-mimelite-report
```

If training is interrupted, rerun the **train** command with `--resume`:

```bash
.venv/bin/python -u -m src.eval.mimelite_comparison train --output outputs/my-mimelite-comparison --resume
```

The runner
checks identical source/configuration/data and runtime, restores optimizer state
and deterministic sample positions, and preserves verified completed jobs. It
does not change settings or silently restart from model weights alone.

The [comparison runner](../../src/eval/mimelite_comparison.py) produces one
`result.json`, checkpoint and development logits file per arm/seed, followed by
an aggregate `summary.json`. A completed result is reused only with matching
job settings, checkpoint hash and logits hash. Source changes require a new
output directory and a new comparison.

The report generator refuses missing runs, changed settings, unequal work,
failed historical/recovery checks and metrics inconsistent with confusion
counts. It produces complete numerical evidence, a per-seed CSV, a quality/FAR
plot, an eight-class recall heatmap and a work-allocation figure. The CSV is an
export of metrics, not a copy of private training inputs.
