# Full-anchor MimeLite and matched baselines

This experiment implements both missing MimeLite configurations and compares
them with Central Adam, Shared Adam, full MIME and FedAvg-Adam. A centralized
MIME-base control is included because MIME's lagged-moment updater is not
ordinary bias-corrected Adam. The old untraceable MimeLite scores are not targets
and do not count as results of this pipeline.

## Protocol fixed before training

The [plan](plan.json) specifies seven configurations and seeds 7, 17 and 27.
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

```bash
.venv/bin/python -m unittest tests.test_mimelite_comparison tests.test_mime_adam tests.test_methodology_comparison -q
.venv/bin/python -m src.eval.mimelite_comparison check
.venv/bin/python -u -m src.eval.mimelite_comparison train --output outputs/my-mimelite-comparison
```

If the process is interrupted, rerun the last command with `--resume`. The runner
checks identical source/configuration/data and runtime, restores optimizer state
and deterministic sample positions, and preserves verified completed jobs. It
does not change settings or silently restart from model weights alone.

The [comparison runner](../../src/eval/mimelite_comparison.py) produces one
`result.json`, checkpoint and development logits file per arm/seed, followed by
an aggregate `summary.json`. A completed result is reused only with matching
job settings, checkpoint hash and logits hash. Source changes require a new
output directory and a new comparison.
