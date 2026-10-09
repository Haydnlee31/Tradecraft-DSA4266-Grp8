# Wider network training results and the validation decision

The subsequent [fixed validation comparison](panel-validation-findings.md)
is complete and does **not** support promoting the wider, zero-dropout model:
its training improvement did not carry over to natural-prevalence validation.
The training-only evidence and original next-step rationale below are retained.

Doubling the two hidden layers from 64/32 to 128/64 improves training macro-F1
in all six paired comparisons. At 300 epochs, the wider model without dropout
reaches a three-seed mean of 0.8175, versus 0.7583 for the smaller model without
dropout. Its mean benign false-alert rate also falls, from 28.26% to 19.75%.

This is useful evidence that width helps the network fit these rows under the
tested recipe. It is **not yet evidence of better validation performance or a
solution to non-IID federated learning**. The next step is a fixed validation
comparison, not another increase in width or training duration. Keep the cloud
machine stopped for now.

## What changed and what stayed fixed

The [frozen plan](width-fit-plan.json) compares two widths at both dropout 0.2
and dropout 0, using seeds 7, 17 and 27. All fits use the same balanced panel of
10,152 training rows, with 1,269 rows per class selected without replacement
from the 2M training cohort. They retain the same 39 features, train-fitted
scaler, LayerNorm, Adam settings, batch size 512 and fixed 300-epoch endpoint.
The loss weights are uniform on this balanced panel; the original imbalanced
cohort weights are not applied a second time.

Each fit processes 3,045,600 examples and performs 6,000 optimizer updates.
These counters include repeated exposure to the same panel, not millions of
unique examples. There is no early stopping or choice of the best epoch.

The smaller model has 5,096 parameters and the wider model has 14,280, about
2.80 times as many. Example exposure and updates are matched; FLOPs, parameter
count and wall time are not. Different widths cannot have identical initial
tensors. The comparison pairs seeds and initialization policy across widths,
and verifies identical starting tensors between the two dropout conditions
within each width.

Before any wider fit starts, all six smaller fits must reproduce their
[duration-control results](duration-fit-findings.md) exactly, including the
100-epoch prefix checks, final parameter hashes, learning curves and work
counters. All six passed. Production architecture defaults remain unchanged.

## Training scores across three seeds

Every score below is **in-sample performance on the balanced training panel**,
not natural-prevalence validation performance. Standard deviations describe
three training seeds on one panel, not independently sampled datasets or
confidence intervals.

| Hidden widths | Dropout | Parameters | Mean macro-F1 | Sample SD | Mean benign false alerts |
|---|---:|---:|---:|---:|---:|
| 64/32 | 0.2 | 5,096 | 0.720529 | 0.002993 | 24.32% |
| 128/64 | 0.2 | 14,280 | 0.756961 | 0.005179 | 23.25% |
| 64/32 | 0 | 5,096 | 0.758286 | 0.007011 | 28.26% |
| 128/64 | 0 | 14,280 | 0.817464 | 0.011086 | 19.75% |

Widening raises mean macro-F1 by 3.64 percentage points with dropout and 5.92
points without it. The zero-dropout improvement comes with an 8.51-point
reduction in mean benign false alerts. The wider zero-dropout model is the
strongest training-fit candidate in this bounded comparison, not a promoted
deployment model.

| Seed | Small with dropout | Wider with dropout | Small without dropout | Wider without dropout |
|---|---:|---:|---:|---:|
| 7 | 0.718538 | 0.761837 | 0.762972 | 0.807595 |
| 17 | 0.719079 | 0.751525 | 0.750225 | 0.829458 |
| 27 | 0.723971 | 0.757522 | 0.761660 | 0.815340 |

## Recall for every class

Values are mean training recall in percent over the same three seeds.

| Class | Small with dropout | Wider with dropout | Small without dropout | Wider without dropout |
|---|---:|---:|---:|---:|
| Benign | 75.68 | 76.75 | 71.74 | 80.25 |
| DDoS | 63.93 | 66.12 | 74.42 | 74.39 |
| DoS | 92.91 | 92.28 | 87.71 | 88.94 |
| Recon | 59.71 | 66.40 | 62.70 | 77.44 |
| Web-based | 59.52 | 64.07 | 62.67 | 76.15 |
| Brute Force | 62.62 | 69.58 | 71.21 | 74.36 |
| Spoofing | 62.17 | 70.87 | 76.36 | 82.51 |
| Mirai | 99.87 | 99.92 | 99.95 | 100.00 |

Without dropout, widening improves Web-based recall in every seed and raises
the mean by 13.48 points. Recon rises by 14.74 points on average. These are
promising fitting improvements for classes that have been difficult in the
natural-prevalence experiments.

The gains are not uniform. Zero-dropout seed 7 loses 7.33 points of Brute Force
recall, from 72.50% to 65.17%, although the other two seeds improve. With dropout,
seed 7's benign false-alert rate worsens from 23.17% to 26.00%; the other two
seeds improve. Mean DoS recall with dropout also falls slightly. Aggregate F1
does not remove these class-level trade-offs.

Even the wider zero-dropout model still misclassifies roughly one in five benign
training rows on average. Its 100% Mirai training recall does not establish
perfect Mirai detection on unseen traffic.

## Runtime and interpretation

The canonical local CPU run measured the following mean seconds per fit,
including training and scheduled training-panel evaluations but excluding
data hashing/loading and receipt writes:

| Dropout | Small network | Wider network | Wider divided by small |
|---|---:|---:|---:|
| 0.2 | 4.01 s | 5.21 s | 1.30 |
| 0 | 3.28 s | 3.69 s | 1.12 |

These are local diagnostic timings, not stable deployment benchmarks, cloud
speed predictions or physical edge measurements. Twelve fits in the canonical
run total 48.55 seconds inside the recorded fit timers. A second complete run
is used for numerical reproducibility, not to select the fastest timing.

Width helps fitting under this optimizer and budget. The experiment does not
separate representational capacity from changes in optimization caused by
widening, and it does not establish that 128/64 is an optimal size. All six wider
fits still improve macro-F1 and training cross-entropy between epochs 200 and
300, so a converged ceiling has not been demonstrated.

The earlier fixed boosted-tree reference reached training macro-F1 0.9444 on
the same rows. The wider zero-dropout mean remains 12.70 points lower. That
comparison is not parameter-, update- or time-matched; stronger memorization
is not proof of better generalization. No tree was refitted in this stage.

## Next fixed validation comparison

Freeze the four neural conditions above, all three seeds and epoch 300 before
opening validation. Reproduce each final fit exactly, then evaluate it on the
existing natural-prevalence validation split in bounded batches. Report every
condition and seed rather than choosing the most flattering seed or epoch.
Use the existing argmax classification rule; do not tune alert thresholds in
the same comparison.

The question becomes: do better training fits preserve rare-class detection
on unseen rows without excessive benign false alerts? Report macro-F1, all
eight classes' precision and recall, class supports and confusion matrices.
Training-panel and validation class proportions differ, so their macro-F1 gap
alone is not a clean measure of overfitting: precision also changes with class
prevalence. This validation split has supported earlier exploratory work;
it is not a newly untouched final test.

If including the existing fixed tree reference, preserve its original recipe
and rows and verify its training-fit replay before validation. No tree search
or federated-tree implementation is needed. Historical models trained on the
full 2M cohort may provide context, but are not data- or compute-matched controls
for these 10,152-row panel fits.

Use that comparison to decide whether a candidate warrants a separately designed
larger-data or federated experiment. A centrally pooled balanced-panel result
does not solve unequal class exposure or model aggregation across simulated
clients. Keep the final test split sealed; no candidate is promoted at this stage.

## Verification and reproduction

Implementation: `src/eval/official_width_fit.py`, with a bounded optional
`hidden_dims` argument in `src/eval/official_training_fit.py`. Existing comments
are retained and the new controls are explained inline.

Nine new tests cover default equivalence, wider-model replay, parameter counts,
within-width initialization pairing, exact prefix reconstruction, work and
configuration guards, checksum checks, no overwrites, and failure before any
wider training when a historical comparison fails. The training-only fixture
contains no validation or test arrays.

All 144 local workspace tests passed, including the nine new width-control
tests. Existing third-party deprecation and CPU-detection warnings did not
cause failures.

An independent real-data repeat reproduced all 12 fits, every observed metric,
parameter hash and paired change exactly, excluding elapsed time. The reference,
plan and recorded source hashes remained unchanged. Neither validation nor test
arrays were opened by this experiment. No dependencies were added, no cloud
session was launched, and no production model or decision policy was changed.

Canonical receipt: `outputs/official39-width-fit-v1.json`, SHA-256
`a40161a169cfc277c1e9fc3912859c90844c0623f15524f1838743e08a7d8919`.
Plan SHA-256:
`7f309da3b3707a080317f5f3c5ec48adf1ff7b73c0ca7f6b6797cbfa5a818e0d`.
The receipt contains each class's precision, recall and F1, confusion matrices,
learning curves, configuration and environment details, hashes and counters.
Generated receipts and data remain outside version control.

To reproduce locally, choose a new output filename:

```bash
.venv/bin/python -m src.eval.official_width_fit \
  --data-root outputs/official39-packed-2m-v1 \
  --reference-path outputs/official39-duration-fit-v1.json \
  --plan-path reports/full_data_extension/width-fit-plan.json \
  --output outputs/official39-width-fit-replay.json
```

These findings remain within the official-39 extension and its within-collection
split. They do not establish performance on unseen capture sessions, identifiable
devices or physical edge hardware.
