# Dropout control results and the remaining fitting gap

The proposed duration-only comparison is now complete. See
[duration control findings](duration-fit-findings.md) for its paired results and
the current next step. The dropout-only experiment is recorded below.

Removing dropout improves balanced training macro-F1 in all three paired seeds,
from a mean of 0.6786 to 0.7131. It also increases mean benign false alerts on
that training panel, and does not improve every class in every seed. Dropout
therefore limits some fitting under this recipe, but its removal is not a complete
solution or a justified production change.

These are training-only results. No validation or test arrays were opened, no
cloud training occurred, and the production models and checkpoints are unchanged.

## What changed and what stayed fixed

The [frozen plan](dropout-fit-plan.json) compares dropout 0.2 with dropout 0 on
the exact panel used in the [previous fitting diagnostic](training-fit-findings.md).
It preserves all 10,152 rows, the 39 features and original train-fitted scaler,
the 64/32 hidden widths, LayerNorm, balanced-panel class weights, Adam settings,
batch size 512 and the 100-epoch budget. Seeds remain 7, 17 and 27.

For each seed, the two conditions start with identical parameter tensors. The
shuffle uses a separate NumPy stream determined by seed and epoch, so disabling
PyTorch dropout cannot alter the row order. Removing stochastic masks during
training is the intended treatment, not an unreported second change. Both
conditions use evaluation mode when measuring training fit.

Before running any zero-dropout fit, the updated helper reproduced every
historical baseline learning curve, parameter hash and work counter exactly.
This checks that adding the optional dropout argument did not change the
default diagnostic. Each fit processes 1,015,200 examples and performs 2,000
optimizer updates; the model still has 5,096 parameters.

## Paired results at epoch 100

The scores below are in-sample metrics on a balanced panel, not natural-prevalence
validation or deployment results. Changes in macro-F1 are percentage points.

| Seed | Macro-F1 with dropout 0.2 | Macro-F1 with dropout 0 | Change | Benign false alerts with 0.2 | Benign false alerts with 0 |
|---|---:|---:|---:|---:|---:|
| 7 | 0.677902 | 0.713824 | +3.59 | 26.40% | 33.33% |
| 17 | 0.681696 | 0.707570 | +2.59 | 27.42% | 28.13% |
| 27 | 0.676149 | 0.717904 | +4.18 | 31.05% | 30.65% |
| Mean | 0.678582 | 0.713099 | +3.45 | 28.29% | 30.71% |

Sample standard deviations of macro-F1 are 0.002836 with dropout and 0.005205
without it. These describe training-seed variation on one fixed panel, not
confidence intervals or variation across independent datasets.

The table below includes every class's mean training recall across the three
seeds. All values are percentages.

| Class | Dropout 0.2 | Dropout 0 |
|---|---:|---:|
| Benign | 71.71 | 69.29 |
| DDoS | 59.78 | 64.91 |
| DoS | 94.56 | 94.72 |
| Recon | 55.35 | 57.95 |
| Web-based | 53.27 | 57.00 |
| Brute Force | 49.23 | 58.92 |
| Spoofing | 59.92 | 68.40 |
| Mirai | 99.63 | 99.79 |

Brute Force and Spoofing recall improve in every seed. Web-based does not:
seed 17 declines from 56.97% to 54.37%, despite higher macro-F1. Benign recall
falls in seeds 7 and 17, and improves slightly in seed 27. Thus an aggregate
gain must not be presented as uniformly better attack detection.

The previous fixed tree fit remains at training macro-F1 0.9444; it was not
rerun or tuned. The gap from the zero-dropout neural mean is still 23.13
percentage points. Because the tree ensemble differs in capacity and model
family, this remains a fitting reference, not a controlled claim about which
architecture will generalize better.

## Interpretation and next step

The result supports a limited conclusion: dropout 0.2 makes it harder for this
small network to fit these balanced training examples at the fixed budget.
That is compatible with dropout's regularizing purpose. It does not show that
removing it improves unseen-data performance, and the rise in mean training
false alerts is another reason not to promote the change now.

Neither condition has clearly stopped learning. Between epochs 50 and 100,
mean training macro-F1 increases from 0.6532 to 0.6786 with dropout and from
0.6896 to 0.7131 without it. In every seed, training cross-entropy also declines
over that interval. The current experiment therefore does not establish a
capacity ceiling for the small network.

The next bounded local control should change **training duration only**, comparing
the existing 100-epoch endpoints with 300 epochs for BOTH dropout conditions
and the same three seeds. Keep the panel, initialization, order and optimizer
unchanged, and require the first 100 epochs to reproduce these results exactly.
Predeclare the 300-epoch endpoint; do not stop at whichever epoch looks best.
This separates an insufficient fitting budget from the dropout effect before
changing widths or preprocessing.

This remains a diagnostic, not a search for a high training score to advertise.
Any resulting candidate still needs a fixed natural-prevalence validation
comparison with all eight recalls and benign false alerts. Do not expand the
dataset, change several settings together, evaluate the held-out test set or
restart cloud scaling on the basis of these fitting scores. The next control
is small enough to run locally.

## Verification and reproduction

Implementation: `src/eval/official_dropout_fit.py`, using the opt-in dropout
argument in `src/eval/official_training_fit.py`. The production runner and
architecture defaults were not changed. Tests in
`tests/test_official_dropout_fit.py` check identical initialization, exact
baseline replay, fixed work, zero-dropout repeatability, rejection of other
dropout settings and prevention of treatment after a failed baseline bridge.
All eight new tests and the full 128-test local workspace suite passed.
An independent repeat of all six real-data fits reproduced every learning curve,
initial/final parameter hash and paired difference exactly, excluding elapsed
time. The recorded source hashes and unchanged reference receipt were verified.

The canonical receipt is `outputs/official39-dropout-fit-v1.json`, SHA-256
`fa36aac0816e5e184e706ff04d02f258026cd89b90523c4f06f233ae3322a5b4`.
It records all six fits, per-class precision/recall/F1, confusion matrices,
learning curves, initial/final parameter hashes, paired differences and source
provenance. Its plan SHA-256 is
`bfaf3274b681e41c50ee2807b3b4ffe4a8456028c85df9e3dba672dc99dd7d02`.
The source fitting receipt is preserved unchanged. These generated receipts
remain outside version control.

Run locally from the repository root, choosing a new output filename:

```bash
.venv/bin/python -m src.eval.official_dropout_fit \
  --data-root outputs/official39-packed-2m-v1 \
  --reference-path outputs/official39-training-fit-v2.json \
  --plan-path reports/full_data_extension/dropout-fit-plan.json \
  --output outputs/official39-dropout-fit-replay.json
```

This is still a within-collection fitting diagnostic. No unseen-session,
physical-device or real edge-hardware performance claim follows from it.
