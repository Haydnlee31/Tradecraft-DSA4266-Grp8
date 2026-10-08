# Training duration control results and the next architecture check

The proposed width-only follow-up is now complete. See
[wider network training results](width-fit-findings.md) for the controlled
64/32 versus 128/64 comparison and the next fixed validation step. The duration
results and original rationale below are retained as the historical record.

Extending the small network from 100 to 300 epochs improves balanced training
macro-F1 in all six fits. Three-seed means reach 0.7205 with dropout 0.2 and
0.7583 without dropout. More training helps under this recipe, but does not close
the fitting gap to the previously measured tree reference. Some class recalls
decline even as macro-F1 rises.

These are training-only diagnostics, not validation or deployment improvements.
No validation or test arrays were opened, no cloud session was used, and no
production architecture, checkpoint or decision policy was changed.

## The fixed duration comparison

The [plan](duration-fit-plan.json) changes only the number of epochs within
each existing dropout condition. It retains the same balanced panel of 10,152
training rows, 39 features, train-fitted scaler, 64/32 hidden widths, LayerNorm,
Adam settings, batch size 512 and seeds 7, 17 and 27. Dropout remains either
0.2 or 0 throughout each fit. The model has 5,096 parameters in both conditions.

All runs start from their original initialization. Immediately after epoch 100,
the code compares the whole earlier record—including learning curves, model
parameter hashes and work counters—with the corresponding archived fit. A
mismatch stops that fit before epoch 101. All six comparisons passed exactly.
The same model and Adam optimizer then continue without a reset. Epoch-specific
shuffling continues through the same deterministic NumPy seed policy.

The endpoint of 300 epochs and the extra observation at epoch 200 were specified
before running. There is no early stopping, best-epoch selection or tree refit.
Each completed fit processes 3,045,600 examples and performs 6,000 optimizer
updates—three times the 100-epoch training work by those counters, not a claim
of exactly three times the wall-clock cost.

## Results across three seeds

All table entries below are **in-sample training metrics on the balanced panel**.
False-alert percentages describe benign rows in that panel, not operational
traffic. Means use the same three seeds, not three independently sampled panels.

| Dropout | Epochs | Mean macro-F1 | Mean benign false alerts |
|---|---:|---:|---:|
| 0.2 | 100 | 0.678582 | 28.29% |
| 0.2 | 200 | 0.704650 | 25.58% |
| 0.2 | 300 | 0.720529 | 24.32% |
| 0 | 100 | 0.713099 | 30.71% |
| 0 | 200 | 0.740299 | 28.11% |
| 0 | 300 | 0.758286 | 28.26% |

From 100 to 300 epochs, mean macro-F1 increases by 4.19 percentage points with
dropout and 4.52 points without it. At epoch 300, the sample standard deviations
across training seeds are 0.002993 and 0.007011 respectively; these are not
confidence intervals or evidence of statistical significance.

| Seed | Dropout 0.2 at 300 | Change from 100 | Dropout 0 at 300 | Change from 100 |
|---|---:|---:|---:|---:|
| 7 | 0.718538 | +4.06 pp | 0.762972 | +4.91 pp |
| 17 | 0.719079 | +3.74 pp | 0.750225 | +4.27 pp |
| 27 | 0.723971 | +4.78 pp | 0.761660 | +4.38 pp |

All eight classes' mean training recalls are shown below, in percent.

| Class | Dropout 0.2 at 100 | Dropout 0.2 at 300 | Dropout 0 at 100 | Dropout 0 at 300 |
|---|---:|---:|---:|---:|
| Benign | 71.71 | 75.68 | 69.29 | 71.74 |
| DDoS | 59.78 | 63.93 | 64.91 | 74.42 |
| DoS | 94.56 | 92.91 | 94.72 | 87.71 |
| Recon | 55.35 | 59.71 | 57.95 | 62.70 |
| Web-based | 53.27 | 59.52 | 57.00 | 62.67 |
| Brute Force | 49.23 | 62.62 | 58.92 | 71.21 |
| Spoofing | 59.92 | 62.17 | 68.40 | 76.36 |
| Mirai | 99.63 | 99.87 | 99.79 | 99.95 |

Web-based and Brute Force training recall improve in every fit. However, DoS
recall drops in all three zero-dropout fits and two of the three dropout fits.
Without dropout, mean DoS recall falls by 7.01 percentage points. The optimizer
is changing decision boundaries and trading errors between classes; it is not
simply making all predictions uniformly better.

Mean benign false alerts decline relative to epoch 100 in both conditions, but
the zero-dropout condition still has higher mean false alerts than dropout 0.2
at epoch 300. The decline is also not universal: zero-dropout seed 17 rises
from 28.13% to 29.55%. This prevents calling the higher macro-F1 condition an
unqualified winner.

## What this establishes

The 100-epoch fitting budget was not sufficient to exhaust improvement under
this recipe. The controlled extension obtains further gains without extra
features, more unique training examples, a larger network or a different
optimizer. The extra work repeats the same panel; it is not a data-scaling result.

At the same time, neither neural condition fits the panel as well as the fixed
tree reference from the earlier diagnostic, whose training macro-F1 was 0.9444.
The gap is still 18.62 percentage points relative to the zero-dropout mean at
300 epochs. Trees can fit or memorize different boundaries and have a different
capacity, so this is not proof that they generalize better or a precise measure
of insufficient neural capacity.

Both neural learning curves are still improving from 200 to 300 epochs, and
training cross-entropy declines in every fit. A converged ceiling or irreducible
error floor has not been demonstrated. The justified conclusion is narrower:
tripling the fitting budget helps but does not close the observed gap within
the tested range.

## Next bounded step

Use one **width-only local control**, comparing hidden widths 64/32 against
128/64 at the same fixed 300 epochs. Keep both dropout conditions and all three
seeds, with the same panel, scaler, normalization, loss, batch size and optimizer
settings. Do not simultaneously change learning rate, preprocessing or class
weights. The smaller-network results above provide the reference.

Different widths cannot share identical parameter tensors, so pair seeds and
initialization policy, not an impossible promise of equal initial weights.
Match example exposure and optimizer-update counts, while explicitly reporting
the larger parameter count and measured runtime; this would not match FLOPs or
compute cost. The question is whether wider layers fit the same rows materially
better at the same update budget, not whether a larger model is universally
superior.

After that bounded architecture check, consolidate the findings into a fixed
natural-prevalence validation comparison rather than indefinitely increasing
training budgets or sweeping architectures. No candidate should be promoted
solely because it scores well on its fitting panel. Keep the final test split
sealed and the cloud machine stopped for the local check.

## Verification and reproduction

Implementation: `src/eval/official_duration_fit.py` and the optional
`prefix_reference` check in `src/eval/official_training_fit.py`. Historical
default fitting behavior and production training defaults remain unchanged.

Seven new tests cover exact prefix matching, stopping before additional work
on a mismatch, unchanged observation schedules, continued-versus-uninterrupted
training equivalence, bounded plans, checksums and training-only receipts.
All 135 local workspace tests passed. An independent real-data repeat matched
all six complete fits, weight hashes, learning curves and paired changes exactly,
excluding elapsed time. The reference receipt, plan and recorded source hashes
were verified unchanged.

The canonical receipt is `outputs/official39-duration-fit-v1.json`, SHA-256
`8181757ef1eb86c284d680dca0befc982918b187128f502a5c9ece6a93eac9f3`.
It contains every class's precision, recall and F1, confusion matrices at all
specified observations, prefix and final weight hashes, work counters and paired
duration differences. Its plan SHA-256 is
`92a1ada84c8b0299f3fcb7bf240b437e6a8f641978d8b822d9e647f955ea00b5`.
Generated receipts and training data remain outside version control.

To reproduce locally, select a new output filename:

```bash
.venv/bin/python -m src.eval.official_duration_fit \
  --data-root outputs/official39-packed-2m-v1 \
  --reference-path outputs/official39-dropout-fit-v1.json \
  --plan-path reports/full_data_extension/duration-fit-plan.json \
  --output outputs/official39-duration-fit-replay.json
```

This remains a fitting diagnostic within the official-39 extension, not a
benchmark on unseen capture sessions, identifiable devices or physical edge
hardware.
