# Balanced training fit and the next neural network diagnostic

The proposed dropout-only follow-up is now complete. See
[dropout control findings](dropout-fit-findings.md) for its paired results and
current next step. The fitting baseline and original recommendation follow.

On the same balanced training panel, the current small network reaches macro-F1
of 0.6786 on average, while a fixed boosted-tree model reaches 0.9444. The network
learns substantially more rare-class training examples than in the earlier
imbalanced runs, but still fits this panel much less well than the trees.

This is evidence to investigate the neural training recipe before further cloud
scaling. It is not evidence that the tree generalizes better, that balancing
solves federated learning, or that the network needs more parameters. No
validation or test examples were opened in this stage.

## The question and fixed protocol

The previous aggregation probe found both weak rare-class learning and biased
local specialists. This experiment removes federated aggregation and gives every
class equal representation, asking a narrower question: can the existing small
network learn the examples it is being trained on?

The [fixed plan](training-fit-plan.json) was written before the real-data fits.
Its SHA-256 is
`f3c7686c5c18bb4e7409f1726eaa2a96699b762336b56b55609b06d8bdbb69ff`.
There was no hyperparameter search, early stopping or best-checkpoint selection.

| Item | Setting |
|---|---|
| Source | Existing 2M official-39 training cohort |
| Panel | 1,269 rows per class, 10,152 total; seeded sampling without replacement |
| Features | Same 39 float32 inputs and existing full-training-cohort scaler |
| Network | 39 → 64 → 32 → 8, LayerNorm, dropout 0.2; 5,096 parameters |
| Neural optimization | Fresh initialization; Adam, learning rate 0.001, weight decay 0.00001 |
| Neural budget | 100 epochs, batch size 512; seeds 7, 17 and 27 |
| Neural observations | Epochs 1, 5, 20, 50 and 100; final endpoint is primary |
| Tree fit | Histogram gradient boosting, 100 iterations, 31 leaves maximum per tree, learning rate 0.1, L2 1.0, seed 7 |
| Execution | Local CPU, two-thread limit; no cloud session |

The panel includes all 1,269 Brute Force training rows and a fixed selection of
1,269 rows from every other class. The same panel is used for every fit.
Square-root class weights are calculated from the balanced panel, making them
uniform. Reusing the original imbalanced cohort's weights here would compensate
for frequency a second time. This changes both exposure and the effective
objective relative to the original training regime; it is not a weight-only
ablation.

The tree estimator's automatic early stopping was explicitly disabled; no
internal validation fraction was held out. Its 100 iterations build 800 trees
for the eight-class problem. These settings follow the
[scikit-learn estimator API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
Tree and neural fits are not matched for capacity, optimizer updates or runtime.
No federated tree implementation was added, and no new packages were installed.

## Results on the training panel

All scores in this section are **in-sample training metrics on an artificial
balanced population**, evaluated on the same rows used for fitting. They must not
be combined with the earlier natural-prevalence validation scores in a model
leaderboard.

| Fit | Training macro-F1 | Training benign false alerts |
|---|---:|---:|
| Small network, seed 7 | 0.677902 | 26.40% |
| Small network, seed 17 | 0.681696 | 27.42% |
| Small network, seed 27 | 0.676149 | 31.05% |
| Boosted trees, seed 7 | 0.944438 | 4.41% |

Network macro-F1 is 0.678582 ± 0.002836, where the plus/minus value is the sample
standard deviation across three training seeds on ONE fixed panel. It is not a
confidence interval or variation over independently sampled panels. The tree
score is 0.265856 higher than the network mean, an in-sample fitting gap of 26.59
percentage points, not a demonstrated deployment improvement.

All entries below are training recall percentages.

| Class | Network mean | Network range across seeds | Boosted trees |
|---|---:|---:|---:|
| Benign | 71.71 | 68.95–73.60 | 95.59 |
| DDoS | 59.78 | 58.39–61.15 | 92.43 |
| DoS | 94.56 | 92.04–96.06 | 96.85 |
| Recon | 55.35 | 53.51–58.63 | 89.68 |
| Web-based | 53.27 | 48.31–56.97 | 92.12 |
| Brute Force | 49.23 | 47.60–50.75 | 91.96 |
| Spoofing | 59.92 | 57.76–62.65 | 96.77 |
| Mirai | 99.63 | 99.61–99.68 | 100.00 |

The network is still learning at the end of the fixed budget. Its seed-7
training macro-F1 rises from 0.6169 at epoch 20 to 0.6576 at epoch 50 and 0.6779
at epoch 100; the other two seeds show similar gains. Thus 100 epochs does not
establish a converged performance ceiling for this architecture.

## What this diagnoses

The missing classes are not completely unlearnable by the existing network:
with this balanced training setup, Web-based and Brute Force recalls are near
one half instead of near zero. However, balancing has not produced a network
that fits this panel well across all classes. It also changes which errors are
favoured: relatively low DDoS and Benign recall remain important, even though
rare recall improves.

The tree fit shows that another nonlinear model can represent decision regions
that fit many more of these same feature vectors and labels. It weakens the
claim that the observed neural failure is explained solely by unavailable
feature information. It does not establish the best achievable generalization
error: a higher-capacity tree ensemble can memorize distinctions that do not
transfer to other examples.

The remaining neural gap could involve regularization, optimization duration,
architecture, the feature representation as used by the network, or several of
these together. Because the tree and network have different model families and
capacities, this comparison cannot isolate one cause. In particular, it does
not justify jumping directly to a large GPU or treating the original dataset
as inherently inseparable.

## Next experiment before cloud scaling

Run a **dropout-only training-fit control locally** first. Pair dropout 0.2 with
dropout 0 on this exact panel, preserving widths, LayerNorm, scaler, row order,
seed-specific initial weights, Adam settings, weight decay and the same 100
epochs. The current three fits supply the reference; the new condition should
use the same three seeds. Compare final training metrics and learning curves,
not a selected best epoch.

Dropout deliberately makes training harder to discourage overfitting. Temporarily
removing it can test whether that regularization materially limits the fitting
check. This is a diagnostic change, not a recommendation to remove dropout in
the deployed classifier. Other settings must stay fixed so a difference has a
clearer interpretation.

If the gap remains, consider a separate predeclared training-duration or width
control. The current learning curves are still improving, so a longer-budget
control is relevant before declaring the small architecture fundamentally
incapable. Do not alter dropout, width, learning rate, preprocessing and class
weights together.

A candidate must subsequently be tested under a fixed protocol on natural
validation prevalence, with all eight recalls and benign false alerts. Success
on this balanced training panel alone is insufficient. Keep the test split
sealed and do not expand to 5M or another federated cloud sweep yet.

## Verification and reproducibility

Implementation: `src/eval/official_training_fit.py`.
Tests: `tests/test_official_training_fit.py`.
All eight new tests and the full 120-test local workspace suite passed. These
are local verification results, not a new GitHub CI run.
The dedicated loader checks training/scaler hashes and class counts and opens
only the two training arrays. Its test fixture contains no validation or test
files. It maps the full training arrays read-only and copies only the bounded
panel, rather than materializing all 2M feature rows.

Each neural fit processed exactly 1,015,200 examples and performed 2,000 optimizer
updates. The tree completed all 100 boosting iterations. Repeating the real-data
experiment reproduced the exact panel, neural learning curves and final parameter
hashes, tree predictions and all reported metrics; elapsed times are naturally
excluded from that equality check. The plan and recorded source hashes were
verified against the final implementation.

Canonical local receipt: `outputs/official39-training-fit-v2.json`.
It records panel row IDs, full eight-class confusion matrices and
precision/recall/F1 at every declared neural observation, optimizer counters,
configuration, package versions and provenance. Its SHA-256 is
`7a2f4f2ec871c1c2586729ca1f97bf887ad125485a06e687a475faea8066b679`.
Panel identity SHA-256 is
`eeb0da60b60a6af56c7a66920dd2877a59b3f5aa768fb79f27bca079b8afab52`.
Generated receipts remain outside version control, as do raw data and trained
model artifacts. No model was promoted or written into a cloud checkpoint.

To reproduce locally, use a new output filename:

```bash
.venv/bin/python -m src.eval.official_training_fit \
  --data-root outputs/official39-packed-2m-v1 \
  --plan-path reports/full_data_extension/training-fit-plan.json \
  --output outputs/official39-training-fit-replay.json
```

All conclusions remain within the 39-feature extension and its
duplicate-grouped within-collection split. They do not establish performance
on unseen capture sessions, identify physical devices, or measure edge hardware.
