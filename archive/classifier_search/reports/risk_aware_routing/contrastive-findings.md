# Contrastive training did not beat the tree

The bounded local contrastive experiment did not pass its continuation rule.
Adding the contrastive loss slightly improves Web F1 over its matched neural
control, but lowers Brute Force F1, increases benign false alerts and leaves
overall macro-F1 below the tree. Keep the tree as the development reference;
do not scale this configuration to cloud or present it as a rare-class solution.

This is one seed on **reused development calibration**, not independent test
evidence or proof that all contrastive approaches fail. Every method below
was evaluated on the same 409,918 flows, including 501 Web-based and 271 Brute
Force examples. No new audit or original-test scoring was performed. These
are centralized classifier experiments, not new federated-learning results.

## What changed

The motivation was to improve separation rather than merely predict rare labels
more often. [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
learns representations by attracting same-class examples and separating different
classes. Its original benchmark evidence concerns images; applicability to these
network-flow statistics was the hypothesis being tested.

Our adaptation uses other real fitting flows as positives, not two artificial
augmentations of a flow. It excludes self-pairs and exact duplicate feature
groups from both positives and negatives, even when a duplicate has conflicting
labels. Other-class examples within each batch are negatives; there is no
special hard-negative mining or evaluation-label feedback.

Both arms use the same numerical embedding basis, 270,326 fitting rows, initial
weights, batches, class weights and update budget. Each batch contains 64
distinct fitting rows per class. Independent draws across batches deliberately
repeat rare examples. The classification weights are corrected for this
sampling: each class gets `8 * sqrt(population_count) / sum(sqrt(counts))`.
That preserves the earlier square-root-weighted classification objective in
expectation; applying the earlier per-row weights unchanged would have
overweighted rare classes a second time.

The control uses classification loss alone. The treatment adds a fixed
contrastive coefficient of 0.1 at temperature 0.1. Both have the same small
training-only projection head, and neither uses it for prediction. Both have
18,584 training parameters and 17,000 deployment parameters. The existing
`Number` and `Tot sum` mask remains in place. There was no hyperparameter grid,
early stopping or choice of the best epoch.

## Calibration results

F1 ranges from zero to one. False alerts are the percentage of benign flows
assigned any attack label, not the fraction of rare predictions that are wrong.

| Method | Macro-F1 | Benign false alerts | Web F1 | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing tree | 0.70031 | 14.43% | 0.26068 | 0.32806 |
| Existing numerical embedding MLP | 0.69036 | 14.99% | 0.22222 | 0.30804 |
| Matched balanced CE control | 0.69020 | 14.77% | 0.22472 | 0.31373 |
| Balanced CE plus contrastive loss | 0.68934 | 15.18% | 0.23423 | 0.29116 |

Against the matched control, Web F1 rises by 0.95 percentage points, but Brute
Force F1 falls by 2.26 points and false alerts rise by 0.40 points. The small
aggregate difference should not be called statistically significant from one
seed. The predeclared practical gate nevertheless fails: it requires better
macro-F1 and both rare F1 scores without higher false alerts against both the
control and the tree. The gate was not relaxed after inspecting the result.

Contrastive Web precision/recall are 23.49%/23.35%; Brute Force precision/recall
are 27.45%/31.00%. The tree reaches 28.88%/23.75% and 35.32%/30.63%, respectively.
None of these results establishes deployment-ready rare-attack detection.

All eight recalls are shown below, in percent.

| Class | Tree | Existing MLP | Matched CE control | Contrastive |
| --- | ---: | ---: | ---: | ---: |
| Benign | 85.57 | 85.01 | 85.23 | 84.82 |
| DDoS | 84.28 | 85.26 | 84.66 | 85.61 |
| DoS | 74.11 | 71.54 | 72.61 | 70.72 |
| Recon | 76.48 | 75.88 | 75.06 | 75.52 |
| Web-based | 23.75 | 20.76 | 17.96 | 23.35 |
| Brute Force | 30.63 | 25.46 | 35.42 | 31.00 |
| Spoofing | 84.51 | 83.83 | 83.55 | 83.41 |
| Mirai | 99.78 | 99.58 | 99.52 | 99.66 |

## What fitting performance tells us

After the screen, an inference-only diagnostic compared the frozen models on
their fitting rows and the same calibration rows. This is post-hoc diagnosis,
not checkpoint selection or another training run. Recall can be compared
directly despite different class proportions. The diagnostic also preserves
confusion matrices standardized to the same fitting-population prior, so raw
F1 from the class-capped sample is not mistaken for natural-prevalence F1.

| Model | Web fitting recall | Web calibration recall | Brute Force fitting recall | Brute Force calibration recall |
| --- | ---: | ---: | ---: | ---: |
| Tree | 29.63% | 23.75% | 48.29% | 30.63% |
| Matched CE control | 22.64% | 17.96% | 45.53% | 35.42% |
| Contrastive | 26.56% | 23.35% | 49.34% | 31.00% |

Web remains poorly detected even on fitting rows. Brute Force has a much larger
fitting-to-calibration recall gap: 18.35 points for contrastive training. The
contrastive representation improves some fitting behavior without producing
the required useful calibration tradeoff. These observations do not isolate
capacity, loss priorities, optimization or feature ambiguity as the sole cause.
They do not prove that the available statistics impose an irreducible ceiling.

Each arm saw every fitting row at least once, but repeated exposure is not new
information. There were 2,027,520 sampled positions per class, yet only 760
distinct Brute Force and 1,502 distinct Web examples. All anchors had at least
one eligible positive. The treatment's mean online contrastive loss fell from
5.45 to 5.03; this shows optimization of the added term, not successful detection.

## Decision and next work

Stop this configuration, preserve the stronger tree reference and keep cloud
compute off. The completed Laya, retrieval and contrastive screens have not
established a new layer that consistently improves the rare-class tradeoff.
Numerical embeddings remain a useful improvement over the earlier plain MLP,
but that is not the same as solving rare attacks.

The next justified local step is a focused error and feature-separability
analysis of rare attacks versus Benign and Recon. Determine which mistakes
persist across the frozen models and whether fitting-only feature neighbourhoods
have consistent labels before choosing another architecture. This is a proposed
diagnostic, not a promise that another model will solve the errors. Avoid
repeating the earlier width/duration experiments or launching a temperature
search just because this coefficient failed.

Any eventual positive extension needs a separately specified independent
evaluation. The original test cannot become untouched again, and the existing
within-collection data cannot establish unseen-capture, per-device or physical
edge-hardware claims. No human-in-the-loop process is required.

## Verification and reproducibility

The [plan](contrastive-plan.json) was committed before training. Both fixed
60-epoch fits completed before calibration scoring. Their initial tensor
states, complete sampled-row order and CE weights match exactly. The tree and
existing MLP reproduce their historical calibration scores exactly.

An independent process reproduced both fits, checkpoint bytes, learning
observations, sampling counts and all four prediction arrays exactly, excluding
elapsed time. The [screen receipt](contrastive-results.json) contains the full
scores and provenance. The [fitting diagnostic](contrastive-fit-diagnostic.json)
contains the post-hoc counts and source identity; its calibration metrics were
also recomputed from the original fixed labels. A separate diagnostic rerun
reproduced the complete diagnostic receipt exactly.

Each fit uses 16,220,160 sampled examples in 31,680 updates. This is 600 more
positions than the earlier permutation recipe because all 528 batches per
epoch now contain 512 rows; the difference is explicit, and both new arms are
matched. Across the screen and replay, four fits consume 64,880,640 positions
and 126,720 updates. These are repeated exposures, not that many unique flows.

The first pass took 204.99 seconds on the local CPU, including provenance checks
and calibration scoring. Its control and treatment fitting timers were 104.49
and 92.51 seconds. Both compute the contrastive diagnostic; its coefficient is
zero for the control. Timings partly overlap regression tests and are not a
controlled speed comparison or physical-edge measurement.

The project passes 545 regression tests, including 13 new contrastive and
receipt tests. Dependency checking reports no broken requirements. No packages
were added, no cloud session was launched, and earlier experiment source and
comments remain unchanged. Checkpoints and row-level arrays stay outside Git.

Implementation: [training and loss](../../src/eval/contrastive_flow.py),
[screen runner](../../scripts/official39_contrastive.py), and
[fitting check](../../scripts/official39_contrastive_fit_check.py).
