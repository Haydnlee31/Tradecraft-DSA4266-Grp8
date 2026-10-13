# Numeric retrieval did not improve the rare attack tradeoff

The local retrieval experiment should not replace the tree baseline. It found
more rare attacks, but it also mistook substantially more other flows for rare
attacks. Macro-F1 fell, both rare-class F1 scores fell against the tree, and
benign false alerts increased. This configuration failed the continuation rule
specified before training, so it does not justify more seeds or cloud scaling.

These are **one-seed exploratory results on reused development calibration**,
not final-test results. All five methods below use the same 409,918 calibration
rows, including 501 Web-based and 271 Brute Force flows. The earlier numerical
embedding report used a different audit subset; its numbers should not be
substituted into this comparison. No new audit or original-test predictions
were made here.

## What the extra model does

Instead of converting a flow into prose or asking a person to review it, the
network looks up eight similar labelled fitting flows and learns how much
their evidence should influence its prediction. The inspiration is
[TabR](https://arxiv.org/abs/2307.14338), whose
[official implementation](https://github.com/yandex-research/tabular-dl-tabr/blob/main/bin/tabr.py)
uses learned keys, attention, label embeddings and key-difference values.

Our bounded adaptation fixes the candidate search in fitting-derived numeric
coordinates and learns the weighting of those candidates. It is **not a full
TabR reproduction**: candidate search is not updated in the learned space,
and the reference bank is capped. A negative result here therefore does not
prove that every retrieval architecture would fail.

The bank contains 14,550 unique fitting flows: 2,048 from each common class,
all 1,502 Web-based fitting examples and all 760 Brute Force fitting examples.
It never contains calibration, audit or test rows. During fitting, a query
cannot retrieve itself or another row with its exact retained feature vector.
The existing duplicate-screened development cohorts remain unchanged.

Both new networks use the 225-dimensional numerical basis fitted previously
from 37 retained features. `Number` and `Tot sum` remain excluded. Both use
the same 270,326 fitting rows, weighted objective, seed 7, Adam settings,
512-row batches and fixed 60-epoch endpoint. There is no checkpoint search.
The reference-free control retains active key and value transformations;
it has 20,232 parameters versus 20,488 for retrieval. The difference is the
256-parameter label embedding, not a large capacity increase.

## Results on the same calibration flows

F1 values range from zero to one. False alerts are the percentage of genuinely
benign flows assigned any attack label. Lower false alerts are better.

| Method | Macro-F1 | Benign false alerts | Web F1 | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing tree | 0.70031 | 14.43% | 0.26068 | 0.32806 |
| Existing numerical embedding MLP | 0.69036 | 14.99% | 0.22222 | 0.30804 |
| New reference-free control | 0.69298 | 15.69% | 0.22614 | 0.31721 |
| Learned retrieval | 0.68010 | 16.37% | 0.23193 | 0.23038 |
| Ordinary eight-neighbour voting | 0.57794 | 27.19% | 0.07548 | 0.13456 |

The rare-rich bank alone is not a solution. Ordinary voting finds 43.31% of
Web attacks, but only 4.13% of its Web predictions are correct. Learning the
reference weights improves that precision to 18.08%, still below the tree's
28.88%. Brute Force precision is 17.53% for learned retrieval versus 35.32%
for the tree. The bank's class mix is a design choice, not an estimate of the
real attack prior or a calibrated risk score.

Every class's recall is reported below, in percent.

| Class | Tree | Existing MLP | Reference-free control | Retrieval | Voting |
| --- | ---: | ---: | ---: | ---: | ---: |
| Benign | 85.57 | 85.01 | 84.31 | 83.63 | 72.81 |
| DDoS | 84.28 | 85.26 | 86.39 | 83.68 | 76.63 |
| DoS | 74.11 | 71.54 | 69.98 | 74.96 | 75.68 |
| Recon | 76.48 | 75.88 | 76.46 | 74.09 | 60.50 |
| Web-based | 23.75 | 20.76 | 21.76 | 32.34 | 43.31 |
| Brute Force | 30.63 | 25.46 | 30.26 | 33.58 | 26.94 |
| Spoofing | 84.51 | 83.83 | 85.05 | 84.17 | 56.34 |
| Mirai | 99.78 | 99.58 | 99.73 | 99.68 | 98.92 |

## Why the increased recall is not enough

The tree correctly identifies 119 Web attacks and produces 293 false Web
labels. Retrieval correctly identifies 162 but produces 734 false Web labels.
That is **43 additional detections for 441 additional false labels**.

For Brute Force, correct detections rise from 83 to 91, while false labels rise
from 152 to 428: **8 additional detections for 276 additional false labels**.
Of retrieval's false Web labels, 231 come from benign flows and 383 from Recon.
For false Brute Force labels, those counts are 178 and 192. These are false
labels from all other classes, not the denominator of the benign false-alert
metric above.

The measured problem is insufficiently selective classification, not a failure
to emit rare labels at all. A plausible interpretation is that this fixed
similarity geometry does not separate rare attacks from their benign and Recon
look-alikes well enough. This is a hypothesis about this configuration, not
proof of irreducible ambiguity in the dataset.

## Next bounded experiment

Keep the tree as the reference and stop this retrieval configuration. The next
useful hypothesis is **supervised contrastive training**: explicitly train an
embedding to bring same-class examples together and separate different classes,
with attention to the benign and Recon confusions above. The original
[SupCon paper](https://arxiv.org/abs/2004.11362) supplies the mechanism and image
benchmark evidence, not a promise of better CICIoT classification.

Compare one fixed contrastive-loss recipe against cross-entropy alone using
the same encoder, batches and work budget. Any class-balanced sampling must be
shared by both arms and its effect on the classification objective made explicit.
Do not invent packet augmentations that could change attack meaning, use
evaluation labels as hard negatives, or expand into a temperature/loss grid.
The same rare-F1, macro-F1 and false-alert gate should apply. This proposed
experiment has not been launched.

Neither a human-review queue nor an LLM API is required. More GPU spending is
not supported by this result. Any later successful extension still needs an
independent evaluation protocol; repeatedly reusing development data does not
produce an untouched test set. These are centralized classifier experiments,
not new federated-learning or physical-edge measurements.

## Work and reproduction

The first complete pass took 228.21 seconds on the local CPU. Its new control
fit took 39.91 seconds and retrieval fit took 150.13 seconds. The calibration
neighbour search took another 15.56 seconds. These are local batch wall-clock
measurements, partly concurrent with regression tests, not a controlled
deployment-latency comparison. Reporting only retrieval's 0.64-second neural
forward pass would omit its search cost.

Each new fit processes 16,219,560 query examples in 31,680 optimizer steps.
The retrieval fit additionally consumes 129,756,480 reference slots; query
example counts alone therefore do not imply equal computation. The plan
allows two fits and their independent replay, with no further training sweep.
All checkpoints and row-level arrays stay in ignored `outputs/`.

The [predeclared plan](retrieval-plan.json) and
[experiment entry point](../../scripts/official39_numeric_retrieval.py) retain
the source identities and stop rule. No dependencies were added and earlier
experiment implementations were left unchanged.

The independent process rebuilt the bank and neighbours and retrained both
networks. Bank membership, all candidate indices, tensor checkpoints, training
observations and all five calibration prediction arrays matched exactly.
Across both passes, the four fits processed 64,878,240 query examples in
126,720 optimizer updates. The complete metrics and provenance are in the
[result receipt](retrieval-results.json).

A receipt-validation test caught an interface mismatch: the score verifier
expects named class supports rather than a list. This was corrected without
changing training, candidate search, scoring or the continuation rule. Both
runner source identities are preserved; the receipt explicitly distinguishes
this verifier-only change from an identical-source replay. The experiment's
`identity`, `save_array` and `run` functions also passed an AST comparison
against the original committed implementation.

The clean committed implementation passed **532 tests**, including 13 retrieval
and receipt tests. Dependency checking found no broken requirements. Private
data, unrelated working-tree changes and checkpoints were excluded from the
clean-snapshot test and the experiment commits. All work stayed local; the
branch remains `experiment/risk-aware-routing`.
