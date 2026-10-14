# Feature aware expert for automatic attack classification

Direct access to flow features produces a stronger automatic classifier on the
reused development audit. The feature-only boosted trees reach mean macro-F1
0.697486, compared with 0.655632 for the existing centralized neural reference,
and reduce benign false alerts from 17.39% to 14.67%. Adding neural scores raises
macro-F1 only slightly, to 0.699460 with the IID model or 0.698635 with the
non-IID model. The main gain comes from the feature-based learner, not a
demonstrated need for a hybrid pipeline.

This is meaningful progress, but not a solved rare-class problem. Web-based
recall improves substantially while Brute Force F1 remains roughly unchanged.
Both predeclared continuation screens fail their requirement to improve both
rare-class F1 scores in every seed. Keep the stronger feature-only model as a
development comparator; do not promote a production model or scale cloud
training on this evidence. Classification is fully automatic, with no human
review, abstention or final-test access.

## Results on the fixed development audit

Values are equal-weight means across seeds 7, 17 and 27 on the same 408,027
rows. They are not final-test scores or three independent datasets. Benign
false alerts mean the percentage of truly benign rows labeled as any attack,
not the percentage of alerts that are false. The feature-only model and central
reference are shared between the two lane comparisons.

The new classifiers fit additional labeled development rows, while the neural
reference remains frozen from the original training. The three new tree
conditions share fitting rows, weights and tree budgets with each other; the
old neural reference is not a matched-training neural-versus-tree benchmark.

| Method | IID macro-F1 | Non-IID macro-F1 | IID benign false alerts | Non-IID benign false alerts |
| --- | ---: | ---: | ---: | ---: |
| Unchanged federated neural model | 0.613689 | 0.404997 | 18.79% | 0.21% |
| Previous linear score fusion | 0.640735 | 0.632954 | 19.49% | 19.46% |
| Nonlinear score-only control | 0.648389 | 0.643007 | 19.76% | 20.17% |
| Existing centralized neural reference | 0.655632 | 0.655632 | 17.39% | 17.39% |
| Feature-only boosted trees | 0.697486 | 0.697486 | 14.67% | 14.67% |
| Features plus neural scores | 0.699460 | 0.698635 | 14.49% | 14.68% |

Feature-only improves macro-F1 by 4.19 percentage points and reduces false
alerts by 2.72 points relative to the central reference. Both improvements
occur in all three seeds. The largest class-F1 gains are Spoofing (+13.38
points) and Web-based (+10.54); this is not an improvement uniformly spread
across every class. Macro-F1 seed SD is 0.002829 for feature-only, 0.002195 for
IID hybrid and 0.002770 for non-IID hybrid.

An improved 14.67% false-alert rate still flags roughly one in seven benign
flows. The study has not established an acceptable operational tolerance;
these results do not justify automatic blocking on a real network.

The nonlinear score-only control remains below the central reference in every
seed and lane. Under this fixed learner and budget, the original features
therefore supply useful predictive information that these two neural score
vectors alone do not recover. This does not establish an information-theoretic
limit on every possible score-based model.

The hybrid's average gain over feature-only is just 0.20 macro-F1 points for
IID and 0.11 for non-IID. It loses to feature-only in non-IID seed 17. The
additional neural inputs are not a consistent improvement across all pairs.

## Rare class recall improves more than precision

Precision measures how often a predicted category is correct; recall measures
how many true examples of that category are found. All entries are percentages,
including F1 on a 0–100 scale.

| Method | Web precision | Web recall | Web F1 | Brute Force precision | Brute Force recall | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized neural reference | 41.67 | 7.34 | 12.37 | 42.29 | 27.14 | 32.94 |
| IID linear score fusion | 20.86 | 11.02 | 14.41 | 34.33 | 30.17 | 32.03 |
| IID nonlinear score-only | 23.08 | 12.64 | 16.30 | 32.15 | 31.23 | 31.63 |
| Non-IID nonlinear score-only | 22.38 | 12.92 | 16.31 | 29.29 | 29.12 | 29.17 |
| Feature-only | 23.96 | 21.96 | 22.91 | 32.04 | 33.60 | 32.76 |
| IID hybrid | 22.88 | 23.80 | 23.31 | 32.77 | 34.39 | 33.52 |
| Non-IID hybrid | 22.73 | 23.52 | 23.11 | 32.07 | 34.52 | 33.21 |

Feature-only finds about three times the proportion of Web-based attacks as
the central reference, but still misses 78% of them. About 24% of its Web-based
predictions identify the correct category. Brute Force recall rises by 6.46
points, but precision falls by 10.25 points, leaving its mean F1 0.18 points
below the reference. Higher recall has not solved the precision trade-off.

The same bottleneck persists in the confusion matrices. Feature-only calls
48.16% of true Web-based examples Recon and 23.73% Benign. It calls 32.67% of
Brute Force examples Benign and 26.48% Recon. Among its incorrect Brute Force
predictions, Benign and Recon account for 492 of 545 errors when the three
seed matrices are summed. Those are repeated predictions on the same audit
population, not 545 distinct independently sampled flows.

The audit contains only 472 Web-based and 253 Brute Force rows. Duplicate
grouping protects split membership, but seed variation does not measure the
uncertainty of performance on a new capture session.

## Recall across all eight classes

Values are mean recall percentages. Every method predicts every row; referrals
or withheld predictions cannot improve these metrics.

| Class | Centralized reference | Feature-only | IID score-only | Non-IID score-only | IID hybrid | Non-IID hybrid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Benign | 82.61 | 85.33 | 80.24 | 79.83 | 85.51 | 85.32 |
| DDoS | 84.66 | 84.32 | 83.32 | 82.94 | 84.20 | 84.04 |
| DoS | 71.04 | 74.51 | 72.58 | 72.26 | 74.96 | 75.07 |
| Recon | 73.53 | 76.55 | 73.05 | 72.62 | 76.41 | 76.59 |
| Web-based | 7.34 | 21.96 | 12.64 | 12.92 | 23.80 | 23.52 |
| Brute Force | 27.14 | 33.60 | 31.23 | 29.12 | 34.39 | 34.52 |
| Spoofing | 70.31 | 84.89 | 66.40 | 66.34 | 85.07 | 85.12 |
| Mirai | 99.60 | 99.81 | 99.70 | 99.63 | 99.82 | 99.80 |

## Why the continuation screens did not pass

The hybrid exceeds both matched nonlinear controls in macro-F1 in five of
six pairs, but not non-IID seed 17. It improves macro-F1, Web F1 and false alerts
against the central reference in every pair, yet Brute Force F1 is lower for
non-IID seed 17 and both seed-27 pairs. Feature-only passes the central-reference
checks in seeds 7 and 17 but loses on Brute Force F1 in seed 27.

The complete per-seed comparison is retained rather than selecting the most
favorable seed or relaxing the screen after scoring:

| Seed | Feature-only macro-F1 | IID hybrid macro-F1 | Non-IID hybrid macro-F1 | Central BF F1 | Feature-only BF F1 | IID hybrid BF F1 | Non-IID hybrid BF F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.698492 | 0.701221 | 0.701630 | 0.326622 | 0.328820 | 0.348515 | 0.356000 |
| 17 | 0.699675 | 0.700158 | 0.698110 | 0.337469 | 0.344689 | 0.337615 | 0.321495 |
| 27 | 0.694292 | 0.697001 | 0.696165 | 0.324190 | 0.309392 | 0.319527 | 0.318681 |

A failed screen does not erase the measured overall gain. It limits the
claim: this is a stronger automatic development baseline, not a consistent
solution to both rare categories or proof that a hybrid is necessary.

## Inference cost is higher than the small neural model

The table uses the IID comparison's mean of three seed-specific median batch
times, in milliseconds. Timings include actual neural inference, score
transformation and the added classifier where applicable, on the local CPU
with two threads. The non-IID measurements show the same ordering.

| Method | One row in ms | Batch of 512 in ms |
| --- | ---: | ---: |
| Centralized neural reference | 0.043 | 0.165 |
| Linear score fusion | 0.154 | 0.511 |
| Feature-only | 3.352 | 9.977 |
| Nonlinear score-only | 3.464 | 9.013 |
| Hybrid | 3.483 | 10.229 |

The feature-only path takes about 78 times the central reference's single-row
time in this implementation. It uses no neural model, but 800 trees are not
free computation. Each fitted tree model occupies 1.40–1.45 MiB on disk and
contains 23,200 nodes. Serialization size is not runtime memory or a neural
parameter count. No low-latency advantage over the small network is established.
The first full comparison took 608.7 seconds locally, including training,
audit scoring, timing and verification inside the recorded run.

## Next local decision

Keep feature-only as the stronger comparator and keep the hybrid result as
an ablation, not the default architecture. Do not spend RONIN credits on a
larger network or paid decision API based on this result.

Before increasing capacity or tuning weights, check whether this tree gain
depends on the known `Number` and `Tot sum` feature routes. Use fresh fits with
the same rows, weights, seeds and tree budget, not inference-only masking.
Audit fit-versus-audit duplicate overlap under the reduced inputs first and
freeze a common eligible comparison population if needed. The earlier neural
ablation cannot establish the new tree model's dependence on those features.

If the gain survives that check, the useful follow-up is a bounded comparison
targeting the Benign/Recon/rare-class confusion, including a neural control
fitted on the same development rows and weighting. Only compare changes that
improve rare F1 while reporting false alerts and added inference cost. A more
elaborate cascade is a hypothesis, not a measured improvement here. Before a
generalization claim, a genuinely new evaluation cohort is needed; neither
this repeatedly used audit nor the already exposed final test can provide it.

## Why this is different from the previous experiments

The [linear score-fusion experiment](fusion-findings.md) raised macro-F1 but
still missed about 89% of Web-based attacks. The remaining errors were often
Recon, Benign and Spoofing predictions. That result did not establish that the
original features lacked useful information, because the fusion head saw only
two sets of eight neural scores.

Broader majority coverage was also
[tested earlier](../full_data_extension/diversity-findings.md). At a matched
balanced neural training budget it produced a small inconsistent gain and more
false alerts. This new run must not be presented as the first diversity test
or evidence that broader negative coverage alone fixes the problem.

The current comparison keeps the recent fusion fitting population and class
weights fixed, and changes the learner and its inputs. The older balanced-panel
tree fit is not reused or relabeled as this result: it had different rows,
class proportions and model settings. No original training code or frozen
neural checkpoint is changed.

## Three matched nonlinear conditions

All conditions use the same histogram-based gradient-boosted classifier:
100 boosting stages, eight trees per stage, at most 15 leaves per tree,
learning rate 0.1, minimum 30 samples per leaf and L2 regularization 1.
Early stopping and internal validation splitting are disabled. The recipe is
fixed before audit scoring; there is no threshold or hyperparameter sweep.

| Condition | Inputs | Neural models required at inference | Question |
| --- | --- | ---: | --- |
| Feature-only expert | 39 scaled flow features | 0 | Is a standalone nonlinear model sufficient? |
| Nonlinear score-only control | 16 log scores from the frozen generalist and specialist | 2 | Was the linear score head too restrictive? |
| Feature-aware hybrid | The same 39 features plus the same 16 log scores | 2 | Do direct feature access and neural scores work better together? |

The feature-only expert is shared across IID and non-IID comparisons for a
given seed. There are three such fits, six score-only fits and six hybrid fits
at seeds 7, 17 and 27: fifteen fits and 12,000 trees in total. The fixed budget
limits tree count and leaves, not identical realized node counts or runtime.
It is not a parameter-matched or compute-matched comparison against the older
neural models.

The earlier linear fusion, both unchanged neural models, their probability
average, both fitted single-model score corrections and the centralized
full-cohort neural reference are retained as historical controls. Their audit
predictions and metrics must replay exactly before the new results are accepted.

The tree estimator uses its seed for binning subsampling; early stopping is
disabled here. These seeds do not represent three different capture sessions
or independent validation datasets. The score and hybrid conditions also pair
the corresponding existing neural-model seeds.
[Estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).

## Fitting data and evaluation boundaries

The 270,326 fitting rows are unchanged from the previous score-fusion run,
selected within the same duplicate-group-preserving fit partition. They include
50,000 Benign, 40,913 Recon, 27,151 Spoofing, 1,502 Web-based and 760 Brute Force
rows, plus the retained DDoS, DoS and Mirai samples. No extra examples are mined
from the audit based on its errors.

Weights remain proportional to sqrt(n_c)/m_c, normalized to mean one, where
n_c is the full fit-partition class count and m_c is the sampled count. This
corrects the class cap and gives square-root-frequency total class weight.
There is no second class-balancing multiplier.

Feature values use the original train-fitted scaler. All 39 features remain,
including `Number` and `Tot sum`; no extra preprocessing, mask selection or
device topology is introduced. Features stay disk-backed outside the bounded
fit sample, and scoring uses batches of at most 8,192 rows.

All fifteen fitted models are saved and frozen before any new audit scoring.
The audit has the same 408,027 rows, including 472 Web-based and 253 Brute Force
examples, across every seed. The existing calibration partition is unused.
This is nevertheless reused exploratory validation, not an independent
confirmation after the many earlier development stages. The completed study's
original final test remains closed.

## Predeclared decision criteria

A hybrid continuation requires it to beat both matched feature-only and
nonlinear score-only macro-F1 in every seed/lane pair. It must also exceed the
existing centralized reference in macro-F1 and in both rare-class F1 scores,
without increasing that reference's benign false-alert rate. Individual checks
are reported separately rather than reducing the experiment to one headline.

If adding the neural scores offers no consistent benefit, the feature-only
expert has a separate predeclared comparison against the centralized reference
using the same macro-F1, rare-F1 and false-alert criteria. A standalone model's
success is not a hybrid success. Passing either development screen does not
approve deployment or establish independent generalization.

These are centrally fitted supervised classifiers using labeled development
rows. Combining one with a federated generalist is assisted inference, not
privacy-preserving federated tree training or improved federated neural
optimization. The old centralized reference is a practical frozen comparator,
not a newly trained neural control matched on these exact rows and budget.

## Reproduction and verification

The [fixed plan](feature-expert-plan.json) binds the earlier fusion results and
diversity evidence. The [aggregate results](feature-expert-results.json) retain
all seeds, all ten methods, eight-class confusion matrices, prediction hashes,
fit identities, serialized model sizes and timing samples. Implementation is in
[official39_feature_expert.py](../../scripts/official39_feature_expert.py) and
[feature_expert.py](../../src/eval/feature_expert.py). Existing byte-pinned
training, routing and fusion implementations are preserved.

An independent full replay refitted all fifteen models: thirty fits and
24,000 trees across the two executions. Serialized models, fitting-input
identities, training and audit prediction hashes, metrics, repair/damage counts
and all non-timing records match exactly. Both runs froze every model before
audit scoring, and every historical control reproduced its saved result.
Latency and elapsed time are deliberately not expected to be identical.

All eighteen new focused tests pass, including source/plan guards, equal fit
rows and weighting, changed-audit-label isolation, saved-model replay, bounded
tree work, no refitting during scoring, tampered metrics and score-only versus
feature-only input isolation. Aggregate auditing does not require private
flows or deserialize model objects. The models and arrays stay in ignored
outputs; neither run opened or evaluated the original final test.

The complete clean-snapshot repository suite passes all 457 tests locally
(439 existing plus 18 new), and `pip check` reports no broken requirements.
The snapshot excludes unrelated working-tree additions. This is a local
macOS verification, not a claim that a new cloud or cross-platform CI run
has occurred.

Audit the published aggregate evidence without rerunning training:

```sh
.venv/bin/python scripts/official39_feature_expert.py audit \
  --record reports/risk_aware_routing/feature-expert-results.json
```

Run locally from the repository root using the existing environment:

```sh
.venv/bin/python -u scripts/official39_feature_expert.py run \
  --cache outputs/official39-routing-gate-v2 \
  --fusion-root outputs/official39-score-fusion-v1 \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --specialists outputs/official39-panel-validation-v1 \
  --archives /Users/haydn/Downloads \
  --output outputs/official39-feature-expert-v1
```

Use a new output directory for a replay. Private model objects remain in
ignored outputs. Joblib objects are loaded only from this run's own
checksum-bound artifacts; never load arbitrary downloaded or uploaded joblib
files. Published aggregate auditing does not unpickle models or open flow arrays.

Latency is measured on the local two-thread CPU at batch sizes one and 512,
with three warmups, thirty repeats and rotating method order. Score-only and
hybrid timings include both actual frozen neural forward passes. Input
preprocessing, network transfer, loading and checkpoint I/O are excluded.
These are warm local implementation timings, not physical-edge measurements
or service latency guarantees.
