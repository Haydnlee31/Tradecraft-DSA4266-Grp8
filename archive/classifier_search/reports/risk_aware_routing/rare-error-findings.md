# Shared rare attack errors and feature overlap

The rare-class problem is not solved. Six frozen classifiers miss the same
60.08% of Web-based examples and 56.46% of Brute Force examples on the reused
development calibration cohort. Their errors concentrate on Benign and Recon.
The rare fitting examples do not contain exact conflicting labels in the 37
retained features, so this diagnostic does not support blaming identical,
contradictorily labeled rare flows for the failures.

The next useful experiment is a bounded test of training on confusing examples,
with a matched ordinary-sampling control. There is no demonstrated reason here
to buy more GPU capacity, add another generic fusion layer or reopen the
original final test. A new training experiment has not been launched.

## The models share most rare class misses

All counts refer to the same 409,918 development calibration rows at model
seed 7, including 501 Web-based and 271 Brute Force rows. The six models are
the feature-only boosted tree, piecewise-linear numerical embedding model
(PLE), retrieval model and its no-reference control, and balanced-batch
cross-entropy (CE) and supervised-contrastive models. Previously saved row
identities, labels, prediction vectors and scores were verified before comparison.

| Diagnostic | Web-based | Brute Force |
| --- | ---: | ---: |
| Correct tree predictions | 119 / 501 | 83 / 271 |
| Correct in at least one of the six models | 200 / 501 | 118 / 271 |
| Missed by all six | 301 / 501 | 153 / 271 |
| All six incorrectly predict Benign | 54 | 59 |
| All six incorrectly predict Recon | 145 | 40 |
| All six incorrectly predict Spoofing | 10 | 3 |

An imaginary selector that knows each true label and chooses a correct existing
hard prediction whenever one exists would reach only 39.92% Web recall and
43.54% Brute Force recall. That selector cannot be deployed. These numbers
limit selection among these six argmax outputs only: they are not a bound on
probability fusion, a newly trained model or achievable performance on this dataset.

For the tree alone, 355 of 382 Web misses and 176 of 188 Brute Force misses go
to Benign or Recon. Retrieval recovers 62 Web examples the tree missed but
loses 19 the tree got right. Its Web false positives grow from 293 to 734.
For Brute Force it recovers 17 and loses 9, while false positives grow from
152 to 428. This explains why a higher rare recall did not pass the earlier
precision and false-alert checks. The issue is not simply that every new model
finds a different useful set of attacks.

## Exact conflicts do not explain rare class failures

The full fitting population contains 1,229,859 rows, not just the 270,326-row
sample used to train the recent models. Numeric equality was checked on the
37 retained packed float32 features, excluding `Number` and `Tot sum`.

There are 1,227,124 distinct vectors and 623 mixed-label groups. Every mixed
group is a DDoS/DoS conflict. At least 623 fitting-row errors would be unavoidable
for a deterministic classifier using exactly this representation, regardless
of capacity. This is a finite fitting-set statement, not an estimate of Bayes
error or a generalization ceiling.

All 1,502 Web-based and 760 Brute Force fitting examples have distinct vectors
within their class, and none shares its exact vector with another class. The
772 rare calibration queries also have no zero-distance match to any full-FIT
reference in either checked geometry. Near overlap and sparse coverage can
still matter; lack of exact duplicates does not prove the labels are easily
separable. Packed float32 equality is not equality at original CSV precision,
and this check does not establish whether labels are semantically correct.

## Reference abundance explains part of the distance result

For every rare calibration query, the diagnostic finds its nearest fitting
example separately in each of the eight classes. It compares the distance to
the true class with the nearest other class. This uses the query's true label
only for post-hoc analysis, not to make a prediction.

Two fixed geometries were checked: Euclidean distance in the original packed,
standardized features and in the historical FIT-quantile coordinates. The
latter clips tails and spaces collapsed quantiles evenly; it is not a true
empirical percentile transform. Neither geometry is learned in this diagnostic.

Using every fitting example, almost every rare query missed by all six models
has another class closer than its true class. However, Benign has 65,706
references and Recon 40,913, versus only 1,502 Web and 760 Brute Force references.
It would be misleading to interpret this result as feature ambiguity alone.

A separately declared sensitivity check therefore samples 760 unique fitting
vectors per class without replacement. It uses three fixed draws, seeds 7,
17 and 27, and the same queries and two geometries. No calibration examples
enter the reference pool. Conflicting vectors across classes are retained;
deduplication is within each class. All available Brute Force vectors remain.

The table counts queries for which another class is strictly closer than the
true class. Ranges span both geometries and all three balanced-reference draws.
They are sensitivity ranges, not confidence intervals.

| Query group | Every FIT reference | Equal reference counts |
| --- | ---: | ---: |
| Web missed by all six, 301 queries | 293–294 / 301, 97.34–97.67% | 199–219 / 301, 66.11–72.76% |
| Brute Force missed by all six, 153 queries | 148–150 / 153, 96.73–98.04% | 111–116 / 153, 72.55–75.82% |
| Web correctly identified by tree, 119 queries | 69–75 / 119, 57.98–63.03% | 35–46 / 119, 29.41–38.66% |
| Brute Force correctly identified by tree, 83 queries | 24–25 / 83, 28.92–30.12% | 4–11 / 83, 4.82–13.25% |

Reference abundance clearly affects the first result. Yet the commonly missed
examples remain more often near competing classes than the tree's correctly
identified examples after count balancing. This is evidence of a difficult
local discrimination problem under these two geometries, not proof of
irreducible ambiguity. Correlated features, class diversity and reference
selection still affect distances. Some correctly classified flows also have
a closer wrong-class neighbour, demonstrating that proximity is not destiny.

## Existing performance remains the benchmark

These are unchanged scores recomputed from frozen predictions, not new training
results. All methods use the same calibration cohort; this is not a fresh,
budget-matched six-model training comparison. False alerts mean truly benign
rows classified as any attack.

| Model | Macro-F1 | Benign false alerts |
| --- | ---: | ---: |
| Tree | 0.700315 | 14.43% |
| PLE | 0.690363 | 14.99% |
| Retrieval control | 0.692977 | 15.69% |
| Retrieval | 0.680103 | 16.37% |
| Balanced CE | 0.690203 | 14.77% |
| Balanced CE plus contrastive loss | 0.689345 | 15.18% |

Recall percentages across all eight classes:

| Class | Tree | PLE | Retrieval control | Retrieval | Balanced CE | Contrastive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Benign | 85.57 | 85.01 | 84.31 | 83.63 | 85.23 | 84.82 |
| DDoS | 84.28 | 85.26 | 86.39 | 83.68 | 84.66 | 85.61 |
| DoS | 74.11 | 71.54 | 69.98 | 74.96 | 72.61 | 70.72 |
| Recon | 76.48 | 75.88 | 76.46 | 74.09 | 75.06 | 75.52 |
| Web-based | 23.75 | 20.76 | 21.76 | 32.34 | 17.96 | 23.35 |
| Brute Force | 30.63 | 25.46 | 30.26 | 33.58 | 35.42 | 31.00 |
| Spoofing | 84.51 | 83.83 | 85.05 | 84.17 | 83.55 | 83.41 |
| Mirai | 99.78 | 99.58 | 99.73 | 99.68 | 99.52 | 99.66 |

All class precision, recall, F1, support and confusion matrices are retained in
[the aggregate diagnostic](rare-error-results.json). The tree remains the
strongest of these comparators on macro-F1 and benign false alerts. A 14.43%
false-alert rate still does not justify operational automatic blocking.

## Next experiment and stopping rule

Test FIT-only hard-example sampling against ordinary sampling with the same
small eight-class model, initialization, class budgets and number of updates.
Hard examples here mean fitting flows the model struggles to distinguish,
especially rare attacks versus Benign and Recon. Preserve exposure to all
eight classes and retain randomly sampled examples rather than concentrating
every update on extreme outliers. Difficulty-based selection changes the
within-class training distribution; disclose it instead of describing it as
the same objective as uniform sampling.

The rationale is to spend learning effort on the observed confusion, rather
than merely amplifying all rare predictions. Hard-example selection is an
established method, not a novel algorithm invented for this project.
[Shrivastava, Gupta and Girshick's OHEM paper](https://arxiv.org/abs/1604.03540)
demonstrates this principle for object detection. Applying it to numeric IoT
flows is a hypothesis; that paper does not establish success here. The earlier
contrastive run used randomly balanced batches without targeted hard-example
mining, so this would answer a different question.

No calibration/audit mistakes may be copied into training or used to select
individual hard negatives. First fix the recipe, run a synthetic recovery
check and freeze both fitted models before scoring. Continue only if the new
method improves both rare-class F1 scores and macro-F1 without increasing
benign false alerts against both its matched control and the existing tree.
Otherwise close this training direction rather than expand a loss/sampling
grid. Even a pass would need confirmation on genuinely new data before a
generalization claim; these repeatedly reused development partitions cannot
become a fresh test by renaming them.

This is a recommendation, not a launched experiment or a deployment change.
It needs no human-in-the-loop inference and does not change the eight-class
target, introduce device identities or claim an improvement in federated
optimization. The original final test stays closed.

## Verification and reproduction

The [main plan](rare-error-plan.json) was committed before the first diagnostic.
The [reference-count plan](rare-reference-plan.json) was committed separately
after the abundance confound was identified and before its control was run.
That follow-up is explicitly post-hoc, not retrospectively described as part
of the first declaration.

Both diagnostics were executed twice. Row selections, distance matrices and
all non-timing aggregates replay exactly; see [the count-control result](rare-reference-results.json).
No new model was trained, no audit predictions were scored, and no original
test arrays or cloud resources were used. Private row identities and matrices
stay in ignored `outputs/`; published results contain aggregates and hashes.

All 557 tests pass in a clean snapshot containing committed project files,
including 12 new synthetic diagnostic/control tests. `pip check` reports no
broken requirements. This is a local macOS check, not a new remote CI or
physical-edge hardware measurement. Existing unrelated working-tree files
were excluded from the snapshot and left untouched.

Reproduce locally, using fresh output directories and the existing private
upstream artifacts:

```sh
.venv/bin/python scripts/official39_rare_error_diagnostic.py run \
  --output outputs/official39-rare-error-new-replay
.venv/bin/python scripts/official39_rare_reference_control.py run \
  --output outputs/official39-rare-reference-new-replay
```

The runners verify upstream provenance and refuse to overwrite existing
outputs. There is no cloud, training or original-test option.
