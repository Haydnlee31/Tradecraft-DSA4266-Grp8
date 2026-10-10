# Locked official39 final-evaluation design — test still closed

The model-development sequence is closed without a deployment-ready replacement.
The next objective is to measure how the frozen research comparison performs on
the held-out within-collection split. Final evaluation will assess the current
models, not repair them. No model training, new threshold, test-row access or
test scoring occurred while preparing this design.

The machine-readable [protocol](final-evaluation-plan.json) pins the
[verified confirmation evidence](ablation-confirmation-results.json), all
candidate checkpoint hashes and data identities. Its current status is
`prepared_test_closed`, not ready-to-run or permission to access test data.
The checker has no evaluation action and no model/data-loading dependency.

## Candidate set and endpoint

Primary comparisons contain nine full39 models: centralized light, IID light
and controlled non-IID light, each at training seeds 7, 17 and 27. The eighteen
Number-only and joint-mask models remain secondary sensitivity comparisons.
No candidate was dropped because its confirmation score was unfavorable.

Every candidate is the exact saved `last.pt` at step 20, with its matching
environment and input mask. Do not substitute a validation-best checkpoint,
pick the best seed, fit an ensemble, retrain, recalibrate probabilities or
apply a prior correction. Partition seed stays 7. The inference architecture
remains the same 39-input, 5,096-parameter LayerNorm light network.

Predictions use the largest logit, resolving exact ties by the first canonical
class in `src/data/label_map.py`. This preserves the existing eight-class task.
No confidence cutoff or alert-rate cap is fitted in this evaluation. The earlier
1% false-alert analysis was exploratory, not an independently validated policy.

## Data identity and the overlap safeguard

The protocol binds the original duplicate-grouped split, shard manifest,
two-million-row train-fitted scaler, train/validation arrays and feature order.
The previously audited test split contains 2,060,864 rows before any float32
overlap screening. This count comes from existing metadata, not a new test read.
Preprocessing must reproduce float64 training-only scaling, then float32 casting,
then the candidate's training-time standardized-input mask. Never refit a scaler.

The known float64-to-float32 overlap issue must not be silently carried into
final evaluation. After separate permission to read test shards:

1. Stream test inputs in bounded batches; use a disk-backed exact-input index.
   Canonicalize float32 byte order and signed zero consistently. Do not load the
   full release, all shards or all three projected datasets into RAM at once.
2. Exclude a test row from the shared primary panel if its model-input vector
   matches any selected 2M training row or any original validation row under
   any of the three masks. The inclusion rule must not use labels, predictions
   or model scores. Validation is included because it guided model selection.
3. Preserve original test rows and their fixed ordering. Report counts by overlap
   source and arm, retained/excluded class supports, and within-test projection
   duplicates. Keep the original float64-unique row as the scoring unit; do not
   change weighting or discard conflicting test labels after seeing predictions.
4. Freeze hashes of the shared panel and its receipt before scoring any candidate.
   If a class has zero retained support, stop before scoring and request review;
   do not redraw, relax the filter or choose a more favorable population.

The common union-exclusion panel is primary for every candidate. The original
test population is a separate secondary view computed from the same forward
predictions. Publish both, even if their conclusions differ. Do not label either
as unseen-session evidence or claim independence from near-duplicates.

The new overlap rule is deliberately stronger than the validation-panel rule:
it checks both training and validation. Raw validation-to-test score differences
therefore also reflect different retained populations, not just generalization.

## Reporting and decisions

For every model and each population, retain the confusion matrix, supports,
macro-F1, benign false-alert rate and all eight classes' precision, recall and
F1. Split attack errors into predictions of Benign versus another attack class.
Report missing predictions explicitly; never omit a difficult class.

Summarize per seed first, then equally weighted mean, sample SD and range. Retain
the within-seed mask-minus-full39 comparisons and the separate prospective-seed
view. Do not use millions of rows or 27 endpoints as independent training
replications. Seeds 17 and 27 were prospective for the validation experiment;
they are not independent test datasets or new client partitions.

Final test performance cannot automatically authorize deployment. Numerical
operational requirements have not been agreed, and this plan does not invent
them after seeing scores. The decision layer must display class-coverage and
false-alert warnings and must not issue automated allow/block actions. A future
operational pilot requires separately agreed costs, targets and independent
validation. The original legacy46 decision defaults are unchanged.

If the test confirms poor rare-class detection, report that limitation rather
than tune against it. Once opened for final model assessment, this holdout cannot
be reused as a fresh final test for additional architecture searches.

## Execution gates and remaining work

The current closeout implements evidence export, offline audits and protocol
checks only. It does not yet implement the evaluator or create the test panel.
Keep this preparation-time plan immutable. A later execution receipt must bind
its hash, the separately approved access, runner identity and prepared panel;
do not make the checker pass by manually changing its closed-state flags.
The remaining work is bounded:

- Implement the evaluator and input-overlap preparation outside the frozen
  training path. Test masks, ties, counts, exclusions and output preservation on
  synthetic fixtures; replay saved validation endpoints before test scoring.
- Obtain explicit permission before opening test shards for the label-blind
  overlap audit. Freeze the resulting panel identity without inspecting scores.
- After all preflight checks pass, obtain explicit approval for one complete
  locked test evaluation. Preserve partial outputs on failure and resume only
  with matching identities; no score-dependent continuation or reruns.
- Publish the full result, then finalize the write-up and policy limitations.

These are inference/data-integrity tasks, not another cloud training campaign.
Keep RONIN stopped for implementation and synthetic checks. Measure validation
replay time locally before deciding whether cloud inference is necessary.

The official39 extension has no matched heavy lane. Keep that limitation visible:
the legacy46 heavy/light/federated study and the official39 light extension answer
different comparisons and must not be merged into a single accuracy leaderboard.
