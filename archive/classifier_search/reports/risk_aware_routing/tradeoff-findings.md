# Frozen-model alert and review trade-offs

11 October 2026 · Stage four · Post-audit exploratory sensitivity

The decision layer has useful but limited headroom. Letting the frozen router
introduce some additional benign alerts improves non-IID macro-F1, but the
rare-attack improvements are inconsistent across seeds. Routing uncertain
flows into a review queue is a different, potentially useful contribution:
it concentrates rare attacks without pretending the model classified them
correctly. Neither result establishes deployment readiness.

Keep this extension local and exploratory. Do not buy more compute, enlarge
the neural models, integrate a paid decision API, or reopen the completed
study's final test on the strength of these results. The next requirement is
a defined alert/review workflow and acceptable burden, not more training.

## What changed, and what did not

The [stage-three router](router-findings.md) failed its predeclared
zero-additional-false-alert screen. That conclusion is unchanged. This
separate [fixed plan](tradeoff-plan.json) asks what the *same frozen models*
do under illustrative alternative policies, after the earlier audit was
already seen. These are not revised success criteria or independent evidence.

All twelve neural models, six fitted trees, seeds (7, 17, 27), feature sets,
predictions and group-preserving partitions are unchanged. No model was
trained and no neural forward pass was run. The experiment uses saved
predictions and small-tree inference on the local CPU.

Thresholds are chosen on the existing 409,919-row calibration partition and
all saved before rescoring the same 408,027-row internal audit. The latter
contains 21,747 benign, 472 Web-based and 253 Brute Force rows. It is reused
validation, not unseen capture sessions or a new test set. All six zero-slack
calibration and audit comparisons reproduce stage three exactly.

Two workflows must stay separate:

- **Automatic consultation:** replace a selected federated prediction with
  the saved balanced-specialist or centralized-light prediction. Count both
  repaired mistakes and damage to correct predictions.
- **Review referral:** flag a flow for potential review, leaving its saved
  classifier label unchanged. Count referrals and unresolved errors. No
  human reviewed these flows; reviewer success and review time are unknown.

Automatic rules include learned, confidence and seeded-random selection for
the balanced specialist, and confidence/random controls for centralized-light.
Each gets calibration consultation ceilings of 1%, 5% and 10%, crossed with
0, 50, 100 or 200 extra false alerts per **10,000 truly benign flows**. Those
allowances correspond to +0, +0.5, +1 or +2 percentage points in benign
false-alert rate. They are illustrative scenarios, not approved tolerances.

Calibration maximizes macro-F1 within both limits, with a no-call option and
fewer calls preferred on equal F1. Review thresholds instead select the
largest whole-score group inside their calibration quota without optimizing
against calibration labels. Review uses learned, confidence and random scores.
The learned score still comes from the earlier label-trained repair/harm tree;
it is not a new model specifically trained to find review cases.

Audit applies each frozen threshold as-is. Equal scores stay together. A
calibration ceiling is not a guaranteed later workload or alert ceiling:
overruns are reported, never repaired by reranking or trimming audit rows.

## Automatic relabelling: gains have costs and seed dependence

Tables show equal-weight three-seed means. The 10% calibration ceiling is
shown to illustrate the largest tested allowance, not to select an operating
point. Every 1%, 5% and 10% result, seed value, sample SD and class metric is in
the [aggregate evidence](tradeoff-results.json). Seed SD is not a confidence
interval over independent datasets.

| Base lane / learned balanced policy | Extra-alert allowance per 10,000 benign | Macro-F1 | Benign false alerts | Actual consultations | Web recall | Brute Force recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IID base only | — | 0.61369 | 18.793% | 0% | 0.92% | 27.27% |
| IID learned | 0 | 0.61817 | 18.769% | 0.021% | 3.18% | 27.27% |
| IID learned | 50 / 100 / 200 | 0.62030 | 18.878% | 0.128% | 9.96% | 27.93% |
| Non-IID base only | — | 0.40500 | 0.215% | 0% | 0.00% | 0.00% |
| Non-IID learned | 0 | 0.40500 | 0.215% | 0% | 0.00% | 0.00% |
| Non-IID learned | 50 | 0.43965 | 0.307% | 0.314% | 2.47% | 7.91% |
| Non-IID learned | 100 | 0.49581 | 0.972% | 4.577% | 6.07% | 17.52% |
| Non-IID learned | 200 | 0.49793 | 1.248% | 4.688% | 6.21% | 17.79% |

All learned-policy rows respect the later consultation limit. At the non-IID
100-alert allowance, however, seeds 7 and 17 produce **112.20 and 102.54**
additional alerts per 10,000 benign flows: both violate that allowance even
though the mean is below it. At 200, each seed is within the later allowance.
That compliance does not establish that the allowance is acceptable.

The non-IID 200 scenario is not a robust rare-attack fix. Seed macro-F1 values
are 0.54043, 0.52708 and 0.42629. Brute Force recalls are 27.67%, 25.69% and
**0%**; Web recalls are 9.53%, 8.69% and 0.42%. A mean improvement must not hide
that last seed. At the smaller 50 allowance, seed 17 has zero Web recall and
seed 27 still has zero Brute Force recall.

IID learned routing improves macro-F1 in every seed under the positive
allowances, but damages more previously correct predictions than it repairs
in every seed (27/70, 51/258 and 102/275 repaired/damaged). Macro-F1 can rise
because rare classes gain while other classes lose; it is not a measure of
net repaired rows. Brute Force is unchanged in seed 27. Allowing more than
50 does not change these IID calibration choices.

Full per-class recall for the illustrative 200-allowance learned policy, in
percent:

| Class | IID base | IID learned | Non-IID base | Non-IID learned |
| --- | ---: | ---: | ---: | ---: |
| Benign | 81.207 | 81.122 | 99.785 | 98.752 |
| DDoS | 84.556 | 84.556 | 98.744 | 95.973 |
| DoS | 67.412 | 67.413 | 20.892 | 32.104 |
| Recon | 72.555 | 71.397 | 19.837 | 20.695 |
| Web-based | 0.918 | 9.958 | 0.000 | 6.215 |
| Brute Force | 27.273 | 27.931 | 0.000 | 17.787 |
| Spoofing | 56.871 | 56.755 | 0.936 | 27.158 |
| Mirai | 99.425 | 99.425 | 98.735 | 98.891 |

Controls prevent crediting all improvements to the learned gate. Under the
same 10%/200 calibration limits, non-IID confidence routing to the balanced
specialist gives F1 0.48768 and 1.928% benign false alerts; its seed-7 later
alert limit is exceeded. Random balanced routing gives F1 0.43151 and 2.120%
false alerts. Confidence routing to centralized-light gives F1 0.49405 and
1.875% false alerts, again exceeding seed 7's alert limit. The learned gate
has a useful mean trade-off here, not universal superiority or a novel
classification breakthrough.

For IID, confidence routing to centralized-light remains stronger in mean
macro-F1 (0.64116) and false-alert rate (17.256%) than learned balanced routing,
but Web recall is lower (4.45%) and two seeds slightly exceed the 10% workload
ceiling. No single displayed policy solves every objective.

## Review queues: useful concentration is not detection

Review coverage means the fraction of a true class entering the queue,
including examples the base already classified correctly. It is **not recall
after correction**. The evidence separately counts base errors referred,
base errors left outside the queue, attacks labelled benign, and benign
referrals. It never assumes that a person or an API fixes referred errors.

Selected illustrative review scenarios:

| Lane / score | Calibration ceiling | Actual referrals per 10,000 flows | Web review coverage | Brute Force review coverage |
| --- | ---: | ---: | ---: | ---: |
| IID learned | 5% | 33.3 | 14.97% | 7.51% |
| IID confidence | 5% | 497.8 | 60.17% | 44.14% |
| IID random | 5% | 500.6 | 5.08% | 6.46% |
| Non-IID learned | 10% | 511.9 | 15.96% | 30.83% |
| Non-IID confidence | 10% | 994.3 | 10.66% | 16.07% |
| Non-IID random | 10% | 1000.2 | 10.10% | 11.33% |

IID confidence concentrates rare examples much more than random referral at
roughly comparable realized workloads. Across seeds its Web coverage is
57.84–62.71% and Brute Force coverage is 42.69–46.64%. Its seed-17 review rate
is 5.019%, slightly beyond the calibration ceiling. The mean queue also
contains 6,422.7 benign flows out of the 408,027-row population. These are
benign referrals, not newly counted classifier false positives.

The learned non-IID score puts more rare cases into the queue than confidence
at this tested ceiling, at a lower mean workload. But realized referral rates
vary from 1.677% to 7.439% across seeds because tree-score ties make some quota
levels unavailable. These are not exactly workload-matched comparisons.
Even this queue leaves approximately 84% of Web examples and 69% of Brute
Force examples outside it. Learned repair/harm scoring is not a general
solution to finding all difficult flows.

All-class review coverage for these confidence/learned examples, in percent:

| Class | IID confidence, 5% ceiling | Non-IID learned, 10% ceiling | Non-IID confidence, 10% ceiling |
| --- | ---: | ---: | ---: |
| Benign | 29.534 | 2.563 | 2.129 |
| DDoS | 1.536 | 2.788 | 6.132 |
| DoS | 2.709 | 11.299 | 28.410 |
| Recon | 35.639 | 7.574 | 17.585 |
| Web-based | 60.169 | 15.960 | 10.664 |
| Brute Force | 44.137 | 30.830 | 16.074 |
| Spoofing | 31.882 | 40.430 | 18.550 |
| Mirai | 0.299 | 1.134 | 0.733 |

The sample is overwhelmingly attack-labelled. A high attack fraction in a
queue would therefore be a misleading headline; it cannot establish useful
precision in a lower-prevalence real network. Neither live workload, analyst
time, end-to-end latency, edge feasibility nor actual reviewer performance
was measured. Saved predictions were available everywhere, so a small
consultation percentage does not demonstrate measured compute savings.

## Recommendation and boundary of the extension

Frame the contribution as **harm-aware selective consultation and honest
review-workload accounting**, not simply “add a hybrid classifier for higher
accuracy.” The results explain why better rare-class specialist coverage did
not translate into a safe automatic router under the previous constraint.

The next implementation, if wanted, should be a local read-only triage
prototype: show the existing prediction and an explicit “needs review” flag,
not an invented corrected class. First specify who or what would review a
flagged flow, the acceptable referral burden, and how to handle quota
overruns. No review ceiling or automatic alert tolerance is chosen here.
An actual review study would need measured reviewer outcomes, including
failure, cost and time. No Jev, Laya or Decisions API was tested.

If the deliverable remains a fully automatic eight-class classifier, retain
the completed study and report these limits rather than promoting this
extension. Stronger claims require a suitable independent evaluation protocol
and data; the exposed final test cannot be made fresh by reshuffling it.
Cloud compute is optional only for a justified final compatibility check or
benchmark after a worthwhile candidate is frozen. It is unnecessary now.

## Verification and reproduction

The successful recorded run section took 9.3 seconds on the local CPU,
excluding startup and initial preflight. A second run reproduces every
non-timing receipt field exactly. All 90 zero-slack policy comparisons
(six base/seed pairs, five rules, three ceilings) match stage three.
There are 360 automatic and 54 review scenarios across those six pairs.

An initial execution completed calculations but stopped at publication:
integer quota dictionary keys became strings when serialized to JSON. The
runner now uses string keys consistently and a round-trip regression test
guards this bookkeeping issue. The initial private output is preserved;
only fresh successful runs supply the published evidence. No threshold grid,
model, allowance or objective changed in response to this error.

No dependencies were added. Aggregate auditing recomputes metrics from
confusion counts, checks correction/damage totals, verifies review workload
arithmetic and scenario completeness, and enforces the no-training/no-test
scope. Published evidence contains no individual flows, probabilities, trees
or credentials. Beginner-oriented comments and unrelated local work remain
intact.

Verification passes all **404 tests** in 39.5 seconds in a clean repository
snapshot excluding unrelated uncommitted work. This includes 17 new tests;
the focused router/trade-off suite passes 46 tests. `pip check`, the published
aggregate audit, exact two-run replay and all-class report-table checks pass.

From the repository root, the aggregate audit needs no private flow arrays:

```bash
.venv/bin/python -m unittest tests.test_policy_tradeoffs -q
.venv/bin/python scripts/official39_policy_tradeoffs.py \
  --audit-summary reports/risk_aware_routing/tradeoff-results.json
```

For a full local replay, use the recorded CPU environment, earlier private
router caches and a fresh output path:

```bash
.venv/bin/python scripts/official39_policy_tradeoffs.py \
  --router-run outputs/official39-routing-gate-v2 \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --output outputs/official39-policy-tradeoffs-reproduction
```

The optional `--publish-summary` accepts a fresh destination only. Existing
experiment outputs and published evidence are never overwritten.
