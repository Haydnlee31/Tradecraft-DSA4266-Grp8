# Automatic score adjustment results

Retain the square-root-weighted, jointly masked tree baseline and close this
bounded score-adjustment search. None of the nine alternatives improves both
rare-class F1 scores and overall macro-F1 without increasing benign false
alerts at every calibration seed. The selected rule is therefore the unchanged
baseline, not a new hybrid improvement.

The useful result is a clearer boundary: increasing rare-class priority and
then correcting the class scores changes which mistakes the model makes, but
this experiment does not find a joint solution to rare detection and false
alerts. It does not establish that every decision-layer architecture must fail.
The [fixed plan](score-adjustment-plan.json) and
[aggregate evidence](score-adjustment-results.json) preserve all candidates,
per-seed metrics and confusion matrices.

## Why test this rule

The [equal-class-weight tree](balanced-tree-findings.md) recognized more rare
attacks, but its rare precision fell to about 6%. A cheap remaining question
was whether its existing scores contained useful distinctions that a different
automatic decision rule could recover. No human review, new classifier,
external API or cloud run was needed.

For each frozen tree, the rule adds `alpha * log(fitting class prior)` to its
log class probabilities, then predicts the highest-scoring class. The prior
comes from the entire development fitting partition, not the capped training
sample or calibration/audit labels. Larger alpha favours more common classes.
Alpha zero preserves the classifier's original prediction exactly. These are
decision scores, not calibrated probabilities of real-world security risk.

The fixed grid contains alpha 0, 0.25, 0.5, 0.75 and 1 for each of the two
existing tree objectives. That is ten rules including the baseline, or nine
alternatives. Both trees mask `Number` and `Tot sum`, keep the other 37 features,
and retain their original preprocessing, fitted parameters and seeds 7, 17
and 27. Selection chooses one common rule across seeds, not an ensemble or a
best seed. Applying any one rule needs only its associated classifier.

## How selection was separated from the audit

The development calibration partition originally contained 409,919 rows. An
exact comparison of the retained feature bytes found one DoS row matching an
unsampled row in the full 1,229,859-row fitting partition. The conservative
screen excluded it even though that matching fitting row was not among the
270,326 classifier-training rows. No remaining calibration row matched the
fixed audit partition. Membership checks used features only, not labels.

Calibration therefore used 409,918 rows, including 501 Web-based and 271 Brute
Force examples. All 408,027 historical audit rows remained unchanged, including
472 Web-based and 253 Brute Force examples. No rows were reassigned.

An eligible alternative had to meet all four requirements separately at each
of the three seeds:

- Higher macro-F1 than the same-seed baseline.
- Higher Web-based F1.
- Higher Brute Force F1.
- No increase in benign false-alert rate.

Among eligible rules, the declared ranking maximizes mean rare-class F1, then
mean macro-F1, then uses the fixed candidate order. With no eligible rule, the
baseline is mandatory. The complete choice and calibration evidence were
saved before new audit scoring. Only the selected rule and the two raw
historical controls were scored on audit; no rejected adjusted rule was
evaluated there. The grid was not expanded after seeing the result.

Both partitions are reused exploratory validation data. The calibration
partition also appeared in earlier routing experiments. This separation
controls selection within this stage; it does not create a pristine holdout
or independent confirmation after the earlier adaptive experiments.

## Calibration results and the tempting near miss

Values are means across the three paired model seeds. F1 values use the 0–1
scale; false alerts are percentages of truly benign flows classified as attacks.

| Frozen classifier | Alpha | Macro F1 | Benign false alerts | Web F1 | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Square-root baseline | 0 | 0.69935 | 14.49% | 0.25900 | 0.32182 |
| Square-root | 0.25 | 0.69529 | 12.83% | 0.19719 | 0.34985 |
| Square-root | 0.5 | 0.68149 | 11.48% | 0.14007 | 0.33835 |
| Square-root | 0.75 | 0.66844 | 10.26% | 0.10812 | 0.31989 |
| Square-root | 1 | 0.65519 | 9.23% | 0.08046 | 0.30448 |
| Equal-class | 0 | 0.63763 | 23.40% | 0.12585 | 0.11180 |
| Equal-class | 0.25 | 0.68021 | 16.33% | 0.23112 | 0.24056 |
| Equal-class | 0.5 | 0.70051 | 13.98% | 0.24667 | 0.34599 |
| Equal-class | 0.75 | 0.69495 | 12.38% | 0.19639 | 0.35340 |
| Equal-class | 1 | 0.67442 | 11.20% | 0.11163 | 0.31643 |

The equal-class tree with alpha 0.5 has slightly higher mean macro-F1 and lower
false alerts than the baseline. However, Web F1 falls from 0.25900 to 0.24667,
and Web recall falls from 23.42% to 18.16%. It fails the Web F1 requirement at
every seed and the macro-F1 requirement at seed 7. Promoting it on the average
headline alone would conceal the rare-class regression that this extension
was intended to fix.

In fact, every alternative has lower Web F1 at every calibration seed. Some
improve Brute Force F1 or reduce false alerts, but none passes even one seed's
full set of requirements. This is not merely a near tie rejected by the
all-three-seeds rule.

## Retained audit baseline

The fallback reproduces every historical baseline prediction exactly. Its
audit macro-F1 remains **0.69876** and benign false-alert rate **14.50%**.
Between-model-seed macro-F1 SD is 0.00373. This is neither an improvement from
the adjustment nor an acceptable deployment claim: roughly one in seven
benign flows still triggers an attack classification.

| Class | Audit examples | Mean precision | Mean recall | Mean F1 |
| --- | ---: | ---: | ---: | ---: |
| Benign | 21,747 | 83.23% | 85.50% | 0.84348 |
| DDoS | 240,791 | 91.39% | 84.39% | 0.87753 |
| DoS | 73,445 | 59.21% | 73.98% | 0.65773 |
| Recon | 13,559 | 74.83% | 76.46% | 0.75638 |
| Web-based | 472 | 24.90% | 21.82% | 0.23249 |
| Brute Force | 253 | 33.87% | 34.52% | 0.34164 |
| Spoofing | 8,868 | 92.55% | 84.51% | 0.88345 |
| Mirai | 48,892 | 99.67% | 99.80% | 0.99735 |

These are repeated scores on the same rows, not three independent capture
sessions or three times as many rare examples. The models are centrally
fitted tree controls, not federated tree training. No physical edge latency,
power, unseen-session generalization or probability calibration is established.

## Decision and study contribution

Close this particular weighting-and-score-adjustment search. Do not launch a
larger GPU or extend the grid to chase a favourable development number. Keep
the existing tree result as a stronger exploratory baseline alongside the
matched neural comparison, and present the equal-weight and score-adjustment
experiments as informative negative controls.

The study can now explain more than an accuracy ranking: the feature-aware
tree improves the earlier neural comparison, but rare-attack performance and
false alerts remain a difficult trade-off. The current evidence does not
support claiming that a hybrid decision layer has solved rare classes. Any
future method would need a separately justified hypothesis and genuinely new
validation evidence, not further selection on this same audit. The completed
official39 final-test study remains unchanged and was not reopened.

## Verification

The runner uses frozen, hash-verified model files and batches feature access.
It fits zero new classifiers. A separate fresh-directory execution repeats
the overlap checks, probability inference, selection and audit comparisons;
publication requires exact equality of all non-timing evidence. The raw
square-root and equal-class audit predictions must also match their previous
published records exactly.

Both executions completed and matched exactly. All 13 focused tests and all
512 tests in a clean repository snapshot passed; dependency checks found no
broken requirements. Report tables were checked against the published numeric
evidence. The snapshot excluded unrelated, uncommitted user work.

The implementation and portable tests are in
[the runner](../../scripts/official39_score_adjustment.py),
[the decision rules](../../src/eval/score_adjustment.py) and
[the tests](../../tests/test_score_adjustment.py). Aggregate verification does
not require private flow arrays or model files:

```bash
.venv/bin/python scripts/official39_score_adjustment.py audit --record reports/risk_aware_routing/score-adjustment-results.json
.venv/bin/python -m unittest tests.test_score_adjustment -q
```
