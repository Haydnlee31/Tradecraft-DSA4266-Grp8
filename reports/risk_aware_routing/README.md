# Routing feasibility for the official39 extension

Updated 11 October 2026. The initial comparison found that selectively
consulting the existing centralized-light model improves the federated models
on reused validation data, but cannot solve their rare-attack failures. Even
a perfect selector between those original models has only 6.87% mean Web-based
recall. Subsequent stages add a balanced specialist and examine routing costs.

**Current direction: fully automatic classification.** The latest
[equal-class-weight experiment](balanced-tree-findings.md) raises Web/Brute
Force recall to 58.76%/53.89%, but precision falls to about 6%, macro-F1 drops
to 0.6366 and benign false alerts rise to 23.19%. Every continuation check
fails at every seed. Retain the square-root-weighted tree at 0.6988 macro-F1
and 14.50% false alerts. The next bounded local question is a calibration-only
automatic score adjustment on frozen classifiers, with a false-alert guard
and a stopping rule, not more model capacity or cloud compute.

The preceding
[matched-data neural comparison](matched-neural-findings.md) retains the tree
advantage: mean audit macro-F1 0.6988 versus 0.6560, with benign false alerts
14.50% versus 17.06%. Neural rare recall is poor even on fitting rows, and a
perfect selector between this pair adds less than 1.5 recall points for either
rare class. The equal-class-weight control above follows that diagnosis;
higher rare recall alone did not provide a better classifier.

The preceding
[feature-removal comparison](feature-ablation-findings.md) shows that the tree
gain survives retraining without `Number` and `Tot sum`: mean macro-F1 0.6988,
benign false alerts 14.50%, Web-based recall 21.82% and Brute Force recall
34.52%. The exact overlap screen retains all 408,027 audit rows. Changes between
tree arms are small; no shortcut-free, deployment or new-session claim follows.
The matched-data neural control above follows that result and diagnoses the
remaining Benign/Recon/rare-category confusion.

The preceding [feature-aware expert comparison](feature-expert-findings.md) reaches mean
macro-F1 0.6975 with feature-only boosted trees, versus 0.6556 for the existing
centralized neural reference, while reducing benign false alerts from 17.39%
to 14.67%. Adding neural scores reaches 0.6995/0.6986 for IID/non-IID, only a
small gain over feature-only. Web-based recall improves to 22–24%, but Brute
Force F1 is not consistently better, and tree inference is much slower than
the small neural model. Both strict continuation screens fail; retain the
feature-only result as a stronger exploratory baseline, not a deployment or
hybrid success. The feature-dependence check above follows that result.

The preceding [score-fusion experiment](fusion-findings.md) combines the frozen
generalist and balanced specialist using a learned eight-class head. Mean
IID/non-IID macro-F1 reaches 0.6407/0.6330, below the centralized reference,
with about 89% of Web-based attacks still missed. That motivated direct feature
access with matched nonlinear score-only and feature-only controls. Neither
automatic classifier requires human review or abstention. The earlier review
work below is preserved as an optional side experiment.

Stage four's [frozen-policy trade-off analysis](tradeoff-findings.md)
finds some gains when extra benign alerts are allowed, but no consistent
rare-attack fix. Review queues concentrate some rare cases without claiming
they are correctly detected. No operating point is approved; work remains
local and no cloud run is needed.

A [local read-only triage prototype](triage-prototype.md) demonstrates
those review policies using unchanged saved predictions. Flags are unresolved
referrals, never automatic corrections or blocking actions. Its generated row
preview stays outside git. It is not in the automatic fusion prediction path.

The [stage-three learned-router experiment](router-findings.md)
did not pass its predeclared success screen. One IID seed improves slightly;
the other IID seeds and all non-IID seeds choose no calls under the conservative
benign false-alert guard. No model is promoted and no cloud run is needed.

Stage two's [balanced-specialist comparison](specialist-findings.md) raised the
Web-based diagnostic recall ceiling to 51.07%, but did not make confidence
routing safe. That motivated the earlier policy analysis. The current
automatic experiments test learned score combination and direct access to flow
features, while still reporting false alerts and all eight classes.
The remainder of this page preserves the stage-one results and rationale.

This is a separate exploratory extension on `experiment/risk-aware-routing`.
The [completed official39 study](../full_data_extension/final-closeout.md) stays
unchanged. Its final test has already been exposed. This diagnostic does not
reopen that test. The stage-one diagnostic described below did not train a
model, fit a learned gate, call an API or authorise deployment. Stage three
fits only the local router, as documented separately. The automatic extension
fits score-combination heads and feature-based classifiers, and now adds three
fresh matched-data neural controls; all original neural checkpoints stay
unchanged. No stage evaluates Jev, Laya or the Decisions API. The new
experts are centrally fitted, not privacy-preserving federated tree training.

## What was compared

The [fixed local plan](headroom-plan.json) pairs each IID and controlled non-IID
model with the same-seed centralized-light model, at seeds 7, 17 and 27. All
nine are the existing full39, step-20 checkpoints trained on two million rows.
There is no seed, mask or checkpoint search. The centralized model is another
5,096-parameter model, not a newly trained heavy model or a rare-attack expert.

Each pair sees the same 2,047,805-row validation panel, excluding the previously
identified exact training-input matches under the ablation projections. The
original data and split membership do not change. Validation has already been
used extensively for development: these are feasibility measurements, not
independent generalisation estimates.

Four types of result answer different questions:

- **Base:** retain the federated prediction for every row.
- **Always centralized:** consult the centralized model for every row.
- **Budgeted controls:** consult it for the least-confident 1%, 5% or 10% of
  federated predictions, or a seeded random sample of exactly the same size.
- **Label-informed oracle:** consult it only when the federated prediction is
  wrong and the centralized prediction is correct. This uses the true labels
  and cannot be implemented at inference time. It never repairs a row both
  models misclassify.

All metrics cover the full panel, including incorrect escalated predictions.
An escalation is not counted as a successful detection merely because someone
could review it. The implementation records corrected predictions and damage
to previously correct predictions, including newly introduced benign alerts.

## Simple routing helps but does not resolve the trade off

Values are equal-weight means across three paired training seeds. False alerts
are the percentage of truly benign rows labelled as attacks. They are not the
percentage of alerts that are false. Seed SDs and every seed's metric values
are retained in [the aggregate results](headroom-results.json).

| Base and routing rule | Escalation budget | Macro F1 | Benign false alerts |
| --- | ---: | ---: | ---: |
| IID without routing | 0% | 0.61386 | 19.07% |
| IID lowest confidence | 1% | 0.62159 | 18.18% |
| IID lowest confidence | 5% | 0.63888 | 17.45% |
| IID lowest confidence | 10% | 0.64197 | 17.43% |
| IID random routing | 10% | 0.61811 | 18.92% |
| Non-IID without routing | 0% | 0.40492 | 0.26% |
| Non-IID lowest confidence | 1% | 0.43087 | 0.53% |
| Non-IID lowest confidence | 5% | 0.46750 | 1.24% |
| Non-IID lowest confidence | 10% | 0.49570 | 2.08% |
| Non-IID random routing | 10% | 0.44415 | 2.01% |
| Always centralized | 100% | 0.65569 | 17.55% |

Confidence routing beats the same-budget random control in mean macro-F1 at
all three budgets, for both lanes. The non-IID result is a genuine measured
development improvement, accompanied by more false alerts. It is not a
solution to rare-attack detection or proof of acceptable operational risk.
The 1%, 5% and 10% settings are illustrative escalation quotas, not approved
false-alert tolerances or a chosen deployment operating point.

These controls rank the whole validation population by uncalibrated maximum
softmax probability. Equal scores keep original row order. This gives an exact
offline quota; it is not yet a fixed threshold for an arriving flow, and
sorting/queueing costs would matter in a real system. The diagnostic evaluates
both models everywhere to count mistakes, so a 10% escalation budget does not
establish a measured 90% compute saving. No edge latency is measured.

## Even perfect selection leaves rare attacks undetected

The table shows per-class recall percentages. Oracle columns are the maximum
recall each class can achieve by selecting between these two models' existing
predictions without a budget limit. The displayed oracle macro-F1 below is the
result of a particular correct-if-either rule, not a general macro-F1 optimum.

| Class | IID base | IID oracle | Non-IID base | Non-IID oracle |
| --- | ---: | ---: | ---: | ---: |
| Benign | 80.93 | 85.25 | 99.74 | 99.75 |
| DDoS | 84.58 | 87.54 | 98.75 | 98.87 |
| DoS | 67.31 | 74.49 | 20.81 | 70.88 |
| Recon | 72.76 | 77.40 | 19.89 | 73.85 |
| Web-based | 0.85 | 6.87 | 0.00 | 6.87 |
| Brute Force | 27.10 | 29.02 | 0.00 | 27.65 |
| Spoofing | 57.52 | 72.82 | 0.94 | 70.78 |
| Mirai | 99.42 | 99.65 | 98.80 | 99.58 |

The label-informed rule reaches mean macro-F1 0.67777 for IID and 0.73249 for
non-IID, requiring respectively 3.79% and 12.50% of rows to be escalated on
average. In particular, the non-IID oracle is **not** achievable within the
10% budget shown above; each of its three seeds needs more than 10% escalation.

For non-IID at the actual 10% confidence quota, Web-based recall is only 0.75%,
Brute Force 5.97% and Spoofing 17.35%. The difference between these recalls and
the oracle shows a routing problem, while the low rare-class oracle itself
shows a specialist problem. A new decision model must not be credited with
fixing mistakes that its downstream specialist cannot correct.

## Stage-one recommendation (now completed in stage two)

Do not start RONIN or install an API SDK yet. The earlier
[balanced-panel comparison](../full_data_extension/panel-validation-findings.md)
already contains candidates with much higher rare-class recall but excessive
false alerts. Their poor standalone performance does not tell us whether
selective routing can use them safely; their complementarity has to be measured
on these same paired rows, with actual harm counted.

The recommended next local stage was to compare the frozen small balanced-panel neural
model against the current centralized fallback. Include every existing seed,
retain the same panel and budgets, and preserve the standalone controls. This
is a second-stage specialist diagnostic, not a promotion of the earlier model,
a new training search or a claim that federated training improved.

That comparison is now complete; see [its findings and next-step protocol](specialist-findings.md).
The subsequent numerical-router experiment is also complete; see the
[results and stopping decision](router-findings.md).

Only if this supplies useful headroom should we fit a simple numerical gate
and compare one typed-decision model with it. Gate training requires out-of-fold
base/expert predictions or a separate duplicate-group-preserving development
protocol; it must not learn from final-test outcomes. Both gate types need the
same inference-available evidence and budget. Using the expert's output to
decide whether to call that expert would invalidate the claimed compute saving.
Sending flows to an external API also changes the data-sharing assumptions.

A later independent evaluation must be specified before final claims. The
old test cannot become untouched again by reshuffling it. If independent data
are unavailable, label this extension exploratory and keep the completed
study's conclusions intact.

## Verification and reproduction

The new implementation is [official39_routing_headroom.py](../../scripts/official39_routing_headroom.py).
It keeps the historical training and evaluation code unchanged. The loader
opens only the two named validation arrays, verifies their checksums and uses
bounded feature batches. Models are loaded from pinned archive members using
restricted weights-only loading. No archived code is executed or extracted.

All nine prediction scans exactly match the independent frozen evaluator on
the local CPU. Six also exactly match the archived CUDA confusion matrices;
three have a confusion-matrix L1 difference of two counts on the primary
panel. Those cross-runtime differences are disclosed in the results rather
than relaxing the historical final-test gate. These are local CPU feasibility
results, not a claimed bitwise CUDA replay.

The private full receipt contains joint true/base/expert counts and confusion
matrices. The checked-in summary contains aggregates and provenance only;
it does not contain flow rows, checkpoints or credentials.

Two separate local runs exactly reproduce the predictions, paired counts,
budgeted controls and summaries. The clean repository snapshot passes 343
tests, including 18 new routing tests. The environment passes `pip check`,
and the completed study's existing JSON provenance audit still passes. No
dependencies were added; unrelated uncommitted work was excluded from the
clean-snapshot verification and left untouched.

Run from the repository root, using fresh output paths:

```bash
.venv/bin/python -m unittest tests.test_official_routing_headroom -q
.venv/bin/python scripts/official39_routing_headroom.py \
  --archives /Users/haydn/Downloads \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --output outputs/official39-routing-headroom-reproduction
```

Change the archive root on another machine. The command needs the two existing
ablation archives, the packed validation files and the shared panel. It does
not need the final-test backup. The optional `--publish-summary` argument
exports aggregate JSON to a fresh destination and never overwrites a report.
