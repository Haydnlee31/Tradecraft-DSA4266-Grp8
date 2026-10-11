# Equal class weighting for rare attack classification

Equal class weighting increases rare-attack recall, but it is not a better
classifier at the fixed argmax decision rule. Audit macro-F1 falls from
0.698757 to 0.636635 and benign false alerts rise from 14.50% to 23.19%.
Rare-class precision falls to about 6%, so rare F1 also deteriorates. Every
continuation check fails at every seed. Keep the square-root-weighted tree
as the reference; do not replace it with this raw equal-weight classifier.

This experiment changes only the loss weights of the existing jointly masked
boosted tree. Each of the eight classes receives equal total fitting weight,
rather than weight proportional to the square root of its full fitting-partition
frequency. The result distinguishes detecting more rare examples from making
useful rare-category predictions.

The preceding [matched neural comparison](matched-neural-findings.md) found
that the tree learned more rare examples than the small neural network, but
still missed most Web-based and Brute Force attacks. Under the previous
objective those two classes received only 1.62% and 1.15% of total class-weight
mass. The treatment gives each 12.5%, while leaving the model and data fixed.

## Results on the common development audit

Values are means across seeds 7, 17 and 27. False alerts mean the fraction of
genuinely benign flows classified as any attack, not the fraction of alerts
that are false. The same 472 Web-based and 253 Brute Force audit examples
occur in every seed.

| Objective | Macro-F1 | Benign false alerts | Web recall | Brute Force recall |
| --- | ---: | ---: | ---: | ---: |
| Square-root baseline | 0.698757 | 14.50% | 21.82% | 34.52% |
| Equal class weights | 0.636635 | 23.19% | 58.76% | 53.89% |

The mean macro-F1 change is -6.21 percentage points and the false-alert
change is +8.68 points. The deterioration holds in every seed, not just in
the average. Macro-F1 seed SD is 0.003727 for the baseline and 0.000854 for
equal weights; reproducibly worse results are not evidence of a better model.

| Objective | Web precision | Web recall | Web F1 | Brute Force precision | Brute Force recall | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Square-root baseline | 24.90% | 21.82% | 23.25% | 33.87% | 34.52% | 34.16% |
| Equal class weights | 6.63% | 58.76% | 11.91% | 5.91% | 53.89% | 10.65% |

About 93% of Web predictions and 94% of Brute Force predictions are wrong
under equal weighting. In seed 7, for example, Web predictions grow from 425
to 4,208: correct predictions rise from 105 to 275, but incorrect predictions
rise from 320 to 3,933. Brute Force predictions grow from 250 to 2,259, while
correct predictions rise only from 85 to 139. This explains why the recall
gain is accompanied by lower F1.

Most new rare predictions come from the already difficult Benign/Recon
boundaries. Averaged across the three seeds, the equal-weight classifier calls
about 1,501 Benign and 1,916 Recon examples Web-based, and 1,043 Benign and
918 Recon examples Brute Force. These are average confusion counts on the
same audit rows, not additional independent samples.

## What fitting and audit errors show

All entries are per-class recall percentages. The common classes are included
so that rare gains cannot hide deterioration elsewhere.

| Class | Baseline fitting | Baseline audit | Equal-weight fitting | Equal-weight audit |
| --- | ---: | ---: | ---: | ---: |
| Benign | 85.54 | 85.50 | 76.84 | 76.81 |
| DDoS | 85.19 | 84.39 | 74.07 | 73.53 |
| DoS | 74.41 | 73.98 | 89.19 | 88.83 |
| Recon | 77.56 | 76.46 | 64.91 | 63.29 |
| Web-based | 30.32 | 21.82 | 77.65 | 58.76 |
| Brute Force | 49.12 | 34.52 | 87.72 | 53.89 |
| Spoofing | 85.10 | 84.51 | 85.11 | 84.90 |
| Mirai | 99.90 | 99.80 | 99.93 | 99.82 |

Reweighting clearly moves the fitted decision boundaries: the architecture
can recover substantially more rare examples under a different objective.
However, it overpredicts rare classes even on fitting rows, where Web and
Brute Force precision are only 10.21% and 11.04%. The false-positive problem
is therefore not solely an audit generalization failure.

Generalization also matters. Equal-weight recall falls by 18.89 points for
Web and 33.83 points for Brute Force between fitting and audit. At the shared
full fitting-partition class prior, macro-F1 goes from 0.654808 to 0.637108,
a 1.77-point drop. The baseline goes from 0.730860 to 0.699036, a 3.18-point
drop. The treatment's smaller aggregate gap does not make it preferable:
its absolute performance is lower and its rare recall gaps are larger.

These observations support an objective/decision trade-off, not a claim that
all remaining errors are caused by class imbalance, label noise or insufficient
model size. Nor do they show that either model's output probabilities are
calibrated estimates of real-world attack risk.

## Next step and stopping rule

Do not launch a larger model or cloud run on the strength of recall alone.
Keep both fitted trees frozen. The next bounded local question is whether a
small, automatic score-adjustment layer can retain useful rare detection
while restoring precision and controlling benign false alerts. This tests a
decision rule, not another training architecture or a human-review queue.

Any score adjustment must be designed in advance and fitted or selected only
on the development calibration partition, after checking that the joint
feature mask introduces no fitting/calibration overlap. Use a small fixed
candidate set and a baseline-relative false-alert constraint; then freeze the
choice before scoring this audit. The calibration partition is development
data, not a new independent confirmation set. No such adjustment was fitted
or evaluated in this stage.

If no candidate improves rare F1 under the alert constraint, retain the
baseline and report the recall/precision trade-off instead of expanding an
unbounded search. Any surviving improvement would still need assessment on a
genuinely new cohort before generalization claims. The exposed original final
test must not become the selector for this extension.

## Fixed comparison

Both conditions use the same 270,326 ordered fitting rows, the old train-fitted
scaler, and the same input mask removing `Number` and `Tot sum`. The same
408,027-row development audit remains at its natural class prevalence; it is
not balanced to make rare precision look better. The earlier exact feature
overlap screen and split membership remain unchanged.

Each new model has 100 boosting stages, eight trees per stage, at most 15
leaves per tree, learning rate 0.1, minimum leaf size 30, and L2 penalty 1.
Seeds are 7, 17 and 27, with no early stopping or best-seed selection. The
three new models are saved before any model audit scoring. The earlier
square-root-weighted models must reproduce their fitting and audit
predictions exactly. An independent complete replay precedes publication.

With N sampled rows and m[c] sampled rows in class c, the new per-row weight
is N/(8*m[c]). Every class therefore contributes total weight N/8 and the
overall mean weight stays one. This avoids changing the overall loss scale
relative to regularization. Full fitting-partition counts still verify the
sample, but do not enter the equal-class weight. No resampling or second
class-weight multiplier is added.

## How the outcome is judged

At every seed, the continuation screen requires higher audit macro-F1, higher
F1 for each rare category, and no increase in benign false alerts relative
to the square-root-weighted tree. This is a descriptive development screen,
not deployment authorization. Increased recall alone is not a pass.

All eight class precision, recall and F1 values, original confusion counts,
and benign false alerts are retained. Fitting-to-audit gaps also use the same
full fitting-partition class prior to avoid confusing prevalence differences
with generalization. Paired correctness counts describe model differences;
their label-informed oracle is not a usable prediction rule.

## Scope and reproduction

This is an automatic eight-class classifier, not a human-review workflow or
new federated tree training. No threshold search, calibration-data use, API
call, cloud compute or neural retraining is part of this comparison. The
original final test is not reopened. Repeated use of this development audit
does not supply independent generalization evidence.

The [fixed plan](balanced-tree-plan.json) binds the model settings and previous
evidence. Existing source files, comments, classifiers and data remain intact.
No additional dependency is needed. A fresh output directory is required:

```sh
.venv/bin/python -u scripts/official39_balanced_tree.py run \
  --cache outputs/official39-routing-gate-v2 \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --trees outputs/official39-feature-ablation-v1 \
  --output outputs/official39-balanced-tree-v1
```

Fitted models stay in ignored outputs and are loaded only as checksum-bound,
locally produced artifacts. Do not load arbitrary downloaded joblib objects.
The public aggregate evidence can be audited without opening private arrays
or deserializing a model.

## Verification

The [published aggregate results](balanced-tree-results.json) were released
only after a separate execution reproduced all three serialized fitted
models, fitting and audit prediction hashes, class metrics, paired counts and
non-timing records exactly. Together the two runs fitted six models and
4,800 trees. Each execution also reproduced all three historical baseline
trees' fitting and audit predictions exactly. This is computational replay,
not confirmation on independent data.

All 14 focused tests pass, and the full suite passes 499 tests in a clean
snapshot containing the preceding commit and this extension, excluding
unrelated working-tree edits. The focused tests check the class-weight arithmetic, no double
correction for the capped sample, a byte-exact bridge to the historical
weighting implementation, source-array preservation, masking, fixed training
work, safe owned-model reload and rejection of changed evidence. The success
screen explicitly rejects higher recall with worse rare F1 or false alerts.
`pip check` reports no broken requirements. No previous source implementation,
private source array or original model artifact was changed.

```sh
.venv/bin/python scripts/official39_balanced_tree.py audit \
  --record reports/risk_aware_routing/balanced-tree-results.json
```
