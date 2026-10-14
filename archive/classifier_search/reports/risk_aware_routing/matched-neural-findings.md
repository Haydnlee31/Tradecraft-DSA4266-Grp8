# Neural and tree classification on the same fitting data

The tree retains its advantage when the small neural model receives the same
fitting rows, feature information and class-weight objective. Mean audit
macro-F1 is 0.698757 for the tree versus 0.656039 for the neural model. The
neural model's rare-class recall is poor even on its fitting rows, while the
tree learns more rare examples but loses some of that performance on the audit.

Do not add a router between this pair: the neural model corrects too few of
the tree's rare-class mistakes to provide much headroom. The next useful
local control is a rare-class-focused loss objective on the existing tree
recipe, with precision and benign false alerts assessed alongside recall.
The original study's completed final test stays closed.

## Matched data results

Values are means across seeds 7, 17 and 27 on the same 408,027-row development
audit. Both models have `Number` and `Tot sum` removed. The tree model here is
the existing joint-removal control, not a newly tuned tree.

| Model | Macro-F1 | Benign false alerts | Web recall | Brute Force recall |
| --- | ---: | ---: | ---: | ---: |
| Matched neural model | 0.656039 | 17.06% | 6.78% | 26.09% |
| Joint-removal tree | 0.698757 | 14.50% | 21.82% | 34.52% |

The tree is ahead by 4.27 macro-F1 percentage points and has 2.56 points fewer
benign false alerts. Both directions hold at every seed. Macro-F1 seed SD is
0.003004 for the neural model and 0.003727 for the tree; this is fitting
variation on the same rows, not uncertainty across independent datasets.

Rare-class precision remains important. Higher recall does not make every
tree prediction reliable:

| Model | Web precision | Web recall | Web F1 | Brute Force precision | Brute Force recall | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched neural | 29.07% | 6.78% | 10.70% | 44.31% | 26.09% | 32.65% |
| Joint-removal tree | 24.90% | 21.82% | 23.25% | 33.87% | 34.52% | 34.16% |

The earlier frozen full-feature central reference scored macro-F1 0.655632
with 17.39% false alerts. That number is context only: its training population,
feature information and budget differ. The matched comparison above removes
the data/objective mismatch between the new neural control and joint-masked
tree, but does not prove that all neural architectures are inferior.

## What the errors diagnose

The following are per-class recall percentages, which remain comparable even
when the capped fitting sample and uncapped audit have different class mixes.

| Class | Neural fitting | Neural audit | Tree fitting | Tree audit |
| --- | ---: | ---: | ---: | ---: |
| Benign | 82.68 | 82.94 | 85.54 | 85.50 |
| DDoS | 85.00 | 84.56 | 85.19 | 84.39 |
| DoS | 70.25 | 70.38 | 74.41 | 73.98 |
| Recon | 74.57 | 73.97 | 77.56 | 76.46 |
| Web-based | 7.12 | 6.78 | 30.32 | 21.82 |
| Brute Force | 28.03 | 26.09 | 49.12 | 34.52 |
| Spoofing | 73.10 | 72.46 | 85.10 | 84.51 |
| Mirai | 99.57 | 99.60 | 99.90 | 99.80 |

After standardizing both sets to the same full fitting-partition class prior,
neural macro-F1 goes from 0.660107 on fitting rows to 0.655976 on audit rows,
a 0.41-point drop. Tree macro-F1 goes from 0.730860 to 0.699036, a 3.18-point
drop. The tree's rare-class recall gaps are larger still: 8.49 points for Web
and 14.60 points for Brute Force. It both learns more and generalizes less
perfectly; the larger fitting score should not be presented as a deployment gain.

The neural result is more consistent with inadequate rare-class fitting under
this recipe than with a large fitting-to-audit collapse. It does not identify
capacity as the sole cause. Loss priorities, optimization and overlapping
feature distributions remain plausible explanations. Two neural seeds still
improve between fitting observations at epochs 40 and 60, while the third
declines slightly; convergence is not established.

The current weighting gives Web-based examples only 1.62% and Brute Force
examples 1.15% of total class-weight mass, versus 35.56% for DDoS. These are
loss-coefficient shares, not measured gradient contributions. Square-root
balancing helps relative to the original prevalence but still gives the two
rare categories little influence. This provides a concrete reason to test
the loss priority before increasing model size.

Most rare errors still go to Benign or Recon. On the audit, the neural model
calls 31.21% of Web examples Benign and 50.00% Recon; for Brute Force, those
rates are 37.02% and 30.83%. The corresponding tree rates are 23.09%/49.44%
for Web and 31.75%/26.09% for Brute Force.

## Why another router is not the next step

Even a perfect, label-informed selector between these two classifiers would
reach only 22.67% Web recall and 35.97% Brute Force recall. That is just 0.85
and 1.45 percentage points above using the tree alone. The neural model
correctly classifies only 3, 6 and 3 additional Web examples across the three
seeds, and 6, 4 and 1 additional Brute Force examples. These are repeated
predictions on the same rows, not new independent examples.

This bound applies to choosing between their existing hard predictions. It
does not bound a new model that learns different decision boundaries or can
predict a third category. Nevertheless, adding routing complexity to this
particular pair cannot produce a large rare-class recall gain.

## Next bounded local experiment

Keep the joint-removal tree as the control and change just its class-weight
objective to equal total weight per class. On the same capped sample, this
means a per-row coefficient proportional to 1/m[c], normalized to mean one;
each class would contribute 12.5% of weight mass. Keep the same fitting rows,
feature mask, three seeds and 100-stage tree recipe. Do not introduce a
weighted sampler as well.

Compare rare precision, recall and F1, overall macro-F1, benign false alerts,
and fitting-to-audit gaps. Increased rare recall alone is not success. If it
comes from labeling many Benign or Recon flows as rare attacks, do not promote
the classifier or disguise the trade-off with accuracy. This control tests
whether the existing architecture can learn more useful rare-class boundaries;
it is not a promise of improvement and has not been launched in this stage.

A successful candidate would still need a predeclared assessment on a genuinely
new cohort before generalization or deployment claims. No larger cloud GPU,
paid decision API or human-review workflow is needed for this local control.

## Fixed comparison

The preceding [feature-removal experiment](feature-ablation-findings.md) found
that the tree improvement survives removing `Number` and `Tot sum`. Its
joint-removal model is the matched control here. All three neural models are
newly initialized, not fine-tuned from a checkpoint exposed to the audit.

| Factor | Neural control | Tree control |
| --- | --- | --- |
| Fitting rows | Same ordered 270,326 rows | Same ordered 270,326 rows |
| Features | Original train-scaled values; Number and Tot sum zeroed | Identical values and mask |
| Loss contribution | Same global mean-one weight per row | Same global mean-one weight per row |
| Seeds | 7, 17, 27 | 7, 17, 27 |
| Architecture | 39 → 64 → 32 → 8, LayerNorm, ReLU, dropout 0.2 | Previously frozen 800-tree classifier |
| Training budget | 60 epochs, 31,680 Adam updates per seed | 100 boosting stages, eight trees each |
| Model selection | Final epoch only | Fixed final boosting stage |
| Evaluation | Same 408,027 audit rows | Same 408,027 audit rows |

Matching data and the predictive loss objective does not match compute,
optimization or regularization. The neural model uses Adam with learning rate
0.001 and weight decay 0.0001, batch size 512. The tree retains its earlier
recipe and L2 penalty. This is a bounded control, not an optimized contest
between model families.

With full fitting-partition class counts n and capped sample class counts m,
each sampled example of class c receives weight sqrt(n[c])/m[c], normalized
to mean one across fitting rows. The neural loss averages weighted per-row
cross-entropy using the batch row count as denominator. Dividing by each
batch's random weight sum or adding weighted sampling would change this
objective. No audit or calibration label determines a training weight.

Sixty epochs provide 16,219,560 row exposures per seed and 48,658,680 overall.
Training-only observations at epochs 1, 5, 20, 40 and 60 expose the learning
trajectory; they do not select a checkpoint. Every model is saved before any
new audit scoring. The complete run is then independently replayed before
publication.

## How fitting and audit errors are compared

The fitting sample caps common classes, while the audit retains the original
development class distribution. Comparing their raw F1 scores directly could
mistake a prevalence difference for a generalization gap.

Both sets therefore retain their original confusion matrices and observed
metrics, with a second diagnostic that rescales each confusion row to the
same full fitting-partition class prior. Per-class recall stays unchanged;
precision and F1 become comparable at that fixed prior. These reweighted
metrics are diagnostics, not additional observed examples or new test scores.

Poor rare-class recall on both fitting and audit rows would suggest that this
recipe has not learned the desired boundaries; high fitting recall followed
by low audit recall would instead suggest a generalization problem. Either
pattern can reflect overlapping feature distributions or ambiguous labels,
so neither alone proves that a bigger model will solve the problem.

Paired correctness counts show how often one model corrects the other's
mistakes. Their label-informed union is only an upper bound on a perfect
selector, not an implementable classifier. A useful automatic selector would
have to recognize those cases without seeing the true label.

## Scope and reproduction

These are centrally fitted eight-class development classifiers, not new
federated training results. The exact feature-overlap screen from the previous
stage is reused, with no row reassignment. Calibration remains unused. Three
seeds on the same rows are not three independent dataset confirmations.

The [fixed plan](matched-neural-plan.json) binds the historical evidence and
recipe. No previous training implementation, checkpoint or completed test is
modified. There is no human-review step, cloud compute or new dependency.

```sh
.venv/bin/python -u scripts/official39_matched_neural.py run \
  --cache outputs/official39-routing-gate-v2 \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --trees outputs/official39-feature-ablation-v1 \
  --output outputs/official39-matched-neural-v1
```

Use a different fresh output directory for the independent replay. Private
weights and row indices stay in ignored outputs. Public aggregate verification
does not need the private arrays or model objects.

## Verification

The [aggregate evidence](matched-neural-results.json) was published after the
independent execution reproduced all three saved neural weight files, training
curves, prediction hashes, paired counts and non-timing results exactly. The
two executions together fitted six models, processing 97,317,360 examples in
190,080 optimizer updates. Each run also reproduced the saved joint-removal
trees' fitting and audit prediction hashes, rather than trusting the report
table alone.

All 14 focused tests pass, and the full suite passes 485 tests in a clean
snapshot containing the previous commit plus this extension, excluding
unrelated working-tree edits. The focused tests cover the fixed loss denominator, matching
class-weight objective, exact row exposure, reproducible fitting, tensor-only
reload, unchanged input arrays, masking, prevalence standardization against an
independent calculation and rejection of altered evidence. `pip check` reports
no broken requirements. Private source arrays and prior model artifacts retain
their original checksums.

```sh
.venv/bin/python scripts/official39_matched_neural.py audit \
  --record reports/risk_aware_routing/matched-neural-results.json
```
