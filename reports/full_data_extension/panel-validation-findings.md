# Validation results reject the wider network upgrade

The subsequent [fixed alert-budget diagnostic](alert-budget-findings.md) is
complete. It quantifies the missed attacks incurred by suppressing false alerts;
it does not establish an independently assessed decision policy. The validation
findings and original next-step rationale below are retained.

The wider network without dropout fits its training panel better, but it does
not provide a useful validation upgrade. On the full natural-prevalence
validation split, its mean macro-F1 is 0.5577 and its benign false-alert rate is
39.52%. The smaller network with dropout reaches 0.5683 with fewer false alerts,
29.52%. The fixed boosted-tree reference reaches 0.6141, but still flags 28.17%
of benign validation rows as attacks.

Do not promote these panel-trained models or spend more cloud credits on the
same recipe. The training controls answered a useful question about fitting
capacity; validation now shows that stronger fitting has not solved the
detection trade-off. The next local diagnostic should examine rare-class score
separation and achievable recall under explicit false-alert constraints.

## Fixed comparison and safeguards

The [validation plan](panel-validation-plan.json) was frozen before opening
validation arrays. It preserves all four configurations from the
[width experiment](width-fit-findings.md): hidden layers 64/32 or 128/64, each
with dropout 0.2 or 0, and seeds 7, 17 and 27. Every neural model is evaluated at
epoch 300, not its best epoch. The single existing boosted-tree recipe is also
retained, without a search or internal early stopping.

All models fit the same balanced training panel: 10,152 rows, with 1,269 from
each class, selected without replacement from the 2M training cohort. The
original 39 features and scaler fitted on that full training cohort remain
unchanged. Each neural fit processes 3,045,600 examples over 6,000 updates.
The tree remains a different model family, with unmatched capacity and work.

Before any validation file was hashed or loaded, all 12 neural fits and the
fixed tree reproduced their historical training receipts exactly. The checks
include neural learning curves, final parameter hashes and work counters, and
the tree's training predictions. The runner then evaluated all 2,059,284
validation rows in batches of at most 8,192. Only those batches, not the full
feature matrix, were materialized in memory.

Predictions use the existing highest-score class rule. There is no threshold
tuning, calibration, new feature engineering, model promotion or test access.
Validation has supported earlier exploratory work; it is not a newly untouched
final test. The held-out test remains sealed.

## Training improvement did not become a validation improvement

Neural values are means across the same three training seeds. The tree is one
fixed seed-7 reference. SD is the sample standard deviation across neural seeds,
not a confidence interval or variation over independent panels.

| Model | Training macro-F1 | Validation macro-F1 | Validation SD | Benign false alerts |
|---|---:|---:|---:|---:|
| Small with dropout | 0.720529 | 0.568271 | 0.003973 | 29.52% |
| Wider with dropout | 0.756961 | 0.568565 | 0.000939 | 32.77% |
| Small without dropout | 0.758286 | 0.561019 | 0.009059 | 38.47% |
| Wider without dropout | 0.817464 | 0.557692 | 0.009878 | 39.52% |
| Fixed boosted trees | 0.944438 | 0.614121 | Not estimated | 28.17% |

With dropout, widening gains only 0.03 percentage points of validation macro-F1
on average while increasing benign false alerts by 3.25 points. That false-alert
increase occurs in every seed. Without dropout, widening reduces validation
macro-F1 in every seed, by 0.33 points on average, and increases mean benign false
alerts by 1.05 points. The substantial training gains do not justify a wider
production model under this panel recipe.

| Seed | Small with dropout | Wider with dropout | Small without dropout | Wider without dropout |
|---|---:|---:|---:|---:|
| 7 | 0.569013 | 0.569604 | 0.558005 | 0.550532 |
| 17 | 0.563979 | 0.568312 | 0.553851 | 0.553582 |
| 27 | 0.571819 | 0.567779 | 0.571201 | 0.568961 |

Training and validation have very different class proportions, so the aggregate
F1 gap is not solely an overfitting measurement: precision depends on prevalence.
There are also substantial class-specific recall gaps. For the wider network
without dropout, Web-based recall falls from 76.15% on training rows to 47.46%
on validation rows, and Brute Force falls from 74.36% to 52.67%. These gaps are
consistent with overfitting or insufficiently representative training examples;
they are not explained merely by changing the class proportions in an F1
calculation. This comparison does not isolate either cause as the only one.

## Recall for every class

Values are validation recall percentages. Neural columns average three seeds;
the tree column is the single fixed fit. Support is identical for every model.

| Class | Support | Small dropout | Wider dropout | Small no dropout | Wider no dropout | Tree |
|---|---:|---:|---:|---:|---:|---:|
| Benign | 109,482 | 70.48 | 67.23 | 61.53 | 60.48 | 71.83 |
| DDoS | 1,213,890 | 63.05 | 64.86 | 71.30 | 70.80 | 72.25 |
| DoS | 373,093 | 92.49 | 91.61 | 85.68 | 86.06 | 83.97 |
| Recon | 68,008 | 53.59 | 53.95 | 49.24 | 50.63 | 56.81 |
| Web-based | 2,475 | 51.07 | 49.39 | 46.49 | 47.46 | 58.95 |
| Brute Force | 1,284 | 54.78 | 58.15 | 58.41 | 52.67 | 61.60 |
| Spoofing | 44,949 | 59.05 | 63.08 | 66.06 | 64.54 | 83.59 |
| Mirai | 246,103 | 99.42 | 99.50 | 99.33 | 99.43 | 99.62 |

## Precision explains the alert burden

Values are validation precision percentages. Low precision means predictions
of that attack category are usually the wrong category. Some are other attacks,
not benign traffic; this is distinct from the benign false-alert rate above.

| Class | Small dropout | Wider dropout | Small no dropout | Wider no dropout | Tree |
|---|---:|---:|---:|---:|---:|
| Benign | 81.77 | 82.91 | 82.23 | 80.86 | 86.66 |
| DDoS | 96.62 | 96.28 | 94.37 | 94.51 | 93.69 |
| DoS | 43.68 | 44.67 | 48.30 | 48.03 | 48.32 |
| Recon | 75.27 | 74.17 | 70.85 | 68.36 | 75.08 |
| Web-based | 3.80 | 4.17 | 3.72 | 3.81 | 5.54 |
| Brute Force | 4.21 | 3.76 | 3.02 | 3.72 | 4.64 |
| Spoofing | 76.86 | 71.21 | 65.80 | 63.21 | 91.01 |
| Mirai | 98.84 | 98.65 | 98.50 | 98.42 | 98.86 |

Web-based and Brute Force each occupied 12.5% of the training panel. In
validation they occupy only 0.1202% and 0.0624%, respectively. The panel models
have learned to predict these rare categories much more often, but many of
those predictions are incorrect.

For the wider network without dropout, a typical mean across seeds is about
43,270 benign rows flagged as attacks out of 109,482 benign rows. About 11,559
benign rows are called Web-based, 8,904 Brute Force, 11,970 Recon and 10,826
Spoofing. These are descriptive mean counts, not one synthetic model's exact
confusion matrix.

The rare-class false predictions also include large numbers of Recon and
Spoofing attacks. Thus the issue is broader than one benign-versus-attack
threshold: the model must distinguish overlapping attack categories as well.
The tree has better rare-class recall and precision than the neural panel
models overall, but its rare precision remains below 6%, and its 28.17% benign
false-alert rate is not an operational success.

## How this relates to the earlier cloud runs

The [full-cohort scaling comparison](scaling-findings-and-controlled-noniid.md)
reported centralized-light validation macro-F1 0.65572 and 17.55% mean benign
false alerts after 20 passes over 2M training examples. Its Web-based recall
was only about 6–8%, and Brute Force about 26–29%. The panel models improve
those rare recalls but worsen overall macro-F1 and benign false alerts.

That is a trade-off, not a replacement win. The historical runs use different
training exposure, effective class weights, budgets and—in some comparisons—
architecture settings. They provide context, not a width-only causal control
for the panel experiments. No claim about improved federated training follows
from these centrally pooled panel fits.

The fitting series remains useful: removing dropout, training longer and
widening all showed that the network could fit more of the balanced panel.
The new validation results set a limit on that conclusion. Simply repeating
the small panel longer or making the model larger is not supported as the next
cloud experiment.

## Next local decision layer diagnostic

Keep the full-cohort study as the main reference and retain the small panel
network with dropout as the neural control for further diagnosis. Neither is
automatically promoted for deployment. Do not restart the cloud or implement
another architecture yet.

Use the saved, frozen neural weights to inspect rare-class score distributions
and the sources of false predictions. Before running the next comparison,
predeclare illustrative benign false-alert budgets, such as 0.1%, 1% and 5%,
and report the corresponding rare-class recall and precision for all three
seeds. These budgets would be diagnostic operating points, not user-approved
deployment requirements. The question is whether useful rare-attack detection
survives a constraint on false alerts, rather than whether recall can be raised
by labelling many benign flows as attacks.

Any thresholds selected using validation must be labelled exploratory, not
independent performance estimates. A subsequent threshold-selection and
assessment protocol needs separate, duplicate-group-preserving partitions;
the final test must not choose thresholds. Do not claim that calibration or
a confidence gate will fix errors without measuring the resulting missed
attacks and attack-category confusions.

If useful separation is absent, design an exposure/objective comparison that
retains much more of the majority-class diversity, with fixed models and
matched work counters, before committing cloud time. Sampling only 1,269 benign
rows is a plausible contributor to these errors, not yet a proven sole cause.
This diagnostic should decide what evidence is missing; it should not start an
open-ended threshold or architecture search.

## Verification and reproducibility

Implementation: `src/eval/official_panel_validation.py`. The training helpers
now optionally return their final fitted objects; their default receipt-only
interfaces and all production defaults remain unchanged. Existing comments are
retained and the validation gates are explained inline.

All 154 local tests passed, including ten new tests for exact training replay,
bounded batches, checksums, input counts, no test access, saved weights, no
overwrites and failure before validation when either model family does not
reproduce its reference. Existing dependency warnings were non-fatal.

An independent second complete run matched all 13 training/evaluation records,
summary metrics, confusion matrices and saved neural checkpoint checksums
exactly, excluding elapsed time. All 12 saved neural checkpoints were separately
loaded with restricted weights-only loading and verified against their final
parameter hashes. Source and plan hashes remained unchanged.

Canonical results are in `outputs/official39-panel-validation-v1/receipt.json`.
Its SHA-256 is
`b5767472daeb19218e2a704c1c3d6edf54a129c96ac8da0a88f8d20698c6df99`.
The frozen plan SHA-256 is
`91bb9f0a5825c2edc4215baee181bc686a1cf88664dd679554d281e9b48bb979`.
The directory also contains all 12 final neural checkpoints. They are diagnostic
artifacts, not production checkpoints, and remain outside version control.
The tree recipe, training prediction hash and validation prediction hash are
recorded in the receipt; its fitted object is not persisted.

The canonical run recorded 49.07 seconds across training fit timers and 25.99
seconds across validation evaluation timers on the local CPU. These exclude
setup, hashing/loading and receipt/checkpoint writes, and are not edge or cloud
benchmarks. No new dependency or cloud session was required.

Reproduce into a new output directory:

```bash
.venv/bin/python -m src.eval.official_panel_validation \
  --data-root outputs/official39-packed-2m-v1 \
  --reference-path outputs/official39-width-fit-v1.json \
  --tree-reference-path outputs/official39-training-fit-v2.json \
  --plan-path reports/full_data_extension/panel-validation-plan.json \
  --output outputs/official39-panel-validation-replay
```

These remain within-collection findings for the official-39 extension. They
are not measurements on unseen capture sessions, identifiable devices or
physical edge hardware.
