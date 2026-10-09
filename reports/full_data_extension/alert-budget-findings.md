# False alert budgets reveal the remaining detection trade-off

The fixed alert gate reduces benign false alerts, but it does not solve rare
attack detection. For the predeclared primary model, the small network with
dropout, a 1% empirical false-alert budget retains only 11.49% Web-based recall
and 34.79% Brute Force recall. These are useful diagnostic results, not a
deployable policy or a reason to scale the same training recipe to cloud.

The next bounded experiment should test whether exposing the small model to
more diverse majority-class training examples improves this trade-off. It
should preserve rare-example exposure, architecture and update counts, rather
than simultaneously changing the model, loss and decision rule.

## One fixed rule and three budgets

The [plan](alert-budget-plan.json) was frozen before scoring. It uses all 12
saved neural models from the [validation comparison](panel-validation-findings.md):
four width/dropout configurations and seeds 7, 17 and 27. The small network with
dropout is the primary control; the other three configurations are sensitivity
checks. There is no new training, tree refit or threshold-family search.

For each row, the rule compares the highest attack logit with the benign logit.
A logit is the model's unnormalized class score. The selected attack category
is retained only if the difference exceeds a nonnegative threshold; otherwise
the prediction becomes Benign. The rule cannot change one attack category into
another or recover an attack category that the original classifier missed.
It also does not provide an uncertainty or human-review outcome: a suppressed
alert is explicitly counted as a benign prediction.

Before any threshold was selected, every model reproduced its original argmax
confusion matrix and validation cross-entropy exactly. A zero threshold also
reproduced those predictions from the saved scores. Thresholds were then set
using only the 109,482 benign validation scores, allowing at most 109, 1,094 or
5,474 benign alerts for budgets of 0.1%, 1% or 5%. Strict greater-than comparisons
handle ties conservatively; ties are never broken using labels or row order.

**These thresholds are fitted and measured on the same validation split.**
The reported false-alert caps hold by construction on these rows. They are not
independent estimates or guarantees for new traffic. No final test, training
array, production model or existing decision-engine criterion was accessed or
changed by this diagnostic.

## Primary model results

Values are means over seeds 7, 17 and 27, on the full 2,059,284-row validation
split. Precision and recall are percentages. The no-gate row is the unchanged
argmax reference.

| Gate budget | Observed benign false alerts | Macro-F1 | Web recall | Web precision | Brute Force recall | Brute Force precision |
|---|---:|---:|---:|---:|---:|---:|
| No gate | 29.51566% | 0.568271 | 51.07 | 3.80 | 54.78 | 4.21 |
| 0.1% | 0.09956% | 0.495187 | 3.26 | 47.06 | 25.34 | 38.86 |
| 1% | 0.99925% | 0.534814 | 11.49 | 16.66 | 34.79 | 17.04 |
| 5% | 4.99991% | 0.553155 | 23.23 | 7.37 | 42.68 | 8.93 |

At 1%, only around one in six predictions of either rare category has the right
category. Some incorrect predictions are other attacks, not benign flows, so
this should not be confused with the separate benign false-alert rate.
At 0.1%, rare-category precision improves further, but the rule misses almost
all Web-based attacks. Raising the budget recovers more attacks while admitting
many more incorrect rare-category predictions.

At 1%, the mean Web-based error breakdown is 11.49% correct category, 83.12%
predicted benign and 5.39% assigned another attack category. For Brute Force it
is 34.79% correct, 62.64% benign and 2.57% another attack category. A low false-alert
rate can therefore coexist with many genuinely malicious flows being called benign.

## Recall for every class

These are mean validation recall percentages for the same primary model.

| Class | No gate | 0.1% budget | 1% budget | 5% budget |
|---|---:|---:|---:|---:|
| Benign | 70.48 | 99.90 | 99.00 | 95.00 |
| DDoS | 63.05 | 63.03 | 63.04 | 63.05 |
| DoS | 92.49 | 92.47 | 92.49 | 92.49 |
| Recon | 53.59 | 17.85 | 21.15 | 31.41 |
| Web-based | 51.07 | 3.26 | 11.49 | 23.23 |
| Brute Force | 54.78 | 25.34 | 34.79 | 42.68 |
| Spoofing | 59.05 | 14.89 | 34.55 | 45.91 |
| Mirai | 99.42 | 99.38 | 99.41 | 99.42 |

The gate preserves most existing DDoS, DoS and Mirai category detections while
suppressing many Recon, Spoofing and rare-class detections. It does not correct
existing DDoS/DoS category confusions. This uneven effect is why a single aggregate
score or an overall attack-detection rate would be insufficient for the eight-class
project.

For completeness, mean precision at the 1% operating point is 56.66% Benign,
96.62% DDoS, 43.68% DoS, 95.04% Recon, 16.66% Web-based, 17.04% Brute Force,
91.38% Spoofing and 98.85% Mirai. The relatively low benign precision reflects
the attack examples now being suppressed into the benign class.

## Sensitivity across the four configurations

Every row below uses the same 1% empirical cap, not a chosen best threshold.
Each entry averages all three seeds.

| Configuration | Macro-F1 | Web recall | Web precision | Brute Force recall | Brute Force precision |
|---|---:|---:|---:|---:|---:|
| Small with dropout | 0.534814 | 11.49 | 16.66 | 34.79 | 17.04 |
| Wider with dropout | 0.541399 | 12.86 | 18.89 | 34.14 | 19.62 |
| Small without dropout | 0.530023 | 11.15 | 13.50 | 30.22 | 15.11 |
| Wider without dropout | 0.515732 | 10.65 | 12.29 | 29.41 | 21.97 |

Widening with dropout gives a modest macro-F1 improvement at this operating
point, but does not resolve the low Web-based recall. Removing dropout does
not provide a consistent solution. Across all configurations, Web recall is
only about 1.1–4.5% at the 0.1% budget and 21.8–23.2% at the 5% budget. This is
a limitation of these frozen models under this particular gate, not proof that
every possible decision rule or architecture must fail.

[Machine-readable counts](alert-budget-results.json) include all 12 original
argmax confusion matrices and all 36 gated matrices, thresholds, budgets,
checkpoint identities and provenance hashes. Every class's precision, recall
and F1 can be recomputed from those counts; there is no hidden best-seed or
best-budget selection. Raw score arrays remain outside Git.

## What the score distributions suggest

For the small network with dropout, median Web-based softmax scores on true
Web-based rows range from 0.305 to 0.330 across seeds. The 99th percentile on
benign rows ranges from 0.509 to 0.545. For Brute Force, corresponding ranges
are 0.340–0.385 and 0.509–0.638. These summaries show substantial overlap
between attack scores and the benign upper tail. They are descriptive score
summaries, not calibrated probabilities of an attack or proof of irreducible
feature overlap.

The gate is therefore not simply removing a set of obvious low-confidence
benign mistakes while preserving every attack. It removes many true attacks
too. The result supports studying how training examples shape these boundaries,
not claiming that confidence filtering has repaired them.

## Next bounded training control

Keep the current cloud machine stopped. Design one local **majority-diversity
control** using the small 64/32 network with dropout 0.2 and seeds 7, 17 and 27.
The current panel repeatedly exposes it to only 1,269 examples of each class,
including Benign, despite a much larger available majority pool.

For the control, retain the original panel. For the treatment, keep the exact
Web-based and Brute Force training rows and their exposure, but draw the other
six classes from their larger existing 2M training pools. Preserve the batch
class sequence, initial seed, train-fitted scaler, loss, optimizer and total
example/update counts. Record unique rows seen and class-specific exposure so
the change is attributable to example diversity at matched work, not a hidden
budget increase or a different rare-class objective.

This tests a hypothesis: a narrow benign and competing-attack sample may help
explain why fitting the small balanced panel transfers poorly. It does not
assume that more majority diversity will solve rare-class detection. Freeze
the protocol before running and require an exact control replay. Do not add
new thresholds, larger widths or a new loss to the same experiment.

Any subsequent operational threshold assessment must separate threshold
selection from assessment with duplicate-group-preserving partitions. Existing
validation reuse must remain explicit, and the final test must not choose
thresholds. No operating point in this report is authorized for deployment.

## Verification and reproduction

Implementation is `src/eval/official_alert_budget.py`, reusing the existing
validation loader, confusion metrics and error-breakdown helper. Model files
are loaded with weights-only loading after checksum checks. All shared source
hashes must match the frozen validation receipt. Existing comments and production
decision rules remain intact; no dependency was added.

Nine new tests cover ties, zero-alert budgets, invalid inputs, monotonic alert
suppression, raw argmax parity, inference-only boundaries, corrupt checkpoints,
protocol restrictions and failure before any threshold selection when a bridge
fails. All 163 local workspace tests passed before publishing the findings.

An independent complete repeat matched all 12 argmax bridges, all 36 thresholds
and confusion matrices, score quantiles and saved-array hashes exactly, excluding
elapsed time. No model was trained or promoted; no training or test arrays were
opened. The run used local CPU inference only.

Canonical receipt: `outputs/official39-alert-budget-v1/receipt.json`, SHA-256
`ea1a7fec35b7923aca3535f7d488fd80785ca9f8e76fc595fccee22616047bc8`.
Frozen plan SHA-256:
`aff97dc85ebaeb2aeb87d0189cc9c6a3367492125ad6b0178614142b0ff234b5`.
The raw receipt contains every class metric and error decomposition, plus score
quantiles and artifact hashes. Saved scores and model files remain local and
ignored by Git.

Reproduce into a new output directory:

```bash
.venv/bin/python -m src.eval.official_alert_budget \
  --data-root outputs/official39-packed-2m-v1 \
  --reference-root outputs/official39-panel-validation-v1 \
  --plan-path reports/full_data_extension/alert-budget-plan.json \
  --output outputs/official39-alert-budget-replay
```

The experiment remains a within-collection official-39 study. It is not an
unseen-session, per-device or physical-edge benchmark.
