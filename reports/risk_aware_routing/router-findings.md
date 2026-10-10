# Local router results and the limits of automatic escalation

11 October 2026 · Exploratory reused-validation experiment

The first small learned router **does not pass the predeclared continuation
screen**. It provides a small IID improvement in one of three seeds, but no
Brute Force improvement and no non-IID improvement under the conservative
false-alert constraint. Do not promote it, increase its capacity automatically,
or spend cloud credits to repeat this design at larger scale.

A specialist can recognise more rare attacks without an upstream router being
able to identify safe consultations. The evidence supports a detection versus
false-alert trade-off, not a free improvement from adding a decision layer.

Update: the separate [stage-four frozen-policy sensitivity study](tradeoff-findings.md)
is complete. It explores illustrative nonzero alert allowances and unresolved
review referrals, without changing this experiment's failed success screen.

## The local experiment

The [fixed plan](router-plan.json) follows the
[specialist comparison](specialist-findings.md). Each saved IID and non-IID
model is paired with its same-seed balanced specialist. The twelve existing
neural checkpoints remain unchanged. Only six small decision trees are trained,
one per base lane and seed (7, 17, 27).

Each tree receives the 39 standardized flow features and eight base-model
softmax outputs. It does not receive labels, row identifiers, group hashes or
the specialist's prediction at inference time. Labels and the specialist's
output construct the training outcomes only: consultation repairs a mistake,
damages a correct prediction, or does neither.

The tree has maximum depth five and at least 128 fit examples per leaf. All
settings are fixed, without an architecture or seed sweep. Training weights
give each true class equal aggregate influence, then double the benign class's
weight. The score is the weighted repair fraction minus weighted harm fraction
in a leaf, not a calibrated probability of safety. Fitted trees have 49–61 nodes.

## Separate fit, calibration and internal audit populations

Identical full39 float32 vectors, including equivalent signed zeros, receive
the same feature-only hash and partition. Labels cannot change that assignment.
The source dataset and original panel remain untouched.

| Role | Rows | Web-based rows | Brute Force rows |
| --- | ---: | ---: | ---: |
| Eligible router fit | 1,229,859 | 1,502 | 760 |
| Threshold calibration | 409,919 | 501 | 271 |
| Internal audit | 408,027 | 472 | 253 |

Within the fit partition only, sampling is capped at 50,000 rows per class.
The actual common sample contains 270,326 rows, including all eligible rare
rows. Feature copies are bounded; source arrays and cached neural probabilities
remain disk-backed.

This separation protects the new gate's fit and calibration, but is **not a
fresh independent test**. Validation has already informed extensive development,
including the choice of specialist. Duplicate grouping does not create unseen
capture sessions or devices. The original final test is not reopened or scored.

## Calibration rules and controls

For each 1%, 5% and 10% maximum consultation budget, calibration chooses the
highest macro-F1 threshold subject to both:

- no more consultations than the declared calibration budget;
- no more benign false-alert cases than the base model on calibration.

Candidates use a fixed 0.1 percentage-point quota grid up to 10%. Equal scores
stay together rather than being split by row ID. Zero consultations are always
allowed; equal-F1 choices prefer fewer calls. The budget is a ceiling, not a
target the router must spend.

Confidence-ranked and seeded random controls receive the same selection rule
and constraint, for both the balanced and full-cohort fallbacks. Separate
quota-only confidence controls show behaviour without the benign guard.
Base-only and always-fallback results are retained too.

All six trees and every threshold are saved and hashed before any new router
audit outcomes are computed. Audit applies the fixed thresholds without
sorting, retuning or trimming calls to the quota. Calibration compliance is
not a guarantee on later data; the results flag later violations explicitly.

The zero-increase benign rule is a conservative research choice, not an
operational requirement supplied by a security team. It is especially strict
for the non-IID base, whose low false-alert rate comes with missed attacks.
It does not mean that the baseline behaviour is safe or satisfactory.

## Internal audit results

These are equal-weight means over the three paired seeds on the same 408,027
audit rows, not the entire previous validation panel. The learned policy is
identical at all three ceilings: a larger budget does not make calibration
find a different acceptable threshold.

| System | Macro-F1 | Benign false alerts | Web-based recall | Brute Force recall | Consulted |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID base | 0.61369 | 18.793% | 0.92% | 27.27% | 0% |
| IID learned router | 0.61817 | 18.769% | 3.18% | 27.27% | 0.021% |
| Non-IID base | 0.40500 | 0.215% | 0.00% | 0.00% | 0% |
| Non-IID learned router | 0.40500 | 0.215% | 0.00% | 0.00% | 0% |

The IID improvement comes entirely from seed 27. It consults the specialist
on 255 audit rows, correcting 60 predictions and damaging 66. Macro-F1 still
increases from 0.60777 to 0.62122 because the changes affect classes differently:
Web-based recall rises from zero to 6.78%, but Brute Force recall is unchanged.
Macro-F1 improvement is not the same as repairing more total rows than are
damaged. Seeds 7 and 17 make zero consultations.

All three non-IID gates make zero consultations. Under this model family,
candidate grid and guard, calibration finds no positive-F1 alternative it can
accept. This does not prove that no router could work, or that another explicit
false-alert tolerance would produce the same result.

All-class recall, in percent:

| Class | IID base | IID learned | Non-IID base | Non-IID learned |
| --- | ---: | ---: | ---: | ---: |
| Benign | 81.207 | 81.231 | 99.785 | 99.785 |
| DDoS | 84.556 | 84.556 | 98.744 | 98.744 |
| DoS | 67.412 | 67.412 | 20.892 | 20.892 |
| Recon | 72.555 | 72.471 | 19.837 | 19.837 |
| Web-based | 0.918 | 3.178 | 0.000 | 0.000 |
| Brute Force | 27.273 | 27.273 | 0.000 | 0.000 |
| Spoofing | 56.871 | 56.796 | 0.936 | 0.936 |
| Mirai | 99.425 | 99.425 | 98.735 | 98.735 |

Guarded confidence and random balanced-fallback controls choose no calls in
either lane. The tree therefore adds a little IID value over those controls,
but not consistently across seeds or both rare classes.

The full-cohort fallback remains useful as a comparator. Guarded confidence
routing at the 10% calibration ceiling reaches IID mean F1 0.64116 and 17.256%
false alerts, with 9.237% average audit consultation. Two seeds slightly exceed
the intended audit budget (10.009% and 10.014%). These violations are reported,
not hidden by retrospectively trimming calls.

Without the benign guard, non-IID balanced-fallback confidence routing at the
10% calibration quota reaches F1 0.49046, Web-based recall 5.30% and Brute Force
recall 12.65%, but benign false alerts rise to 2.117%. That is a trade-off, not
a result meeting the guard or grounds for promoting the model.

[Aggregate evidence](router-results.json) preserves all seed values, sample
SDs, standalone controls, calibration decisions, confusion matrices and
correction/harm counts. Seeds share the same audit rows; their SD is not an
independent-data confidence interval.

## Decision and next step

No lane passes the fixed screen at any budget. All three seeds would need to
respect realised budget and benign guard, improve macro-F1 and both rare
recalls over the base, and match or beat guarded confidence and random
balanced-fallback controls. We do not loosen that screen after seeing the
audit or start a larger search.

The next useful decision is the intended security workflow and its acceptable
false-alert or review burden. Automatic relabelling and human-review referral
have different costs. If a nonzero alert increase is acceptable, predeclare a
separate cost/alert-budget study before fitting another policy; reused-data
results remain exploratory. Review referrals must be counted as referrals,
not fabricated correct classifications or measured human-review successes.

A paid decision model faces the same information and error-cost problem; this
experiment supplies no evidence that an API would solve it. A decision tree
is not methodological novelty by itself. The useful study contribution is the
controlled account of when selective consultation helps and when it harms.

Keep work local. Consider cloud only after a worthwhile candidate and evaluation
protocol are frozen, for a specific final compatibility check or benchmark.
Cloud is not required merely to complete this extension. Confirmatory claims
need suitable independent data, not a stronger processor or a reshuffled old test.

## Verification and reproduction

The recorded run section took 27.8 seconds on the local CPU, excluding startup
and initial preflight checks. No local GPU, cloud machine or external API was
used. This is experiment runtime, not per-flow latency or an edge measurement.
All frozen models were scored across validation for diagnosis, so consultation
percentages do not establish measured compute savings.

Two complete runs exactly reproduce every non-timing receipt field, including
tree hashes, calibration freeze, group assignment, predictions, thresholds and
audit counts. All twelve neural predictions match the prior specialist run,
and neural weights remain unchanged.

An initial execution stopped before calibration freeze or audit: the JSON
tree reader redundantly normalized already-normalized probabilities. Export
now records each leaf's exact public `predict_proba` output, with a synthetic
regression test. Data, tree settings, guard and success screen were not changed
in response. The failed partial output is preserved, not reused as evidence.

No dependency was added. Tests cover group isolation, audit-independent fitting,
exact JSON inference, whole-score ties, no-call fallbacks, honest budget breaches,
fixed calibration choices and aggregate tamper detection. Full-suite verification
passes all 387 tests in 35.4 seconds in a clean snapshot, excluding unrelated
uncommitted work. There are 29 new tests and 62 focused routing tests. The
environment passes `pip check`, and the published count/provenance audit passes.

Run locally from the repository root, using a fresh output path:

```bash
.venv/bin/python -m unittest tests.test_routing_gate tests.test_routing_gate_report -q
.venv/bin/python scripts/official39_routing_gate.py \
  --archives /Users/haydn/Downloads \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --specialists outputs/official39-panel-validation-v1 \
  --output outputs/official39-routing-gate-reproduction
.venv/bin/python scripts/official39_routing_gate_report.py audit \
  --evidence reports/risk_aware_routing/router-results.json
```

Replay needs the recorded CPU environment and prior private inputs. The
aggregate audit needs neither flow arrays nor checkpoints and performs no new
inference. The report command's `publish` action takes the private `--output`
directory and a fresh `--evidence` destination; it refuses to overwrite evidence.
No final-test backup is needed. Beginner-oriented comments and unrelated local
work are preserved.
