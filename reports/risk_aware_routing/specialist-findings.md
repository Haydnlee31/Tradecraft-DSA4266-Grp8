# A better rare-attack specialist still needs a better router

11 October 2026 · Local, exploratory validation evidence · No deployment approval

The saved balanced-panel model recognises substantially more rare attacks than
the full-cohort centralized model. However, sending low-confidence federated
predictions to it also creates mistakes. It is useful evidence for studying
routing, not a replacement we should promote or scale to cloud now.

The next bounded experiment should ask: **can a small numerical router predict
when consulting this specialist will help, rather than merely when the base
model is uncertain?** A paid decision API has not yet earned a place in the
pipeline. This comparison does not evaluate Jev, Laya or any external service.

Update: the proposed local router experiment is now complete and did not pass
its predeclared success screen. See [the router findings](router-findings.md).
The specialist results below remain unchanged.

## What changed, and what stayed fixed

This follows the [first routing diagnostic](README.md). The
[predeclared specialist plan](specialist-plan.json) changes only the fallback:

- Control: the existing centralized-light model trained on the full 2M cohort.
- Alternative: the existing `small_dropout` model trained centrally on the
  balanced 10,152-row panel, with 1,269 rows per class, for 300 epochs.

Both have 5,096 parameters, 39 inputs, two hidden layers of 64 and 32 units,
LayerNorm and dropout 0.2. The alternative is a different training-distribution
specialist, not a bigger network or a newly federated-trained model. Its saved
final checkpoint is used; there is no new training or checkpoint search.

All three seeds (7, 17, 27), both federated base lanes, the shared 2,047,805-row
validation panel, preprocessing and 1%, 5%, 10% routing quotas stay fixed.
Every fallback receives exactly the same selected rows for each seed, lane,
rule and budget. We retain seeded random-routing and always-fallback controls.
All eight output classes remain in scope.

The confidence controls rank the complete population offline; they are not
fixed thresholds for arriving flows. Every model is scored on every row to
measure complementarity. A 10% routing quota therefore does not establish a
90% compute saving, and no edge latency, memory or power was measured here.

The panel has already been used for development, and the original final test
was exposed in the completed study. These are exploratory validation results,
not independent generalisation estimates. This run neither opens nor scores
the test arrays, fits a gate, calls an external API nor changes the completed
study. No raw flows or model weights are included in this report's JSON export.

## Actual routing: rare recall improves, but overall performance does not

The table uses the largest *predeclared* quota to make the trade-off visible;
it is not a selected deployment setting. All numbers are equal-weight means
over the three paired seeds. False alerts mean attacks predicted among truly
benign rows, not the fraction of alerts that are false.

| Base / fallback | Routed | Macro-F1 | Benign false alerts | Web-based recall | Brute Force recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID / no fallback | 0% | 0.61386 | 19.07% | 0.85% | 27.10% |
| IID / full-cohort | 10% | 0.64197 | 17.43% | 4.51% | 27.31% |
| IID / balanced-panel | 10% | 0.58904 | 27.15% | 39.29% | 42.08% |
| Non-IID / no fallback | 0% | 0.40492 | 0.26% | 0.00% | 0.00% |
| Non-IID / full-cohort | 10% | 0.49570 | 2.08% | 0.75% | 5.97% |
| Non-IID / balanced-panel | 10% | 0.49049 | 2.17% | 5.87% | 11.14% |

For IID, the balanced fallback is clearly harmful under this simple gate: it
corrects an average of 30,464 predictions but damages 69,319 previously correct
ones. Its macro-F1 falls relative to both the original base and the full-cohort
fallback. More rare detections are not a free improvement.

For non-IID, it corrects about 86,214 predictions and damages 64,670. That helps
relative to the collapsed base, but does not beat the full-cohort fallback in
mean macro-F1. The paired F1 difference is -0.00522 (sample SD 0.01167);
individual differences are -0.01833, -0.00134 and +0.00402. With only three
training seeds on the same rows, this is a descriptive trade-off, not a claim
of statistically established superiority. False alerts increase in all three
seeds, by a mean 0.098 percentage points versus the full-cohort fallback.

The pattern across every predeclared confidence quota is:

| Base | Quota | Full-cohort F1 | Balanced F1 | Full-cohort false alerts | Balanced false alerts |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID | 1% | 0.62159 | 0.60069 | 18.18% | 21.69% |
| IID | 5% | 0.63888 | 0.59040 | 17.45% | 26.77% |
| IID | 10% | 0.64197 | 0.58904 | 17.43% | 27.15% |
| Non-IID | 1% | 0.43087 | 0.43332 | 0.53% | 0.55% |
| Non-IID | 5% | 0.46750 | 0.46713 | 1.24% | 1.26% |
| Non-IID | 10% | 0.49570 | 0.49049 | 2.08% | 2.17% |

Always using the balanced specialist yields F1 0.56823 and 29.52% false alerts,
versus 0.65569 and 17.55% for the full-cohort centralized model. At the 10%
random quota the balanced fallback yields F1 0.60456 for IID and 0.44191 for
non-IID. Thus confidence routing beats random routing for non-IID, but is worse
than random routing for IID with this specialist. The gate's usefulness
depends on which mistakes its fallback makes.

All-class recall at the 10% confidence quota, in percent:

| Class | IID full-cohort | IID balanced | Non-IID full-cohort | Non-IID balanced |
| --- | ---: | ---: | ---: | ---: |
| Benign | 82.57 | 72.85 | 97.92 | 97.83 |
| DDoS | 84.48 | 80.66 | 94.50 | 93.63 |
| DoS | 68.53 | 73.94 | 39.92 | 40.59 |
| Recon | 73.02 | 59.76 | 31.53 | 27.44 |
| Web-based | 4.51 | 39.29 | 0.75 | 5.87 |
| Brute Force | 27.31 | 42.08 | 5.97 | 11.14 |
| Spoofing | 67.17 | 58.51 | 17.35 | 16.45 |
| Mirai | 99.47 | 99.48 | 99.13 | 98.97 |

## There is useful complementarity, not yet an implementable solution

An oracle can use the true label to consult the specialist only when it repairs
the base model. This is a diagnostic that cannot be deployed. Its class recall
is the union-of-correct-predictions ceiling for this pair; its macro-F1 is not
a general macro-F1 optimum.

With the balanced fallback, this rule reaches mean F1 0.75940 for IID and
0.81950 for non-IID. Web-based recall could reach 51.07% in either lane, versus
6.87% with the original fallback. Brute Force recall could reach 54.86% for
IID and 54.78% for non-IID. Even perfect selection still misses many rare
attacks; routing cannot repair rows both models misclassify.

The oracle requires 4.96% average escalation for IID and 15.46% for non-IID.
Every non-IID seed needs more than 10%, so its oracle must not be presented as
an attainable result at the tested 10% quota. Even the IID oracle fitting
within a quota does not establish that an inference-time router can find it.

At 10%, the non-IID confidence gate selects only 11.50% of the Web-based
mistakes the specialist could repair, and 20.33% of the repairable Brute Force
mistakes. These fractions pool counts over paired seeds, unlike the metric
means above. The gate misses most useful consultations. For IID it finds more
rare repairs, but routes many cases where the specialist is worse. This is why
simply adding another classifier or raising the escalation rate is insufficient.

## Next local experiment and stopping rule

Keep RONIN stopped. Before another training run or paid API integration:

1. Freeze a development-only routing protocol. Keep duplicate input groups
   together across gate fit, calibration and audit partitions; do not call an
   internal holdout a new independent test after this extensive validation use.
2. Fit one small numerical gate using flow features and base-model outputs
   available *before* escalation. Train it to distinguish useful consultations
   from harmful ones, with an explicit treatment of rare classes and benign
   false alerts. Do not use expert outputs as free gate inputs: obtaining them
   already incurs the consultation cost.
3. Select calibration thresholds without audit labels, then freeze them.
   Retain base-only, always-specialist, confidence and random controls. Report
   all-class recall, false alerts, actual audit escalation rates, corrections
   and damage. A calibration quota need not equal the rate on later traffic.
4. Continue only if the gate shows a useful, reasonably consistent trade-off
   across the paired seeds under declared budgets and false-alert constraints.
   If gains require excessive false alerts or disappear on the internal audit,
   stop this extension and report the negative result. Do not keep searching
   architectures until one wins on reused validation.

This is a protocol recommendation, not a gate that has already been trained.
No operational false-alert tolerance has been approved. A typed-decision model
could later compete with the numerical gate on the same permitted inputs and
budget; external data sharing and measured end-to-end cost would need separate
review. Any confirmatory performance claim still requires suitable independent
data. Reshuffling the old final test cannot supply it.

## Verification and reproduction

The [runner](../../scripts/official39_routing_specialist.py) uses bounded
validation batches, hash-pinned checkpoints and restricted weights-only
loading. The historical training/evaluation files are unchanged.

All nine stage-one model predictions and the full stage-one routing summary
replay exactly. All three specialist checkpoints reproduce their historical
all-validation confusion matrices exactly at the original batch size of
8,192. Scoring at the common batch size of 512 also produces identical
confusion matrices for these specialists. Every current scan matches the
independent frozen evaluator; all twelve model states remain unchanged.

One whole-file provenance difference was reviewed explicitly: commit `5cc8206`
added opt-in ablation training support to `official_streaming.py` after the
specialist run. Its `evaluate` function is byte-identical to the historical
`bfab230` version. The plan pins both file hashes and the unchanged function
hash, and still requires exact historical replay. It does not waive arbitrary
source differences. The initial preflight stopped on this mismatch before
creating an experiment output; no training was launched to repair it.

Two independent local runs exactly reproduce all non-timing receipt fields.
The [aggregate JSON](specialist-results.json) preserves seed-level values,
sample SDs, all-class recalls, paired corrections/damage and provenance.
The comparison checks the same selected-row hashes for both fallbacks.
The focused suite passes 33 tests, including 15 new specialist tests. The
clean repository snapshot passes all 358 tests in 34 seconds, excluding
unrelated uncommitted work. The environment passes `pip check`, and the
published result's source hashes and all twelve model checks are verified.

Run from the repository root with the existing local environment and fresh
output names:

```bash
.venv/bin/python -m unittest tests.test_official_routing_specialist tests.test_official_routing_headroom -q
.venv/bin/python scripts/official39_routing_specialist.py \
  --archives /Users/haydn/Downloads \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --specialists outputs/official39-panel-validation-v1 \
  --output outputs/official39-routing-specialist-reproduction
```

This exact replay is pinned to the stage-one local CPU runtime. Reusing the
code on a different runtime needs a separately recorded compatibility review,
not deletion of the guard. The original specialist receipt and three named
checkpoints are required in addition to the stage-one inputs. No final-test
backup is needed. An optional `--publish-summary` exports aggregate JSON to a
fresh destination and refuses to overwrite existing evidence. No dependencies
were added, and unrelated uncommitted work is preserved.
