# Class prior correction improves aggregate scores but sacrifices attack recall

The subsequent [experiment ledger and decision summary](decision-summary.md)
now consolidate these findings with the full-cohort comparisons. The fixed
prior experiment and its original recommendation are preserved below.

The fixed class-prior correction improves macro-F1 and substantially reduces
benign false alerts for all six saved models. It also collapses Web-based and
DoS category recall. This is a clearer trade-off, not a satisfactory replacement
model. Close the current balanced-panel tuning sequence, preserve the existing
full-cohort references, and keep cloud compute stopped.

For the original panel models, mean macro-F1 rises from 0.568271 to 0.608169 and
false alerts fall from 29.52% to 15.09%. But Web-based recall falls from 51.07%
to 3.50%, and DoS recall from 92.49% to 31.89%. The diverse-training models show
the same pattern. No production model or decision rule has been changed.

## The fixed rule and its assumptions

The [plan](prior-correction-plan.json) was frozen before corrected scoring.
All six models come from the [training-diversity control](diversity-findings.md):
two training conditions, each with seeds 7, 17 and 27. We reused their fixed
300-epoch weights. There was no retraining, checkpoint selection, temperature
tuning, threshold search or validation-derived prior estimate.

Each model saw equal optimization exposure for the eight classes, so its source
class proportion is `q(c) = 1/8`. The diagnostic target proportion `p(c)` is the
original class count divided by 2,000,000, read from the hashed training manifest.
The score rule is:

```text
corrected_score(c) = original_logit(c) + log(p(c) / q(c))
prediction = class with the highest corrected score
```

A logit is an unnormalized class score. The constant `-log(1/8)` is identical
for every class, so the rule is equivalent in real arithmetic to the proposed
`logit(c) + log(p(c))`. Retaining the denominator makes equal priors an exactly
zero adjustment. The implementation computes the log ratio in float64, then
casts the fixed offsets to float32 before adding them to the model logits.

| Class | Original training proportion | Applied logit offset |
|---|---:|---:|
| Benign | 5.30740% | −0.856627 |
| DDoS | 58.92805% | +1.550589 |
| DoS | 18.12115% | +0.371351 |
| Recon | 3.31385% | −1.327618 |
| Web-based | 0.11955% | −4.649749 |
| Brute Force | 0.06345% | −5.283232 |
| Spoofing | 2.20475% | −1.735115 |
| Mirai | 11.94180% | −0.045684 |

This favors classes that are more common in the original cohort relative to
equal exposure. It can change one attack category into another, unlike the
earlier alert gate, which could only suppress an attack prediction into Benign.
It does not impose any particular false-alert cap.

Adjusting a classifier's outputs for changed class frequencies without refitting
is an established idea; see [Saerens, Latinne and Decaestecker, 2002](https://research.dial.uclouvain.be/entities/publication/ff0a9d02-c8d6-4e6a-8320-5665e5cb85ba).
This experiment uses known training counts, not their iterative unknown-prior
estimator. The posterior interpretation assumes useful probability estimates
and stable within-class feature distributions. Those assumptions are not
established here. The cohort proportions are not measurements of deployment
traffic, and the adjusted scores are not certified calibrated risks.

Equal-class exposure was deliberate, not a software bug. Reversing its effect
can improve some precision measures while undermining the rare-attack objective.

## Results across the same three seeds

These are means on the complete 2,059,284-row validation split. It has already
supported exploratory work, so this is not independent confirmation. The prior
offsets do not use validation labels or predictions, but the overall research
sequence has used validation feedback. The official-39 final test remains sealed.

| Training condition and scoring rule | Macro-F1 | Benign false alerts | Web recall | Brute Force recall | DoS recall |
|---|---:|---:|---:|---:|---:|
| Original panel, raw | 0.568271 | 29.52% | 51.07% | 54.78% | 92.49% |
| Original panel, corrected | 0.608169 | 15.09% | 3.50% | 25.10% | 31.89% |
| Diverse training, raw | 0.572322 | 30.85% | 52.70% | 56.59% | 89.77% |
| Diverse training, corrected | 0.611480 | 14.54% | 4.48% | 21.91% | 28.66% |

Every corrected model improves macro-F1 and lowers false alerts relative to
its own raw predictions. Every one also loses Web-based, Brute Force and DoS
recall. This consistent direction is descriptive evidence across these six
models, not a statistical significance claim over independent datasets.

| Condition | Seed | Corrected macro-F1 | Corrected false alerts |
|---|---:|---:|---:|
| Original panel | 7 | 0.605897 | 14.96% |
| Original panel | 17 | 0.609346 | 15.41% |
| Original panel | 27 | 0.609263 | 14.89% |
| Diverse training | 7 | 0.608950 | 16.01% |
| Diverse training | 17 | 0.610574 | 13.24% |
| Diverse training | 27 | 0.614917 | 14.39% |

Corrected macro-F1 sample standard deviations are 0.001968 for the panel and
0.003085 for diverse training. These describe seed variation, not confidence
intervals. Secondary overall accuracy increases from 72.70% to 82.60% for the
panel and from 74.12% to 82.78% for diverse training. Those increases do not
negate the severe category-level recall losses.

## Recall for all eight classes

Values are mean percentages across the three seeds.

| Class | Panel raw | Panel corrected | Diverse raw | Diverse corrected |
|---|---:|---:|---:|---:|
| Benign | 70.48 | 84.91 | 69.15 | 85.46 |
| DDoS | 63.05 | 96.50 | 66.28 | 97.72 |
| DoS | 92.49 | 31.89 | 89.77 | 28.66 |
| Recon | 53.59 | 68.42 | 53.77 | 69.87 |
| Web-based | 51.07 | 3.50 | 52.70 | 4.48 |
| Brute Force | 54.78 | 25.10 | 56.59 | 21.91 |
| Spoofing | 59.05 | 58.09 | 61.20 | 57.02 |
| Mirai | 99.42 | 99.34 | 99.52 | 99.30 |

The DDoS/DoS confusion changes sharply. For the panel models, the fraction of
true DoS rows predicted as DDoS rises from 7.24% to 67.88%. For diverse training,
it rises from 10.01% to 71.24%. These flows are still flagged as attacks, but
their assigned attack category is wrong. A binary detection headline would hide
this failure in the project's eight-class task.

## Precision and missed attack categories

Values below are also mean percentages, not pooled predictions from a single
ensemble. Higher rare-category precision means the fewer remaining rare alerts
are more often correct; it does not mean most rare attacks are detected.

| Class | Panel raw | Panel corrected | Diverse raw | Diverse corrected |
|---|---:|---:|---:|---:|
| Benign | 81.77 | 73.47 | 82.48 | 73.69 |
| DDoS | 96.62 | 82.18 | 95.56 | 81.62 |
| DoS | 43.68 | 74.49 | 45.16 | 79.80 |
| Recon | 75.27 | 70.20 | 76.06 | 70.49 |
| Web-based | 3.80 | 59.68 | 4.07 | 73.22 |
| Brute Force | 4.21 | 48.45 | 4.11 | 78.27 |
| Spoofing | 76.86 | 77.85 | 73.90 | 80.32 |
| Mirai | 98.84 | 99.09 | 99.05 | 99.77 |

For corrected panel models, Web-based flows split into 3.50% correct category,
38.13% predicted Benign and 58.37% another attack category. Brute Force splits
into 25.10% correct, 39.25% Benign and 35.64% another attack category. For corrected
diverse models, those breakdowns are 4.48% / 36.58% / 58.94% for Web-based and
21.91% / 40.60% / 37.49% for Brute Force. Both undetected attacks and incorrect
attack categorization remain important limitations.

All 12 raw/corrected confusion matrices and the exact prior offsets are in
[machine-readable results](prior-correction-results.json). No best seed or
favorable operating point was selected for the headline comparison.

## Decision and next stage

The correction is a useful diagnostic: class prevalence materially changes
the decisions of these fixed networks. It does not repair their eight-class
performance. Stop this balanced-panel tuning sequence without promoting either
corrected model. Do not turn this result into a coefficient or threshold sweep.

The earlier [full-cohort comparison](scaling-findings-and-controlled-noniid.md)
remains the reference evidence: centralized-light achieved mean macro-F1 0.65572
with 17.55% false alerts after 20 passes over 2M rows, while IID federated-light
achieved 0.61395 with 19.07%. Those settings are not work- or objective-matched
to these panel models, so they provide context rather than a causal superiority
claim. Their rare-class weaknesses also remain; none is certified deployment-ready.

Next, consolidate the official-39 experiment ledger and decision-layer evidence:
retain the fixed centralized and federated references, distinguish matched-work
controls from contextual comparisons, and state which classes fail under each
setting. Keep the original 46-feature study separate. Any explainability work
should describe frozen models' feature-level behavior and errors, not make
per-device, causal or physical-edge claims. Keep the final test sealed until
the evaluation protocol is explicitly frozen; no further cloud session is
needed to compile this evidence.

## Verification and reproduction

Every model file passed a checksum and metadata check before weights-only
loading. All six raw confusion matrices and cross-entropies reproduced exactly
before any corrected model was evaluated. Model parameters, checkpoints,
validation arrays, source files, plan and reference hashes were unchanged.
Training arrays were neither loaded nor rehashed. Priors came from their
previously verified manifest counts, not from a new fit or validation estimate.

An independent full repeat produced an identical receipt byte for byte, including
all 12 evaluations and provenance fields. Nine new tests cover probability-ratio
direction, uniform-prior identity and ties, inference-only access, corrupt
checkpoints, metadata/path rejection, failed-bridge sequencing and scope guards.
All 180 local workspace tests passed. No dependency, production decision criterion,
cloud job or physical-edge measurement was introduced.

Plan SHA-256:
`877474d04e492eb3321c7eb984bbc569bcf5ae92cccc8b800c2eaa0cf3163510`.
Canonical receipt: `outputs/official39-prior-correction-v1/receipt.json`, SHA-256
`c892b5621067fcaf50449a3d05b3c23e7dbe34f74d5d3e209da0a0b2bc5fb274`.

```bash
.venv/bin/python -m src.eval.official_prior_correction \
  --data-root outputs/official39-packed-2m-v1 \
  --reference-root outputs/official39-diversity-v1 \
  --plan-path reports/full_data_extension/prior-correction-plan.json \
  --output outputs/official39-prior-correction-replay
```

Reproduction requires the recorded local data/checkpoints and package versions;
they are intentionally not committed. The unit tests use synthetic fixtures.
Exact historical replay is a same-environment check, not a cross-version
bitwise-equality guarantee. This is a within-collection comparison, not an
unseen-session benchmark.
