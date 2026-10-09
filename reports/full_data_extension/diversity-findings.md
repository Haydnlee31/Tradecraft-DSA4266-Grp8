# Broader training coverage does not fix the false alert burden

The subsequent [fixed class-prior correction](prior-correction-findings.md)
is complete. It improves aggregate metrics but sacrifices Web-based and DoS
recall; the balanced-panel tuning sequence is now closed without model promotion.
The diversity evidence and original next-step rationale below are retained.

Exposing the small neural network to more diverse training examples produces a
small, inconsistent macro-F1 gain and more benign false alerts. Across seeds 7,
17 and 27, mean validation macro-F1 changes from 0.568271 to 0.572322, while the
benign false-alert rate increases from 29.52% to 30.85%. False alerts increase
in every seed. This is not a useful replacement for the existing model or a
reason to resume cloud scaling with the same balanced training recipe.

The result narrows the diagnosis: repeatedly seeing only a small benign and
common-attack sample was not sufficient to explain the alert problem. It does
not establish that training diversity never matters, or that any single class
caused the change. All six nonrare class pools changed together.

## What the experiment changed

The [frozen plan](diversity-plan.json) followed the
[alert-budget diagnostic](alert-budget-findings.md). That diagnostic showed
that suppressing false alerts also suppressed many real attacks. The next
question was whether the training sample itself was too narrow.

Both conditions use the same 64/32 LayerNorm network, dropout 0.2, Adam settings,
39 features and scaler fitted on the 2M training cohort. Each seed processes
3,045,600 examples in 6,000 optimizer updates. Each of the eight classes appears
380,700 times. The exact class sequence, batch boundaries, initial parameters,
and Web-based and Brute Force row identities and positions are matched.

The control repeatedly visits the original balanced panel of 10,152 rows,
1,269 per class. The treatment retains those exact rare-class rows but draws
the other six classes from their larger training pools. Each class has its own
seeded shuffle; every available row is visited before that class reshuffles.
No validation examples enter these pools. The full feature matrix stays
disk-backed, with only bounded training batches copied into memory.

Here, an epoch still means 10,152 scheduled draws, not one pass over the 2M
cohort. Both conditions end after exactly 300 such epochs. No model is selected
by its best epoch, and the original production training code is unchanged.

## Coverage increased substantially at the same work budget

The counts below hold for each seed. Exact row identities differ across the
treatment seeds, but coverage and class exposure counts agree.

| Class | Available training rows | Unique control rows | Unique treatment rows | Treatment visits per seen row |
|---|---:|---:|---:|---:|
| Benign | 106,148 | 1,269 | 106,148 | 3–4 |
| DDoS | 1,178,561 | 1,269 | 380,700 | 1 |
| DoS | 362,423 | 1,269 | 362,423 | 1–2 |
| Recon | 66,277 | 1,269 | 66,277 | 5–6 |
| Web-based | 2,391 | 1,269 | 1,269 | 300 |
| Brute Force | 1,269 | 1,269 | 1,269 | 300 |
| Spoofing | 44,095 | 1,269 | 44,095 | 8–9 |
| Mirai | 238,836 | 1,269 | 238,836 | 1–2 |

The treatment sees 1,201,017 unique training rows versus 10,152 for the control.
It sees every available benign row. It deliberately does not add the remaining
Web-based rows, because changing rare-class exposure would confound this control.
Brute Force already uses all 1,269 available training examples in both conditions.

This matches example counts and optimizer updates, not data-loading cost or
wall-clock time. The saved per-row exposure arrays and sequence hashes document
both coverage and repetition; those generated arrays stay outside Git.

## Validation results

Every model is evaluated on the same 2,059,284-row natural-prevalence validation
split, using raw argmax predictions. There is no alert threshold, probability
calibration, tree fit or final-test evaluation in this experiment. The split has
been used in earlier exploratory work, so these are not independent confirmation
results. SD below describes variation across three training seeds, not across
new capture sessions or a confidence interval.

| Condition | Mean macro-F1 | Sample SD | Benign false alerts | Web recall | Brute Force recall |
|---|---:|---:|---:|---:|---:|
| Original panel control | 0.568271 | 0.003973 | 29.52% | 51.07% | 54.78% |
| Broader nonrare coverage | 0.572322 | 0.005875 | 30.85% | 52.70% | 56.59% |

The mean macro-F1 gain is only 0.004052, or 0.41 percentage points. It is positive
in two seeds and negative in the third. The mean false-alert increase is 1.33
percentage points, and its direction is consistent across all three seeds.

| Seed | Control macro-F1 | Treatment macro-F1 | Control false alerts | Treatment false alerts |
|---|---:|---:|---:|---:|
| 7 | 0.569013 | 0.578890 | 28.49% | 31.17% |
| 17 | 0.563979 | 0.567568 | 28.74% | 29.28% |
| 27 | 0.571819 | 0.570510 | 31.32% | 32.10% |

Web-based recall improves in every seed, but its mean precision remains only
4.07%. Brute Force recall improves in two seeds, while mean precision declines
slightly to 4.11%. Many rare-category predictions are still incorrect. Some
incorrect category predictions are other attacks, not benign traffic, so these
precision values must not be confused with the benign false-alert rate.

## Every class retains an explicit comparison

Precision and recall are mean percentages across the same three seeds.

| Class | Control recall | Treatment recall | Control precision | Treatment precision |
|---|---:|---:|---:|---:|
| Benign | 70.48 | 69.15 | 81.77 | 82.48 |
| DDoS | 63.05 | 66.28 | 96.62 | 95.56 |
| DoS | 92.49 | 89.77 | 43.68 | 45.16 |
| Recon | 53.59 | 53.77 | 75.27 | 76.06 |
| Web-based | 51.07 | 52.70 | 3.80 | 4.07 |
| Brute Force | 54.78 | 56.59 | 4.21 | 4.11 |
| Spoofing | 59.05 | 61.20 | 76.86 | 73.90 |
| Mirai | 99.42 | 99.52 | 98.84 | 99.05 |

The treatment trades some benign and DoS recall for modest gains elsewhere.
It has not produced uniformly better separation. The
[machine-readable results](diversity-results.json) publish all six confusion
matrices, training budgets, class exposures and checkpoint hashes, allowing
every class metric to be recomputed without choosing a favorable seed.

## Interpretation and the next bounded check

The equal class exposure remains very different from natural prevalence. Each
rare class receives 12.5% of the optimization examples; in the 2M training cohort,
Web-based accounts for 0.11955% and Brute Force for 0.06345%. More diverse examples
did not change that objective. This makes class prevalence a useful next question,
but it does not prove that prevalence is the only remaining issue. Equal exposure
was deliberate, not a coding error. A natural-prior correction changes the decision
trade-off and is not guaranteed to improve macro-F1 or rare-class recall.

Keep cloud compute stopped. Before any further training, predeclare one
inference-only comparison of the six saved models: raw argmax versus a fixed
class-prior adjustment based only on the original 2M training counts. For this
uniform-exposure recipe, the proposed score is `logit[c] + log(p_train[c])`;
the uniform training-prior term is a common constant and does not change argmax.
There should be no fitted coefficient, temperature search or validation-derived
class frequency. Report all eight recalls and false alerts, not just accuracy.

Adjusting outputs for changed class frequencies without refitting is an
established approach; see [Saerens, Latinne and Decaestecker, 2002](https://research.dial.uclouvain.be/entities/publication/ff0a9d02-c8d6-4e6a-8320-5665e5cb85ba).
Our proposed fixed-count check is not their iterative unknown-prior estimator.
Its interpretation assumes useful posterior scores and sufficiently stable
within-class feature distributions. Neither is guaranteed here. The cohort
frequencies are a diagnostic proxy, not estimates of real deployment traffic.
The correction may lower rare recall sharply; that would be a trade-off to
report, not something to hide through another threshold search.

If that bounded check also offers no useful compromise, close this balanced-panel
tuning path and preserve the stronger full-cohort reference for the project's
main comparison. Further cloud runs should require a new, specific hypothesis
and a predeclared local result that justifies their cost. No claim about a
federated improvement follows from these centrally pooled training controls.

## Verification and reproduction

All three controls reproduced the complete historical training observations,
the 100-epoch parameter checkpoint and the final 300-epoch weights before any
treatment began. All six fits finished before validation opened. Every control
then reproduced its historical validation cross-entropy and confusion matrix
before any treatment was scored. Paired checks require identical initial weights,
class weights, work counts, label sequences and rare-row position sequences.

An independent complete repeat matched every non-timing receipt field exactly,
including checkpoint files, exposure arrays, confusion matrices and summaries.
The eight new tests cover sampling cycles, singleton batches, exact control
replay, repeatability, tampering, scope restrictions, failed-bridge sequencing
and sealed test access. All 171 local workspace tests passed. No dependency was
added, no production model was promoted, and no physical-edge performance was
measured. This remains a within-collection official-39 experiment.

Frozen plan SHA-256:
`15dc34b4334d3dca15341f204a9ac921891d961e237fef384fdd10c3c34eeffb`.
Canonical receipt: `outputs/official39-diversity-v1/receipt.json`, SHA-256
`0d1d36434c4ba4921d80a4f0c155cf4a49de7918a54c384bd5d239fcadbf36f4`.

Reproduce into a new directory using the recorded package versions:

```bash
.venv/bin/python -m src.eval.official_diversity \
  --data-root outputs/official39-packed-2m-v1 \
  --reference-path outputs/official39-panel-validation-v1/receipt.json \
  --training-reference-path outputs/official39-width-fit-v1.json \
  --plan-path reports/full_data_extension/diversity-plan.json \
  --output outputs/official39-diversity-replay
```

This intentionally requires local packed data and prior receipts, not just a
Git clone. The committed tests use synthetic fixtures and require neither the
large dataset nor generated model files. Exact historical replay is a local
same-environment check, not a promise of bitwise equality across Torch versions.
