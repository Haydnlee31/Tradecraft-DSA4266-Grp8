# Automatic score fusion for rare attack classification

The fully automatic fusion classifier improves both federated baselines, but
does not solve rare-attack classification or outperform the existing centralized
model overall. Its non-IID gain largely disappears when compared with a simpler
recalibrated specialist instead of the weak unchanged federated baseline.
Do not promote this two-model pipeline or scale it to cloud training yet.

Human review, referral queues and abstention are not part of this classifier.
The earlier triage prototype remains an optional historical artifact, not a
requirement. This local experiment addresses whether an added learned layer can
improve the actual eight-class prediction.

## Results across all three seeds

Numbers below are equal-weight means over seeds 7, 17 and 27 on the same
408,027-row internal audit. They are exploratory development results, not new
final-test scores. Benign false alerts mean the percentage of truly benign
flows classified as any attack; they are not the fraction of alerts that are
false. Every seed and all eight classes are retained in the
[aggregate results](fusion-results.json).

| Method | IID macro-F1 | Non-IID macro-F1 | IID benign false alerts | Non-IID benign false alerts |
| --- | ---: | ---: | ---: | ---: |
| Unchanged federated base | 0.613689 | 0.404997 | 18.79% | 0.21% |
| Unchanged balanced specialist | 0.567591 | 0.567591 | 29.33% | 29.33% |
| Equal probability average | 0.613290 | 0.554476 | 20.93% | 2.00% |
| Recalibrated base alone | 0.615518 | 0.575256 | 19.96% | 25.25% |
| Recalibrated specialist alone | 0.632242 | 0.632242 | 19.07% | 19.07% |
| **Learned fusion** | **0.640735** | **0.632954** | **19.49%** | **19.46%** |
| Existing centralized reference | 0.655632 | 0.655632 | 17.39% | 17.39% |

Fusion raises IID macro-F1 by 2.70 percentage points and non-IID macro-F1 by
22.80 points relative to their respective unchanged bases. Both gains occur in
every seed. But the non-IID false-alert rate also rises by 19.25 percentage
points: its original low-alert behavior included calling nearly all rare
attacks benign, not accurately recognizing them.

The stronger controls change the interpretation. IID fusion beats the
recalibrated specialist by 0.85 macro-F1 points in every seed, with 0.42 points
more benign false alerts. Non-IID fusion exceeds that simpler specialist by
only 0.07 points on average and loses to it in seed 17. Both fusion variants
lose to the existing centralized model in every seed, while generating more
benign false alerts. The predeclared continuation screen therefore fails in
all six seed/lane comparisons. This does not erase the baseline gains, but it
does rule out claiming that the more complex combination is the overall winner.

## Rare attacks improve but remain mostly missed

Precision answers whether a predicted category is correct; recall answers how
many true attacks of that category are found. Values below are percentages,
including F1 expressed on a 0–100 scale for readability.

| Method | Web precision | Web recall | Web F1 | Brute Force precision | Brute Force recall | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IID base | 16.67 | 0.92 | 1.74 | 30.55 | 27.27 | 28.68 |
| Non-IID base | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Balanced specialist | 3.76 | 52.75 | 7.02 | 4.13 | 55.20 | 7.68 |
| Recalibrated specialist | 21.89 | 11.65 | 15.20 | 30.54 | 31.23 | 30.83 |
| IID fusion | 20.86 | 11.02 | 14.41 | 34.33 | 30.17 | 32.03 |
| Non-IID fusion | 20.31 | 11.02 | 14.28 | 30.37 | 30.43 | 30.36 |
| Centralized reference | 41.67 | 7.34 | 12.37 | 42.29 | 27.14 | 32.94 |

Fusion improves both rare-class F1 scores over the corresponding base in every
seed. That is real automatic classification improvement, not credit for a
referral to a person. However, it still misses about 89% of Web-based attacks
and 70% of Brute Force attacks. Only about one in five Web-based predictions is
the correct category. This is not a solved rare-class problem.

The high-recall balanced specialist illustrates why recall alone misleads: it
finds over half the rare attacks, but fewer than 5% of its rare-category
predictions are correct. The fitted heads suppress many such predictions and
increase precision, at a substantial recall cost. The specialist-only head
actually has better Web-based F1 than joint fusion in both lanes on average.

The remaining errors are not merely a benign-versus-attack threshold problem.
For IID fusion, 44.42% of true Web-based rows are called Recon, 27.90% Benign
and 16.53% Spoofing. For non-IID fusion the corresponding figures are 42.66%,
29.87% and 16.24%. Brute Force is also frequently confused with Benign and
Recon. These percentages aggregate the three fixed-seed confusion matrices;
the same underlying audit rows occur in each seed, not three new datasets.

## What this diagnoses and what to do next

The added classifier can use information hidden beneath the neural models'
top labels. Across the three seeds it correctly predicts a third label on
1,674–1,934 IID rows and 2,853–3,552 non-IID rows per run, where neither frozen
model selected the correct top label. Thus the earlier hard-selector recall
ceiling is not an upper bound for score fusion.

Nevertheless, the two-model non-IID system contributes little beyond a fitted
specialist alone. More copies of these models, a larger fusion head, or a paid
decision API are not supported by this comparison. It also does not prove that
the original features lack information: fusion only sees sixteen compressed
scores, not the 39 flow features.

The next justified local experiment is a **feature-aware rare-attack expert**:
let a small nonlinear classifier use the original flow features and learn
against diverse Benign, Recon and Spoofing examples rather than relying only
on the balanced specialist's scores. The current specialist saw only 1,269
examples from each class during neural training. Broader negative-example
coverage is a plausible way to reduce category confusion, not a proven fix.

Keep that comparison bounded and fully automatic. Compare the feature-aware
expert alone with the same expert combined with the frozen generalist, retain
the current score-only fusion and centralized reference, and hold the fitting
rows and loss weighting fixed where possible. Include all three seeds and
natural-prevalence evaluation. A gain must survive the standalone-expert
control and report rare precision, recall, F1 and benign false alerts. Do not
repeat the earlier wider balanced-panel network or present the earlier tree
baseline as a new experiment; those recipes already failed to resolve this
trade-off.

No feature-aware expert is trained by this run, and no cloud session is
needed. Independent data or a separately justified untouched evaluation
protocol will still be necessary before confirmatory claims; another split
of the already-used audit or original final test does not create that evidence.

## What the additional model learns

For each flow, the frozen federated generalist and the frozen balanced
specialist each supply eight class scores. A small multinomial logistic
classifier combines their sixteen log scores and predicts one final class.
It has 136 fitted coefficients and intercepts. Unlike selecting one model's
top label, it can select a third class when the softer scores contain useful
evidence. This is standard supervised stacking, not a claim to have invented
a new general-purpose algorithm or evaluated Jev, Laya or a Decisions API.

Both neural networks run for every flow. The generalist and specialist each
have 5,096 parameters, so the combined system has 10,328 fitted parameters,
plus 32 stored normalization values for the fusion inputs. There is no claimed
90% compute saving from the earlier limited-consultation design.

## Controls that distinguish fusion from simpler corrections

Each IID and non-IID generalist is paired with its same-seed specialist at
seeds 7, 17 and 27. The comparison includes the unchanged generalist, unchanged
specialist, an equal average of their probabilities, and the existing
centralized full-cohort light model. Two further controls learn the same kind
of classifier using only one model's eight scores. Each has 72 parameters and
the same fitting labels, objective and regularization as the joint head.
Here "recalibrated" means a supervised score-to-class correction, not a claim
that the resulting probabilities are calibrated real-world risk estimates.

The specialist-only head is shared across the two lanes at each seed. In total
there are six joint heads, six generalist-only heads and three specialist-only
heads. Beating an unchanged base alone would not establish that two models are
needed; fusion should also outperform the single-model fitted controls.

## Fixed fitting and evaluation protocol

The [pre-run plan](fusion-plan.json) fixes log-score clipping at 0.000001,
fit-only standardization, L2-regularized multinomial logistic regression with
C equal to one, and no threshold or hyperparameter sweep. Only the heads are
trained. The existing checkpoints, neural architecture, train/validation
split, duplicate grouping and preprocessing remain unchanged.

Fitting uses the existing 270,326-row class-capped sample from the 1,229,859-row
grouped fit partition. Within that sample, each row of class c receives weight
proportional to sqrt(n_c)/m_c, where n_c is the full fit-partition class count
and m_c is the retained sample count. This corrects the sampling cap and makes
each class's total loss contribution proportional to the square root of its
original count. It does not balance the capped sample a second time. Weights
are normalized to mean one, and input means and standard deviations are fitted
only on the retained fitting rows.

All fifteen heads are saved and hashed before scoring the existing 408,027-row
internal audit partition. Its 472 Web-based and 253 Brute Force examples are
the same examples across all seeds; repeated seeds are not three independent
datasets. The 409,919-row calibration partition is unused in this experiment.
No audit labels are supplied to fitting or inference. Exact feature-vector
groups stay in one partition.

These are nevertheless **exploratory results on reused validation data**.
Earlier experiments have already informed the choice of models and this
extension. The internal split does not erase that development history, and the
original final test remains closed. Results do not establish generalization
to unseen capture sessions or production networks.

## What counts as progress

The predeclared continuation screen asks whether, in every seed and lane,
fusion has higher macro-F1 than every control and higher Web-based and Brute
Force F1 than the unchanged base. False-alert changes are reported alongside
this screen; it does not approve an operational false-alert threshold or
authorize deployment. Every class's precision, recall and F1 is retained,
including damage to predictions the base originally got right.

The central specialist and centrally fitted score head make this **assisted
inference using a federated generalist**, not a new pure federated training
result. Labeled fitting data are required for the head; neither privacy
preservation nor decentralized head training is demonstrated here.

## Reproduction and verification

The implementation is [official39_score_fusion.py](../../scripts/official39_score_fusion.py)
with numerical helpers in [score_fusion.py](../../src/eval/score_fusion.py).
New probability caches and fitted head parameters stay under ignored outputs,
not in git. Historical byte-pinned training and evaluation files are unchanged.

Use the existing local environment from the repository root. This is not a
cloud command and no RONIN connection is needed:

```sh
.venv/bin/python -u scripts/official39_score_fusion.py run \
  --cache outputs/official39-routing-gate-v2 \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --specialists outputs/official39-panel-validation-v1 \
  --archives /Users/haydn/Downloads \
  --output outputs/official39-score-fusion-v1
```

The output directory must not exist. A separate replay uses the same arguments
with a fresh output directory. Portable synthetic tests use no private flows:

```sh
.venv/bin/python -m unittest tests.test_score_fusion -q
```

Latency is measured on the local CPU, with two threads, already-preprocessed
inputs in memory, batch sizes one and 512, three warmups and thirty timed
repeats per method. Method order rotates between repeats. The timer includes
actual neural forwards, score transformation, head inference and schema checks;
it excludes loading, input preprocessing, network transfer and checkpoint I/O.
These are local implementation measurements, not physical edge measurements.

In the canonical run, the mean of the six paired median single-flow latencies
was 0.0241 ms for the base, 0.0502 ms for the recalibrated specialist and
0.0842 ms for fusion. For a batch of 512, the corresponding values were
0.1051 ms, 0.1766 ms and 0.3199 ms per batch. Fusion was about 3.5 times the
base's single-flow cost and 3.0 times its batch cost in this implementation.
Absolute latency is small on this local machine, but the added work still
needs a predictive benefit. These warm in-process timings are neither service
response-time guarantees nor measurements on RONIN or a Jetson device.

The canonical run completed in 360.35 seconds including fitting, scoring,
checksums and inference timing. An independent complete replay reproduced all
fifteen fitted head files, three specialist score caches, prediction hashes,
confusion matrices, metrics and non-timing receipts exactly. Timings are
measured again, not required to match bit for bit. The six base/specialist/
centralized audit comparisons also reproduce the earlier historical controls.

All **439 tests** passed in a clean snapshot of committed code plus this
extension, including nineteen new synthetic and published-evidence tests.
The broader working-tree suite also passed all 443 tests; the four additional
tests belong to pre-existing uncommitted work, which this change does not stage.
Dependency checking found no broken requirements, and no package was added.
Matplotlib/font-cache warnings from the sandbox were non-fatal.

Publish only aggregate evidence after a separate replay:

```sh
.venv/bin/python scripts/official39_score_fusion.py publish \
  --first outputs/official39-score-fusion-v1 \
  --replay outputs/official39-score-fusion-replay-v1 \
  --destination reports/risk_aware_routing/fusion-results.json
```

The published destination must not already exist. It now contains the audited
result; do not overwrite it when rerunning. Its aggregate checks require no
private flow arrays:

```sh
.venv/bin/python scripts/official39_score_fusion.py audit \
  --record reports/risk_aware_routing/fusion-results.json
```
