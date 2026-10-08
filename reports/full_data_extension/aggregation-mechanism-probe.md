# Controlled non IID results and the aggregation diagnosis

The proposed balanced training-fit follow-up is now complete. See
[training-fit findings](training-fit-findings.md) for its results and the next
local control. The aggregation experiment and its original recommendation are
recorded below.

The controlled 2M run learns some attacks better than the 500k reference after
20 rounds, but still misses nearly all Web-based, Brute Force, and Spoofing
examples. A subsequent local diagnostic finds rare-attack signals in individual
clients, often alongside severe false alarms. Averaging their predictions instead
of their parameters does not fix the failure. Keep the cloud machine stopped;
the next useful question is whether the unchanged small network can fit a bounded,
balanced training panel before another full-data intervention.

## What the controlled cloud run established

The frozen partition keeps all 500,000 original rows with their original clients
and expands each class's allocation without redrawing the heterogeneous client
mixtures. GPU recovery checks and the historical 500k bridge passed. The 2M run
completed 20 rounds, with all 20 clients processing their assigned examples each
round. Its test metrics remain null.

| Endpoint | Processed training examples | Macro-F1 | Benign false alerts |
|---|---:|---:|---:|
| Historical 500k reference at round 20 | 10,000,000 | 0.36683 | 0.0904% |
| Controlled 2M at round 5 | 10,000,000 | 0.31447 | 0.0904% |
| Controlled 2M at round 20 | 40,000,000 | 0.40064 | 0.4110% |

At equal processed examples, the larger setup is worse by 5.24 macro-F1
percentage points. At 20 rounds it is better by 3.38 points, but has processed
four times as many examples. This is one seed and one anchored partition, not
a general scaling guarantee. Local epoch length, cohort-specific scalers and
class weights still differ; frozen ownership removes a partition-redraw confound,
not every difference between the training regimes.

The final 2M model sends 96.48% of Web-based, 95.87% of Brute Force and 99.34% of
Spoofing validation flows to Benign. Its low benign false-alert rate therefore
does not indicate reliable security detection: many attacks pass unnoticed.
Benign precision is only 51.85%.

## The local experiment

Starting from a copy of the saved round-20 model, each client trains once on its
own assigned examples. Nothing is written back into the cloud run or checkpoint.
The experiment keeps the existing LayerNorm architecture, square-root weighted
cross-entropy with the original batch denominator, optimizer settings, full
participation and row-count aggregation weights. Each client's Adam optimizer
is freshly initialized, matching the cloud recipe. The diagnostic round uses
the round-21 training seeds.

The SAME 20 trained client models supply two comparisons:

1. **Parameter averaging:** ordinary FedAvg, producing one small network.
2. **Probability averaging:** each client network predicts class probabilities;
   those probabilities are averaged with the same client row-count weights.

The second method is a diagnostic ensemble, not a replacement edge model. It
requires 20 network evaluations per example, rather than one. No physical edge
latency, memory, or energy was measured.

One diagnostic execution processes 2,000,000 training examples and performs
3,917 optimizer updates. It runs locally on CPU, not as a CUDA continuation.
CPU inference from the starting checkpoint exactly reproduces all saved GPU
endpoint validation metrics. Repeating the diagnostic locally produced identical
observations, including client metrics and both aggregation results. The source
archive remained unchanged; the test split was not opened.

## Averaging predictions does not recover the missing classes

These scores use the entire natural validation population of 2,059,284 rows.
Neither a balanced validation subset nor a favourable intermediate checkpoint
is used for these headline metrics.

| Model | Macro-F1 | Benign false alerts |
|---|---:|---:|
| Saved global model before the diagnostic | 0.400636 | 0.4110% |
| FedAvg after the diagnostic round | 0.404050 | 0.4412% |
| Probability ensemble of the same clients | 0.390761 | 0.0210% |

All entries below are validation recall percentages.

| Class | Saved global model | FedAvg after one round | Probability ensemble |
|---|---:|---:|---:|
| Benign | 99.5890 | 99.5588 | 99.9790 |
| DDoS | 99.0451 | 99.0236 | 99.0577 |
| DoS | 17.5039 | 18.0550 | 20.0071 |
| Recon | 21.8077 | 22.1136 | 14.2483 |
| Web-based | 0.0000 | 0.0000 | 0.0000 |
| Brute Force | 0.0000 | 0.0000 | 0.0000 |
| Spoofing | 0.1624 | 0.8610 | 0.0000 |
| Mirai | 98.7806 | 98.8566 | 98.5624 |

FedAvg improves slightly, without repairing the rare-class failures. The
probability ensemble reduces false alerts by becoming more conservative but
misses all three target attack categories. It is worse in macro-F1 despite its
extra inference work. This rules out this simple row-weighted ensemble as the
immediate remedy at this endpoint; it does not rule out every ensemble or every
federated method.

## What individual clients learn

All local models see the same validation probe: every Web-based, Brute Force
and Spoofing row, plus 2,000 seeded examples from each other class. Thus rare
recalls below cover the full validation class, but benign false alerts use only
2,000 benign rows. Precision and macro-F1 on this altered class mixture are
deliberately not reported. Client identifiers are zero-based virtual clients,
not identifiable IoT devices.

| Client | Training weight in FedAvg | Web recall | Brute Force recall | Spoofing recall | Benign probe false alerts |
|---|---:|---:|---:|---:|---:|
| 0 | 4.40% | 10.51% | 0.00% | 0.00% | 0.70% |
| 1 | 1.39% | 0.00% | 24.92% | 0.05% | 0.25% |
| 10 | 3.75% | 0.00% | 27.34% | 0.02% | 0.45% |
| 5 | 3.47% | 0.00% | 27.10% | 70.20% | 71.00% |
| 8 | 3.27% | 0.00% | 13.86% | 98.46% | 100.00% |
| 11 | 4.50% | 0.00% | 44.00% | 80.92% | 99.95% |

These are illustrative clients selected after inspection, not validation-selected
deployment candidates. Conditional recall and training diagnostics for all 20
clients are in the receipt.

For Web-based, only three clients obtain any correct validation predictions.
Even choosing a correct client separately for every example using its TRUE label
would recover only 10.67% of the class. That oracle calculation is not deployable;
it shows how little correct local argmax signal exists after this round. The best
Web-based client's own-training recall is also only 8.65%.

Brute Force is more mixed. Clients 1 and 10 recover roughly one quarter of the
validation class while producing few false alerts on the benign probe. Yet both
global aggregation methods have zero Brute Force recall. Their row weights are
only 1.39% and 3.75%, respectively. This is evidence that some local signal does
not survive pooling, but it does not prove that simply increasing their weights
would produce a good eight-class model.

Spoofing has strong local recall in several clients, but those clients also
misclassify most or all benign probe rows. For example, client 8's 98.46%
Spoofing recall comes with 100% benign false alerts. Calling this a collection
of good detectors destroyed solely by parameter averaging would be misleading.
Local class bias, weak rare-class learning and aggregation all remain relevant.

## The next bounded experiment

Do not expand to 5M, buy a larger GPU, deploy the ensemble, or repeat a broad
loss-weight sweep on the strength of these results. First run the previously
planned training-fit diagnostic locally:

1. Select a fixed, seeded, class-balanced panel ONLY from the 2M training split,
   with all eight classes represented. Keep the 39 features, train-fitted scaler,
   architecture and label mapping unchanged. Record every selected row.
2. Give the existing small network a fixed, modest training budget and examine
   per-class training recall. This tests fitting ability on an easier balanced
   panel, not performance under natural attack prevalence. Do not use validation
   to choose the panel, stopping point or hyperparameters.
3. Compare a modest nonlinear tabular baseline on the same panel. If it fits
   substantially better, investigate representation or optimization in the small
   network before more federated scaling. If both struggle, investigate feature
   overlap and fit settings without claiming the classes are inherently
   indistinguishable.
4. Only after those checks, specify one controlled full-cohort intervention and
   evaluate it against the unchanged reference on natural validation prevalence,
   with all eight recalls and benign false alerts. Improved panel training recall
   alone is not enough to authorize a large cloud experiment.

This is a diagnostic branch in the reasoning, not a promised solution or a
production sampling policy. Earlier stronger class weights, shorter local
updates and loss-denominator controls have already been tried; rerunning them
without a new distinguishing hypothesis would add little evidence. No new cloud
training is required to answer the immediate training-fit question.

## Reproduction and evidence

Implementation: `src/eval/official_aggregation_probe.py`.
Tests: `tests/test_official_aggregation_probe.py` cover deterministic sampling,
probability validation and weighting, absent-class reporting, one-client
equivalence, repeatability, unchanged starting weights and exact work counters.
All six new tests and the full 112-test local workspace suite passed. These are
local checks, not a claim of a new GitHub CI run.

Run locally from the repository root; choose a new output filename on each run:

```bash
.venv/bin/python -m src.eval.official_aggregation_probe \
  --data-root outputs/official39-packed-2m-v1 \
  --archive /Users/haydn/Downloads/tradecraft-official39-controlled-noniid-backup.tgz \
  --member-root outputs/official39-controlled-noniid-2m-v1/dirichlet-seed7 \
  --output outputs/official39-aggregation-probe-replay.json
```

Canonical local receipt: `outputs/official39-aggregation-probe-v2.json`.
It includes full-population confusion matrices and per-class precision/recall/F1,
all client diagnostics, probe row indices, package versions and provenance.
Its SHA-256 is
`22e1154abc8588ad9fa98317095ca3de81c437696eea704c3ce45852ed43737d`.
The input archive SHA-256 is
`c5b43c8d7fbdefde543b94b364c407707abe9655df081de7d31c014e55473152`.
Raw data, checkpoints and generated receipts stay outside version control.

The local probe used PyTorch 2.14.0, NumPy 2.5.3 and Flower 1.38.0. It is a
single-seed CPU mechanism experiment with exact local replay, not proof that a
GPU round 21 would have identical updates. All conclusions concern the
duplicate-grouped within-collection split, not unseen capture sessions or
physical devices. The held-out test split remains sealed.
