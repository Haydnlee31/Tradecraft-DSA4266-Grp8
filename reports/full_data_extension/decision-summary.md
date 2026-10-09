# Official 39 feature findings and decision recommendations

The cloud and local experiments now support a clear research conclusion: a
larger training budget helps the centralized and IID light networks, but none
of the investigated changes resolves the combination of missed rare attacks,
incorrect attack labels and benign false alerts. Close this tuning sequence.
Keep the full-cohort references for analysis, not deployment. Leave the cloud
machine stopped and the official-39 final test sealed.

This is a useful result for Tradecraft's trade-off question, even without a new
winning architecture. It shows where the small federated model loses useful
detection, and why an aggregate score or quiet alert stream can give the wrong
impression. It does not establish that federated learning in general cannot
work, or that these 39 features cannot support better models.

## Which comparisons answer the main question

All entries below use the same 2,059,284 validation flows and final-at-budget
checkpoints, not each run's best intermediate checkpoint. Centralized and IID
values average training seeds 7, 17 and 27. Seed 7 motivated the comparison;
17 and 27 were prospective confirmations. SD describes variation across these
seeds, not uncertainty across independent datasets or client populations.

| Setting | Training rows | Passes or rounds | Processed examples | Macro-F1 mean and sample SD | Benign false alerts |
|---|---:|---:|---:|---:|---:|
| Centralized light | 500k | 20 | 10M | 0.63223 ± 0.00568 | 19.93% |
| Centralized light | 2M | 5 | 10M | 0.63515 ± 0.00216 | 19.68% |
| Centralized light | 2M | 20 | 40M | 0.65572 ± 0.00102 | 17.55% |
| IID federated light | 500k | 20 | 10M | 0.58526 ± 0.00450 | 22.31% |
| IID federated light | 2M | 5 | 10M | 0.58257 ± 0.00560 | 21.90% |
| IID federated light | 2M | 20 | 40M | 0.61395 ± 0.00413 | 19.07% |
| Controlled non-IID, seed 7 only | 500k | 20 | 10M | 0.36683; SD not estimated | 0.0904% |
| Controlled non-IID, seed 7 only | 2M | 5 | 10M | 0.31447; SD not estimated | 0.0904% |
| Controlled non-IID, seed 7 only | 2M | 20 | 40M | 0.40064; SD not estimated | 0.4110% |

At equal work, the larger dataset does not consistently help: centralized
macro-F1 rises in two seeds and falls in one, while IID falls in all three.
At 20 passes, both improve in all three seeds, but process four times as many
examples. That supports the larger training setup and budget, not a claim that
extra unique rows alone caused the gain. Cohort-specific scalers, class weights
and local epoch lengths also change. Equal processed examples does not mean
equal communication, wall time, aggregation rounds or exactly equal updates.

The controlled non-IID comparison preserves the original rows' client ownership
and closely preserves class allocations as data grows. It removes the earlier
partition-redraw confound. It remains one seed and one simulated heterogeneous
partition, not equally strong confirmation. IID is a diagnostic control, not
a stand-in for the intended non-IID scenario. See the [scaling findings](scaling-findings-and-controlled-noniid.md)
and [controlled non-IID results](aggregation-mechanism-probe.md).

## Detection failures that the averages hide

These are class recalls at the 2M, 20-pass endpoints. Centralized and IID columns
are three-seed means; non-IID is the single controlled seed. Recall asks, “Of the
flows that really belong to this class, how many receive the correct label?”

| True class | Validation support | Centralized light | IID light | Controlled non-IID light |
|---|---:|---:|---:|---:|
| Benign | 109,482 | 82.45% | 80.93% | 99.59% |
| DDoS | 1,213,890 | 84.50% | 84.44% | 99.05% |
| DoS | 373,093 | 71.06% | 67.54% | 17.50% |
| Recon | 68,008 | 73.80% | 72.76% | 21.81% |
| Web-based | 2,475 | 6.87% | 0.85% | 0.00% |
| Brute Force | 1,284 | 27.65% | 27.10% | 0.00% |
| Spoofing | 44,949 | 70.79% | 57.52% | 0.16% |
| Mirai | 246,103 | 99.58% | 99.42% | 98.78% |

There are two different ways to get an attack wrong. Calling it Benign means
missing an attack; calling it another attack category can produce the wrong
response even if it still raises an alert. These percentages partition each
true class and sum to 100%, allowing for rounding.

| Setting and true class | Correct category | Predicted Benign | Wrong attack category |
|---|---:|---:|---:|
| Centralized Web-based | 6.87% | 31.22% | 61.91% |
| IID Web-based | 0.85% | 29.94% | 69.21% |
| Non-IID Web-based | 0.00% | 96.48% | 3.52% |
| Centralized Brute Force | 27.65% | 34.03% | 38.32% |
| IID Brute Force | 27.10% | 33.49% | 39.41% |
| Non-IID Brute Force | 0.00% | 95.87% | 4.13% |
| Non-IID Spoofing | 0.16% | 99.34% | 0.50% |

The non-IID model is quiet partly because it calls these attacks Benign. Its
0.4110% benign false-alert rate is therefore not evidence of a safe detector.
Conversely, centralized light labels roughly 18 of every 100 genuinely benign
validation flows as attacks. This is not the percentage of all alerts that are
false; that also depends on attack prevalence. Validation proportions are not
an operational traffic forecast.

## What each experiment contributed

Early cloud controls are exploratory seed-7 results. Later panel controls
average three training seeds unless stated otherwise. Recovery runs check
implementation compatibility; they are not model-quality experiments.

| Stage and motivation | What it established | Why the investigation moved on |
|---|---|---|
| Data audit and GPU recovery: can this release be used safely? | This release has 39 features, not the legacy 46. Exact duplicate groups were kept apart across splits; bounded loading and interrupted/resumed training were verified. | Establish separate references rather than reuse incompatible scalers or models. |
| Normalization: does BatchNorm contribute to non-IID collapse? | LayerNorm raised non-IID macro-F1 from 0.20812 to 0.36683 and recovered Mirai recall, but Web, Brute Force and Spoofing remained missed. | Keep LayerNorm and investigate rare-class learning. |
| Stronger class weights: can rare attacks get more influence? | Centralized Web recall reached 50.67%, but precision was 3.57% and benign false alerts reached 26.98%. Non-IID still missed the three problematic classes. | More rare predictions were not necessarily useful predictions; inspect client updates. |
| Five-batch client cap: does frequent averaging help? | At matched optimizer updates, IID macro-F1 fell to 0.55444 and non-IID rose to 0.38229. Both had zero Web and Brute Force recall; exposure replay confirmed all rare training rows were visited. | Neither complete row omission nor long local epochs alone explained the failure. |
| Fixed loss denominator: does batch-dependent weighting explain the discrepancy? | Centralized, IID and non-IID macro-F1 changed by about +0.172, −0.025 and +0.016 percentage points. Missing non-IID classes did not recover. | Retain the original denominator and test data size with explicit work budgets. |
| 500k versus 2M plus seed confirmation: does scaling help? | Gains were consistent at 20 passes, not at equal examples. Rare-class failures persisted. | Separate data scaling from client-mixture changes and inspect rare-class training fit. |
| Controlled non-IID scaling and aggregation: does pooling destroy good detectors? | The 2M endpoint reached 0.40064. One extra FedAvg round reached 0.40405; averaging the same clients' probabilities reached only 0.39076, still missing Web, Brute Force and Spoofing. Some local specialists also produced extreme false alarms. | Parameter averaging alone was not a sufficient explanation; test fitting ability on a balanced panel. |
| Balanced fit, duration, dropout and width: can the model learn these classes? | Longer or wider fits improved training scores. Natural-validation macro-F1 was 0.56827 for small dropout, 0.56857 for wider dropout, and 0.55769 for wider without dropout, with 39.52% false alerts. A fixed tree scored 0.61412 with 28.17% false alerts. | Better training fit did not supply a satisfactory replacement; examine decisions and alert burden. |
| Alert-budget diagnostic: can a gate retain useful rare detections? | At an illustrative 1% benign-alert cap, the small-dropout control retained 11.49% Web recall and 34.79% Brute Force recall, with macro-F1 0.53481. | Thresholds were selected and assessed on the same validation flows, not independently validated. |
| Matched-work majority diversity: is repeating a tiny panel the main problem? | More unique majority-class rows changed macro-F1 from 0.56827 to 0.57232, but increased mean benign false alerts from 29.52% to 30.85%. | This intervention did not fix the trade-off; inspect one fixed prior adjustment without retraining. |
| Fixed training-prior correction: can natural class proportions correct overprediction? | The diversity model reached macro-F1 0.61148 and 14.54% false alerts, but Web recall fell to 4.48%, Brute Force to 21.91% and DoS to 28.66%. | Close panel tuning: improving aggregate metrics by sacrificing classes is not the required solution. |

Detailed methods and limits: [client controls](client-diagnostic.md),
[bounded updates](bounded-updates.md), [loss denominator](loss-denominator-control.md),
[training fit](training-fit-findings.md), [panel validation](panel-validation-findings.md),
[alert budgets](alert-budget-findings.md), [diversity](diversity-findings.md), and
[prior correction](prior-correction-findings.md). Earlier plans retain their
original pre-run wording; completed findings and this ledger give current status.

Panel experiments are centrally pooled diagnostics, not federated improvements.
Their training exposure, class balance and budgets differ from full-cohort
references. The tree is single-seed with unmatched model family and work.
The probability ensemble evaluates 20 models per flow. These experiments must
not be combined into an unqualified fair-compute architecture leaderboard.

## Recommendations for the decision layer

Keep two decisions separate. The research reference is the unchanged full-cohort
comparison, with centralized light providing the strongest aggregate reference
among the evaluated scaled light settings. Deployment remains unapproved:
these results do not justify automated allow/block actions or reliable eight-class
attack response. This is a research conclusion, not a newly invented numerical gate.

The official-39 summary must not silently enter the legacy 46-feature decision
engine. That engine's illustrative recall and size gates are unchanged, and it
has no false-alert ceiling. There is no completed, matched official-39 heavy
comparison here: the wider panel network is not that missing heavy lane. Do
not claim the extension proves a small model can replace a heavy one.

For a comparison or demo, retain each candidate's study, split, endpoint, seed
coverage and partition context. Display all eight recalls, benign false alerts,
and the split between attacks missed as Benign and attacks given the wrong
category. Keep the non-IID coverage warning visible even when its false-alert
rate looks attractive. Do not substitute IID for non-IID or select the best seed.

The LayerNorm light architecture has 5,096 parameters: 20,384 bytes of float32
parameter values, about 0.0194 MiB. Centralized and federated training produce
the same-sized inference architecture. Federated training does not itself make
the model smaller or establish a privacy guarantee. Parameter storage excludes
runtime allocations, activations and framework overhead. No physical-edge
latency, energy or deployment memory was measured in this extension.

## Next analysis and the stopping rule for compute

The [full 219-case explanation study](explanation-study.md) now passes numerical
QA for all seven frozen references, including controlled non-IID. It preserves
the balanced core separately from the targeted supplement and compares the
centralized/IID seeds. Number ranks first in every balanced-core model summary,
but other feature rankings and error explanations vary across seeds. Non-IID
rare-attack misses show recurring Number and ack_flag_number contributions to
Benign-minus-attack margins on the selected groups; this is a single-model
diagnostic, not a validated detection rule. Earlier [Monte Carlo warnings](explanation-convergence.md)
and all measured cloud scores remain unchanged.

Next, check important feature definitions, related-feature behavior and
sensitivity to a separately frozen training-only background on matched cases.
The present background has no Web-based or Brute Force examples. Numerical
convergence does not resolve that reference sensitivity. Keep this audit local
and bounded before proposing security-monitoring hypotheses.

The question is what feature patterns distinguish a correct prediction from a
mistake. SHAP describes model behavior, not model quality, vulnerable device
identity or attack causation. Feature-level recommendations should be monitoring
hypotheses for human review, not automatically generated firewall rules.
Explanations cannot recover identifiers absent from the release.

Do not launch 5M/full-release training, a stronger GPU or another broad sweep
merely because credits remain. Further compute needs a distinct hypothesis,
a frozen comparison and a success criterion covering rare-class detection and
false-alert cost. Before final evaluation, separately agree the candidates,
operating requirements and threshold-selection protocol; freeze them before
accessing the held-out test. Validation has been reused extensively, and even
a within-collection test cannot prove unseen-session or real-device generalization.

## Reproduce the evidence audit

The [published evidence](decision-evidence.json) contains 72 validation endpoints,
not 72 independent training runs. Some are prefixes, exact bridges or repeated
scoring of the same weights. It retains confusion matrices, all-class metrics,
error counts, declared work, source hashes and separate comparison groups.
Single-seed standard deviation is null, not zero.

This local command needs only the checked-in JSON and Python standard library,
not the dataset, checkpoints or a GPU:

```bash
python -m src.eval.official_evidence --audit reports/full_data_extension/decision-evidence.json
```

The [source plan](decision-evidence-plan.json) pins exact JSON paths and archive
members. Rebuilding also requires the existing ignored local receipts and six
original cloud archives in Downloads. Use a fresh output file:

```bash
.venv/bin/python -m src.eval.official_evidence \
  --plan reports/full_data_extension/decision-evidence-plan.json \
  --archive-dir /Users/haydn/Downloads \
  --output outputs/official39-decision-evidence-recheck.json
```

Rebuilding checks source hashes, explicit status/scope guards, class counts and
confusion-derived metrics. For archive-history endpoints it also rechecks work
counters. Scaling work comes from the previously verified frozen protocol.
The portable audit checks published counts and summaries; it does not re-run
training, authenticate the dataset publisher or re-verify every historical
checkpoint. Neither command opens test data or changes training, scoring or
deployment defaults.
