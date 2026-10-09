# Background sensitivity and feature redundancy findings

The audit identifies a **strong dataset-construction shortcut concern**:
`Number`, the most influential feature in the preceding explanation study,
almost separates two sets of attack categories by taking values near 10 or
100. This association was measured on all two million training rows. It is
consistent with the published scenario-dependent aggregation procedure, but
does not prove the exact mechanism in this downloaded release.

Several input columns are also redundant. Their individual attributions can
be large while cancelling strongly when combined. The second training
background preserves Number's leading rank on the eight matched pilot cases,
but changes contribution magnitudes materially. These findings qualify the
earlier explanations; they do not change any model scores or establish a safe
security policy.

**Next, prepare a matched shortcut-ablation control**, including checks for
remaining proxies and train/validation overlap after the proposed feature
change. Do not enlarge the dataset or buy a stronger GPU on the basis of these
explanations. No training, feature removal, model promotion or test evaluation
was performed in this audit.

## Published definitions and release limits

The original paper describes Number as a packet-count feature. It distinguishes
flag values such as `ack_flag_number` from flag-count summaries, and presents
IPv, ARP and LLC as protocol indicators. Its processing section specifies
10-packet aggregation for benign traffic and several lower-volume scenarios,
and 100-packet aggregation for DDoS, DoS and Mirai scenarios.
[Paper, Table 4 and Section 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/).

Those definitions provide context, not an authenticated extractor for this
39-column release. The [official page's displayed statistics](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
describe a different feature layout, including Number values up to 15. The
current training pack reaches approximately 100. Keep the studies separate;
do not silently equate every column, unit or extraction step. No third-party
reimplementation was used as authority for the downloaded files.

## Number nearly identifies the scenario group

The audit read only the frozen two-million-row training cohort in chunks of
8,192 rows. It verified the packed-file and train-fitted scaler hashes and
reconstructed Number approximately from the standardized float32 values.
Values within 0.0001 of 10 or 100 were counted in those bins; all other values
were retained separately. The tolerance was fixed before reading the data.

| True training class | Rows | Number near 10 | Number near 100 | Other values |
|---|---:|---:|---:|---:|
| Benign | 106,148 | 99.9501% | 0% | 53 |
| DDoS | 1,178,561 | 0.0025% | 99.7180% | 3,295 |
| DoS | 362,423 | 0.0025% | 99.8099% | 680 |
| Recon | 66,277 | 99.9759% | 0% | 16 |
| Web-based | 2,391 | 99.9164% | 0% | 2 |
| Brute Force | 1,269 | 99.9212% | 0% | 1 |
| Spoofing | 44,095 | 99.9524% | 0% | 21 |
| Mirai | 238,836 | 0.0059% | 99.3975% | 1,425 |

A learner could use this feature to distinguish the mostly-100 group from
the mostly-10 group without learning a general attack mechanism. That is a
**shortcut hypothesis**, not evidence of train/test row leakage or proof that
the model's entire performance is artificial. Number does not distinguish
Benign from Web-based or Brute Force within the mostly-10 group; the existing
rare-attack failures remain important.

The previous comparisons still measure how these training settings perform
under the same within-collection feature construction. Their interpretation
must not expand to label-independent live traffic or unseen capture sessions.

## Redundant features and cancelling explanations

These relationships hold for every one of the two million **stored standardized
training vectors**, with maximum absolute residual exactly zero:

| Relationship in model inputs | Rows satisfying it |
|---|---:|
| IPv equals LLC | 2,000,000 / 2,000,000 |
| AVG equals Tot size | 2,000,000 / 2,000,000 |
| ARP equals negative IPv | 2,000,000 / 2,000,000 |

The last relationship is in standardized coordinates; it does not mean raw
protocol values are negative. The audit also found Pearson correlations of
0.9974 between fin_flag_number and fin_count, 0.9919 between syn_flag_number
and syn_count, and 0.9895 between rst_flag_number and rst_count. Correlation
does not make those flag pairs exact duplicates.

We summed the already saved signed attributions within two predeclared groups:
ARP/IPv/LLC and AVG/Tot size. Cancellation is
`1 − sum(abs(group sum)) / sum(sum(abs(member contributions)))`, over all
case/output combinations in each stratum. This explicitly distinguishes
adding signed contributions from adding their absolute importances.

Across the seven models' balanced cores, **64.3–86.8%** of the combined absolute
ARP/IPv/LLC contribution cancels within the group. The targeted supplements
show 61.5–85.1% cancellation. The AVG/Tot size group also exhibits cancellation,
but less consistently. A prominent individual protocol column is therefore
not evidence of three separate network mechanisms. Group sums are descriptive
algebra on the existing attributions, not grouped Shapley values, feature
ablation results or causal effects.

## One new background on exactly matched cases

The [protocol](explanation-sensitivity-plan.json) fixed seed 42663908 and drew
128 unique training rows uniformly, excluding the original background rows.
There was one draw, with no class-based selection or redraw. The model weights,
scaler, eight original pilot cases, two integration resolutions and all QA
thresholds stayed unchanged. Both backgrounds use equal weight per row.

| Class | Original background | New background |
|---|---:|---:|
| Benign | 12 | 7 |
| DDoS | 65 | 81 |
| DoS | 22 | 19 |
| Recon | 8 | 4 |
| Web-based | 0 | 0 |
| Brute Force | 0 | 0 |
| Spoofing | 6 | 3 |
| Mirai | 15 | 14 |

For the new background, **448/448 class-score and 29/29 error-margin checks**
pass at both resolutions across seven models. Featurewise resolution and
global reconstruction checks also pass. Frozen case logits and predictions
are unchanged. An independent run reproduced the full receipt and all eight
saved NPZ files byte-for-byte, including the training audit and new background.
Replay demonstrates reproducibility, not sampling uncertainty.

| Model | Relative featurewise change between backgrounds | Pilot own-true-score top-five overlap |
|---|---:|---:|
| Centralized seed 7 | 16.29% | 0.667 |
| Centralized seed 17 | 14.86% | 1.000 |
| Centralized seed 27 | 16.28% | 1.000 |
| IID seed 7 | 15.50% | 1.000 |
| IID seed 17 | 14.14% | 1.000 |
| IID seed 27 | 14.11% | 1.000 |
| Controlled non-IID seed 7 | 16.16% | 0.667 |

Relative change is mean absolute featurewise difference divided by the new
background's mean absolute attribution magnitude, over all eight cases and
eight outputs. Top-five overlap is intersection divided by union, ignoring
order. The ranking uses contributions to each case's true-class score, whether
correctly classified or not. These eight-case rankings must not be confused
with the prior 64-case balanced-core rankings.

Number remains first for every model under both pilot backgrounds. One of its
56 own-true-score contributions changes sign: centralized seed 7's Spoofing
case moves from -0.0162 to +0.0184 logit units, a small effect near zero. This
does not overturn the broader magnitude finding. On the controlled non-IID
pilot's missed Web-based, Brute Force and Spoofing cases, Number continues to
contribute positively to the Benign-minus-attack contrast under both references.

Changing the background changes what an explanation is relative to. The
14–16% shifts are **reference sensitivity, not failed numerical integration**.
Five models keep the same top-five set and two keep four of five features, but
two backgrounds and one case per class cannot establish broad robustness.
Both backgrounds still omit Web-based and Brute Force; that limitation has
not been solved.

## What should change next

Retain the current runs as the original full-feature references, with an
explicit scenario/windowing limitation. Do not delete columns from existing
artifacts, relabel the results as deployment validation or turn Number values
into blocking thresholds. The strongest evidence here changes the research
question: how much of the measured behavior survives when this suspected
shortcut is unavailable?

Prepare one matched, prospective Number-ablation control, preserving the
training cohort, fixed comparison endpoints, seed pairing, loss, optimizer
and compute budget. Before launching it, check whether packet-count information
is recoverable from other summaries, such as totals and averages, and audit
train/validation feature collisions after the proposed change. Removing the
explicit column alone may not remove the shortcut. That recoverability has
not been measured by this audit.

Keep duplicate-feature removal as a separate control if needed, so a score
change is not attributed to several simultaneous alterations. An ablation may
lower macro-F1 while producing a more credible comparison; it should also
report all class recalls and benign false alerts. Even a successful ablation
cannot prove real-device or unseen-session generalization. A label-independent
extraction validation would require additional data/extraction provenance,
not merely a larger GPU.

The next action is local protocol preparation. Keep RONIN stopped until that
control is ready. Broad architecture searches and further data scale-up remain
unjustified by this evidence.

## Evidence and reproduction

The [published results](explanation-sensitivity-results.json) contain every
training count, flagged correlation, identity residual, numerical check,
matched-case contribution and background comparison. The canonical raw receipt
in `outputs/official39-explanation-sensitivity-v1/receipt.json` has SHA256
`2aa71f763b1fb089ef35fe0098e2e29583a37c191db041cddc6a2be128fb329d`.
The independently replayed output lives in
`outputs/official39-explanation-sensitivity-replay-v1/`. Both are ignored by Git.

```bash
.venv/bin/python -m src.explain.official_sensitivity \
  --preparation-root outputs/official39-explanation-preparation-v2 \
  --convergence-root outputs/official39-explanation-convergence-v1 \
  --integration-root outputs/official39-explanation-integration-v1 \
  --study-root outputs/official39-explanation-study-v1 \
  --archive-root /Users/haydn/Downloads \
  --packed-root outputs/official39-packed-2m-v1 \
  --preparation-plan-path reports/full_data_extension/explanation-cpu-plan.json \
  --convergence-plan-path reports/full_data_extension/explanation-convergence-plan.json \
  --integration-plan-path reports/full_data_extension/explanation-integration-plan.json \
  --study-plan-path reports/full_data_extension/explanation-study-plan.json \
  --plan-path reports/full_data_extension/explanation-sensitivity-plan.json \
  --output outputs/official39-explanation-sensitivity-recheck
```

One run integrates 1,376,256 forward path rows and 11,010,048 output-gradient
rows, plus endpoint checks. The independent replay repeats that budget. No
validation feature-array scan or test access occurs; the eight validation cases
come from their previously pinned snapshot. Tests exercise scope guards,
sampling, chunked statistics, training-only reads, memmap cleanup, cancellation,
reference changes, end-to-end synthetic execution, replay and tamper rejection.
