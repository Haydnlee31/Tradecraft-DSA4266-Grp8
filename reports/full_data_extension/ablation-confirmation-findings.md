# Completed feature-mask confirmation — 10 October 2026

The experiment worked; feature masking did not supply a reliable model fix.
All 18 prospective runs completed. Together with the nine exploratory seed-7
models, they show almost unchanged centralized performance, consistently worse
IID performance after masking, and no convincing repetition of the small seed-7
non-IID gain. Keep full39 as the research reference; do not promote a mask.

This closes the bounded feature-mask experiment. It does not establish that
federated learning generally cannot work, or that a better model is impossible.
It does establish that these interventions did not solve this setup's failures.
Keep RONIN stopped while preparing the final research evaluation.

## What was checked

The original [confirmation plan](ablation-confirmation-plan.json) is unchanged.
Seeds 17 and 27 each completed all nine lane/arm combinations. Each model used
the same 2M training cohort, 20 passes/rounds, 5,096-parameter LayerNorm network,
batch size 512, original square-root-weighted loss and exact client assignments.
Initial weights match across arms within each lane/seed and differ between the
two new seeds. Partition seed remains 7: this is training-seed replication,
not replication across different client mixtures.

The archive review verified all 18 new last/best checkpoint pairs, completion
receipts, paired initialization, full client participation, fixed work, and 86
training-source hashes. It reproduced the cloud summary exactly and independently
recomputed 1,080 full/panel metric views across all 27 scientific histories.
The new runs processed 720M repeated training examples and 1,409,280 optimizer
updates, not 720M unique flows. Recorded training-plus-validation time sums to
1.540 hours, excluding setup/checkpoint I/O and idle machine time; it is not a
billing measurement. No model was trained or scored in the local review, and no
official39 test evaluation is recorded in these runs.

The confirmation archive SHA256 is
`9e7640f4d8953ce913dc3f99edfc04c2bc95af30e6b7c9abede0eec0f7c636eb`.
The [portable evidence](ablation-confirmation-results.json) retains all 27 final
endpoints, all eight class metrics, confusion matrices, separate error counts,
work, timings, archive members and checkpoint hashes. Its JSON-only auditor
checks provenance and metric consistency; it does not repeat the checkpoint audit.

## Same endpoint, same validation population

Every score below uses the final step-20 checkpoint on the shared 2,047,805-row
panel. It excludes exact float32 training-input matches under any of the three
arms; all 2,475 Web-based and 1,284 Brute Force validation cases remain. Historical
all-validation results on 2,059,284 rows remain separate and unchanged.

Full39 keeps all inputs. Number masks that standardized column to its training
mean; joint masks Number and Tot sum. All arms were retrained, not merely masked
at inference. The input width and parameter count remain identical.

| Setting | Mask | Macro-F1 mean ± sample SD | Mean benign false alerts |
|---|---|---:|---:|
| Centralized light | None | 0.65569 ± 0.00103 | 17.55% |
| Centralized light | Number | 0.65139 ± 0.00534 | 19.18% |
| Centralized light | Joint | 0.65582 ± 0.00147 | 17.82% |
| IID federated light | None | 0.61386 ± 0.00412 | 19.07% |
| IID federated light | Number | 0.60856 ± 0.00369 | 18.95% |
| IID federated light | Joint | 0.60540 ± 0.00330 | 19.10% |
| Controlled non-IID | None | 0.40492 ± 0.00347 | 0.26% |
| Controlled non-IID | Number | 0.40510 ± 0.00242 | 0.18% |
| Controlled non-IID | Joint | 0.40725 ± 0.00201 | 0.44% |

False alerts are the fraction of genuinely benign rows predicted as attacks,
not the fraction of all alerts that are false. Means weight training seeds
equally; they do not pool confusion matrices or select the best seed.

## Did the first-seed effect repeat?

Differences below are masked minus the same-seed full39 reference, in macro-F1
percentage points (0.01 macro-F1 equals one percentage point).

| Setting / mask | Seed 7: exploratory | Seed 17: confirmation | Seed 27: confirmation | Three-seed mean ± SD | New-seed mean only |
|---|---:|---:|---:|---:|---:|
| Centralized / Number | −0.908 | −0.308 | −0.073 | −0.430 ± 0.431 | −0.190 |
| Centralized / Joint | −0.028 | +0.129 | −0.061 | +0.014 ± 0.102 | +0.034 |
| IID / Number | −0.296 | −0.915 | −0.378 | −0.530 ± 0.336 | −0.646 |
| IID / Joint | −0.638 | −1.373 | −0.527 | −0.846 ± 0.459 | −0.950 |
| Non-IID / Number | +0.107 | +0.109 | −0.163 | +0.018 ± 0.156 | −0.027 |
| Non-IID / Joint | +0.736 | −0.046 | +0.011 | +0.233 ± 0.436 | −0.018 |

The joint non-IID gain is driven by exploratory seed 7. The prospective average
is slightly negative, and joint masking increases benign false alerts in all
three seeds (+0.037, +0.168, +0.343 points). This is not a repeated improvement.
Number-only masking consistently lowers centralized F1 and increases its false
alerts; both masks consistently lower IID F1. Centralized joint-mask false-alert
changes have mixed signs, so the seed-7 increase must not be described as universal.
Three seeds cannot prove equivalence, significance or generalization.

## Class coverage is still the deciding weakness

These are mean recalls in percent; all seeds' precision/recall/F1 and supports
are retained in the evidence JSON. Rounded 0.00 can hide tiny nonzero values;
Web-based and Brute Force non-IID recalls are exactly zero in every arm/seed.

| Setting / mask | Benign | DDoS | DoS | Recon | Web-based | Brute Force | Spoofing | Mirai |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Centralized / None | 82.45 | 84.64 | 70.86 | 73.80 | 6.87 | 27.65 | 70.78 | 99.58 |
| Centralized / Number | 80.82 | 83.78 | 72.68 | 75.03 | 6.38 | 28.40 | 69.62 | 99.55 |
| Centralized / Joint | 82.18 | 84.63 | 70.82 | 74.27 | 5.78 | 28.32 | 74.40 | 99.56 |
| IID / None | 80.93 | 84.58 | 67.31 | 72.76 | 0.85 | 27.10 | 57.52 | 99.42 |
| IID / Number | 81.05 | 83.77 | 69.07 | 71.94 | 0.00 | 27.31 | 56.82 | 99.39 |
| IID / Joint | 80.90 | 83.60 | 69.11 | 71.98 | 0.00 | 27.28 | 55.43 | 99.30 |
| Non-IID / None | 99.74 | 98.75 | 20.81 | 19.89 | 0.00 | 0.00 | 0.94 | 98.80 |
| Non-IID / Number | 99.82 | 98.38 | 23.99 | 18.57 | 0.00 | 0.00 | 0.00 | 98.57 |
| Non-IID / Joint | 99.56 | 98.61 | 21.73 | 18.85 | 0.00 | 0.00 | 1.78 | 98.91 |

For the non-IID joint mask, mean attack-as-benign rates are 97.28% for Web-based,
97.14% for Brute Force and 89.89% for Spoofing. Full39 rates are 97.54%, 97.22%
and 98.34%, respectively. The Spoofing change mostly moves mistakes to other
attack categories, rather than yielding correct Spoofing classification.
These are different response risks; low false alerts alone cannot establish
useful detection. Centralized light, meanwhile, alerts on about 18 of every 100
benign validation rows and misses most correct rare-category labels.

## What this means for explanations and the decision layer

The earlier [explanation study](explanation-study.md) and
[redundancy audit](explanation-sensitivity.md) describe existing model behavior.
The present retraining experiment asks whether models can learn without explicit
access to selected information. A high attribution and little retraining loss
are compatible: the newly trained model can learn substitutes. The joint mask
breaks the demonstrated Number ratio routes, but removes legitimate size
information too; other proxies may remain. Neither shortcut-free learning nor
wholly artificial original performance has been established.

The practical decision is research-only comparison, not automatic allow/block:

| Decision | Supported action | What not to claim |
|---|---|---|
| Research reference | Keep unchanged centralized full39; show IID and non-IID separately, with all seeds and recalls | It is not a validated deployment winner |
| Federated coverage | Keep the rare-attack blind-spot warning visible even when false alerts are low | Quiet predictions do not imply safe benign traffic |
| Feature explanations | Treat protocol/flag/size patterns as monitoring hypotheses for human review | No per-device vulnerability, causal attack rule or firewall threshold |
| Further tuning | Close this bounded sequence and preserve unsuccessful controls | More GPU credits or a larger model are not demonstrated remedies |

Do not change the legacy 46-feature decision engine or its illustrative gates.
There is no matched heavy official39 lane, no measured physical-edge benchmark,
and no privacy guarantee. The extension supports a light-setting comparison,
not the claim that a light model can replace a heavy model.

## Final local handoff

The [final-evaluation protocol](final-evaluation.md) now fixes all nine full39
references and all eighteen sensitivity controls, their exact final checkpoints,
preprocessing and reporting. It introduces no fitted threshold, ensemble or new
training. Test inputs remain closed until explicitly approved; no test scoring
runner is supplied in this closeout. A validation replay and label-blind test
overlap audit must pass before one separately approved final evaluation.

The original cloud plans and training code remain untouched. Teammates can run
these checks with Python 3.11+ and the standard library, without a GPU:

```bash
python scripts/official39_closeout.py audit
python scripts/official39_closeout.py check-final-plan
```

To rebuild the portable JSON from the hash-pinned confirmation archive and the
published seed-7 evidence, use a fresh ignored destination:

```bash
python scripts/official39_closeout.py build --archive-dir /path/to/backups --output outputs/official39-confirmation-rebuild.json
```

The existing 72-endpoint [historical ledger](decision-evidence.json) remains a
separate all-validation/diagnostic record. Do not add 27 to 72 and describe the
sum as independent runs: several records reuse models or different metric views.
