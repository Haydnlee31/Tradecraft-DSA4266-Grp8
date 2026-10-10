# Official39 final results and teammate handoff

Updated 11 October 2026. The official39 extension is complete through final
evaluation, research decision integration and this handoff. All 27 frozen
models completed the final test. Centralized light remains the strongest
predeclared reference, but rare-attack failures and benign false alerts prevent
these results from supporting operational deployment. No more cloud training
or parameter search is planned for this experiment.

This is a useful negative result: additional data, compute and several targeted
controls improved parts of the comparison without solving the security trade-off.
It does not establish that federated learning generally fails or that no better
architecture is possible. It establishes the limits of the models and conditions
actually tested here.

## What the comparison covers

The extension uses the downloaded 39-feature release, eight classes and a
duplicate-grouped within-collection split. Every final model used two million
training rows and the same 5,096-parameter LayerNorm network, at the fixed
step-20 endpoint. Centralized light, IID federated light and controlled non-IID
federated light each have training seeds 7, 17 and 27. The federated experiment
is a simulation with 20 virtual clients; the client partition seed is fixed.

The nine full39 models are the primary references. Eighteen separately trained
Number-only and Number-plus-Tot-sum masks are sensitivity controls. They retain
the same input width and parameter count. No best seed, validation-best checkpoint,
new ensemble or test-selected threshold replaces these predeclared candidates.

The official39 extension has no matched heavy model. It must not be combined
with the original 46-feature heavy/light/federated study into one leaderboard,
or used to claim that a light model replaces a heavy one. There are no measured
physical-edge results, per-device vulnerability conclusions or privacy guarantees.

## Final test findings

The common primary population has **2,040,729 rows**. The preparation removed
20,135 exact model-input matches to training or validation under any mask, while
preserving all eight classes. Every one of the 2,437 Web-based and 1,310 Brute
Force test rows remains. The original 2,060,864-row population is also reported
as a secondary view from the same predictions; no original rows were deleted.

The table uses equal-weight means across three training seeds. SD is the sample
standard deviation across those seeds, not uncertainty across independent
datasets, devices or client partitions. False-alert rate means the fraction
of truly benign rows predicted as attacks, not the fraction of alerts that are false.

| Full39 reference | Primary macro-F1 mean and SD | Mean benign false alerts | Original-population macro-F1 mean |
| --- | ---: | ---: | ---: |
| Centralized light | 0.65673 ± 0.00091 | 17.49% | 0.65671 |
| IID federated light | 0.61169 ± 0.00349 | 19.08% | 0.61176 |
| Controlled non-IID light | 0.40494 ± 0.00328 | 0.25% | 0.40390 |

Both populations support the same main conclusion. Scores also resemble the
preceding validation findings, but the test overlap rule excludes matches to
validation as well as training. Small validation-to-test differences cannot be
interpreted as pure generalization effects on identical populations.

### Recall for every class

Recall is the fraction of a true class assigned its correct category. These are
three-seed means on the primary population; zeros for non-IID Web-based and Brute
Force are exact in every seed, not rounded small detections.

| True class | Support per model | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: | ---: |
| Benign | 109,133 | 82.51% | 80.92% | 99.75% |
| DDoS | 1,203,157 | 84.80% | 84.70% | 98.77% |
| DoS | 364,574 | 70.74% | 67.23% | 21.01% |
| Recon | 68,363 | 73.93% | 72.73% | 19.83% |
| Web-based | 2,437 | 8.00% | 0.75% | 0.00% |
| Brute Force | 1,310 | 26.29% | 25.34% | 0.00% |
| Spoofing | 45,471 | 71.04% | 57.42% | 0.93% |
| Mirai | 246,284 | 99.54% | 99.41% | 98.81% |

Centralized light correctly labels only about eight of every hundred Web-based
test attacks, while flagging about seventeen of every hundred benign rows as
attacks. These are empirical test proportions, not forecasts of live traffic.

Non-IID looks quiet, but calls **97.55% of Web-based attacks, 96.92% of Brute Force
attacks and 98.29% of Spoofing attacks benign**, averaged across seeds. A wrong
attack label can still raise an alert; a benign prediction can suppress one.
The [generated appendix](final-test-tables.md) separates those error counts for
every full39 model and attack category. Low false alerts alone are not success.

### Feature masking did not fix the trade-off

Number-only masking reduces centralized macro-F1 in every seed. Both masks
reduce IID macro-F1 in every seed. Joint masking changes centralized mean F1
from 0.65673 to 0.65845, a small sensitivity result with mixed per-seed changes,
not grounds to select a new winner after opening the test.

Non-IID joint masking reaches mean F1 0.40740 versus 0.40494 for full39, but
the paired gain is dominated by exploratory seed 7. Web-based and Brute Force
recall remain exactly zero in all nine non-IID arm/seed combinations. Its mean
false-alert rate also rises from 0.25% to 0.41%. The masked models remain controls.

## How the experiments led to this conclusion

The [detailed experiment ledger](decision-summary.md#what-each-experiment-contributed)
records the early motivations, measurements and unsuccessful alternatives.
The full sequence has three parts:

1. **Establish reliable comparisons.** Data auditing exposed the 39-versus-46
   feature mismatch. Recovery pilots, shared preprocessing and fixed budgets
   established reproducible centralized and simulated federated lanes.
   LayerNorm addressed part of the early non-IID collapse, but not rare-class coverage.
2. **Diagnose the remaining weaknesses.** Weighting, bounded local updates,
   denominator controls, 500k-to-2M scaling, balanced-fit controls and related
   diagnostics separated several plausible explanations. Improvements in fit
   or aggregate scores did not provide satisfactory rare-class detection and
   alert burden together. Scaling at 20 passes also increases training work;
   it is not evidence that extra unique rows alone caused the improvement.
3. **Explain and test the frozen models.** Validation explanations and feature
   redundancy motivated a paired mask experiment. Three-seed confirmation did
   not establish a reliable fix. CUDA validation replay reproduced all 27
   historical endpoints; a label-blind overlap audit fixed the test population;
   one final evaluation measured every frozen candidate on both populations.

The compute helped distinguish reproducible weaknesses from implementation
errors. It did not turn additional GPU spending into a guaranteed better model.

## Explanations and security policy hypotheses

The existing [219-case explanation study](explanation-study.md) concerns selected
validation rows, not these final test predictions. Its method is
**background-averaged integrated gradients**, not exact conditional Shapley
values or a newly run SHAP-library analysis. Numerical checks pass for its seven
models: centralized and IID at three seeds, and non-IID at seed 7 only.

Number leads the balanced-core rankings; protocol, flag and size features also
recur. The [background and redundancy audit](explanation-sensitivity.md) shows
that reference choice and correlated features materially affect contributions.
Number may encode scenario/windowing information; the mask results neither
prove that all useful signal is artificial nor demonstrate shortcut-free learning.
The later three-seed non-IID prediction results do not expand the explanation
study into three-seed non-IID attribution evidence.

| Evidence | Human-review hypothesis | Unsupported action |
| --- | --- | --- |
| Rare attacks frequently receive benign predictions | Keep independent authentication and web/application alerts visible when the model says benign | Let this classifier automatically clear or suppress those alerts |
| Protocol, flag and size features recur but have correlated contributions | Review them with service and flow context when investigating an alert | Turn one attribution rank into a causal firewall rule or numerical threshold |
| Number is influential and associated with scenario construction | Audit the feature extractor and evaluate on independently collected traffic in future work | Treat Number values as attack definitions or make device-specific vulnerability claims |

These are investigation hypotheses, not tested operational policies. The
[research decision view](final-test-decision.json) displays all class metrics,
per-seed error counts and scope warnings, with automatic allow/block and model
promotion disabled. It deliberately does not feed this test cohort into the
legacy46 validation decision engine or invent deployment gates after seeing scores.

## Evidence and reproducibility

The [portable final evidence](final-test-results.json) contains the 62 aggregate
JSON documents from the reviewed backup: final cases, validation-replay cases,
sessions, recovery journals and frozen plan/preparation receipts. It contains
no traffic rows, model weights, executable archive code or per-row predictions.
Preserving each JSON document's original order and formatting allows its
archived byte hash to be reconstructed and checked.

The auditor verifies all 27 final model-state identities against the successful
CUDA replay, fixed candidate membership, source/runtime identities, population
supports and recovery journals. It recomputes 108 validation/test metric views
from confusion counts, then both test summaries and paired comparisons. This
is a provenance/count audit, not new inference or a fresh dataset inspection.

The closeout passes 325 tests in a clean repository snapshot, including 12 new
tests for provenance, missing/tampered results, per-class metrics, output
preservation and the non-operational decision boundary. The audit also runs
with Python site packages disabled, confirming it needs no training dependencies.
Unrelated local work was excluded from this verification and left untouched.

Run these from the repository root with Python 3.11 or newer. They require only
the standard library, checked-in source and JSON; no data download or GPU:

```bash
python scripts/official39_final_report.py audit
python scripts/official39_final_report.py decision
python scripts/official39_final_report.py tables
```

For a quieter file-based handoff, choose fresh ignored destinations:

```bash
python scripts/official39_final_report.py decision --output outputs/official39-final-review/decision.json
python scripts/official39_final_report.py tables --output outputs/official39-final-review/tables.md
```

Rebuild the portable evidence from the private backup, without extracting it:

```bash
python scripts/official39_final_report.py build --archive /path/to/tradecraft-official39-final-test-backup.tgz --output outputs/official39-final-review/evidence.json
python scripts/official39_final_report.py audit --evidence outputs/official39-final-review/evidence.json
```

Existing output files are protected. The reviewed archive SHA-256 is
`91e240ffe1bbc07e2a4573a03d1112bd600c6bb8148ff49b01f97e2ca416431e`.
The final completion receipt SHA-256 is
`ca8b2679e5b8255737fe6b4bfac0d9a75e4a2cba3d054ea370639a51a0a976d1`.
The original design and preparation receipts keep their historical flags;
the final execution receipt records the subsequent approved test evaluation.

## Completion and remaining team work

| Roadmap stage | Status | Deliverable |
| --- | --- | --- |
| 6 Final test evaluation | Complete | All 27 frozen models and both populations audited |
| 7 Results and research decision integration | Complete | Portable evidence, all-class tables, explicit error risks and separate research decision view |
| 8 Official39 report and teammate handoff | Complete | This report, detailed appendix, links to prior experiments and offline reproduction commands |

Teammates should use branch `experiment/full-data-scaling`. Their remaining
work is to incorporate this extension into the overall submission and review
any merge into the shared branch. The original 46-feature comparison remains a separate
section. This closeout does not claim that the entire semester manuscript or
all teammates' work is finished, and does not merge branches automatically.

Keep RONIN stopped. Preserve the private archives and raw data outside Git.
The final holdout is now exposed; do not use it to choose another architecture,
threshold, seed or mask and then report it as an untouched final test. A future
research extension needs a new protocol and independent evaluation evidence.
