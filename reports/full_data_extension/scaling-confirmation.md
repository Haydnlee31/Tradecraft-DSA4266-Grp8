# Scaling confirmation across training seeds

The next experiment checks whether the 2M cohort's improvements repeat under
different training randomness. It keeps the existing architecture, loss, data,
validation population, and client partition recipe fixed. The machine can remain
stopped while the handoff is prepared.

## Evidence motivating confirmation

At seed 7, centralized-light reached final macro-F1 0.65460 at 2M versus 0.62615
at 500k, with false alerts increasing from 17.01% to 18.00%. IID reached 0.61566
versus 0.58912, with false alerts declining from 21.29% to 19.16%. These endpoints
use 20 passes, so the larger cohort also receives four times the processed examples.
At equal processed examples, the 2M scores were 0.63648 centrally and 0.58708 under
IID. Web-based recall remained 6.55% centrally and zero under IID at 20 passes.

These findings justify confirmation, not a claim that scaling has solved the task.
The reviewed archive preserves both five-pass checkpoints and their exact history
prefixes in the completed runs. Its source hashes and work counters passed review.

## Frozen run matrix

Reuse seed 7; add training seeds 17 and 27, for eight new runs:

| Cohort | Lanes for each new seed | Saved comparison steps | Final processed examples per run |
|---|---|---|---:|
| 500k | Centralized-light and IID | 20 | 10,000,000 |
| 2M | Centralized-light and IID | 5 and 20 | 40,000,000 |

Use the settings and hashes in [the machine-readable plan](scaling-confirmation-plan.json).
Run sequentially on the existing T4. Never launch concurrent GPU training for this
comparison. Retain one 20-pass run per setting; pause each 2M run after step 5,
verify the budget, save its entire directory separately, then resume the original.
Do not alter source files or dependencies between pause and resume.

Partition seed remains 7 independently of training seed. Check that IID assignment
arrays match the seed-7 reference for the SAME cohort. Membership differs across
cohort sizes; do not require or claim nested client assignments. No non-IID run is
included until its changing client mixtures have a separate controlled design.

## Validation gates

Before training, verify transferred cohort manifest hashes, GPU access, installed
requirements, and source version. Use new output directories; do not overwrite the
seed-7 evidence. Preserve environment and partition metadata with every checkpoint.
After training, require 20 completed steps, exact example and optimizer counters,
finite losses and metrics, and full IID participation with every assigned row
processed once per round. Test metrics must remain null. Check the step-5 snapshot
history against the completed history prefix. Stop for a failed gate; do not silently
substitute a seed, skip a failed run, or extend its training budget.

The existing seed-7 training sources predate this documentation but must match all
training-affecting code and settings. New planner or test files alone need not force
a rerun. Record source differences explicitly when importing reference evidence.

## Analysis and decision rules

Within each lane, report two paired differences for each seed: 2M step 5 minus
500k step 20 (primary equal-example comparison), and 2M step 20 minus 500k step 20
(secondary equal-pass comparison). Report all seed values, mean, sample standard
deviation, range, and each paired difference. Include macro-F1, benign false-alert
rate, and precision and recall for all eight classes. Best-checkpoint scores do not
replace final-at-budget scores.

Seed 7 motivated the experiment; label it exploratory and show the two prospective
seeds separately as well as in the three-seed summary. Consistent positive macro-F1
differences on both new seeds support repeatability under these settings, not
statistical significance or general deployment superiority. False-alert or rare-class
regressions must be reported alongside aggregate gains. Mixed signs mean the result
is uncertain; do not respond by choosing the favourable seed. Three seeds are a
small diagnostic replication, not a precise uncertainty estimate.

Training seeds vary initialization, dropout, and training order. Fixed partitions
mean this stage does not measure robustness across client mixtures. Separately
fitted scalers and class weights mean the size comparison includes preprocessing
changes. The cleaned within-collection validation split is not an unseen-session
benchmark. Keep the test split sealed until model selection is frozen.

Review after all eight runs. Do not automatically launch 5M, heavy models, new
hyperparameters, or additional seeds. No cloud training is launched by this plan.
