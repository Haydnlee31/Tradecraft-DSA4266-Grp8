# Feature removal in the automatic boosted tree classifier

The boosted-tree improvement survives retraining without `Number` and
`Tot sum`. Removing both gives mean macro-F1 0.698757 and benign false alerts
14.50%, compared with 0.697486 and 14.67% using all 39 features. The changes
between tree arms are small; the important result is that removing these
known feature routes does not erase the larger gain over the frozen central
neural reference.

The exact overlap check retains all 408,027 original audit rows. Performance
differences therefore do not come from changing the evaluation population.
This is evidence about a bounded information-removal intervention, not proof
of shortcut-free learning, new-session generalization or solved rare classes.
Keep the work local; this result does not justify a cloud capacity increase.

## Performance after removing the features

Values are means across paired seeds 7, 17 and 27 on the same reused development
audit. False alerts are the proportion of genuinely benign flows labeled as
any attack, not the fraction of alerts that are false.

| Model | Macro-F1 | Benign false alerts |
| --- | ---: | ---: |
| Frozen centralized neural reference | 0.655632 | 17.39% |
| Trees with all 39 features | 0.697486 | 14.67% |
| Trees with Number removed | 0.699078 | 14.50% |
| Trees with Number and Tot sum removed | 0.698757 | 14.50% |

Joint removal leaves macro-F1 4.31 percentage points above the central
reference and lowers its benign false-alert rate by 2.89 points. Both
advantages hold in every seed. This remains an assisted-development comparison:
the new classifiers fit extra labeled development rows, while the central
reference has not been retrained on those rows.

Neither masked arm consistently improves macro-F1 over the full-feature tree.
Both lose very slightly in seed 7 and improve in seeds 17 and 27. Paired
changes below are percentage points, not relative percentages:

| Removal | Seed 7 macro-F1 change | Seed 17 change | Seed 27 change | Mean change | Mean false-alert change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Number | -0.010 | +0.213 | +0.275 | +0.159 | -0.172 |
| Number and Tot sum | -0.014 | +0.299 | +0.096 | +0.127 | -0.167 |

Both masks reduce false alerts in all three seeds. The tiny difference between
their average macro-F1 scores is not a reason to select Number-only as a new
winner; that arm deliberately leaves the demonstrated ratio routes available.
Macro-F1 seed SDs are 0.002829, 0.002457 and 0.003727 for full39, Number-only
removal and joint removal respectively. They describe repeated fitting on the
same data, not uncertainty across independent capture sessions.

## Rare categories remain the main limitation

All entries below are percentages, including F1 on a 0–100 scale.

| Model | Web precision | Web recall | Web F1 | Brute Force precision | Brute Force recall | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Central neural reference | 41.67 | 7.34 | 12.37 | 42.29 | 27.14 | 32.94 |
| Full39 trees | 23.96 | 21.96 | 22.91 | 32.04 | 33.60 | 32.76 |
| Number removed | 25.31 | 23.02 | 24.10 | 32.04 | 34.12 | 33.02 |
| Joint removal | 24.90 | 21.82 | 23.25 | 33.87 | 34.52 | 34.16 |

Joint removal improves Brute Force F1 relative to the full-feature tree in
every seed: +0.92, +2.33 and +0.96 points. However, seed 27 still falls below
the central neural reference (0.318949 versus 0.324190). Thus it does not
satisfy the earlier requirement to exceed the central reference on both rare
F1 scores in every seed. That check is descriptive here, not a newly relaxed
promotion criterion.

The joint-removal model still misses about 78% of Web-based examples and 65%
of Brute Force examples. Only about one quarter of its Web predictions and
one third of its Brute Force predictions identify the correct category.
The improved 14.50% benign false-alert rate still flags roughly one in seven
benign flows. None of these measurements establishes an acceptable deployment
operating point or permission to block real traffic automatically.

Confusion is still concentrated in the same categories. With joint removal,
49.44% of true Web-based rows are called Recon and 23.09% Benign; 31.75% of
Brute Force rows are called Benign and 26.09% Recon. These percentages come
from the three summed confusion matrices, which reuse the same underlying
audit rows. Removing the known feature routes has not fixed these boundaries.

## Recall across every class

Values are three-seed mean recall percentages. The full per-seed precision,
recall, F1 and confusion matrices remain in the aggregate evidence.

| Class | Central neural reference | Full39 trees | Number removed | Joint removal |
| --- | ---: | ---: | ---: | ---: |
| Benign | 82.61 | 85.33 | 85.50 | 85.50 |
| DDoS | 84.66 | 84.32 | 84.33 | 84.39 |
| DoS | 71.04 | 74.51 | 74.27 | 73.98 |
| Recon | 73.53 | 76.55 | 76.69 | 76.46 |
| Web-based | 7.34 | 21.96 | 23.02 | 21.82 |
| Brute Force | 27.14 | 33.60 | 34.12 | 34.52 |
| Spoofing | 70.31 | 84.89 | 84.43 | 84.51 |
| Mirai | 99.60 | 99.81 | 99.81 | 99.80 |

## Next local experiment

Retain full39 and joint-removal trees as the reference pair. The joint-removal
version offers a useful conservative feature view without sacrificing the
observed overall gain; it is not being selected as a proven best final model.
There is no evidence here for spending more on GPU capacity, adding a paid
decision API, or returning to human review.

The next useful step is a matched-data neural control and a focused diagnosis
of the Benign/Recon/rare-category boundaries. Fit the small neural architecture
on the same development sample, class-weight objective and joint-masked inputs,
keeping its recipe bounded before scoring. Compare training and audit errors
by class to distinguish poor fitting from poor generalization. Keep the
existing full-feature reference and tree controls rather than attributing every
gain to architecture when training populations differ.

That evidence can guide a fully automatic rare-category repair model, if
warranted. A new specialist or cascade must improve rare F1 without hiding
false alerts or its inference cost; its benefit is not established by the
current ablation. Any eventual generalization claim also needs a genuinely
new evaluation cohort, not another pass over this audit or the exposed final
test. This stage trains no such additional model and launches no cloud run.

## Question and controls

The [preceding feature-expert comparison](feature-expert-findings.md) reached
mean macro-F1 0.697486 with feature-only boosted trees, compared with 0.655632
for the frozen centralized neural reference. Most of the hybrid's gain came
from direct feature access; adding neural scores helped only slightly. This
stage therefore examines the simpler feature-only model, not another hybrid.

| Arm | Information removed in both training and inference | What it checks |
| --- | --- | --- |
| Full39 | None | Reproduce the earlier feature-only result exactly |
| Number removed | `Number` | Dependence on the explicit packet count while ratio proxies remain |
| Joint removal | `Number`, `Tot sum` | Dependence after also removing the shared numerator of the two demonstrated count ratios |

Every arm is trained from scratch. Masked standardized columns are held at
zero, their original training mean, while the other values and all 39 input
slots remain unchanged. Applying a mask only after fitting would answer a
different question about input perturbation, not adaptation without those
features. Neural score vectors are excluded because they could reintroduce
information from an otherwise removed feature.

The recipe stays at 100 boosting stages, eight trees per stage, at most 15
leaves per tree, learning rate 0.1, minimum 30 samples per leaf and L2 penalty
1. There is no early stopping, best-iteration selection, threshold search or
hyperparameter sweep. Nine models are fitted per execution: three arms at
seeds 7, 17 and 27. All nine are saved before any model audit scoring.

The ordered 270,326 fitting examples and their square-root-frequency weighting
are unchanged. The old train-fitted scaler remains fixed. The full39 arm must
reproduce each historical fitted model, training prediction hash, and original
audit predictions and metrics exactly. Constant masked features are checked
to ensure that none appears in a tree split.

## One common population after exact overlap screening

Removing inputs can make previously different flow vectors identical. The
screen therefore uses exact little-endian float32 feature bytes, treating
positive and negative zero as equal and ignoring labels for membership.
A disk-backed index holds the 37 remaining columns for the entire 1,229,859-row
fitting partition. This is more conservative than checking only the 270,326
sampled training rows: unsampled fitting rows still supplied class-prior counts.

Because the masks are nested, every exact match under 38 or 39 features also
matches under 37. Excluding every 37-feature match would protect all three
arms using one common population. Separate diagnostics distinguish matches
to sampled fitting rows from matches only to unsampled fitting rows.

The actual scan finds zero matches of either kind. No fitting or audit row is
reassigned; no source array is modified. The retained counts are:

| Class | Audit rows |
| --- | ---: |
| Benign | 21,747 |
| DDoS | 240,791 |
| DoS | 73,445 |
| Recon | 13,559 |
| Web-based | 472 |
| Brute Force | 253 |
| Spoofing | 8,868 |
| Mirai | 48,892 |
| Total | 408,027 |

Panel membership is frozen before fitting. All masks and the frozen centralized
neural comparator use this same population. The separately recorded original
audit view is identical in membership here, rather than a second independent
evaluation. Calibration remains unused.

## Interpretation limits

The [earlier arithmetic audit](../full_data_extension/shortcut-preflight.md)
showed that `Tot sum / AVG` and `Tot sum / Tot size` reconstruct `Number` on
the packed training and validation data within the prespecified tolerance.
Deleting `Number` alone therefore cannot remove that information. Joint
removal also deletes legitimate total-length information; its effect cannot
be attributed purely to an undesirable shortcut. Other correlated features
can remain even after both columns are removed.

This is reused exploratory validation, not a new final test or proof of
generalization across captures. The same 472 Web-based and 253 Brute Force
rows occur in all three seeds. Seed variation is not independent dataset
uncertainty. The original study's completed test stays closed.

These are centrally fitted supervised classifiers. They use additional
labeled development rows, while the central neural comparator remains frozen
from the original study. The within-tree ablation is matched on fitting data
and work; the neural-versus-tree comparison is not a matched-training benchmark
or a demonstration of improved federated optimization. All inference is
automatic, with no human review, abstention or external decision API.

## Reproduction

The [fixed plan](feature-ablation-plan.json) binds the earlier evidence and the
three arms. The implementation is in
[official39_feature_ablation.py](../../scripts/official39_feature_ablation.py)
and [feature_ablation.py](../../src/eval/feature_ablation.py). Earlier byte-pinned
training and evaluation code is left unchanged. No new dependency is needed.

Use a fresh output directory for each local execution:

```sh
.venv/bin/python -u scripts/official39_feature_ablation.py run \
  --cache outputs/official39-routing-gate-v2 \
  --packed outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-panel-v1 \
  --output outputs/official39-feature-ablation-v1
```

Private row indices, the collision index and fitted model objects remain in
ignored outputs. Only aggregate evidence is published in Git. Model objects
are loaded only from the run's own checksum-bound output; do not load arbitrary
downloaded joblib files. Published evidence can be audited without opening
private arrays or deserializing models.

## Verification

The [aggregate results](feature-ablation-results.json) were published only after
an independent local execution reproduced all nine serialized model hashes,
panel membership, prediction hashes, metrics and other non-timing records
exactly. Across the two executions, 18 models and 14,400 trees were fitted.
Each execution also reproduced all three historical full39 models and their
audit predictions exactly. This establishes computational reproducibility,
not confirmation on a new dataset.

All 14 focused tests pass, including exact projection matching against an
independent reference, source-array preservation, masked-feature invariance,
historical bridges, fixed work accounting and evidence-tampering checks.
The full suite passes 471 tests in a clean snapshot containing the previous
commit plus this extension, without including unrelated working-tree edits.
`pip check` reports no broken requirements. Input data and previous private
prediction caches retain their original checksums; the final test was not
opened. No cloud run, additional dependency or human-review workflow was used.

The public record can be verified locally without the private data:

```sh
.venv/bin/python scripts/official39_feature_ablation.py audit \
  --record reports/risk_aware_routing/feature-ablation-results.json
```
