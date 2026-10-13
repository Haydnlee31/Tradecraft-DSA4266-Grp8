# Hard example sampling reduced classification performance

The fixed local hard-example experiment failed its continuation criteria.
Compared with the matched neural control, macro-F1 fell from 0.69020 to
0.61765 and benign false alerts rose from 14.77% to 15.73%. Web-based recall
fell from 17.96% to 7.98%. Greater precision on the remaining rare predictions
did not compensate for the missed attacks. Close this configuration without
a sampling-fraction or warm-up search, and keep the tree as the stronger
development comparator.

Both full runs replay exactly, and the ordinary control reproduces the earlier
balanced cross-entropy model's weights and calibration predictions exactly.
No cloud compute, new audit scoring or original-test evaluation was performed.
These are exploratory centralized classifier results on reused development
data, not new federated-learning or independent-test evidence.

## What the experiment tested

The [previous diagnostic](rare-error-findings.md) found substantial shared
rare-class confusion, particularly with Benign and Recon, without exact
conflicting-label duplicates involving rare fitting vectors. The hypothesis
was that directing more updates toward difficult examples would improve
discrimination.

[Online hard example mining](https://arxiv.org/abs/1604.03540) has demonstrated
benefits in object detection. This experiment borrows the selection principle;
it is not a reproduction of that paper or evidence that its image-domain
results transfer to network flows. Here, difficulty is recomputed once per
epoch, rather than continuously within a detection pipeline.

Both arms use the same 270,326 fitting rows, seed-7 initialization, numerical
embedding inputs, 64/32-unit LayerNorm network, dropout 0.2, Adam settings and
60-epoch update budget. `Number` and `Tot sum` remain masked. Each batch has
64 distinct examples from each of the eight classes. Existing class weights
and the fixed batch-size loss denominator are unchanged.

For the first five epochs, both arms use the historical uniform balanced
sampler. At the start of every epoch, both score every fitting row using
unweighted classification loss with dropout disabled and no gradients. This
scoring does not change weights or consume training random numbers. Within
each class, the hardest quarter is ranked by loss, with ties broken by fitting
row position. The ordinary control ignores the ranking. After warm-up, the
new arm draws 32 uniform examples and 32 additional distinct examples from
that hard pool for each class.

Selection uses all eight classes, not only rare examples or examples called
Benign/Recon by the historical model. It uses only fitting losses; individual
calibration mistakes are never fed back into training. Class exposure remains
equal across the two arms, but the within-class training distribution changes
deliberately. There is no importance correction that would make the hard
mixture an unbiased estimate of the old objective.

Both arms retain the historical zero-weight contrastive computation solely to
preserve the exact control bridge. There is no nonzero contrastive loss here.
The training-only projector is discarded at inference: both arms have 18,584
training parameters and 17,000 deployed classifier parameters.

## Calibration results

Every model is scored on the same 409,918 calibration examples, including 501
Web-based and 271 Brute Force examples. All fitted models were frozen before
new calibration scoring. The endpoint is epoch 60, not the most favorable
intermediate epoch. Tree and PLE predictions are verified historical comparators.

| Model | Macro-F1 | Benign false alerts | Web F1 | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing tree | 0.70031 | 14.43% | 0.26068 | 0.32806 |
| Existing PLE MLP | 0.69036 | 14.99% | 0.22222 | 0.30804 |
| Matched ordinary control | 0.69020 | 14.77% | 0.22472 | 0.31373 |
| Hard-example mixture | 0.61765 | 15.73% | 0.13986 | 0.30809 |

The new method loses 7.26 macro-F1 percentage points against its control and
8.27 points against the tree. It fails both comparisons in the fixed gate:
macro-F1 and both rare-class F1 scores must increase, with no increase in
benign false alerts. The gate was not changed after seeing the scores.

Rare precision and recall, in percent:

| Model | Web precision | Web recall | Brute Force precision | Brute Force recall |
| --- | ---: | ---: | ---: | ---: |
| Tree | 28.88 | 23.75 | 35.32 | 30.63 |
| Matched ordinary control | 30.00 | 17.96 | 28.15 | 35.42 |
| Hard-example mixture | 56.34 | 7.98 | 52.68 | 21.77 |

The mixture makes fewer rare predictions. It correctly identifies only 40 Web
examples rather than the control's 90, and 59 Brute Force examples rather than
96. Its wrong Web predictions fall from 210 to 31, and wrong Brute Force
predictions from 245 to 53. Precision therefore rises while recall falls.
Those reductions do not imply lower overall benign false alerts: a benign
flow can still be incorrectly labeled as another attack category.

All eight recall percentages:

| Class | Tree | Existing PLE | Ordinary control | Hard mixture |
| --- | ---: | ---: | ---: | ---: |
| Benign | 85.57 | 85.01 | 85.23 | 84.27 |
| DDoS | 84.28 | 85.26 | 84.66 | 76.71 |
| DoS | 74.11 | 71.54 | 72.61 | 51.79 |
| Recon | 76.48 | 75.88 | 75.06 | 53.61 |
| Web-based | 23.75 | 20.76 | 17.96 | 7.98 |
| Brute Force | 30.63 | 25.46 | 35.42 | 21.77 |
| Spoofing | 84.51 | 83.83 | 83.55 | 86.40 |
| Mirai | 99.78 | 99.58 | 99.52 | 99.77 |

The damage is not limited to rare classes: DoS and Recon recall fall sharply.
The aggregate result is consistent with an unfavorable shift in what the
model learns when high-loss regions are repeatedly emphasized. This experiment
does not isolate whether ambiguous examples, optimization or another mechanism
caused that deterioration. It does not establish that all forms of hard mining
must fail.

## More repetition did not add rare examples

Each fit uses 16,220,160 training positions and 31,680 optimizer updates, with
2,027,520 positions per class. Each also scores 16,219,560 fitting positions
for difficulty. Both models see all 270,326 distinct fitting rows at least once.
The hard mixture forces 7,434,240 positions to come from the hard pools after
warm-up. Including uniform draws that happen to land there, 59.37% of its
training positions are in the current hardest quarter, versus 25.01% for the
ordinary control.

These are repeated exposures to only 1,502 distinct Web-based and 760 distinct
Brute Force fitting rows. The treatment adds no new observations. Its online
loss cannot be compared directly with the control's as a measure of model
quality because the batches have different difficulty distributions.

Across the original pair and independent replay, the four fits use 64,880,640
training positions and 64,878,240 mining positions. Mining is additional
no-gradient inference, not additional optimizer updates. The first pair took
about 225 seconds locally; repository tests ran concurrently during part of
that execution, so this is not a clean timing benchmark or GPU comparison.

## Next step is rare training coverage

A read-only check of the original training-shard inventory identified more
rare rows than the recent experiments used. The relevant seven training files'
SHA-256 hashes and Parquet row metadata match their saved manifest. Test shards
were not opened.

| Data scope | Web-based rows | Brute Force rows |
| --- | ---: | ---: |
| Recent extension fitting cohort | 1,502 | 760 |
| Earlier original 2M training subset | 2,391 | 1,269 |
| Original materialized training shards | 19,699 | 10,450 |

The last row is availability, not verified new unique coverage. It must not be
added to the other rows as though the populations were disjoint. The original
training shards and recent development-fit cohort have different provenance;
existing split boundaries and retained-feature overlap need checking before
any combination. More rows from the same collection do not establish more
independent capture sessions. Web spans six source files, while Brute Force
still comes from one source file.

The next local preparation should inventory unique rare vectors and their
source coverage, exclude cross-partition retained-feature matches, and design
a cohort that retains rare training observations while bounding majority-class
sampling. This differs from proportionally scaling every class and from giving
the same few rare examples more optimizer updates. It does not require loading
the full raw dataset into memory or expanding to packet captures or device IDs.

If the coverage check supports it, compare capped versus rare-retaining
cohorts with the same learner and update budget, keeping the current tree as
a comparator and the eight-class target unchanged. Define evaluation before
training and treat reused calibration evidence as exploratory. A claim about
new-session generalization still needs genuinely independent data. No new
cohort, larger model or further training run has been launched in this stage.

The inventory is `outputs/official39-shards-v1/manifest.json`, SHA-256
`1f35e1c29599e056e0758554094425b2d3f744c4853928bafc44bc2f800a1ed3`.
The original 2M counts are from `outputs/official39-packed-2m-v1/manifest.json`.
These private artifacts are not added to Git.

## Verification and reproduction

The [fixed plan](hard-example-plan.json), implementation and tests were
committed before full fitting. [Aggregate results](hard-example-results.json)
retain all class metrics, confusion matrices, work counters, sampling digests,
per-epoch loss/pool hashes and source identities. Checkpoints, row arrays and
predictions remain in ignored `outputs/`.

Synthetic tests verify sampling without within-batch duplicates, per-class
counts, stable ranking ties, mining without RNG/weight changes, and exact
epoch-boundary recovery including Adam state, RNG, exposure counters and the
sampling digest. Recovery is a synthetic check, not a claim that the full
real-data runner was interrupted and resumed.

Both complete fits were repeated independently. Model weights, mining losses
and pools, sampled-row digests, predictions and all non-timing receipt fields
match exactly. The ordinary control's weights and predictions also match the
historical balanced-CE reference. The warm-up states and first six mining
passes agree between arms before their training paths diverge.

All 571 tests pass in a clean committed snapshot, including 14 new focused
tests. `pip check` reports no broken requirements. Unrelated working-tree
changes were preserved and excluded from the snapshot. This is local macOS
verification, not a new remote CI, cloud or physical-edge measurement.

To reproduce locally with the existing private upstream artifacts and a fresh
output directory:

```sh
.venv/bin/python scripts/official39_hard_example.py run \
  --output outputs/official39-hard-example-new-replay
```

There is no parameter sweep, cloud or original-test option. Existing output
directories are preserved rather than overwritten.
