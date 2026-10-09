# Official data scaling extension

## Latest research status on 9 October 2026

The [background and redundancy audit](explanation-sensitivity.md) identifies a
strong scenario/windowing shortcut concern: Number is near 10 for almost all
Benign and rarer-category training rows, and near 100 for almost all DDoS, DoS
and Mirai rows. Several protocol/size columns are exactly redundant in the
two-million-row model inputs. A second uniform training background preserves
Number's leading pilot rank but changes feature contributions by 14–16%.
Numerical QA and independent replay pass; the findings do not authorize policy
deployment or invalidate the measured within-collection comparisons.

The preceding [full 219-case explanation study](explanation-study.md) passes for all seven
models: 12,264/12,264 class-score checks and 835/835 error-margin checks at both
resolutions, with featurewise resolution agreement. Balanced-core rankings are
reported separately from the targeted supplement. Number is the leading core
feature in every model, but other rankings and error explanations vary across
seeds; these are model-behavior findings, not validated security rules. The prior
[integration pilot](explanation-integration.md), [Monte Carlo warnings](explanation-convergence.md),
[CPU reference](explanation-preparation.md) and all cloud scores remain unchanged.
No new training or final-test evaluation is needed.

The [experiment ledger and decision summary](decision-summary.md) consolidate
the cloud controls, scaling confirmation and local diagnostics. The balanced-panel
tuning sequence is closed without promoting a model or decision rule. The best
scaled light reference still has weak rare-attack recall and substantial false
alerts; the controlled non-IID model misses almost all examples from several
attack categories. These are informative research findings, not deployment readiness.

The [portable evidence receipt](decision-evidence.json) retains 72 validation
endpoints in separate comparison groups, including all eight class recalls.
These are not 72 independent training runs. Teammates can audit the metrics with
`python -m src.eval.official_evidence --audit reports/full_data_extension/decision-evidence.json`
without the dataset, models or a GPU.

Next: locally prepare a prospective matched shortcut-ablation control, checking
remaining proxies and train/validation feature collisions first. Do not launch
cloud training until that protocol is ready. No broad training sweep is recommended. The official-39
final test remains unused for model/threshold evaluation, and the original
46-feature study stays separate. Raw data, checkpoints and per-row scores remain
outside Git. The [prior correction](prior-correction-findings.md) and
[diversity control](diversity-findings.md) retain the preceding detailed findings.

## Purpose and data preparation background

This experimental extension asks whether additional unique training data improves
rare-class performance and the centralized versus federated trade-off. It preserves
the original study, splits, scalers, models, comments and thresholds. More rows or
more expensive hardware are not themselves evidence of a more credible result.

The downloaded release is a separate **39-feature** study, not a drop-in expansion
of the legacy 46-feature data. Both the extracted source files and the inspected
Merged01.csv use 39 features; the merged file adds a Label column. Nine legacy
features are missing and two additional columns are present. Do not zero-fill the
missing features, equate Time_To_Live with Duration without evidence, reuse old
scalers/checkpoints, or attribute cross-study score differences to data size alone.

Implemented: structural auditing, explicit official-label normalization, nonfinite
counts, source hashing, and an on-disk SHA256 duplicate index that detects conflicting
eight-class targets. The protocol builder defines a duplicate-grouped within-collection
split. Numerical receipts state their completion status; no training readiness or
model improvement follows from passing the data-tool tests.

The [completed audit and frozen split counts](results/README.md) report all 309
files and the eligible counts for every class. All derived shards have now been
materialized and read-back verified: 16,474,212 training, 2,059,284 validation and
2,060,864 test vectors. Exact feature digests are unique across the collection and
disjoint across splits. Training/recovery checks have run on small synthetic
fixtures. The 500k cohort has since completed cloud reference, normalization and
loss controls. The [client diagnostic](client-diagnostic.md) summarizes those
results. The [bounded-update plan](bounded-updates.md) records that stage's gated
experiment; the subsequent [2M scaling comparison](scaling-findings-and-controlled-noniid.md)
and current diagnostic findings are linked above.

The [full local loader check](loader-check.json) passed on all 16,474,212 training
and 2,059,284 validation vectors. On this Mac, fitting the diagnostic train-only
scaler and reading both transformed splits took 17.22 seconds, with a measured
peak process RSS of 358,481,920 bytes (about 342 MiB). This measures data loading
and scaling only, potentially benefiting from the filesystem cache; it is not
training throughput, a cloud speedup or an edge-device measurement. The loader
check did not open the test split. The materializer separately read test shards
only to verify their integrity, coverage and split assignment.

At the initial preparation checkpoint, all 75 local repository tests passed,
including the recovery-pilot wrapper and fixture
checks for cross-file duplicates, conflicting targets, nonfinite/malformed rows,
interrupted materialization recovery, checksum rejection, deterministic bounded
shuffling and parity with an in-memory scaler. The 3,541 derived shards and their
selection files remain under ignored `outputs/official39-shards-v1/`; only small
evidence receipts accompany this guide. No new dependencies were added.

## Obtain the extracted CSV release

The [official dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
describes separate CSV and PCAP directories. Use CSV features, not packet captures.
On 7 October 2026 its download link led to a registration form. The user must
complete that form; no personal information has been submitted by this extension.
Keep the download URL, acquisition date, archive checksum and release listing.
Do not claim that an arbitrary folder with the right schema is the official release.

The archive has been extracted under `data/raw/CICIoT2023-official/CSV/`, separate
from the Kaggle mirror. It contains 309 CSV files and 8,943,841,070 uncompressed
bytes including archive entries. Structural auditing counted 46,776,697 readable
records and three malformed records separately. The original ZIP and raw files
remain untouched and excluded from Git. Local provenance is recorded in
`outputs/official-schema-audit/archive.json`; publisher checksum verification was
not available, so do not confuse a locally recorded hash with publisher authentication.

## Inventory before conversion

Run from the repository root with the existing environment and new output names:

```bash
python -m src.data.official_schema_audit --source data/raw/CICIoT2023-official/CSV --output outputs/official-schema-recheck/summary.json
python -m src.data.official_quality --source data/raw/CICIoT2023-official/CSV --schema-receipt outputs/official-schema-recheck/summary.json --output outputs/official-quality-recheck
```

The structural audit uses the old feature metadata only to identify differences.
The quality audit uses the actual 39-column schema and rechecks every source hash.
Canonical folder labels come from `official_raw_label` in the shared label-map
module; unknown folders fail closed. Case variants and Benign_Final/BENIGN aliases
do not loosen the original label parser. All 34 labels in Merged01.csv mapped to
the canonical labels during a read-only check.

After a complete quality audit, freeze the approved within-collection recipe:

```bash
python -m src.data.official_protocol --audit outputs/official-quality-recheck --output outputs/official-protocol-recheck/protocol.json
```

The quality audit streams 8 MiB CSV blocks and uses a disk-backed SQLite index
with a configurable bounded page cache (512 MiB by default, 16–1024 MiB allowed).
`--resume` rechecks the source hashes, completed-file prefix and committed row
counts before continuing an incomplete audit. A boundary mismatch fails closed;
never bypass that check. Interrupted-file transactions roll back, while completed
files are preserved. The elapsed time after resuming covers only the final execution
segment, not the entire multi-segment audit. Exact executed scripts are retained
alongside the local receipts. This is not a measured peak-RAM guarantee. Nonfinite rows
and malformed records are counted, not imputed. Finite numerical vectors use
little-endian float64 with signed zeros normalized; SHA256 hashes ignore labels so
identical vectors with contradictory targets remain visible. Exact hash matching
does not detect near-duplicates or prove absence of correlated capture traffic.

The earlier `official_inventory` staging prototype is intentionally left strict
for 46-feature labelled inputs. Do **not** run it on these 39-feature CSVs. Use the
separate `official_shards` adapter below. The audit/protocol commands above do not
create cleaned data or evaluate a model.

## Materialization and streaming checks

The adapter keeps one representative per eligible feature vector, assigns it to
the frozen split, and writes compressed Parquet shards. It preserves all 39 numeric
features as float64, plus the feature digest, eight-class target, raw-label mask,
source file ID and source record ordinal. That ordinal counts successfully parsed
CSV records starting at zero; it is not a physical line number after malformed rows.
Only the numeric features enter the model loader.

```bash
python -m src.data.official_shards --source data/raw/CICIoT2023-official/CSV --audit outputs/official-quality-audit --protocol outputs/official-protocol/protocol.json --output outputs/official39-shards-v1
python -m src.eval.official_loader_check --data outputs/official39-shards-v1 --output outputs/official39-loader-check.json
```

These commands refuse to overwrite completed evidence. The first exports eligible
digests from the immutable index in batches of 65,536, then loads at most 400,000
eligible digest entries for one source file. CSV parsing uses 8 MiB blocks. Each
written shard is read back: its feature hashes, labels, provenance and split must
match the selected index entries exactly once. Since the index has globally unique
digest keys assigned to one source file, exhausting all those selections establishes
exact-digest uniqueness across the complete collection and disjointness across
splits. This does not establish independence of similar flows or capture sessions.

After a completed source file, an atomic manifest records its verified shards.
`--resume` is available for an incomplete build with matching code, protocol and
memory cap; it verifies completed files before continuing. If interrupted during
the initial selection export before the first manifest, use a new output directory.
Never remove the original raw archive or audit database to recover a derived build.

The loader rejects changed shard checksums, streams numeric batches and fits a
new `StandardScaler` incrementally using training data only. It standardizes in
float64 before converting model inputs to float32. The local checker reads every
training and validation row and verifies class totals and finite transformed inputs;
it does not open test shards or train a model. Its full-training scaler is temporary
and is not saved: future learning-curve subsets must fit their own scalers.

With a seed, the loader shuffles shard order and each bounded 65,536-row buffer.
This is deterministic, but not a uniform shuffle of all rows. Class-heavy windows
can remain. The newer packed adapter below addresses global batch mixing and
adds client assignments and checkpoint-recovery tests. Restarting the basic
loader with the same seed restarts an epoch; it does not resume mid-epoch. A verified
materialization therefore deliberately keeps `training_ready` false.

## Nested training subsets

The learning-curve design uses exactly 500,000, 2,000,000 and 5,000,000 unique
training vectors. Each smaller subset is contained in every larger subset. A fixed
hash ranking selects rows within each class, with proportional quotas rounded by
the largest-remainder rule. This preserves the cleaned training distribution up to
rounding instead of introducing an oversampling policy at the same time as scaling.
The ranking does not depend on source-file order or validation performance.

All three subsets have been materialized and verified. Their exact class counts are:

| Class | 500,000 rows | 2,000,000 rows | 5,000,000 rows |
|---|---:|---:|---:|
| Benign | 26,537 | 106,148 | 265,370 |
| DDoS | 294,640 | 1,178,561 | 2,946,402 |
| DoS | 90,606 | 362,423 | 906,057 |
| Recon | 16,569 | 66,277 | 165,693 |
| Web-based | 598 | 2,391 | 5,979 |
| Brute Force | 317 | 1,269 | 3,172 |
| Spoofing | 11,024 | 44,095 | 110,238 |
| Mirai | 59,709 | 238,836 | 597,089 |

The smallest subset still has limited rare-class training coverage. Interpret its
results as an initial learning-curve point, not evidence that those classes have
been adequately learned. The larger subsets increase rare examples as well as
common examples without changing the intended class proportions.

The [frozen subset recipe](subset-recipe.json) records the exact quotas and hash
thresholds. The [completed subset check](subset-check.json) confirms all three
training views and all three transformations of the shared validation set. Peak
process RSS for that local check was 355,483,648 bytes (about 339 MiB). Scaler files
named in the receipt live in ignored `outputs/official39-subset-check/`, not beside
the shareable report. These checks do not evaluate model quality.

```bash
python -m src.data.official_subsets --parent outputs/official39-shards-v1 --output outputs/official39-subsets-v1
python -m src.eval.official_subset_check --parent outputs/official39-shards-v1 --subsets outputs/official39-subsets-v1 --output outputs/official39-subset-check
```

The builder counts 4,096 hash buckets per class, then sorts only keys in the quota
boundary buckets. A hard cap of 131,072 boundary keys limits this working set;
the full-data recipe needed 12,060 keys. It stores disjoint tiers so a row included
at 500,000 is reused, not duplicated, in the larger views. Read-back checks compare
every retained column with its parent row and recompute feature fingerprints.
An interrupted subset build remains incomplete and cannot be loaded; restart it
under a new output name. Unlike source-shard materialization, this builder does
not currently offer resume.

Each size fits its own train-only scaler. The checker streams the selected training
rows and the same complete 2,059,284-row validation set through that scaler, verifies
counts and finite inputs, and saves a scaler receipt tied to the exact selection.
It neither trains a model nor calculates validation performance. Test data is not
opened. The subset loader exposes selected class counts so future loss weights
cannot accidentally use the full-training counts.

The exact tiers occupy 8,040 small Parquet shards (five million stored vectors in
total). This layout prioritizes traceable selection and read-back checks. The packed
adapter below removes these repeated file opens from the training loop. Data checks
alone are not a GPU training benchmark.

These are proportional learning-curve subsets, not a rare-class balancing treatment.
Any later oversampling or loss change needs a separate control. More examples also
mean more optimizer updates at a fixed epoch count; retain the planned matched-update
control when comparing sizes.

## Disk backed training adapter

`official_packed` converts a selected cohort and its shared validation set into four
NumPy arrays: training features/labels and validation features/labels. Features are
already standardized using that cohort's verified train-only scaler. Every packed
row is checked against a second source replay, and the arrays and scaler are hashed.
Training uses read-only memory maps and copies only selected batches, rather than
loading the entire feature matrix into RAM. Shuffle/client indices still take O(N)
memory; one int64 index per five-million-row cohort costs 40 MB, with additional
temporary arrays during partition construction. The operating system also caches
mapped pages, so this is not a constant-RAM or edge-memory claim.

The 500,000-row cohort has been packed locally in
`outputs/official39-packed-500k-v1/` (about 400 MiB including validation). The
larger cohorts remain in their verified tier format until needed. Packing commands
for a fresh output are:

```bash
python -m src.data.official_packed --parent outputs/official39-shards-v1 --subsets outputs/official39-subsets-v1 --checks outputs/official39-subset-check --size 500000 --output outputs/official39-packed-500k-v1
python -m src.eval.official_packed_check --data outputs/official39-packed-500k-v1 --output outputs/official39-packed-check.json
```

The [real-cohort packed check](packed-check.json) covered every training and
validation row without training. With 20 clients, seed 7 and alpha 0.5, the IID
partition has 25,000 rows per client; the non-IID partition ranges from 5,200 to
49,561. Both cover the training data exactly once. Brute Force occurs in 9/20
non-IID clients and Web-based in 11/20, versus 20/20 for both classes in IID.
These are simulated client groups, not physical IoT devices.

`src.models.official_streaming` supports heavy, light, IID FedAvg and non-IID
FedAvg. It reuses the shared architectures/losses, the existing Dirichlet algorithm
and Flower's weighted aggregation. Central Adam state persists across epochs;
local Adam resets each client-round. Global index shuffling mixes each selected
training view; every row is processed once per epoch or full-participation round.
A final singleton is merged into the preceding batch for BatchNorm, allowing a
batch of size `batch_size + 1`. Unlike the legacy loader's singleton drop, this
retains every row; treat it as an explicit new-protocol policy.

Evaluation accumulates an eight-class confusion matrix and sample-mean unweighted
cross entropy, without retaining all predictions. Reports include macro-F1, every
class's precision/recall/F1/support, benign false-alert rate, examples processed and
optimizer steps. Checkpoint selection is strict maximum validation macro-F1 starting
at step one. There is no test-data path in the runner.

CPU fixtures verified exact optimizer/model parity with the existing in-memory
trainer on identical batches, and interrupted-versus-uninterrupted recovery for
all four lanes. Checkpoints include model/optimizer/RNG state and best-model history;
resume rejects changed settings, source code, environment, packed data or client
assignments. Recovery repeats an interrupted epoch/round from the last completed
boundary; it does not resume mid-batch. CPU exactness does not certify CUDA exactness.

## Next GPU compatibility pilot

Before starting the cloud machine, decide whether to publish this local extension
branch or transfer a fixed code snapshot. Transfer the packed 500,000-row directory
separately; data arrays must not be committed. Freeze the code and environment
throughout each recovery test. No cloud execution or push was performed here.

After the code and packed data are present in the cloud repository and the intended
environment has verified CUDA access, run inside a persistent terminal session:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.eval.official_packed_check --data outputs/official39-packed-500k-v1 --output outputs/official39-cloud-packed-check.json
python -m src.eval.official_streaming_recovery --data outputs/official39-packed-500k-v1 --device cuda --output outputs/official39-gpu-recovery
```

The recovery pilot performs two central-light epochs and two non-IID FedAvg rounds,
each both uninterrupted and paused/resumed, then compares model, optimizer, RNG,
best checkpoint and non-timing history exactly. It is a compatibility test, not a
sweep or a reason to promote a model. Review its `checks.json` and measured runtime
and memory before running the learning curve. CUDA determinism is strict: an
unsupported operation should fail rather than silently weaken the check.

The runner now offers an opt-in federated local-batch cap and exact aggregate
update budget; full epochs/rounds remain the default. See the
[bounded-update plan](bounded-updates.md) for budget restrictions, CPU recovery
checks and the required CUDA gate. This does not yet authorize the 500k/2M/5M
comparison: its data-scale budgets and stopping policy still need a separate
freeze. Keep the test split sealed.

## Split interpretation and cleaning policy

Several attack types, including Brute Force, have only one source file. A strict
file holdout cannot cover every class across all three partitions. Following the
user's request to choose, the selected design is a deterministic feature-digest
group split with approximate 80/10/10 train/validation/test proportions and fixed
domain seed 426639. The seed is not searched or changed based on model performance.

Exclude malformed and nonfinite records from derived data while preserving raw
sources. Exclude entire duplicate groups with conflicting eight-class labels.
Retain one representative for each other unique numerical feature vector; differing
raw labels within one eight-class target retain their provenance mask. This means
the sampling unit is a unique feature vector, not original repeated flow frequency.
Keep exclusion counts by class visible; deduplication may change class proportions.
Removing conflicting targets also removes ambiguity; performance on the cleaned
subset must not be advertised as performance on every original flow.

All copies of a vector receive the same partition. A frozen recipe records
per-class coverage and input hashes but does not materialize shards. Keep the test
partition sealed for model selection. No capture/session independence or absence
of overlap with the legacy mirror is claimed. Shared feature names alone are not
enough to establish equivalent extraction semantics for a cross-version overlap
audit. Treat this as a new within-collection experiment, not an external validation
of the old models.

## Research gates before training

1. Verify official origin, full file listing, schema, invalid values and per-label
   coverage. Confirm whether the release adds unique rare examples to the mirror.
2. Implement a disk-backed duplicate and mirror-overlap audit using canonical
   numeric feature values. Audit feature-only matches as well as feature-plus-label
   matches; conflicting labels must be visible. Do not mistake different file hashes
   for non-overlapping records. Exact matching still does not exclude correlated flows.
3. Review usable capture/session grouping from actual provenance. Freeze splits and
   duplicate handling before model selection; file grouping alone does not establish
   independence. Separate held-out data from historical mirror exposures. Do not
   pool the mirror's existing train/validation/test files into new training data.
4. Implement bounded training batches, train-only incremental scaling, reproducible
   shuffle/sampling, disk-backed client assignments and exact resume tests. Run a
   small parity check against the existing in-memory trainer before cloud execution.
5. Use the frozen nested training subsets and shared validation set. Sizes are exactly
   0.5M, 2M and 5M; log both source counts and unique
   rare-class coverage. Refit preprocessing only on each training subset and record
   it. Use consistent cleaning, loss and partition rules across matched lanes.
6. Benchmark one short T4 run before considering a stronger GPU. Measure ingestion,
   training throughput, peak host/GPU memory and end-to-end cost, not just GPU usage.
7. Run the learning curve with centralized-light and non-IID light, using IID as a
   diagnostic control. Record examples processed and optimizer steps; include a
   matched-update control to separate additional data from additional optimization.
8. Confirm promising settings with three seeds, preserve failed runs and inspect all
   eight recalls and false alerts. Tune only after the data-scale comparison, one
   hypothesis at a time. Do not relax the original eligibility checks after results.

The expanded study needs its own frozen evaluation protocol. Its results cannot be
directly subtracted from the old mirror scores and called a data-size effect: source
coverage, splits and composition also change. Make learning-curve comparisons within
the new protocol. Keep any genuinely new test partition sealed until final selection.

## Spending limits and next handoff

The proposed $170 envelope is $25 for preparation/pilots, $45 for learning curves,
$40 for targeted controls, $40 for confirmation and $20 contingency. These are caps,
not measured costs or authorization to exhaust the balance. Confirm actual RONIN
rates and storage costs before launching. No instances or downloads are automated.

Next checkpoint: review and commit the client diagnostics and bounded-update
extension, then transfer a fixed snapshot and pass its CUDA recovery gate before
running the planned same-code bridges and capped comparison.
Do not run full-scale training yet. The branch
`experiment/full-data-scaling` remains local until the user chooses to publish it.

The audited source-shard milestone was committed locally as `3773e51`.
Keep raw data, derived shards, environments and audit databases out of that commit.
Review the existing synthesis/demo changes separately when staging, because they
pre-date this extension work. A push remains a separate decision; passing data
checks does not yet establish that the extension improves the research results.
