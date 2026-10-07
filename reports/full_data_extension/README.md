# Official data scaling extension

## Purpose and current status

This local-only extension asks whether additional unique training data improves
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
disjoint across splits. No extension model has been trained.

The [full local loader check](loader-check.json) passed on all 16,474,212 training
and 2,059,284 validation vectors. On this Mac, fitting the diagnostic train-only
scaler and reading both transformed splits took 17.22 seconds, with a measured
peak process RSS of 358,481,920 bytes (about 342 MiB). This measures data loading
and scaling only, potentially benefiting from the filesystem cache; it is not
training throughput, a cloud speedup or an edge-device measurement. The loader
check did not open the test split. The materializer separately read test shards
only to verify their integrity, coverage and split assignment.

All 68 local repository tests passed after these additions, including fixture
checks for cross-file duplicates, conflicting targets, nonfinite/malformed rows,
interrupted materialization recovery, checksum rejection, deterministic bounded
shuffling and parity with an in-memory scaler. The 3,541 derived shards and their
selection files remain under ignored `outputs/official39-shards-v1/`; only the small
loader receipt accompanies this guide. No new dependencies were added.

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
can remain. Training subset construction, client sampling, and model checkpoint
recovery still need separate implementation and parity tests. Restarting the basic
loader with the same seed restarts an epoch; it does not resume mid-epoch. A verified
materialization therefore deliberately keeps `training_ready` false.

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
5. Freeze nested training subsets and one validation set. Proposed sizes are roughly
   0.5M, 2M and 5M, subject to audited availability; log both raw counts and unique
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

Next engineering task: freeze nested training subsets, implement matched centralized
and federated streaming training with client assignments, and verify optimization
and checkpoint-recovery parity on a small local fixture. Only then prepare a short
cloud pilot. Do not run full-scale training yet. The branch
`experiment/full-data-scaling` remains local until the user chooses to publish it.

This verified data-engineering milestone is a suitable local commit checkpoint.
Keep raw data, derived shards, environments and audit databases out of that commit.
Review the existing synthesis/demo changes separately when staging, because they
pre-date this extension work. A push remains a separate decision; passing data
checks does not yet establish that the extension improves the research results.
