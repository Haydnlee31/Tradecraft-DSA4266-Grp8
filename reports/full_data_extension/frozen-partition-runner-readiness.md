# Frozen partition runner readiness

The controlled non-IID assignments can now be loaded without redrawing clients.
Local CPU integration checks pass; CUDA recovery and the historical GPU bridge
remain required before interpreting a controlled 2M cloud experiment.

## Implemented safeguards

`src/models/official_streaming.py` accepts optional `--partition-root`. With no
argument, the original path and settings remain unchanged. The imported control
is limited to the Dirichlet lane, full local epochs, partition seed 7, and alpha
0.5. Training seed may still vary independently in a later approved replication.

`src/federated/official_frozen.py` checks all three partition artifact hashes,
the matching packed-cohort manifest, expected client count, sorted unique int64
indices, full non-overlapping row coverage, and per-client class counts against
the packed training labels. Verification happens before creating the run output.
It does not trust a filename alone as proof of cohort identity.

The environment receipt records the partition policy, selected cohort, plan hash,
and imported assignment hash. Checkpoint recovery requires that metadata to match
and the run's saved client arrays to equal the frozen arrays. A changed plan,
corrupted artifact, wrong cohort, missing frozen option, or changed assignments
stops recovery rather than overwriting a checkpoint. Moving identical artifacts
to another directory is allowed because file contents, not absolute paths,
identify them. Never edit a plan to mark it ready after training has begun.

## Completed local checks

| Check | Result |
|---|---|
| Full local suite | 106 tests passed |
| Previous commit versus current default, miniature CPU fixture | Exact match in light, heavy, IID, and Dirichlet modes |
| Real 500k, two rounds, frozen versus native partition | Model, optimizer, RNG, best state and non-timing history match exactly |
| Real 500k, two rounds, uninterrupted versus pause/resume | Exact match |
| Real 2M, two rounds, uninterrupted versus pause/resume | Exact match |
| Plan/cohort/count/client/duplicate/corruption guards | Rejected invalid inputs in tests |
| Artifact directory relocation | Same arrays and provenance |

These CPU compatibility pilots are not model-selection runs or scaling evidence.
CPU results should not be compared directly with archived CUDA scores as a claim
about model quality. They do not replace CUDA recovery checks on the cloud image.

Receipts live in `outputs/official39-frozen-cpu-gates-v1/500k/checks.json` and
`outputs/official39-frozen-cpu-gates-v1/2m/checks.json`. The 500k pilot processes
1,000,000 examples and 1,968 optimizer updates per two-round run; the 2M pilot
processes 4,000,000 and 7,834 respectively. Timing differences are intentionally
excluded from exact recovery comparisons.

`src/eval/official_frozen_recovery.py` performs the same checks on either CPU or
CUDA. Before the 500k bridge, it verifies that the ordinary seed-7 partition is
array-for-array identical to the imported anchor. It does not attempt such a
comparison at 2M because the newly drawn legacy partition is the confound this
control is designed to remove.

## Remaining cloud gates

1. Commit the scoped implementation, tests, and research evidence; package the
   code bundle and the unchanged frozen partition directory. No raw data or
   checkpoints belong in Git. Preserve unrelated workspace changes.
2. Verify transfer hashes, source version, both packed cohorts, installed
   requirements, and GPU access in the existing cloud environment.
3. Run the two-round CUDA frozen recovery pilot for 500k and 2M, sequentially.
   The 500k pilot must also pass its same-code native/frozen bridge.
4. Run a 20-round 500k frozen-assignment bridge at seed 7. Compare the saved
   model state and non-timing history with the archived 500k original-denominator
   reference, not with the fixed-denominator treatment. Expected work: 10M
   examples and 19,680 optimizer updates. Explain any mismatch before proceeding.
5. Only after those gates, run the controlled 2M seed-7 experiment with the
   previously frozen settings. Preserve step 5, then continue the same run to 20.
   Expected updates: 19,585 at step 5 and 78,340 at step 20. Do not add a cap,
   change losses, redraw clients, or inspect the test split.

Adding source files changes the runner's strict source inventory. Finish or archive
old runs before updating; do not try to resume a checkpoint across code revisions.
Historical scaling summaries can still be audited against their pinned source
using `official_scaling_review --source-ref
886a7b82a32c98981a205f9a4338e53b8fa83913` with the original archive/plan arguments.
This validates the old code rather than waiving a source mismatch.

The partition plan's preparation-time `runner_ready: false` field remains
unchanged in the immutable handoff. Readiness is tracked here and in separate
pilot receipts, not by rewriting a hash-bound plan.

The previous-commit comparison used runner source from
`886a7b82a32c98981a205f9a4338e53b8fa83913`, with the same local dependencies and
miniature data. It compared model/optimizer/RNG/best state and non-timing history,
not source-identity metadata. Its receipt is
`outputs/official39-frozen-cpu-gates-v1/legacy-default-checks.json`.

The prepared partition transfer archive is
`outputs/cloud-handoff-frozen-v1/official39-nested-partition-v1.tgz` (about 4.1 MB).
SHA256: `b58911a9408b1d04aa6d844cf317c343aaca7726a18f003e52d7f6128bb0ebe2`.
The code still needs a scoped commit and bundle before cloud transfer.
