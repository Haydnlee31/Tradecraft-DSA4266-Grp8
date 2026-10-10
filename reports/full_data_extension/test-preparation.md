# Official39 test preparation

**Step 5 is complete.** The shared primary test population contains 2,040,729
rows, with all eight classes represented. The original 2,060,864 rows are
preserved for the secondary comparison. No test predictions, performance
metrics, model training or new parameter choices were produced in preparation.

The earlier [CUDA replay](final-validation-cuda-checks.json) reproduces all 27
frozen models' historical validation counts exactly. The subsequent approved
**step 6 final evaluation is also complete**, followed by the local research
decision view and [final handoff](final-closeout.md). Keep RONIN stopped.
The preparation results below remain unchanged historical evidence.

## What was prepared

Following the user's approval after the eight-step roadmap, the existing
preparation program streamed the original test shards and used the saved
two-million-row training scaler. It verified source checksums, feature order,
class counts and a second replay of the packed rows. Scaling remains float64
before conversion to float32; no statistics were fitted on test inputs.

The inclusion rule excludes a row from the primary population if its exact
float32 model-input vector matches selected training data or any original
validation row under any of the three feature configurations. Signed zero is
canonicalized, and full vectors are compared in disk-backed indices. Labels
and predictions do not determine membership. Every model will use the same
retained-row list; no original row or source file was deleted.

The final union excludes **20,135 rows, or 0.9770%** of the original test split.
Most exclusions belong to DDoS and DoS. None belongs to Web-based or Brute Force.

| Class | Original test rows | Excluded from primary | Retained primary rows |
| --- | ---: | ---: | ---: |
| Benign | 109,137 | 4 | 109,133 |
| DDoS | 1,213,933 | 10,776 | 1,203,157 |
| DoS | 373,680 | 9,106 | 364,574 |
| Recon | 68,364 | 1 | 68,363 |
| Web-based | 2,437 | 0 | 2,437 |
| Brute Force | 1,310 | 0 | 1,310 |
| Spoofing | 45,485 | 14 | 45,471 |
| Mirai | 246,518 | 234 | 246,284 |
| Total | 2,060,864 | 20,135 | 2,040,729 |

These are data-support counts, not detection results. Keeping rare classes in
the test population does not mean the models can detect them.

## Overlap by feature configuration

| Configuration | Matches training | Matches validation | Matches either source |
| --- | ---: | ---: | ---: |
| All 39 features | 11,450 | 11,699 | 20,120 |
| Number masked | 11,450 | 11,699 | 20,120 |
| Number and Tot sum masked | 11,458 | 11,706 | 20,135 |

Some rows match both sources, so the first two columns must not be added.
Number-only masking adds no exclusions; the joint mask adds 15. Because the
projections are nested, the joint-mask exclusion set is the full union.

Within the original test population, each projection has 2,054,580 distinct
float32 vectors, 5,544 duplicate groups and 6,284 excess duplicate rows. Those
within-test duplicates are not independently deduplicated or reweighted:
the original float64-unique row remains the unit of analysis. The common
train/validation overlap rule still applies to every row.

This audit addresses exact model-input overlap, not near-duplicates, shared
capture conditions or unseen-device/session generalization. The final results
remain a within-collection comparison. No favorable effect on model scores is
assumed before evaluation.

## Verification and frozen artifacts

The [preparation receipt](test-panel-preparation.json) is an unchanged copy of
the generated receipt. It binds the frozen protocol, scaler, source manifest,
implementation and all three array hashes. The preparation used the same
implementation bytes as the successful CUDA replay; no evaluator or training
source was changed. Reopening through the production loader verified the
array checksums, dimensions, membership totals and all class supports.

Independent read-only reductions of the stored indices reproduce each
configuration's training/validation overlap totals. The joint reduction also
reproduces the unique-vector and duplicate counts. The full-feature database
passes SQLite's structural integrity check. Twenty-one synthetic evaluator and
preparation tests pass, as do four new offline receipt checks. The clean
repository snapshot passes all **313 tests**; unrelated local edits were
excluded from that verification and remain untouched.

Preparation took approximately 11.4 minutes locally, including disk-backed
index work. The optional macOS resource profiler could not query a kernel clock
counter under the sandbox and returned a nonzero wrapper status after the
program completed. The generated receipt and independent data checks pass;
there is no need to rerun preparation. No peak-memory measurement is claimed.

The immutable final-plan JSON retains its original `prepared_test_closed`
state. It describes the pre-execution design; it is not the live status page.
The new receipt records approved data access with `test_opened=true`,
`test_evaluated=false` and `model_trained=false`. Do not rewrite the frozen
plan's flags to reflect later progress.

Local prepared inputs are in `outputs/official39-test-panel-v1/`. The transfer
archive is `outputs/official39-test-panel-v1.tgz`, 48,487,993 bytes, approximately
46.2 MiB. It contains only `x.npy`, `y.npy`, `retained.npy` and `receipt.json`
under the prepared folder name. All four archived payloads were read back and
matched to their original checksums. The larger SQLite indices stay local.

| Artifact | SHA-256 |
| --- | --- |
| Preparation receipt | `e21315311f3af6158a22f581a108d83af3033e6fff88f4e62d952cb0ad370e7d` |
| Transfer archive | `5d9765a465cf8786efb19a36c621e26fe7ac65ac528ceaee5524cbd79158e3dc` |

Raw inputs, checkpoints, database indices and the archive are not committed
to Git. Only aggregate receipts, tests and the handoff are published.

## Handoff at the end of preparation

The following were the remaining stages when this receipt was created. They
have since completed for the official39 extension; see the final handoff above.

6. **Final evaluation:** after separate approval, transfer the prepared archive
   to the existing T4 machine. Verify its checksum and the saved CUDA replay,
   then score all 27 fixed checkpoints once as a complete campaign. The nine
   full-feature models are primary references; the eighteen masked models are
   sensitivity controls. Preserve partial outputs and use the existing guarded
   resume path if interrupted. No new training or parameter search is planned.
7. **Results and decision-layer integration:** publish both test populations,
   all eight classes' metrics and seed variation; combine them with the existing
   explanation findings and make missed-attack/false-alert warnings visible.
   The official39 extension has no matched heavy model and remains separate
   from the original 46-feature study. No automatic traffic-blocking policy is
   authorized by these research results.
8. **Final report and teammate handoff:** complete the comparison tables,
   figures, experiment narrative, limitations and reproducibility instructions;
   verify the final artifacts and commit/push the agreed deliverables.

These are the remaining stages, not an open-ended tuning sequence. If final
performance remains poor, report that result rather than use this test split
to search for a better model.
