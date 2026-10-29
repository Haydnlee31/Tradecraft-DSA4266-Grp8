# Cleanup and maintenance

## Preserved starting point

Committed baseline: `8ccbe87`, branch `snapshot/pre-repository-cleanup-2026-10-29`.
Cleanup branch: `cleanup/focused-study`. Nineteen locally modified/untracked files
and the tracked diff were backed up under ignored
`outputs/repository-cleanup-preservation-2026-10-29-v1/`. Unrelated drafts remain
in place; they are not automatically staged or published. The root README was
intentionally replaced with the focused guide; its pre-cleanup working version
and patch are preserved in that local backup.

## Changes

The primary study is now the front door. Context files, result/script indexes,
scope, methodology, data and reproduction guides distinguish primary,
supplementary and historical studies. `scripts/study.py` delegates to the existing
frozen runner and compares all 33 scientific endpoints. No optimizer, feature,
row, loss, threshold, seed or training-budget change is part of this cleanup.

Four superseded single-seed table-audit files moved to `archive/table_audit/`.
Five previous documentation files are preserved in `archive/superseded_docs/`,
with explicit active replacements. Archive verification/restoration covers all
172 preserved files and retains the 22 archived regression modules.

The [source preservation record](../studies/primary/preserved-sources.json) covers
25 pinned files. The primary runner has a static local import closure of 43
modules. Older-looking compatibility sources and checksum-linked reports remain
in place deliberately. They are classified as historical/supporting in the
navigation, not physically moved in a way that would break reproduction.

## Verification contract

Completed on candidate commit `9fa11c9` in a separate committed-only checkout.
The [verification receipt](verification.json) records:

- 661 active tests and 246 restored historical tests passed.
- 172 preserved archive files verified; all 25 protected sources unchanged.
- The public input asset downloaded and matched its SHA-256; all 16 members verified.
- All 33 primary models trained afresh; 33 serialized recovery checks passed.
- All 33 development/confirmation scientific endpoints matched the published report
  exactly, including weights, sample streams, logits, metrics, settings and work.
- All 363 confusion-matrix endpoints independently checked.
- The scientific report analysis, Markdown table and CSV cells matched; elapsed
  times are excluded from equality. CSV CRLF/LF differences are recorded.
- Rebuilt figures were visually inspected; fresh-run PNGs match those inspected
  images byte-for-byte. No workspace data/checkpoints/untracked modules supplied
  the clean run. The existing pinned Python environment was reused.

The remaining post-validation edits are documentation links and this receipt;
the validated numerical implementations and workflow adapter are unchanged.

The MimeLite numerical code/evidence is unchanged and remains a separate study;
cleanup does not require another search. The raw ~46M-record reconstruction is
not rerun here. CI's cross-platform synthetic checks remain distinct from strict
real-data numerical replay. Clean-checkout validation does not constitute a new
external holdout or a new hardware benchmark.

## Further maintenance

Keep study registries as references to one frozen protocol, not duplicate editable
settings. New results get versioned output directories. If a hash-pinned source
must change, preserve its historical version and establish a new verified
implementation; never replace old evidence hashes to manufacture equivalence.
Use `python scripts/archive_tools.py verify` before and after archival changes.
