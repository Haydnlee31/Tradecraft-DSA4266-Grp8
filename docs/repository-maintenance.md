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

Before publishing cleanup, use a separate checkout containing only committed
files. Validate source identities, all active tests, archived restoration/tests,
the approved input download, strict preflight, fresh training/evaluation of all
33 primary endpoints, exact scientific comparisons, and report reconstruction.
Use the same recorded numerical environment; a shared interpreter is acceptable,
but no workspace datasets, checkpoints or untracked modules may supply the test.
Record actual checks in `verification.json` after they complete.

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
