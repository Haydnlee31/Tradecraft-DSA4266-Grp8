# FedNTD implementation audit — 2026-10-21

Scope: self-review of the new extension, numerical checks on synthetic data,
and regression checks on existing public evidence. No second reviewer or
production/real-data training run is implied.

## Isolation

- New implementation, plan, tests and documentation live under `extensions/fedntd/`.
- SHA-256 comparison against a snapshot taken before implementation checked
  **210 original Python source/script/test and selected dependency/config files**;
  none changed. `git diff --exit-code` also passed: no tracked file changed.
- Original source and script entry points have no imports of `extensions.fedntd`.
  `pyproject.toml` still points Flower at its existing client/server applications.
- No dependencies were added or installed. The extension uses existing NumPy
  and PyTorch plus standard-library modules.
- The runner rejects output paths outside its own `runs/` subtree. Git's
  existing ignore rule covers these generated artifacts.
- The adapter reads an explicit fit/development whitelist. It does not open
  confirmation data even if reserved files appear in the upstream manifest.
  Existing scalers, feature order, row mappings and partitions are read-only.
- Existing untracked research notes and the user's unrelated DOCX were retained.

## Numerical and recovery checks

Command: `python3 -m unittest extensions.fedntd.test_fedntd -v`.

**11 tests passed**, covering:

1. Teacher-to-student KL against an independent probability formula, temperature
   scaling, true-logit exclusion, zero gradient on excluded logits, detached teacher.
2. Finite gradients for extreme logits and a local batch containing one class.
3. Rejection of invalid temperatures, beta values and class targets.
4. Bit-exact beta-zero weights and sample chains against the original FedAvg
   numerical function bodies over three rounds, with unequal client populations,
   imbalanced pooled class counts and unchanged caller RNG state.
5. One frozen round-start teacher, unchanged input model and teacher parameters,
   positive-beta trajectory differences, and teacher/payload work accounting.
6. Frozen 108-round budget and 5,096-parameter model, plus invalid-plan rejection.
7. Memory-mapped disk input, whitelist-only access, local sampling and checksum
   tampering rejection.
8. Fit/development overlap and overlapping client ownership rejection even when
   the test fixture updates its file checksums.
9. End-to-end execution of all three arms through the actual disk-backed entry
   point on tiny synthetic data; metrics match the original metric implementation.
10. Interruption during a positive-beta arm, actual checkpoint serialization and
    recovery to bit-identical weights, metrics, sample hashes and work counters;
    changed contracts and completed runs are refused.
11. Original output-path protection and absence of reverse imports from original
    source/script entry points.

The baseline/metric comparisons compile only the selected unchanged function
bodies from the original source AST. They do not import or execute the full
Flower application. This allows numerical comparisons without installing the
unavailable optional application dependencies.

During review, completed-arm recovery was strengthened to verify the saved
checkpoint checksum before reusing a result. Interrupted checkpoint writes use
atomic replacement, and the CLI holds an exclusive run-directory lock.

## Existing regression checks

- `python3 -m unittest tests.test_official_final_report tests.test_archive_layout -q`:
  **19 tests passed**.
- `python3 scripts/official39_final_report.py audit`: **verified** 27 candidates
  and 108 metric views from existing JSON evidence. This performed no new model
  inference or test-data evaluation.
- `python3 -m extensions.fedntd.run --help`: passed.
- `python3 -m compileall -q extensions/fedntd`: passed.

## Remaining empirical work

The required prepared official39 optimizer fit/development bundle was not found
in this workspace's `data/` directory. The real 64.8-million-backward-example
screen was **not run**. No FedNTD accuracy, macro-F1 gain, network measurement or
edge-device performance is claimed. The entire repository test suite was not
run; focused numerical and isolation checks above are the verified scope.

Run the documented command once the existing prepared bundle is available.
Compare the paired beta-zero control against beta .1 and 1 using macro-F1,
all-class recall and false alerts. This runner uses argmax decisions, not the
previous study's fitted 5% alarm rule. Development results remain exploratory;
the previously exposed test and confirmation sets cannot select the extension.
