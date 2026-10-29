# Historical experiments archive

This archive separates retired experiments and superseded documentation from the
active centralized-versus-federated study. Nothing was deleted. All 172 preserved
files retain their original bytes, including comments, results and provenance.
The [manifest](manifest.json) maps each old path to its archive location and SHA-256.

| Collection | Contents | Status |
| --- | --- | --- |
| [Classifier search](classifier_search/reports/risk_aware_routing/README.md) | 110 files: routing, specialists, score fusion, trees, Laya, embeddings, retrieval and associated tests | Retired exploratory directions, not the final method |
| [Early runs](early_runs/reports/full_run/RESULTS.md) | 49 files: initial model reports, early decisions, smoke result and teammate merge notes | Superseded runs; the synthetic smoke result is not study evidence |
| [Proposals](proposals/reports/Central%20aligned%20federated%20updates.md) | Four tracked proposal and literature-note files | Design history; not claims that every proposal succeeded |
| Superseded documentation | Five previous root/context/index documents under `superseded_docs/` | Byte-identical snapshots; focused active replacements remain at the original paths |
| Single-seed table audit | Four files under `table_audit/` | Superseded by the complete primary and MimeLite studies; the original missing-MimeLite finding remains historical |

Start with the [primary study](../studies/primary/README.md). The older integrated
[thesis](../reports/thesis/README.md), [official39 results](../reports/full_data_extension/final-closeout.md)
and [original shared-optimizer confirmation](../reports/federated_shared_optimizer/confirmation-findings.md)
remain available. Reports and source dependencies required by their audited evidence stay in place.
Untracked personal drafts, private datasets, checkpoints and local output folders
were not moved or committed by this cleanup.

## Verify and inspect the history

From the repository root:

Use a Python 3.13 environment for the full active and archived audit suites.
The frozen source bridge uses the historical `ast.dump` representation, which
differs from Python 3.12 even when source bytes are identical. See the
[Python AST documentation](https://docs.python.org/3.13/library/ast.html#ast.dump).
This is an audit-runtime constraint, not an operating-system-specific dependency.

```bash
python scripts/archive_tools.py verify
python scripts/archive_tools.py test
```

The second command reconstructs the historical paths in a temporary directory and
runs the archived regression tests. It does not start a real training experiment,
use the cloud or require the retired Laya runtime. CI runs this check separately
from the active tests; archiving does not remove regression coverage.

For a persistent review copy:

```bash
python scripts/archive_tools.py restore --output outputs/archive-review-v1
```

Open the reports under that copy to follow their original relative links. Archived
Markdown, commands and JSON retain the paths used at the time of the experiment;
do not execute archived scripts directly from this folder or treat old run
instructions as the current workflow. Restoring a copy does not copy private data
or authorize another evaluation of an exposed test population.

The source snapshot is commit `f5ae051`. It remains available through Git history
for a complete, exact historical checkout. Reconstruction combines the unchanged
archived files with current shared helpers; receipt checks fail if those helpers
later become incompatible. Do not rewrite old hashes to silence such a failure.

The 29 October additions use snapshot `8ccbe87`, recorded per entry. Restoring
superseded documents intentionally installs their historical contents in the
disposable review copy, not their active replacements. The previous locally
modified README and unrelated drafts were separately backed up under ignored
`outputs/repository-cleanup-preservation-2026-10-29-v1/`; those working files are
not published as historical committed evidence.

For online browsing with the original relative links, use the
[pre-archive snapshot](https://github.com/Haydnlee31/Tradecraft-DSA4266-Grp8/tree/f5ae051).
