# Data sources and cohort boundaries

There are two source workflows: the historical partial Kaggle mirror (46 features)
and the downloaded official CSV collection (39 features). They are not interchangeable.
The primary study uses the official representation. Development and confirmation
are cohorts derived from that source, not additional dataset providers.

```text
Official CSVs (streamed)
  -> schema/source audit, finite values, duplicate grouping, conflict exclusion
  -> deterministic original official train / validation / test partitions
  -> nested 2M-row view of original official training
       -> 500,000 fitting rows
       -> 399,552 development rows
            -> 200,022 calibration rows (threshold uses benign rows only)
            -> 199,530 audit rows
       -> 398,502 confirmation rows
```

Whole exact-feature groups define the original split. The 2M subset is ranked by
fixed hashes within proportional class quotas. A second label-independent hash
defines fitting eligibility/development/confirmation; only 500k eligible fitting
rows are selected. Remaining fitting-eligible rows are unused. Fit-only scaling
and class weights precede label-independent float32 collision removal (674
development, 1,126 confirmation rows). Client ownership is inherited and verified.

The original official validation/test partitions belong to a different historical
study. The primary runner does not reopen them. Development was used during
selection, so its audit subset is not an independent test. Confirmation was
reserved for the original optimizer extension but has since been evaluated; its
rows had also appeared in historical training. Report within-collection,
retrospective confirmation, not unseen-device/session generalization.

## Default: prepared inputs

`python scripts/study.py setup` uses the existing pinned release bootstrap. It
verifies the archive and every member before accepting the prepared cohorts,
scaler, row IDs, ownership and reference artifacts. Existing different inputs
cause failure rather than overwrite. No raw CSV download or retraining occurs.
The release includes confirmation arrays for explicit later evaluation; their
availability does not authorize using them for model selection.

## Optional: reconstruct from raw CSVs

The source acquisition and historical reconstruction tools remain supported as
advanced building blocks, not as the default teammate workflow. Read the
[original source/audit/materialization guide](../reports/full_data_extension/README.md)
and the [detailed methodology](../reports/methodology_comparison/methodology-draft.md).
Do not use the legacy 46-feature `official_inventory` conversion prototype for
39-feature raw files; some active code imports only its hashing helpers.

| Stage | Existing implementation |
| --- | --- |
| Schema and quality audit | `src.data.official_schema_audit`, `src.data.official_quality` |
| Grouped partition and provenance shards | `src.data.official_protocol`, `src.data.official_shards` |
| Nested views and historical packed inputs | `src.data.official_subsets`, `src.data.official_packed` |
| Frozen ownership | `src.federated.official_nested_partition` |
| Fit/development builder | `src.data.shared_optimizer_data` |
| Confirmation reconstruction | `src.data.shared_optimizer_confirmation` |

The last builders require the historical shard/subset manifests, packed row
identities, scaler and partition to match their pinned hashes. A different CSV
download or newly drawn partition is not an exact reconstruction. Never bypass
the checks or rewrite receipts to accept it. A future new source version needs
its own protocol and outputs.

This cleanup validates reproduction from the published prepared inputs; it does
not rerun the complete ~46M-record raw reconstruction. All existing raw builders
and their regression tests are retained. Raw arrays, caches and checkpoints stay
outside Git. [Reproduction guide](reproduction.md).
