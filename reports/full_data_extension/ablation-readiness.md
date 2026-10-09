# Shared validation panel and ablation runner readiness

The local preparation is complete. The shared validation panel retains
**2,047,805 rows and all eight classes** after excluding 11,479 rows that match
training under the strictest proposed input mask. Removing both Number and
Tot sum adds just eight overlapping validation rows to the earlier full-feature
count. All Web-based and Brute Force validation examples remain.

The optional ablation runner now passes CPU recovery for all nine arm/lane
combinations. Its full-feature arm also matches the runner archived in commit
`f6e0bc3` exactly on a synthetic fixture. These checks establish implementation
compatibility, **not an improvement in model quality**. No official-data model
was trained or scored here, and the test split remains unopened.

Next, review the panel counts below. The next compute step after that review is
a two-step CUDA recovery pilot, not the full experiment or a parameter sweep.
RONIN has not been accessed or started by this preparation.

## The same validation rows for every arm

The [frozen panel plan](ablation-panel-plan.json) implements the membership rule
from the [ablation design](shortcut-ablation-plan.json): remove a validation row
if its inputs match any training row under any of the three masks. The masks
are nested, so matching on the 37 columns left after dropping Number and Tot sum
covers every match under either smaller mask. This uses exact canonical float32
values, with signed zero treated consistently, rather than approximate similarity.

Labels, predictions and model scores do not determine membership. Labels are
counted afterward to check coverage. The original training data, source splits,
scaler, class weights and within-validation duplicate multiplicity remain
unchanged. This is an evaluation subset, not a newly randomized split.

| Class | Original validation rows | Excluded | Retained |
|---|---:|---:|---:|
| Benign | 109,482 | 2 | 109,480 |
| DDoS | 1,213,890 | 5,870 | 1,208,020 |
| DoS | 373,093 | 5,478 | 367,615 |
| Recon | 68,008 | 2 | 68,006 |
| Web-based | 2,475 | 0 | 2,475 |
| Brute Force | 1,284 | 0 | 1,284 |
| Spoofing | 44,949 | 4 | 44,945 |
| Mirai | 246,103 | 123 | 245,980 |
| Total | 2,059,284 | 11,479 | 2,047,805 |

The joint projection has 9,838 shared train/validation vector groups, involving
11,462 training rows and 11,479 validation rows. Among these, 2,792 groups contain
multiple class labels. The panel excludes all matching validation rows whether
their labels agree or disagree with training. It does not selectively remove
hard cases based on predictions, and it does not repair contradictory training
inputs or remove possible near-duplicates.

The [published audit receipt](ablation-panel-results.json) records two independent
rebuilds with byte-identical receipts and identical index-file hashes. Both scans
use a disk-backed index and batches of 8,192 feature rows. The training and
validation matrices are never loaded together into Python memory; small O(N)
row-index/membership arrays are retained. The production loader separately
verified index bounds, ordering, disjointness, exact coverage, class counts and
artifact hashes without constructing or scoring a model.

This panel remains **reused within-collection validation**, not a fresh holdout
or evidence of unseen-session generalization. Its scores must be reported with
the changed cohort identified. Keep the original all-validation view separately
for historical comparisons; do not overwrite old result files or relabel them
as overlap-excluded results.

## What changes in the runner

The three opt-in arms are `full39`, `number_masked`, and `number_total_masked`.
They retain the same 39 input slots and 5,096 parameter slots. Selected columns
are held at zero after scaling—equivalent to the corresponding training means,
not raw traffic values of zero. Masks apply to both training and validation.
Source arrays are copied before masking and remain unchanged on disk.

The controlled non-IID arm requires the frozen client assignments. It cannot
silently draw a fresh Dirichlet partition. Partial arm/panel requests, heavy
models, BatchNorm and capped local updates are rejected by this experiment path.
With neither option supplied, the historical runner remains the default.

Both evaluation views come from **one full-validation model forward pass**.
The panel filters those same predictions, avoiding extra inference and possible
floating-point differences from changing batch composition. The console labels
the scores separately as `all_val_macro_f1` and `panel_val_macro_f1`.

For the planned experiment, the primary endpoint is the final step-20 model on
the shared panel. In each result JSON:

- `primary_checkpoint` is `last.pt`.
- `primary_validation_metrics` and `final_panel_validation_metrics` identify
  the final shared-panel endpoint, including every class's support and recall.
- `final_validation_metrics` retains the final original all-validation view.
- The legacy `validation_metrics` and `best.pt` remain validation-best on the
  original full validation set. They are secondary, not the primary comparison.

Both checkpoints and the environment record the input transform. Resume rejects
changed masks, panels, code, settings, partitions or initial model state. A
standalone inference consumer must apply `transform_inputs` using the saved
arm metadata; loading masked-model weights and supplying unmasked features
would be a different experiment. No deployment integration is authorized here.

## Local verification and its limits

The [CPU gate receipt](ablation-local-gate-results.json) uses a deliberately small
synthetic pack: 160 training rows, 80 validation rows, four simulated clients and
two epochs/rounds. Its inherited `500k` partition-schema label is only a fixture
key; it does not mean the real 500k cohort was used.

| Check | Local outcome |
|---|---|
| Pre-change archived runner versus full-feature ablation | Exact in centralized-light, IID and frozen non-IID |
| Uninterrupted versus paused/resumed training | Exact for all three arms in all three lanes |
| Model, optimizer, RNG, best state and non-timing history | Match in the applicable bridge/recovery comparisons |
| Initialization, work and assignments across arms | Identical within each lane |
| Panel membership under changed labels | Unchanged in the synthetic check |
| Masks and evaluation | Only selected columns change; one forward supplies both views; source packs unchanged |
| Corrupt or changed panel and initialization | Rejected without overwriting the saved checkpoint |

The gate processed 6,720 synthetic examples across its reference and recovery
runs. Those models and their scores are software tests, not scientific
classification results. CPU success does not establish CUDA determinism. Full
20-step historical GPU bridges also remain pending.

## The next cloud gate is bounded

After panel review, use `src.eval.official_ablation_recovery` for the two-step
CUDA compatibility check. It requires an explicitly supplied reviewed-panel
checksum. It tests all three lanes and masks, with full-feature bridges,
uninterrupted/resumed comparisons and paired initialization/partition/work checks.
It does **not** launch the 20-step comparisons or a sweep.

On the existing 2M training cohort, that gate has a declared total of 84M processed
training examples: each lane has one two-step default reference and three pairs
of two-step full/resumed runs. The resumed count includes the paused first step
once, not twice. This is a compatibility budget, not a claim that lanes use
identical optimizer-step counts, wall time or communication.

Only after the CUDA gate passes should the three 20-step full-feature historical
bridges run. Review those before the six seed-7 masked comparisons. Seeds 17/27,
model promotion, test evaluation and any larger-data experiment remain separate
decisions. Removing Number plus Tot sum breaks the demonstrated ratio route but
also removes legitimate size information; other window-size proxies may remain.

No cloud commands are needed during this local handoff. When scheduling the
pilot, use the new committed code in fresh output directories: strict source
guards intentionally reject resuming old checkpoints with changed code.

## Local artifacts and reproduction

The transfer archive is
`outputs/cloud-handoff-ablation-v1/official39-ablation-panel-v1.tgz`.
It contains only `receipt.json`, `retained_rows.npy` and `excluded_rows.npy` under
`official39-ablation-panel-v1/`, not the audit database or dataset.

- Archive SHA256: `520cc0b47980904213cda506ceae1754ad342767288e6fce8fd4b56c18bbce76`
- Reviewed-candidate panel receipt SHA256: `d2ae8eb1cdbd37974dd2e1c5c880e56858adc69181121760b16ea2f641309e77`

These local commands do not require a GPU. Rebuilds need fresh output directories;
the examples use the canonical names, so choose new names if they already exist.

```bash
.venv/bin/python -m unittest tests.test_official_ablation_panel \
  tests.test_official_ablation tests.test_official_ablation_receipts -q

.venv/bin/python -m src.data.official_ablation_panel build \
  --plan reports/full_data_extension/ablation-panel-plan.json \
  --packed outputs/official39-packed-2m-v1 \
  --output outputs/official39-ablation-panel-v1

.venv/bin/python -m src.data.official_ablation_panel build \
  --plan reports/full_data_extension/ablation-panel-plan.json \
  --packed outputs/official39-packed-2m-v1 \
  --output outputs/official39-ablation-panel-replay-v1

.venv/bin/python -m src.data.official_ablation_panel publish \
  --first outputs/official39-ablation-panel-v1 \
  --replay outputs/official39-ablation-panel-replay-v1 \
  --plan reports/full_data_extension/ablation-panel-plan.json \
  --output outputs/official39-ablation-panel-publication-recheck.json
```

The archived-reference gate uses the exact pre-change runner file from commit
`f6e0bc3`, with SHA256
`82eebe2e342c727fb79811278c9847e9debd0daeb9c2ff7debf7c29988a29646`.
The published gate pins that file, the current runner/pilot, supporting code and
fixture sources. CI exercises the same recovery/default-path checks with only
synthetic data; the archived historical comparison is the separately recorded
local check. No dataset, models, panel indices, archive or SQLite index is added
to Git by this stage.
