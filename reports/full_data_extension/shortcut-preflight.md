# Packet count proxies and model input overlap

Removing `Number` alone would **not remove packet-count information** from this
dataset. On every one of the 2,000,000 training and 2,059,284 validation rows,
`Tot sum / AVG` reconstructs `Number` within the fixed audit tolerance. Replacing
AVG with its duplicate, Tot size, gives the same result. A model could therefore
recover the information without the explicit Number column, although this audit
does not establish that the current neural network computes that ratio.

There is a second issue to address before further training. **11,471 validation
rows, or 0.5570%, already have an exactly matching training input at the float32
precision used by the models.** The original split was grouped at float64
precision; distinct source vectors can become indistinguishable after scaling
and conversion to float32. Removing Number creates no additional collisions in
these two cohorts. Historical scores remain recorded as measured, but their
validation inputs must no longer be described as entirely disjoint at model
precision.

Keep RONIN off. The next step is local: audit the two-column control, freeze a
common collision-excluded validation panel for review, then implement and test
the optional input masks. No new model was trained, input pack changed, split
reassigned or test array opened in this audit.

## What the ratio check establishes

The [preflight plan](shortcut-audit-plan.json) fixed the arithmetic checks and
tolerances before the scan. The audit approximately reverses the frozen
train-only scaler on the stored float32 inputs. It then compares the candidate
ratio against the similarly reconstructed Number. This is an arithmetic check,
not a trained attack classifier or a test of model performance.

A row passes when the absolute reconstruction error is at most
`0.001 + 0.00001 × abs(Number)`. Denominators must exceed 0.0001; none were
invalid. These tolerances were not adjusted after seeing the results.

| Cohort | Rows passing for each ratio | Mean absolute error | Maximum absolute error |
|---|---:|---:|---:|
| Training | 2,000,000 / 2,000,000 | 0.000004168 | 0.0000168173 |
| Validation | 2,059,284 / 2,059,284 | 0.000004167 | 0.0000168173 |

Every class passes separately, including all 2,391 training and 2,475 validation
Web-based rows and all 1,269 training and 1,284 validation Brute Force rows.
For the rows previously identified as near 10 or 100, the reconstructed ratio
also preserves the corresponding side of the prespecified midpoint, 55, in
every case: 1,994,507 training and 2,053,683 validation rows. That agreement is
not an eight-class accuracy score.

The [earlier background audit](explanation-sensitivity.md) links the 10/100
pattern to a scenario/windowing concern. The new measurement adds a concrete
reason why simply deleting the highest-ranked explanatory feature would be
an incomplete control. It does not prove that every remaining feature is safe,
nor that all measured classification performance comes from the shortcut.

## Exact input collisions are not the same as raw duplicate records

The audit indexes all 4,059,284 train/validation rows in a disk-backed SQLite
database, using complete canonical little-endian float32 vector bytes as keys.
Positive and negative zero are treated as equal. Labels are deliberately absent
from the key, so inconsistent labels cannot hide a duplicate. These are exact
model-input equalities, not approximate nearest neighbors or hash matches.

| Measurement | All 39 inputs | Number removed |
|---|---:|---:|
| Unique vectors across the two cohorts | 4,037,330 | 4,037,330 |
| Training duplicate excess rows | 5,950 | 5,950 |
| Validation duplicate excess rows | 6,174 | 6,174 |
| Distinct groups shared across train and validation | 9,830 | 9,830 |
| Training rows in shared groups | 11,454 | 11,454 |
| Validation rows in shared groups | 11,471 | 11,471 |
| Shared groups containing multiple class labels | 2,790 | 2,790 |
| Newly overlapping training or validation rows after removal | — | 0 / 0 |

Duplicate excess counts mean rows beyond one representative per vector within
that split. A shared group can contain several rows, so groups and affected rows
must not be interchanged. There are 13,649 cross-split row pairs with identical
inputs; 3,240 pairs have different labels. Across both cohorts, including
same-split-only groups, 4,899 distinct input vectors have multiple class labels.

The affected validation rows are concentrated in DDoS and DoS:

| Class | Training rows in shared groups | Validation rows in shared groups |
|---|---:|---:|
| Benign | 3 | 2 |
| DDoS | 5,850 | 5,866 |
| DoS | 5,471 | 5,474 |
| Recon | 2 | 2 |
| Web-based | 0 | 0 |
| Brute Force | 0 | 0 |
| Spoofing | 3 | 4 |
| Mirai | 125 | 123 |

This finding qualifies the earlier no-overlap statement: it remains a statement
about the original float64 vectors, not their lossy model representation.
Identical inputs with contradictory labels can also make classification harder.
The size or direction of the effect on macro-F1 has not been measured here;
do not subtract 0.557% from a score or assume all overlap inflated it.

Removing the 11,471 overlapping validation rows would leave 2,047,813 rows before
checking any further projection. That is a count, not a newly scored result or
an already-approved replacement validation set. The Number-plus-Tot-sum arm
could create additional matches and needs its own audit first. Exact matching
does not rule out near-duplicates or establish capture/session independence.

## A matched experiment with three distinct questions

The [prospective design](shortcut-ablation-plan.json) freezes the following arms.
It is a design specification, **not an executable training authorization**.

| Arm | Standardized inputs set to zero | Question it can answer |
|---|---|---|
| Full reference | None | Does the new optional-mask runner reproduce the historical code path? |
| Explicit count removed | Number | How much does direct access to Number help, while the ratio proxy remains? |
| Count and ratio route removed | Number, Tot sum | What changes when both the explicit count and the shared numerator of the two demonstrated ratios are unavailable? |

Zeroing standardized columns means holding them at their training means, not
setting the corresponding raw traffic statistics to zero. Train each arm from
scratch under that intervention; masking only at inference would measure a
different, out-of-distribution perturbation. Preserve all 39 input slots and
5,096 parameter slots so initialization and architecture do not change.

The last arm also removes legitimate total-length information. Its performance
change cannot be attributed solely to removing an undesirable shortcut. Other
counts and distributional features may still reveal window size. Call this
a bounded information-removal control, not a shortcut-free model.

Use the same 2M training rows, scaler, class weights, LayerNorm light architecture,
batch size 512, Adam settings, fixed client assignments and 20-epoch/round budget.
Compare centralized-light, IID and controlled non-IID separately. Each run
processes 40M examples; optimizer steps remain lane-specific at 78,140, 78,400
and 78,340 respectively. Matching within a lane does not make centralized and
federated optimization equivalent.

The first prospective stage is seed 7 only: three full-feature bridges followed
by six ablation runs, after all gates pass. The primary endpoint is the final
step-20 checkpoint. Report macro-F1, all eight class metrics, false alerts,
confusion matrices and resource/work counters; keep validation-best scores
secondary. Seeds 17 and 27 are a separate confirmation decision, not an
automatic sweep or permission to select the best seed.

## Gates before any cloud run

1. Audit the Number-plus-Tot-sum projection locally. Do not infer its collision
   behavior from the Number-only result.
2. Freeze one common, label-blind collision-excluded validation panel for all
   three arms: exclude a validation row if it matches training under any arm.
   Retain within-validation duplicate multiplicity and report per-class counts.
   Use this shared panel for the prospective primary comparison, keeping the
   original all-validation view separately for historical comparison. Review
   membership before scoring or training; do not silently resplit, delete
   training rows or alter old receipts. Neither view is a fresh holdout.
3. Implement opt-in masks with immutable source arrays, explicit feature metadata
   and resume guards. Check the empty-mask bridge, identical initialization and
   partitions, unchanged work counters and exact recovery for every arm.
4. Only then review cloud readiness. Require a small CUDA recovery check and the
   three full-feature historical bridges before the seed-7 comparison. Any
   provenance, recovery or work mismatch stops the experiment.

No performance outcome automatically promotes a model. A large ablation loss
would show dependence on the removed information within this collection; a small
loss would not prove robustness to new sessions. A stronger GPU or larger dataset
does not resolve either the proxy or validation-overlap issue. Test-based model
selection and deployable security rules remain out of scope.

## Reproduction and evidence

The [published receipt](shortcut-audit-results.json) contains per-class proxy
statistics, exact collision counts, input/source hashes and two matching audit
receipt hashes. Raw arrays and the two SQLite indexes remain in ignored outputs,
not Git. The reader checks every input checksum before mapping arrays and
explicitly closes all mappings; synthetic tests cover signed zero, label
conflicts, overlap expansions, independent dictionary counts and replay guards.

From the repository on the Mac, using fresh output directories:

```bash
.venv/bin/python -m unittest tests.test_official_shortcut_audit -q

.venv/bin/python -m src.data.official_shortcut_audit \
  --plan reports/full_data_extension/shortcut-audit-plan.json \
  --packed outputs/official39-packed-2m-v1 \
  --output outputs/official39-shortcut-preflight-v1

.venv/bin/python -m src.data.official_shortcut_audit \
  --plan reports/full_data_extension/shortcut-audit-plan.json \
  --packed outputs/official39-packed-2m-v1 \
  --output outputs/official39-shortcut-preflight-replay-v1

.venv/bin/python -m src.data.official_shortcut_report \
  --first outputs/official39-shortcut-preflight-v1/receipt.json \
  --replay outputs/official39-shortcut-preflight-replay-v1/receipt.json \
  --plan reports/full_data_extension/shortcut-audit-plan.json \
  --output outputs/official39-shortcut-preflight-publication-recheck.json
```

Publication requires separate, byte-identical completed receipts. It checks
scope, current audit-source hash and count conservation without opening the data.
The scan uses batches of 8,192 rows and a 128 MiB SQLite cache, rather than loading
all feature rows into Python memory. Operating-system mapped pages and framework
imports are additional memory; the cache setting is not a measured process-RAM cap.
