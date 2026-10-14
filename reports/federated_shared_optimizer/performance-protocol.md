# Local shared optimizer performance comparison

14 October 2026. This exploratory screen asks whether shared adaptive optimizer information closes the non-IID macro-F1 gap under a fixed amount of gradient work. It follows the successful [numerical bridge](bridge-findings.md), but is a new from-scratch experiment rather than a continuation of the completed official-data test study.

## Data and evaluation boundaries

Use only the historical two-million-row training view. Hash canonical raw feature vectors into 60% fit-eligible, 20% development and 20% reserved confirmation groups. Select exactly 500,000 fit vectors by the frozen hash ordering. Require raw-vector uniqueness, preserve historical client ownership, and verify row ordering against the old training arrays. The old scaler is used only for this identity check. Fit a new StandardScaler and class weights on the new 500,000-row fitting cohort.

After transformation to float32, exclude development vectors identical to fitting vectors and confirmation vectors identical to either earlier partition. This label-independent rule prevents numerical rounding from introducing exact input overlap. Confirmation row IDs are reserved, but confirmation feature arrays are not materialized for scoring. Original validation and test arrays stay unopened. These are previously used historical training rows, so the new split is exploratory within-collection evidence, not an independent capture-session test.

## Fixed controls and resources

All candidates start from the same seed-7 39–64–32–8 LayerNorm model, without dropout, clipping or weight decay. Disabling dropout makes the full MIME anchor a deterministic full-data gradient. All use square-root class weighting with one fixed fitting-population loss denominator. Each method gets learning rates 0.001 and 0.003; select its highest final-development macro-F1, breaking ties toward the smaller learning rate. There is no best-epoch selection or post-score learning-rate expansion.

| Method | Update convention | Units per candidate | Backward examples |
| --- | --- | ---: | ---: |
| Central Adam | Ordinary pooled batches of 500, persistent moments | 43,200 updates | 21,600,000 |
| Shared-gradient Adam | Population-weighted gradients from 20 client batches, one persistent server optimizer | 2,160 synchronizations | 21,600,000 |
| FedAvg with reset Adam | 20 local updates per client, reset local moments each round | 108 rounds | 21,600,000 |
| Central MIME base | Published lagged-second-moment update on pooled batches | 43,200 updates | 21,600,000 |
| Full MIME | Full fitting-data anchor and 20 paired local updates per client | 24 rounds | 21,600,000 |

Standard Adam uses betas 0.9/0.99 and epsilon 0.001. MIME and its central base use those coefficients, no bias correction, and an initial second moment of 0.0001. That initialization bounds the first-step denominator; it is a declared experiment choice, not a searched value. The full MIME kernel keeps moments fixed throughout each local block and updates them once from its full anchor, following the previously reviewed [Algorithm 1 and Table 5](https://arxiv.org/html/2008.03606).

With 500,000 anchor rows and 200,000 local examples, each MIME round costs 900,000 backward examples: 500,000 + twice 200,000. All ten candidates therefore cost 216 million processed backward examples. Repeated sampling does not create new unique data. Full anchors use batches of 8,192, so equal backward examples do not imply identical kernel work, wall time or optimizer-update counts. This experiment compares whole update strategies, not an isolated causal test of optimizer memory alone.

The shared-gradient arm is numerically matched to the separately tested centralized accumulation calculation, not to ordinary sequential Adam. It does not run a redundant full-quality accumulation arm. FedAvg, shared Adam and MIME reuse corresponding per-client sampled-row schedules, with population weights recomputed from fitting rows. More synchronized units consume longer schedule prefixes. Central pooled methods share their own sampling schedule.

Report all eight class recalls, precisions and F1 scores, confusion matrices, benign false-alert rate, elapsed training time, backward work and modeled payload. Count two logical barriers per MIME round and one per shared-Adam or FedAvg round. Network bytes are estimates excluding transport overhead; no real network or edge measurements are implied. The primary comparison is backward-work matched, not communication matched.

## Continuation criteria

A selected shared-Adam or MIME candidate warrants paired-seed confirmation only if it improves macro-F1 over selected FedAvg by at least 0.02, falls no more than 0.02 below selected central Adam, increases false alerts by no more than 0.005 against either reference, and lowers no class recall by more than 0.05 against FedAvg. These thresholds are absolute fractions: 0.02 means two percentage points.

Rare-class improvement is not required to continue, but zero recall remains a failure to detect that class. A passing development result is not final success. Keep confirmation closed until the settings and next comparison are reviewed. Do not add adaptive anchor timing, more rates or larger models during this screen.

The machine-readable [plan](performance-plan.json), data builder, training runner and synthetic recovery tests are committed before the first quality run. Preserve all completed outputs and failed attempts. No cloud launch or push is required.
