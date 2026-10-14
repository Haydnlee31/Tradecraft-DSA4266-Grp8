# Confirmation comparison is locked and locally ready

14 October 2026. **All twelve saved checkpoints reproduced their historical development predictions exactly. All 48 frozen operating points matched.** No model was retrained, no threshold was refitted, and the reserved confirmation cohort remains unopened. There is no reason to start a cloud machine for the next step.

## What was completed

The [protocol](confirmation-protocol.md), [plan](confirmation-plan.json), preparation code and ten preliminary tests were committed as `adeddbb` before the real-data replay. The lock includes centralized-light Adam, shared-gradient Adam, FedAvg and blockwise four-bit shared Adam at seeds 7, 17 and 27. Each checkpoint, deployed weight state, initialization, sample-chain identity, work record and four existing thresholds is pinned to the previously audited evidence.

The compressed model was loaded from its delivered `visible` weights, not the hidden full-precision server target. Fresh inference matched the saved logits exactly for every model. Sorted benign-score checks and an independent direct-logit decision calculation reproduced the previously stored thresholds and outcomes without calling a calibration helper. This establishes that the models we intend to compare are the models already studied; it does not add a new performance result.

The [aggregate receipt](confirmation-preparation-results.json) contains the lock, source hashes, checks and measured local timing. Private arrays, logits, row identities and checkpoints remain in ignored local output directories. The lock digest is `879b1ceacaaf3164523a752e4842009db64242169593c2f7c209c171e0146aa0`.

## Local work and cloud budget

| This preparation | Measured result |
| --- | ---: |
| Checkpoints / frozen operating points verified | 12 / 48 |
| Development examples evaluated across models | 4,794,624 |
| Combined model-inference time | 0.866 seconds |
| Total preparation time | 2.975 seconds |
| Process peak resident memory | 604.5 MiB |
| New model fits / training examples | 0 / 0 |

These measurements use the existing local CPU runtime with two threads, not a GPU or cloud instance. They exclude any future confirmation-data reconstruction. The later inference budget is nearly identical: 4,782,024 forward examples across the twelve checkpoints. The small model and bounded data processing make a larger GPU unnecessary for that task. Keep the remaining cloud credits unused unless a separate, justified experiment later needs them.

## Verification and repository status

All **811 tests passed in 60.218 seconds** in an isolated source archive of commit `3cc4122`, excluding unrelated uncommitted work. The fifteen new preparation/evidence tests cover preserved thresholds and identities, delivered-versus-hidden model state, malformed artifacts, metadata-only access, no refitting/training, output preservation and the public receipt. All report links and Git whitespace checks also passed. This is local regression verification, not a new GitHub CI run.

The implementation and aggregate evidence are committed locally on `experiment/federated-shared-optimizer`. This extension has not been pushed, and unrelated working-tree changes were left untouched.

## One provenance issue handled before evaluation

The temporary raw-training staging cache has no recorded checksum. The lock therefore does not trust or open that cache. It records the historical Parquet manifest/recipe identities and the existing fit-only scaler. Future reconstruction must stream the checksum-verified nested training shards, verify the frozen reserved row IDs and ordering, and use the same scaler without refitting it.

Only metadata was checked at this stage; the raw shards and reserved row-ID array were not opened. The manifest records 398,502 confirmation rows, with 500 Web-based and 260 Brute Force examples. Those are existing counts, not newly evaluated outcomes.

The cohort was held aside for this extension but comes from previously used research data. It cannot support a claim about new capture sessions or an independent external dataset. The previously evaluated original test remains closed to this extension.

## What remains, in order

1. **Authorize the single reserved-confirmation evaluation.** Preparation alone has not authorized opening it. The next implementation will consume this lock, reconstruct the same cohort and evaluate all twelve endpoints locally, without training or threshold changes.
2. **Report the result against the fixed rules.** All three seeds must meet the unchanged five safeguards at the existing 5% calibration target. Report actual false alerts, every class recall, rare-class counts and centralized-light gaps. If a safeguard fails, keep the failure; do not choose another threshold or average it away.
3. **Finish the thesis trade-off synthesis.** Explain the aggregate quality gain, communication savings and remaining synchronization/rare-class limitations. No further classifier or parameter search is part of this plan.

The promising development conclusion is unchanged: blockwise shared-gradient Adam is about 11.08 macro-F1 points above this FedAvg configuration and 1.94 points below centralized-light on average, while reducing modeled shared-gradient application payload by 86.4%. Those are the [previous development results](block-replication-findings.md), not new confirmation findings. Confirmation will test whether they carry forward; it cannot establish privacy, rare-class resolution or measured real-network/edge efficiency.
