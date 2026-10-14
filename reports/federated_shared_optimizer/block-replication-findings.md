# Blockwise compression passes both additional training seeds

14 October 2026. **The fixed four-bit block128 method passes every predeclared safeguard on seeds 17 and 27, as it did on discovery seed 7.** Its three-seed mean macro-F1 is 0.6219, compared with 0.6236 for uncompressed shared-gradient Adam, 0.6414 for centralized-light and 0.5112 for FedAvg. It retains most of the shared-optimizer aggregate quality while reducing modeled application payload by 86.4%.

This is now replicated across training seeds, not just one promising run. It is still development evidence on one fixed client partition, and it does not solve rare-class recognition. The next step is to prepare a locked comparison for the reserved confirmation split—not launch another parameter search or increase cloud compute.

## What this stage tested

Only initialization and client sampling seed changed. The serializer, block length, residual feedback, persistent server Adam, learning rate, architecture, fitting rows, client ownership, loss, calibration rule and work budget stayed fixed. Each compressed candidate is paired with its existing same-seed centralized-light, shared-Adam and FedAvg references. Matching initialization and shared-client sample-chain hashes were verified.

The [protocol](block-replication-protocol.md), [plan](block-replication-plan.json), runner and eleven preliminary tests were frozen in commit `2088f78` before new real-data scores were computed. Both additional seeds ran to their prescribed endpoints. Seed 7 was reused as discovery evidence; the nine reference models were also reused, not counted as new training.

The rule was deliberately per seed: every seed must meet all five safeguards at the primary 5% calibration false-alert target. A favorable mean cannot rescue a class-recall failure in one seed. No block size, learning rate, checkpoint, threshold rule or acceptance limit was changed after inspecting these results.

## Every seed passes the unchanged primary safeguards

| Seed | Block macro-F1 | Actual false alerts | F1 loss from shared Adam | Largest class recall loss | All five checks |
| --- | ---: | ---: | ---: | --- | --- |
| 7, reused discovery | 0.6241 | 5.39% | 0.054 points | 4.697 points, Spoofing | Pass |
| 17, new replication | 0.6164 | 5.36% | 0.447 points | 3.333 points, Web-based | Pass |
| 27, new replication | 0.6252 | 5.25% | 0.010 points | 3.494 points, Spoofing | Pass |

The limits remain: at most one macro-F1 point lost from same-seed shared Adam, at most 0.5 percentage points of actual false-alert difference, no class losing more than five recall points, at least eight macro-F1 points gained over same-seed FedAvg, and at most 15% of the historical shared-gradient raw payload.

The new seeds gain 10.31 and 11.88 macro-F1 points over FedAvg respectively. Their actual false-alert differences from uncompressed shared Adam are 0.122 and 0.188 points, within the 0.5-point allowance. Seed 7 still has the narrowest class-safeguard margin; passing does not erase that residual trade-off.

The calibration target is 5%, but actual audit false-alert rates are 5.25–5.39%. This is an empirical operating-point comparison, not a guaranteed deployment alarm budget. The earlier failed uncalibrated training gate remains unchanged.

## Three-seed quality comparison

These statistics describe seeds 7, 17 and 27, with seed 7 explicitly retained as discovery. The centralized model here is **centralized-light using the same architecture**, not the project's separate heavy model. The standard deviation is the sample standard deviation across three training seeds, not a confidence interval.

| Method | Mean macro-F1 | Sample SD | Mean actual false alerts | Payload per run |
| --- | ---: | ---: | ---: | ---: |
| Centralized-light Adam | 0.6414 | 0.0053 | 5.30% | Not modeled |
| Uncompressed shared Adam | 0.6236 | 0.0024 | 5.37% | 1,761.18 MB |
| Blockwise four-bit shared Adam | 0.6219 | 0.0048 | 5.33% | 240.26 MB |
| FedAvg with client Adam | 0.5112 | 0.0041 | 5.40% | 88.06 MB |

On average, compression costs **0.17 macro-F1 points** relative to uncompressed shared Adam. The compressed method remains **11.08 points above FedAvg and 1.94 points below centralized-light**, closing approximately 85.1% of the mean FedAvg-to-centralized gap. This is not statistical equivalence: three seeds are too few for a strong general claim, and all use the same development panel and partition.

The central gap is 2.15, 1.90 and 1.78 points for seeds 7, 17 and 27 respectively. The mean being below two points must not be presented as every seed being within two points, or used to rewrite an older failed gate. The compressed model also varies more across these seeds than the uncompressed shared model.

## What improves and what remains weak

Mean correct eight-class recalls at the primary target show where the aggregate gains come from:

| Class | Centralized-light | Shared Adam | Blockwise shared Adam | FedAvg |
| --- | ---: | ---: | ---: | ---: |
| Benign | 94.70% | 94.63% | 94.67% | 94.60% |
| DDoS | 86.06% | 82.78% | 83.73% | 99.09% |
| DoS | 67.82% | 73.80% | 72.18% | 20.46% |
| Recon | 49.26% | 46.07% | 47.08% | 42.42% |
| Web-based | 13.83% | 10.00% | 9.51% | 0.37% |
| Brute Force | 22.82% | 20.51% | 19.23% | 19.74% |
| Spoofing | 66.82% | 59.52% | 56.27% | 21.01% |
| Mirai | 99.60% | 99.52% | 99.50% | 99.04% |

Compared with this FedAvg configuration, the shared-optimizer methods recover much more DoS and Spoofing recall. FedAvg retains higher DDoS recall, so the shared method does not dominate every class. The compressed method's mean Brute Force recall is slightly below FedAvg's despite its much better macro-F1.

Rare-class identification remains poor even centrally. The compressed candidate correctly identifies 23, 23 and 31 of the 270 Web-based audit examples, and 25, 24 and 26 of the 130 Brute Force examples. These are the same audit rows scored by three models, not three independent datasets. Compression generally preserves the earlier model's behavior; it does not create missing information or establish a rare-attack solution.

## Secondary thresholds and communication limits

All four predeclared alarm targets and all per-class metrics are retained in the [aggregate evidence](block-replication-results.json). Across those twelve seed-target endpoints, aggregate macro-F1 differs from same-seed uncompressed shared Adam by no more than 0.448 points in magnitude. However, at the **20% calibration target**, seeds 17 and 27 have actual false-alert differences of approximately −0.593 and +0.593 percentage points. These exceed the 0.5-point comparability tolerance. We therefore do not claim alarm-matched performance at every secondary target; the primary 5% comparisons pass individually.

Each run transmits 240,255,840 modeled application bytes. This counts the actual packed messages, tensor counts, block scales, sequence headers, initial FP32 broadcast, and all twenty client uploads and twenty delivered downlinks per step. It is 13.64% of the historical shared-gradient raw payload, but **2.73 times FedAvg's historical raw payload**. The older raw references exclude startup and message metadata, making this particular savings comparison conservative.

The method still uses **2,160 synchronization steps versus FedAvg's 108 rounds**. Smaller messages do not remove round-trip latency. The simulated study does not measure real network throughput, secure aggregation, transport overhead or physical edge performance, and compression is not a privacy guarantee. These are remaining costs to discuss, not details to hide behind the F1 gain.

## Verification and work performed

The two new runs processed 43.2 million backward examples in total. Recorded local CPU training-loop times were 54.42 and 54.44 seconds, about 109 seconds combined, excluding evaluation. Independent recovery added 40,000 backward examples. The data and runtime match the historical references; no cloud session was used.

In a separate process, both checkpoints recovered exactly from two steps before their endpoints. The audit compared delivered weights, server targets, Adam moments, client compression residuals, sample chains and byte counters. Reloaded delivered-model inference reproduced the saved logits, and independent sorted-threshold/direct-logit calculations reproduced all eight new operating points. The publication retains confusion matrices, per-class scores, pairing records and hashes; private rows, logits and checkpoints stay outside Git.

Reserved confirmation and original validation/test arrays stayed closed. This is training-seed robustness on an already inspected development panel and one fixed non-IID partition, not independent capture-session or partition generalization. The [runner](../../src/eval/shared_optimizer_block_replication.py), [preliminary tests](../../tests/test_shared_optimizer_block_replication.py) and [evidence tests](../../tests/test_shared_optimizer_block_replication_evidence.py) preserve those boundaries.

## Next stage without further tuning

The result supports preparing a single locked confirmation comparison of centralized-light, FedAvg, uncompressed shared Adam and the blockwise candidate. Freeze the checkpoint identities, already calibrated primary thresholds, metrics and success interpretation before opening reserved data. Do not recalibrate or select checkpoints on the confirmation results.

After that preparation is reviewed, evaluate the locked comparison once, then compile the final quality, class-recall, payload and synchronization trade-offs. Keep the rare-class weakness explicit. No new classifier search, larger GPU or automatic block-size sweep is justified before this stage.

The useful thesis is now more specific: **in this simulation, aggregating gradients with a persistent shared optimizer approaches centralized-light aggregate quality; blockwise residual-feedback compression preserves most of that improvement at much lower payload.** The combination still costs frequent synchronization, and its underlying mechanisms are established prior art rather than a claim of a newly invented algorithm. Reserved confirmation will test whether the development conclusion carries forward.
