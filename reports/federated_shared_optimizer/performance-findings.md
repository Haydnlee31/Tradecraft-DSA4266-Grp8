# Shared gradient Adam approaches centralized light performance

14 October 2026. **Shared-gradient Adam reaches 0.6521 development macro-F1 versus 0.6608 for centralized-light Adam**, compared with 0.4837 for the matched non-IID FedAvg control. This closes 95.1% of the observed FedAvg-to-centralized macro-F1 gap in one exploratory seed. It shows that this small model can approach centralized aggregate quality while training through federated gradient aggregation. It does not establish a new algorithm, reliable rare-class detection or a low-cost deployment solution.

The costs remain substantial. Shared-gradient Adam has a 19.55% benign false-alert rate, versus 1.42% for FedAvg, and exchanges 20 times as much modeled training payload. Full MIME reduces communication and improves on FedAvg, but reaches only 0.5466 macro-F1. **Neither candidate passes the complete predeclared continuation screen.** Do not launch an adaptive-anchor extension or more cloud training on the strength of these results.

## What was compared

The [protocol](performance-protocol.md), [machine-readable plan](performance-plan.json), data builder, runner and initial synthetic tests were committed as `428d9a1` before quality measurements. Five methods each received two learning-rate choices, the same fresh seed-7 initialization, a 5,096-parameter LayerNorm model and exactly 21.6 million backward examples. The complete local screen therefore processed 216 million backward examples, counting MIME's full anchors and both gradients in its correction. This is repeated training work, not 216 million unique flows.

All inputs came from the historical two-million-row training view. The new fitting cohort contains 500,000 raw unique vectors; the new development set contains 399,552 rows. Another 398,502 rows remain reserved for confirmation. A new scaler and class weights were fitted only on fitting rows. The label-independent float32 overlap check excluded 674 development rows and 1,126 confirmation rows before scoring. Original validation and final-test arrays remained unopened.

These rows have been used in earlier research as training data, and the new development set selects the learning rates reported below. This is **within-collection, single-training-seed, single-client-partition development evidence**, not an untouched final test or a claim about unseen devices or capture sessions. The centralized comparator is the small matched model, not centralized-heavy. The historical completed-study scores are not interchangeable with these scores.

## Selected final endpoints

Each row uses the better final-development macro-F1 of learning rates 0.001 and 0.003. There was no best-epoch selection. Both rates and all confusion matrices are preserved in the [public evidence](performance-results.json).

| Method | Selected learning rate | Macro-F1 | Benign false alerts | Local training seconds |
| --- | ---: | ---: | ---: | ---: |
| Centralized-light Adam | 0.001 | 0.6608 | 21.96% | 50.25 |
| Shared-gradient server Adam | 0.003 | 0.6521 | 19.55% | 35.79 |
| FedAvg with reset client Adam | 0.003 | 0.4837 | 1.42% | 46.35 |
| Centralized MIME base updater | 0.003 | 0.6519 | 23.67% | 76.87 |
| Full MIME | 0.003 | 0.5466 | 21.90% | 29.86 |

All ten training loops took about 472.50 seconds in total on the local CPU with two intra-op threads. Timings include checkpoint writes but exclude setup and development evaluation. They are single-process local observations on a shared computer, not a dedicated timing benchmark or a measured network deployment. Equal backward examples do not equal identical kernel work; MIME streams its anchors in larger batches. The shorter simulated MIME time is not evidence of faster distributed deployment.

## What improved and what did not

Think of FedAvg as letting each client walk in its own direction for twenty updates, then averaging where they ended up. With different attack mixtures, those walks can disagree. The shared-gradient control instead asks all clients for their current direction at the same model, combines those directions by client population, and takes one step using the server's persistent Adam memory. The previous numerical audit established its correspondence with centralized gradient accumulation; this trial establishes that its predictive quality can also get close to ordinary centralized training under this recipe.

Shared-gradient Adam improves macro-F1 over FedAvg by 0.1685, while remaining only 0.0087 below centralized-light. Its FAR is 2.41 percentage points lower than the central control. These comparisons are encouraging, but the method changes synchronization frequency, update count and optimizer-state placement together. They do not isolate Adam reset as the sole cause of the historical federated gap.

| Class recall | Central Adam | Shared Adam | FedAvg | Central MIME base | Full MIME |
| --- | ---: | ---: | ---: | ---: | ---: |
| Benign | 78.04% | 80.45% | 98.58% | 76.33% | 78.10% |
| DDoS | 87.71% | 83.91% | 99.42% | 87.45% | 83.43% |
| DoS | 64.73% | 72.43% | 18.36% | 65.54% | 59.00% |
| Recon | 77.93% | 68.90% | 26.79% | 74.32% | 58.66% |
| Web-based | 15.15% | 12.32% | 0.00% | 12.53% | 0.00% |
| Brute Force | 33.60% | 28.00% | 19.60% | 33.20% | 29.60% |
| Spoofing | 79.25% | 76.70% | 15.47% | 82.79% | 34.10% |
| Mirai | 99.51% | 99.49% | 98.96% | 99.59% | 98.76% |

The main shared-gradient gains are broader attack recognition, especially DoS, Recon and Spoofing. Its Web-based recall rises above zero, but it still misses almost 88% of those development examples; Brute Force recall remains 28%. Centralized training is also weak on these classes. The fitting cohort contains only 599 Web-based and 332 Brute Force examples, distributed across 11 and 9 of the 20 clients respectively. This experiment does not resolve their data or representation limitations.

FedAvg's low false-alert rate comes alongside poor recognition of several attacks. The shared-gradient improvement is therefore not a free Pareto improvement: it sacrifices Benign and DDoS recall relative to FedAvg while recovering several other categories. A low FAR alone does not establish a useful detector, and a high macro-F1 alone does not establish an acceptable alarm workload.

## Why full MIME did not close the gap here

Full MIME supplies global gradient corrections and frozen optimizer statistics during local steps. The implementation uses the reviewed lagged-second-moment formulation, not ordinary bias-corrected Adam, and its own central base control reaches 0.6519. Its coefficients, initial second moment and epsilon were fixed before training. [MIME Algorithm 1 and Table 5](https://arxiv.org/html/2008.03606).

Under the common work budget, MIME completes only 24 rounds and hence 24 global moment refreshes. Twelve million of its 21.6 million backward examples are full anchors; its 4.8 million local examples are differentiated twice. The central base makes 43,200 stochastic updates. This explains an important budget difference, but does not prove that too few refreshes caused the quality gap: that would require a separate matched ablation. This one bounded configuration is evidence against advancing it now, not proof that MIME cannot work on CICIoT2023.

## Communication remains costly

| Federated method | Model aggregation or gradient rounds | Logical barriers | Modeled payload per selected run |
| --- | ---: | ---: | ---: |
| FedAvg | 108 | 108 | 88,058,880 bytes |
| Shared-gradient Adam | 2,160 | 2,160 | 1,761,177,600 bytes |
| Full MIME | 24 | 48 | 58,705,920 bytes |

Payload assumes 20 participating clients and 20,384 bytes per FP32 model-sized tensor. Shared Adam and FedAvg each exchange two model-sized tensors per client per round; this same-client MIME protocol exchanges six, with cached anchor uploads. MIME requires an anchor-gradient barrier followed by an updated-model barrier. All figures exclude transport overhead, security protocols and synchronization latency. They are not measured traffic, and this simulated setup supplies no differential-privacy guarantee.

MIME uses one-third fewer modeled bytes than FedAvg at this backward-work budget, while improving macro-F1 by 0.0629. Its much higher false-alert rate prevents that pair of gains from establishing an overall improvement. Shared Adam's near-central quality costs twenty times FedAvg's modeled payload. No communication-matched quality comparison was run.

## Frozen decision and next step

Shared Adam passes the macro-F1 improvement, central-gap and central-FAR criteria. It fails the FedAvg-FAR allowance and maximum per-class recall-regression criterion. MIME passes the FedAvg macro-F1 improvement and central-FAR criteria, but fails the central-gap, FedAvg-FAR and class-regression criteria. The thresholds were not relaxed after observing scores.

The most useful next local diagnostic is to compare the saved models at common false-alert operating points. That would ask whether shared Adam retains its attack-recognition advantage when all methods are allowed the same benign alarm burden. It needs a separately frozen rule and must be labeled development analysis; it is not permission to claim that the current gate passed or to score the reserved confirmation set. It requires no new classifier, human-review layer or cloud training.

Keep adaptive MIME and broader architecture searches on hold. If a useful alarm-matched advantage survives, a subsequent study could investigate reducing the communication cost of the shared-gradient reference. For now, the defensible conclusion is narrower but useful: **near-central aggregate performance is attainable by this federated calculation path; achieving it without expensive synchronization or a large alarm burden remains unresolved.**

## Verification and artifacts

The saved endpoints are independently reloaded and rescored by [the audit command](../../src/eval/shared_optimizer_audit.py). Its receipt, training results, source hashes, data hashes, exact work counts and all class metrics are in [performance-results.json](performance-results.json). This replay verifies saved-model inference, not an independent training seed. Synthetic tests cover exact serialized recovery for all five runners, client ownership, splitting, scaler isolation, precision-collision handling and decision criteria.

All ten saved endpoints reproduced their development metrics exactly. The 39 focused tests passed. The complete committed suite passed **707 tests in 56.725 seconds** in an isolated archive of `4022956`, excluding unrelated uncommitted files. Local report links and Git whitespace checks also passed. These are local results, not a new GitHub CI run.

Private arrays, row IDs and model weights remain under ignored `outputs/` directories. Original source comments and unrelated working-tree changes are preserved. No cloud machine was used and this experimental branch remains local.

The proposed alarm-target follow-up is now complete: see [the separate diagnostic findings](alarm-findings.md). It preserves this experiment's results and failed continuation screen.
