# Shared optimizer confirmation and final trade-offs

14 October 2026. **The improvement carries to the reserved confirmation cohort.** Blockwise four-bit shared-gradient Adam passes all five predeclared safeguards on each of the three training seeds. Its mean macro-F1 is **0.6325**, compared with **0.5187 for FedAvg** and **0.6535 for centralized-light**. It preserves most of the uncompressed shared optimizer's quality while reducing that method's modeled application payload by **86.4%**.

This is a positive result for the centralized-versus-federated thesis, not a solution to rare attacks. The method still has weaker rare-class recognition than centralized-light, costs 20 times as many synchronization rounds as FedAvg, and has not been tested on physically distributed clients or new capture sessions. The planned search ends here; the next use of these results is synthesis, not more tuning on this cohort.

## What the comparison means

All four methods use the same small 39-input, 64/32-hidden-unit, eight-output LayerNorm network with 5,096 parameters. Centralized-light has pooled access during training; the federated methods simulate twenty clients with a fixed non-IID distribution. This is **not** a comparison against the project's separate centralized-heavy model.

FedAvg lets clients take several local Adam steps before averaging their models. Shared-gradient Adam instead combines their gradients at each synchronization and applies a persistent server-side Adam update. The compressed version sends four-bit values in blocks of 128 coordinates, with a scale per block; residual feedback carries omitted information into later messages. Both uploads and delivered model updates are compressed. Evaluation uses the delivered model, not the hidden full-precision server target.

The comparison holds architecture, fitting rows, loss and backward-example budget fixed at 21.6 million examples per trained endpoint. It does not hold synchronization count, effective aggregate batch size or optimizer-step schedule fixed. The result therefore supports this combined training design; it does not isolate a single causal explanation such as client drift. Gradient aggregation, Adam and error-feedback quantization are established mechanisms. The contribution is their tested combination and explicit trade-off analysis in this study, not a claim to have invented those mechanisms.

## Final quality at the locked operating point

The primary threshold for each model was fixed earlier using the **5% development-calibration false-alert target**. No threshold was changed using confirmation outcomes. All primary compressed-versus-reference comparisons fall within the predeclared 0.5-percentage-point allowance for actual false-alert differences.

| Method | Mean macro-F1 | Sample SD across seeds | Mean actual benign false alerts |
| --- | ---: | ---: | ---: |
| Centralized-light Adam | 0.6535 | 0.0076 | 5.30% |
| Uncompressed shared-gradient Adam | 0.6361 | 0.0033 | 5.33% |
| Blockwise four-bit shared-gradient Adam | 0.6325 | 0.0038 | 5.36% |
| FedAvg with client Adam | 0.5187 | 0.0033 | 5.27% |

The compressed method gains **11.39 macro-F1 points over FedAvg**, loses **0.36 points to uncompressed shared Adam**, and remains **2.10 points behind centralized-light** on average. It closes approximately **84.4% of the mean FedAvg-to-centralized gap**. This is descriptive closeness, not statistical equivalence or a guarantee of being within two points: the per-seed centralized gaps are 2.80, 1.44 and 2.06 points.

Actual false alerts are not guaranteed to equal the calibration target. The compressed model's rates range from 5.29% to 5.45%, so it does not satisfy a strict deployment requirement of at most 5%. No confirmation recalibration was used to make that number look better.

## Every seed passes the unchanged safeguards

| Seed | Compressed macro-F1 | Actual false alerts | F1 loss to shared Adam | Largest class recall loss to shared Adam | All five safeguards |
| --- | ---: | ---: | ---: | --- | --- |
| 7 | 0.6303 | 5.45% | 0.475 points | 4.021 points, Spoofing | Pass |
| 17 | 0.6304 | 5.33% | 0.307 points | 4.200 points, Web-based | Pass |
| 27 | 0.6369 | 5.29% | 0.286 points | 3.431 points, Spoofing | Pass |

The fixed limits were at most one F1 point lost to shared Adam, at most 0.5 points of actual false-alert difference, at most five recall points lost in any class, at least eight F1 points gained over FedAvg, and at most 15% of the historical shared-gradient payload. Each seed passed individually; averaging was not used to rescue a failure. Seed 7 remains the discovery training seed, although its confirmation outcomes are new.

A tolerance is not the absence of a cost. In seed 17, Web-based recall drops from 12.6% to 8.4% under compression: 21 fewer correctly named Web-based examples out of 500. That passes the five-point safeguard but remains a meaningful weakness. The earlier failed uncalibrated training screen is not rewritten by this later, separately defined compression result.

## Where the improvement comes from and what remains weak

Mean correct eight-class recalls at the primary operating point are:

| Class | Centralized-light | Shared Adam | Compressed shared Adam | FedAvg |
| --- | ---: | ---: | ---: | ---: |
| Benign | 94.70% | 94.67% | 94.64% | 94.73% |
| DDoS | 86.24% | 83.02% | 83.93% | 99.07% |
| DoS | 67.90% | 73.85% | 72.25% | 20.61% |
| Recon | 50.27% | 46.86% | 47.53% | 42.70% |
| Web-based | 15.60% | 11.87% | 9.87% | 0.20% |
| Brute Force | 30.00% | 27.18% | 26.28% | 24.87% |
| Spoofing | 65.92% | 58.23% | 55.17% | 20.66% |
| Mirai | 99.62% | 99.57% | 99.51% | 99.00% |

The strongest improvements over this FedAvg configuration are in DoS and Spoofing, with a smaller Recon improvement. DDoS recall moves in the opposite direction: FedAvg remains better for that dominant class. The shared methods improve the balance of eight-class performance rather than winning every class.

The compressed model correctly names 41, 42 and 65 of the 500 Web-based examples, and 66, 73 and 66 of the 260 Brute Force examples. These same rows are evaluated by three models; the supports must not be multiplied by three and presented as independent data. Even centralized-light only reaches mean recalls of 15.6% and 30.0% for those classes. The study has not resolved their information, imbalance or representation limitations.

Detecting an attack is easier than naming its category. For the compressed method, an average 45.0% of Web-based examples trigger some attack label, but only 9.87% receive the correct Web-based label. For Brute Force, those figures are 47.56% and 26.28%. Its overall attack-detection recall is 97.22%, dominated by common attacks; that high binary figure must not replace macro-F1 and the rare-class results.

The ordinary argmax predictions are retained in the evidence. They give compressed-model mean macro-F1 0.6513 but 17.91% benign false alerts; centralized-light gives 0.6646 and 20.28%, while FedAvg gives 0.4888 and 1.17%. Those are substantially different operating points. The primary calibrated comparison is therefore more informative than treating the raw scores as an alarm-matched contest.

## Communication savings do not remove synchronization cost

| Federated method | Application payload per training run | Synchronizations |
| --- | ---: | ---: |
| Uncompressed shared Adam, historical raw accounting | 1,761.18 MB | 2,160 |
| Compressed shared Adam, serialized accounting | 240.26 MB | 2,160 |
| FedAvg, historical raw accounting | 88.06 MB | 108 |

The compressed accounting includes packed values, block scales, tensor counts, sequence headers, the initial FP32 broadcast and all twenty recipients. It is 13.64% of the historical uncompressed shared payload, but **2.73 times FedAvg's payload**. The historical raw baselines exclude startup/message metadata; this makes the savings comparison against shared Adam conservative, not a measurement of end-to-end network traffic.

The useful trade-off is therefore: better aggregate quality than this FedAvg configuration, with much smaller messages than uncompressed gradient sharing, but still frequent synchronization. A high-latency or intermittently connected deployment may prefer FedAvg despite the quality loss. This simulation measures neither real network delay nor physical edge feasibility, and compression supplies no privacy guarantee.

All four threshold targets remain published. At the secondary 20% target, actual false-alert differences exceed the 0.5-point comparability allowance for seed 17 against shared Adam and FedAvg, and seed 27 against shared Adam. Those comparisons are explicitly marked not alarm-matched. Secondary results do not change the primary decision.

## What was verified and what the evidence supports

The [execution plan](confirmation-execution-plan.json), [reconstruction](../../src/data/shared_optimizer_confirmation.py), [scoring and audit runner](../../src/eval/shared_optimizer_confirmation.py) and eight preliminary tests were frozen in commit `6cb8629` before opening the cohort, after the user's explicit authorization. The prior [lock](confirmation-preparation-results.json) was consumed unchanged: twelve checkpoint identities and 48 existing thresholds, with no model fits, threshold refits, scaler fits or checkpoint selection.

The loader streamed two million historical training rows from checksum-verified Parquet shards, reproduced the 899,552 fitting/development anchors exactly, replayed all original split memberships and precision-collision removals, and found no standardized input groups crossing cohort boundaries. The confirmation cohort contains 398,502 rows and 398,281 distinct standardized input groups. The unhashed temporary staging cache was not used.

One inference pass per model evaluated 4,782,024 examples in total. The local CPU run took 15.01 seconds including reconstruction and scoring; inference itself took 0.73 seconds. Process peak resident memory was 843.1 MiB. A separate audit process verified saved logits and independently reproduced all 60 metric endpoints: twelve argmax results plus 48 frozen-threshold results. It did not retrain, refit or run another model search. No cloud was used.

The [aggregate evidence](confirmation-results.json) retains confusion matrices, per-class precision/recall/F1/support, attack-detection diagnostics, seed comparisons, hashes and audit checks. Private arrays, logits and checkpoints remain outside Git. Original validation and test were not reopened in this extension.

This is a within-collection confirmation reserved for the current extension, not a completely untouched external benchmark: the source rows belong to historical training data used earlier in the project. All seeds share one client partition and one confirmation cohort. The three-seed standard deviations describe training variation; they do not establish population-level statistical significance or capture-session generalization.

## Final study conclusion and next action

The defensible conclusion is: **in this simulated non-IID setting, sharing gradients through a persistent server optimizer substantially improves balanced eight-class performance over the tested FedAvg configuration. Blockwise residual-feedback compression retains most of that improvement while reducing the shared-gradient payload, at the cost of frequent synchronization and continuing rare-class weakness.**

Use this as the completed extension to the centralized-versus-federated thesis. Keep centralized-light as the aggregate-quality reference and preserve the original heavy/light experiment as a separate comparison; do not merge metrics from different cohorts into one leaderboard. Do not reopen the discarded classifier search or use this confirmation panel to tune another method.

No additional cloud session is needed to finish this comparison. The remaining work is integrating these tables and the algorithm explanation into the team report, alongside the project's existing efficiency and explainability findings. Further partition robustness, real-network latency or external-capture validation would be separate future studies with fresh evaluation data—not requirements to keep spending the remaining credits now.
