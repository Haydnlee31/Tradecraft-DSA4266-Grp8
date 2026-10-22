# Evaluation of FedNTD and Synchronization-Interval Extensions

Tradecraft — CICIoT2023 eight-class network-flow classification  
22 October 2026

## Abstract

Two extensions were evaluated to investigate whether the existing federated training approach could be improved without modifying the original pipeline. The first added not-true distillation (FedNTD) to the FedAvg training procedure. The second increased the interval between client–server synchronizations using a normalized FedOpt-inspired update. Neither extension improved macro-F1 over its matched baseline under the tested settings. FedNTD reduced benign false alerts but slightly reduced macro-F1, while longer synchronization intervals substantially reduced modeled communication at the cost of larger macro-F1 and class-recall losses. These results identify practical trade-offs rather than establish that either method is ineffective in general. The extension experiments used one seed and an existing development cohort; their outcomes are exploratory.

## 1. Objective and experimental setting

The task is to classify CICIoT2023 network flows into eight classes: Benign, DDoS, DoS, Recon, Web-based, Brute Force, Spoofing and Mirai. Macro-F1 is the principal classification metric because dominant attack categories can conceal poor performance on less common classes when only accuracy is reported. Per-class recall measures how many examples of each category are correctly recognized. The benign false-alert rate measures the proportion of benign flows classified as any attack.

Both extensions used the official39 prepared data: 500,000 fit rows, 399,552 development rows and 20 simulated clients with a fixed non-IID partition. The model was a 39→64→32→8 MLP with LayerNorm, zero dropout and 5,096 parameters. Training used the existing square-root inverse-frequency weighted cross-entropy with a fixed training-mean normalization. Aggregation weighted clients by their assigned population sizes.

Every arm used seed 7 and processed 21.6 million backward examples. Ordered client minibatch streams were matched within the comparisons. Batch size was 500, and the applicable learning rates were fixed at 0.003. Adam used betas (0.9, 0.99), epsilon 0.001 and zero weight decay. Final development predictions used ordinary argmax. No extension selected a best checkpoint, fitted an alarm threshold or scored confirmation/test data.

### Distinguishing the original results

The previously reproduced **0.632537 mean macro-F1** belongs to compressed Shared Adam across three seeds on the confirmation cohort, with a calibrated alarm rule. It is not directly comparable to the extension development scores because the evaluation cohort, decision rule and seed aggregation differ.

Two matched controls are therefore used in this report:

- **FedAvg with reset client Adam:** macro-F1 0.483683, the control for adding FedNTD.
- **Uncompressed shared-gradient Adam at synchronization interval K=1:** macro-F1 0.652142, the control for changing synchronization intervals.

Both controls reproduced their historical seed-7 endpoints exactly. The FedAvg control was also identical across the two extension studies. FedNTD was tested as an addition to FedAvg, not as an addition to the compressed Shared Adam method.

## 2. Extension one: FedNTD

### 2.1 Method and motivation

FedNTD adds a teacher–student consistency term during local training. At the beginning of each round, the global model is copied into a frozen teacher. Each client trains its local student using labeled examples and the teacher's predictions.

For an example with true class y, the true-class logit is removed from both models before their remaining logits are normalized. Distillation therefore transfers the teacher's relative probabilities among the seven incorrect classes. The implemented loss is:

`loss = weighted_cross_entropy + β × T² × KL(teacher_nontrue || student_nontrue)`

The teacher runs in evaluation mode and receives no gradients. Temperature T was fixed at 1, and β was tested at 0, 0.1 and 1. The β=0 arm skips teacher computation and matches the original FedAvg control. Each arm used 108 rounds with 20 local Adam steps per client per round; local Adam state was reset each round.

The motivation is to reduce the loss of global knowledge during training on heterogeneous client data. A client with a narrow attack-category distribution may otherwise adapt strongly to its own examples. Preserving relationships among non-true classes could help retain distinctions learned globally. This implementation is an adaptation to the Tradecraft model and training recipe, rather than a claim to reproduce all settings of the original FedNTD paper.

### 2.2 Measured results

- **FedAvg control, β=0:** macro-F1 **0.483683**; benign false-alert rate **1.425%**; 108 synchronization rounds.
- **FedNTD, β=0.1:** macro-F1 **0.478103**; benign false-alert rate **0.840%**; 108 synchronization rounds.
- **FedNTD, β=1:** macro-F1 **0.470483**; benign false-alert rate **0.429%**; 108 synchronization rounds.

Relative to FedAvg, β=0.1 reduced macro-F1 by **0.558 percentage points**, and β=1 reduced it by **1.320 points**. Both settings reduced benign false alerts.

All-class recall is shown below in the order **FedAvg / FedNTD β=0.1 / FedNTD β=1**:

- **Benign:** 98.575% / 99.160% / 99.571%; development support 21,193.
- **DDoS:** 99.418% / 99.289% / 99.366%; development support 235,894.
- **DoS:** 18.362% / 19.846% / 18.851%; development support 72,090.
- **Recon:** 26.793% / 24.543% / 21.425%; development support 13,246.
- **Web-based:** 0.000% / 0.000% / 0.000%; development support 495.
- **Brute Force:** 19.600% / 17.200% / 22.000%; development support 250.
- **Spoofing:** 15.467% / 14.018% / 10.568%; development support 8,696.
- **Mirai:** 98.960% / 98.941% / 99.023%; development support 47,688.

Recon and Spoofing recall declined as β increased, while Web-based recall remained zero. Effects were not uniformly negative: β=1 improved Brute Force recall from 19.6% to 22.0%. Nevertheless, improvements in individual classes did not offset the overall loss in macro-F1, which reflects both precision and recall for all classes.

### 2.3 Why improvement may not have occurred

**The teacher may have preserved weak distinctions.** The round-start teacher is the existing global model, not an independently trained stronger model. The FedAvg endpoint already performed poorly on several categories. Distillation can preserve existing predictions, but it does not introduce new labeled evidence. If those predictions poorly distinguish minority categories, preserving them may not improve classification. This is a plausible explanation, not a measured diagnosis of teacher quality throughout training.

**The extra loss may have constrained useful local adaptation.** Increasing β gives the teacher-consistency term more influence relative to supervised learning. The observed combination of lower false-alert rates and lower macro-F1 is consistent with a trade-off in the learned decision boundaries. It does not show that every attack became harder to detect, as the Brute Force result demonstrates. Loss-component and gradient diagnostics would be needed to verify this mechanism.

**Rare-class data remain limited.** The fit set contains only 599 Web-based and 332 Brute Force examples out of 500,000 rows. Distillation does not add independent observations of these classes. This scarcity plausibly limits learning, particularly under heterogeneous client distributions. However, scarcity alone cannot explain the differences between arms because they used the same data.

Finally, only two nonzero β values and one temperature were tested. The results support a conclusion about this fixed recipe, not all possible FedNTD configurations.

### 2.4 Computational implications

Each nonzero-β arm added 21.6 million teacher-forward examples. Communication remained 108 rounds and 88.059 MB of modeled tensor payload for every arm. Measured host training-loop times were 59.10 seconds for FedAvg, 83.67 seconds for β=0.1 and 120.09 seconds for β=1. These are single-run timings, not repeated benchmarks; both nonzero β values require the same number of teacher forwards, so their timing difference does not establish a computational-complexity difference.

The tested settings therefore provided lower false alerts but no macro-F1 or communication improvement over FedAvg.

## 3. Extension two: synchronization intervals

### 3.1 Method and motivation

This extension investigated whether clients could communicate less frequently while retaining the original shared-gradient Adam model's quality. The synchronization interval K specifies how many local steps occur before the next aggregation. Values K=1, 4 and 20 were tested.

At K=1, clients compute gradients at common global weights. The server forms their population-weighted aggregate and applies persistent Adam. This reproduces the original uncompressed shared-gradient Adam update exactly.

At K>1, each client follows K plain local SGD steps and sends the average gradient along its local path. For constant local learning rate η, this is mathematically equivalent to:

`client_message = (round_start_weights − final_local_weights) / (η × K)`

The implementation accumulates gradients directly to avoid numerical cancellation from subtracting nearly equal weights. The server population-weights these messages and applies persistent Adam. This is a normalized FedOpt-inspired adaptation. It changes the optimization trajectory as well as synchronization frequency; it is not simply a delayed transmission of otherwise identical updates.

Compression was disabled. Two additional K=20 controls used FedAvg with local SGD and FedAvg with reset client Adam.

### 3.2 Measured results

- **Shared Adam, K=1:** macro-F1 **0.652142**; benign false-alert rate **19.554%**; 2,160 synchronization rounds.
- **Normalized FedOpt, K=4:** macro-F1 **0.611025**; benign false-alert rate **15.425%**; 540 synchronization rounds.
- **Normalized FedOpt, K=20:** macro-F1 **0.574382**; benign false-alert rate **19.799%**; 108 synchronization rounds.
- **FedAvg with local SGD, K=20:** macro-F1 **0.493947**; benign false-alert rate **29.198%**; 108 synchronization rounds.
- **FedAvg with reset client Adam, K=20:** macro-F1 **0.483683**; benign false-alert rate **1.425%**; 108 synchronization rounds.

Relative to K=1, K=4 lost **4.112 macro-F1 percentage points**, while K=20 lost **7.776 points**. Their synchronization counts and modeled tensor payloads decreased by **75%** and **95%**, respectively. Payload was 1,761.178 MB at K=1, 440.294 MB at K=4 and 88.059 MB at K=20. MB denotes 1,000,000 bytes; these totals exclude network protocol overhead.

All-class recall is shown below in the order **K=1 / K=4 / K=20 / FedAvg SGD / FedAvg Adam**:

- **Benign:** 80.446% / 84.575% / 80.201% / 70.802% / 98.575%; development support 21,193.
- **DDoS:** 83.910% / 95.384% / 87.952% / 83.721% / 99.418%; development support 235,894.
- **DoS:** 72.434% / 42.029% / 54.278% / 52.317% / 18.362%; development support 72,090.
- **Recon:** 68.904% / 68.164% / 67.930% / 68.640% / 26.793%; development support 13,246.
- **Web-based:** 12.323% / 3.434% / 0.000% / 0.000% / 0.000%; development support 495.
- **Brute Force:** 28.000% / 29.600% / 34.800% / 0.000% / 19.600%; development support 250.
- **Spoofing:** 76.702% / 59.131% / 45.435% / 25.368% / 15.467%; development support 8,696.
- **Mirai:** 99.495% / 99.392% / 98.956% / 98.329% / 98.960%; development support 47,688.

The quality loss involved several categories. DoS recall fell from 72.43% at K=1 to 42.03% at K=4. Spoofing recall fell from 76.70% at K=1 to 45.44% at K=20. Web-based recall declined from 12.32% to 3.43% and then zero. Brute Force recall improved as K increased, but this did not compensate for the broader losses.

Both interval extensions exceeded FedAvg Adam's macro-F1. However, their benign false-alert rates were substantially higher, so this does not establish superiority under a common operational alarm requirement.

### 3.3 Why improvement may not have occurred

**Local training paths diverge between synchronizations.** At K=1, gradients are evaluated at shared weights. At larger K, clients move to different weights as they train on their own data distributions. Aggregating those local-path gradients can produce a different direction from repeated shared-weight updates. Non-IID class mixtures make this a plausible source of degradation, although the experiment did not directly measure client drift.

**There are fewer server Adam updates.** The fixed example budget yields 2,160 server updates at K=1, 540 at K=4 and 108 at K=20. Processing the same number of examples does not guarantee the same optimization progress: the larger-interval models receive fewer global corrections and different Adam-state evolution.

**Learning rates were not adapted to the interval.** Local and server rates were held at 0.003. A recipe suited to frequent aggregation may not be equally suitable for longer local trajectories. Separate rate tuning might change the outcome, but no such improvement has been demonstrated by these runs.

These factors change together, so the study cannot attribute the loss solely to client drift, update count or learning rate. It establishes the net effect of the tested interval recipes.

### 3.4 Efficiency and acceptance criteria

The interval experiment required at least fourfold fewer synchronizations while allowing no more than 0.01 macro-F1 loss, 0.005 false-alert-rate increase or 0.05 recall loss in any class. Both interval candidates passed the communication and false-alert checks but failed the macro-F1 and class-recall checks.

No CPU speedup was demonstrated: the five arms took approximately 53–60 seconds of host training-loop time. Most training computation remains even when communication frequency falls. The communication reduction is an accounting result for simulated tensor exchanges, not measured network latency or edge-device performance. Its reference is uncompressed K=1, not the original four-bit compressed method.

## 4. Combined interpretation and limitations

The two extensions addressed different problems. FedNTD tried to preserve global predictive knowledge during local training; longer synchronization intervals tried to reduce client–server exchange frequency. Their respective outcomes were a false-alert–classification trade-off and a communication–classification trade-off.

Shared Adam K=1 achieved the highest development macro-F1 among these endpoints, but its 19.55% benign false-alert rate is operationally significant. It should not be called the best deployment choice solely on macro-F1. A comparison at a common alarm policy would be needed to assess that question.

The evidence is limited to one seed, one client partition, fixed hyperparameters and an already used development cohort. It does not establish statistical significance or fresh held-out generalization. The small development supports for Web-based and Brute Force also limit the stability of their recall estimates. No new confirmation/test scoring was conducted for the extensions. Suggested mechanisms in this report are explanations consistent with the observations, not experimentally isolated causes.

## 5. Implementation isolation and verification

Both approaches were implemented under separate `extensions/` directories. The original training flow was preserved. The recorded synchronization audit found all 662 previously tracked files unchanged, including the completed FedNTD implementation.

The FedNTD audit reloaded all three checkpoints and independently recomputed development metrics with scikit-learn. Its β=0 control exactly matched original weights, metrics and sampling. Eleven FedNTD tests passed.

The synchronization audit independently recomputed all five endpoint metrics and replayed each final update from saved recovery state. Weights, optimizer state where applicable, counters and sample hashes reproduced bit for bit. Eight new synchronization tests, eleven reused FedNTD tests and nineteen existing report/archive regression tests passed. Historical Shared Adam and FedAvg Adam controls matched exactly. These were self-audits supported by executable checks, not an external review or a run of the entire repository test suite.

This report summarizes those existing verified runs; preparing it did not retrain models or modify either implementation.

## 6. Conclusion

Neither extension improved macro-F1 over its matched original baseline under the tested conditions. FedNTD reduced false alerts but introduced additional computation and slightly reduced macro-F1. Longer synchronization intervals reduced modeled communication substantially but caused larger classification losses and failed the predefined quality-retention criteria.

The appropriate decision is to preserve the original flow and retain both extensions as documented experiments. Their contribution is evidence about the limits and trade-offs of this training recipe. Further work could evaluate repeated seeds, interval-specific learning rates, or teacher-quality diagnostics under a separately specified protocol, followed by comparison at a common alarm policy. Such follow-up remains future work rather than an improvement established by the present results.

## Evidence and reproducibility records

- [FedNTD frozen experiment plan](../../extensions/fedntd/plan.json)
- [FedNTD implementation and instructions](../../extensions/fedntd/README.md)
- [FedNTD audited result receipt](fedntd/real-data-results.json)
- [FedNTD detailed findings and audit reproduction](fedntd/real-data-findings.md)
- [Synchronization frozen experiment plan](../../extensions/sync_intervals/plan.json)
- [Synchronization implementation and instructions](../../extensions/sync_intervals/README.md)
- [Synchronization audited result receipt](sync_intervals/audit.json)
- [Synchronization findings and verification scope](sync_intervals/findings.md)

All numeric results in this report derive from the local audited receipts above. Percentage-point changes in macro-F1 are computed as 100 times the difference between scores on the 0–1 scale.
