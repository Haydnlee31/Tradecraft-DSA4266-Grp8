# Improving simulated federated intrusion classification through shared optimization

## Methods, results and discussion draft

Tradecraft | CICIoT2023 official39 extension | 14 October 2026

### Abstract

Federated intrusion classification must balance class coverage, false alerts and communication cost. In the earlier official39 experiment, a controlled non-IID model substantially underperformed a matched centralized-light model, particularly for several minority attack categories. This extension investigated whether aggregating client gradients through a persistent server optimizer could recover balanced classification quality, and whether blockwise error-feedback compression could reduce the resulting communication burden.

Four training designs used the same 5,096-parameter network, a 500,000-row fitting cohort and 21.6 million processed training examples per endpoint. Following development-only selection, twelve checkpoints and their alarm thresholds were frozen. Evaluation on 398,502 reserved confirmation rows showed mean macro-F1 of 0.6535 for centralized-light, 0.6361 for uncompressed shared Adam, 0.6325 for compressed shared Adam and 0.5187 for FedAvg. The compressed method passed all five predeclared safeguards at each of three training seeds. Its counted application payload was 86.4% lower than uncompressed gradient sharing, but remained 2.73 times the FedAvg payload and required 20 times as many synchronization rounds.

The results support a useful quality-communication trade-off in this simulated setting, rather than a general rare-class solution. Compressed-model Web-based and Brute Force recalls remained 9.87% and 26.28%. The cohort was reserved for this extension within a previously studied collection; it was not an independent capture-session benchmark. The contribution is a reproducible empirical design and analysis, not a newly established optimization theorem or a deployment-ready detector.

### Research questions

1. Does persistent shared optimization improve eight-class macro-F1 over the tested FedAvg configuration at comparable observed false-alert rates?
2. Can blockwise four-bit compression retain most of that quality while reducing the shared-gradient communication payload?
3. Which class-level and synchronization costs remain after the aggregate improvement?

This chapter concerns the shared-optimizer extension. The original 46-feature heavy/light study, the earlier official39 baseline and the current reserved-confirmation study have different data and evaluation protocols. Their results must remain separately labeled.

<!-- pagebreak -->

## 1. Training designs and fairness of the comparison

![Figure 1. Training designs and reserved data workflow](figures/01_design.png)

Figure 1. All methods use one inference architecture. Top arrows show workflow order, not movement of rows between cohorts. The lower diagrams describe a single-machine simulation, not physical networking.

The network maps 39 standardized inputs through hidden layers of 64 and 32 units to eight classes. LayerNorm is used without dropout. The fitting-only scaler is shared across methods. A square-root class-weighted cross-entropy loss uses one fixed training-mean denominator. Twenty virtual clients retain one fixed non-IID ownership partition. Each endpoint processes 21.6 million training examples; this is not equality of FLOPs, wall time, effective batch size or optimization trajectory.

The FedAvg control performs 20 local Adam steps before model averaging and resets each client optimizer at the start of a round. Centralized-light retains a pooled Adam optimizer. Shared Adam combines population-weighted client gradients at each synchronization and applies one persistent server Adam update. It uses 2,160 synchronized updates, whereas centralized Adam performs 43,200 minibatch updates and FedAvg uses 108 communication rounds. Consequently, this comparison changes optimizer placement and update scheduling together; it cannot attribute the entire gain to optimizer memory alone.

Federated averaging follows the iterative model-aggregation principle introduced by [McMahan et al. (2017)](https://proceedings.mlr.press/v54/mcmahan17a.html). Adam uses adaptive first- and second-moment estimates [Kingma and Ba (2015)](https://arxiv.org/abs/1412.6980). These mechanisms are established prior work; the experimental contribution here is their controlled comparison for this task.

<!-- pagebreak -->

## 2. Cohort construction and evaluation controls

![Figure 2. Fitting and confirmation class counts](figures/09_class_support.png)

Figure 2. Counts use logarithmic axes to keep rare classes visible. They are cohort sizes, not counts multiplied by the number of model seeds.

A two-million-row historical training view supplied the extension. A deterministic raw-feature hash assigned fitting-eligible, development and confirmation membership. Exactly 500,000 rows were retained for fitting, 399,552 for development and 398,502 for confirmation after the predeclared precision-collision rule. Scaling was fitted only on the fitting rows. Standardized feature groups crossing cohort boundaries were removed with fitting taking priority, then development, then confirmation. No such cross-cohort groups remained.

Development data supported method choices and benign-alarm calibration. For each model, an attack score is the largest attack logit minus the benign logit. The highest-scoring attack category is predicted only when that score strictly exceeds the fixed threshold; threshold ties remain benign. Thresholds were calibrated at 1%, 5%, 10% and 20% benign false-alert targets. The primary comparison was fixed at 5% before confirmation opened.

The twelve checkpoints cover four methods at training seeds 7, 17 and 27. No checkpoint, seed, threshold or scaler was selected again on confirmation. The cohort contains only 500 Web-based and 260 Brute Force rows, making their counts particularly important. The same rows are scored by all models, so three training seeds do not constitute three independent datasets. Historical use of the underlying collection also limits the strength of any generalization claim.

<!-- pagebreak -->

## 3. Primary confirmation performance

![Figure 3. Macro-F1 and actual false-alert rates](figures/02_primary.png)

Figure 3. Small markers identify individual training seeds; black diamonds show means. Connecting spans show the observed range, not confidence intervals. The dashed line marks the development calibration target, not a guaranteed confirmation rate.

| Method | Macro-F1 mean | Sample SD | Actual false alerts |
| --- | ---: | ---: | ---: |
| Centralized-light Adam | 0.6535 | 0.0076 | 5.30% |
| Shared Adam | 0.6361 | 0.0033 | 5.33% |
| Compressed shared Adam | 0.6325 | 0.0038 | 5.36% |
| FedAvg with client Adam | 0.5187 | 0.0033 | 5.27% |

Compressed shared Adam improved mean macro-F1 by 11.39 percentage points over FedAvg, retained all but 0.36 points of uncompressed shared Adam's quality, and remained 2.10 points below centralized-light. It closed approximately 84.4% of the mean FedAvg-to-centralized gap. This describes closeness in this comparison, not statistical equivalence. Centralized gaps at seeds 7, 17 and 27 were 2.80, 1.44 and 2.06 points respectively.

All primary compressed-versus-reference comparisons had actual false-alert differences within the predeclared 0.5-percentage-point allowance. Nevertheless, compressed-model actual rates ranged from 5.29% to 5.45%. The method therefore does not meet a strict deployment requirement of at most 5% false alerts. Three-seed sample standard deviations describe training variation, not uncertainty across devices, captures or client partitions.

<!-- pagebreak -->

## 4. Coverage across the eight classes

![Figure 4. Per-class recall](figures/03_recall.png)

Figure 4. Cells show equal-weight means of correct-category recall across seeds. Every class shares a 0-100% color scale. Supports refer to one confirmation population.

The macro-F1 improvement is not confined to the two rarest categories. Compared with FedAvg, compressed shared Adam raises mean DoS recall from 20.61% to 72.25% and Spoofing recall from 20.66% to 55.17%. Recon also improves, from 42.70% to 47.53%. These changes recover useful coverage of categories that FedAvg handles poorly in this fixed non-IID configuration.

The improvement is not classwise dominance. DDoS recall falls from FedAvg's 99.07% to 83.93%, while Mirai remains near 99.5%. Web-based recall rises from almost zero but remains below 10%; Brute Force recall reaches 26.28%, compared with 24.87% for FedAvg and 30.00% centrally. A statement that the method "solves rare attacks" would therefore be inconsistent with the observed results.

Macro-F1 is useful because it gives each class equal weight, but it cannot replace class-level inspection. High detection rates for common attacks can coexist with serious minority-class failures, and a benign false-alert rate must be reported alongside category performance.

<!-- pagebreak -->

## 5. Error destinations and interpretation

![Figure 5. Mean row-normalized confusion matrices](figures/04_confusion.png)

Figure 5. True classes are rows and predicted classes are columns. Each matrix averages three row-normalized matrices with identical supports. Blank cells are exact zero; small nonzero cells are marked below 0.5%. The common color scale permits method comparisons.

The benign column distinguishes suppressed attack alerts from incorrect attack labels. A DoS flow predicted as DDoS remains an eight-class error, but can still raise an attack alarm. A flow predicted as benign may not raise one. This distinction explains why binary attack-detection recall can look strong even when macro-F1 and rare-category identification remain inadequate.

<!-- pagebreak -->

## 6. Rare-class detection and identification are different tasks

![Figure 6. Rare-class error decomposition](figures/05_rare_errors.png)

Figure 6. The three outcomes sum to 100% for each method and class. Correct identification is a subset of all attack alerts; the red segment represents missed attacks predicted benign.

Compressed shared Adam correctly identifies 41, 42 and 65 of the 500 Web-based examples at the three seeds. For Brute Force, the corresponding counts are 66, 73 and 66 out of 260. In mean terms, 45.0% of Web-based examples receive some attack label, but only 9.87% receive the correct category. Brute Force attack-alarm recall is 47.56%, compared with 26.28% correct-category recall.

Its overall binary attack-detection recall is 97.22%, heavily influenced by common classes. Presenting that value as the headline would conceal the remaining rare-class problem. Even centralized-light achieves only 15.60% Web-based and 30.00% Brute Force recall. The evidence is compatible with limitations in representation, overlap or available minority information, but this experiment does not isolate their causes.

Compression can also have a meaningful local cost while passing an aggregate safeguard. At seed 17, Web-based recall falls from uncompressed shared Adam's 12.6% to 8.4%: 21 fewer correctly identified examples. This remains within the predeclared five-percentage-point tolerance, but should not be described as "no loss."

<!-- pagebreak -->

## 7. Compression and communication trade-offs

![Figure 7. Communication and synchronization costs](figures/06_communication.png)

Figure 7. Left: mean primary macro-F1 with sample SD, against counted application payload on a logarithmic axis. Right: synchronization counts. The centralized reference line represents quality only; centralized raw-data collection cost is unknown, not zero.

The compressed method quantizes consecutive blocks of 128 tensor coordinates using a block scale and symmetric four-bit values. Client residual feedback retains the difference between the corrected gradient and its transmitted approximation. The server keeps full-precision Adam moments and a target model; compressed target-minus-delivered updates carry undelivered information into later broadcasts. Gradients are computed at the delivered approximation, so the compressed trajectory is not mathematically identical to full-precision Adam.

Error feedback has established foundations in gradient-compression research [Karimireddy et al. (2019)](https://proceedings.mlr.press/v97/karimireddy19a.html). That motivates retaining omitted information; it does not transfer an EF-SGD convergence theorem automatically to this compressed Adam implementation.

| Federated method | Application payload | Synchronizations |
| --- | ---: | ---: |
| Shared Adam | 1,761.18 MB | 2,160 |
| Compressed shared Adam | 240.26 MB | 2,160 |
| FedAvg | 88.06 MB | 108 |

Compression saves 86.4% of historical shared-gradient payload but still uses 2.73 times the FedAvg payload and 20 times its synchronization rounds. Serialized compressed accounting includes startup and message metadata; the historical raw baselines do not. These are application-byte counts, not measured network traffic, latency, energy or privacy. The trade-off could be unfavorable on high-latency or intermittently connected clients despite improved macro-F1.

<!-- pagebreak -->

## 8. Operating points and fixed success criteria

![Figure 8. The four predeclared operating points](figures/07_operating_points.png)

Figure 8. Points show actual confirmation rates at thresholds previously fixed on development. Lines connect four observed points only; they are neither fitted curves nor an ROC analysis. Large rings mark the primary target. No new threshold sweep was performed.

Ordinary argmax produces compressed-model mean macro-F1 of 0.6513 with 17.91% benign false alerts. Centralized-light gives 0.6646 and 20.28%, whereas FedAvg gives 0.4888 and 1.17%. Those raw results represent materially different alarm burdens. The primary calibrated comparison reduces that mismatch and retains a substantial federated quality gain.

Each seed independently passed the unchanged five safeguards: at most one macro-F1 point lost to shared Adam; at most 0.5 points of actual false-alert difference from shared Adam; no class losing more than five recall points; at least eight F1 points gained over FedAvg; and at most 15% of historical shared-gradient payload. A favorable mean was not used to rescue a failed seed. These are engineering tolerances, not statistical equivalence tests or operational deployment gates.

At the secondary 20% target, actual false-alert differences exceed the 0.5-point comparability allowance for seed 17 versus shared Adam and FedAvg, and seed 27 versus shared Adam. Those secondary comparisons are not alarm-matched. They do not change the primary conclusion, and no secondary threshold was promoted after confirmation.

<!-- pagebreak -->

## 9. Relation to the earlier official39 study

![Figure 9. Earlier baseline results under their original protocol](figures/08_historical_context.png)

Figure 9. This historical comparison uses 2,040,729 final-test rows, two million training rows, 20 training steps/rounds and argmax predictions. It remains a separate population and protocol, not a directly comparable panel of the optimizer experiment.

The earlier full39 baseline yielded mean macro-F1 of 0.65673 for centralized-light, 0.61169 for IID federation and 0.40494 for controlled non-IID federation. Its non-IID benign false-alert rate was only 0.25%, but both Web-based and Brute Force recalls were exactly zero. Low false-alert burden alone therefore did not establish a useful detector. This result motivated investigation of the update procedure as well as rare-class representation.

The later experiment cannot be described as a test-set score rising from 0.40494 to 0.6325. It uses a different fitting cohort, update schedule, model-selection boundary and alarm calibration. The valid improvement is the paired within-extension comparison against its own FedAvg control, 0.5187 to 0.6325. Keeping the two studies separate preserves both the earlier failure and the later positive finding.

The original 46-feature study also included a heavy model, but the official39 experiments do not contain a matched heavy arm. They cannot establish that the light architecture replaces the heavy model. Existing explanation work should retain its actual method labels: the official39 background-averaged integrated-gradients analysis is not a newly performed SHAP analysis of these twelve optimizer endpoints.

<!-- pagebreak -->

## 10. Validity, limitations and reproducibility

### Internal validity

The confirmation comparison used the same input schema, fitting cohort, loss definition, per-endpoint training-example budget and reserved population. Twelve checkpoints and 48 development-calibrated thresholds were frozen before scoring. Every checkpoint reproduced its saved development predictions. Confirmation reconstruction replayed the original row-membership and precision-collision rules from checksummed Parquet shards. A separate audit reproduced all twelve argmax and 48 thresholded metric endpoints from saved outputs.

The study is not a single-factor test. Shared optimization changes synchronization frequency and effective update batching as well as optimizer placement. FedAvg specifically uses reset-client Adam; other persistence policies, optimizers or partial participation could produce different results. One non-IID partition was used in the final extension. These restrictions prevent a claim that the method is universally superior to federated averaging.

### External validity and deployment limits

The source collection was used in earlier project research. Reserving a subset for this extension reduces direct model-selection reuse but does not create an independently collected benchmark. Duplicate-group checks address exact model-input overlap, not temporal, scenario or capture-session independence. Models evaluated at different training seeds share the same rows and client partition, so their spread is not a confidence interval across deployments.

All clients were simulated on one machine. There are no measured physical-edge, network-delay, energy or privacy results. The compressed detector still produces more than 5% actual benign false alerts and misses many rare attacks. It should not be presented as an autonomous production allow/block system. Feature-level security hypotheses from earlier explanations remain hypotheses, not causal rules or device-specific vulnerability claims.

### Reproducibility of the visual analysis

All nine figures in this draft are rebuilt from committed aggregate JSON and frozen manifest counts. No model was retrained, no inference was repeated, and no private array or checkpoint was loaded for visualization. Source hashes, derived numerical figure data, PNG/SVG hashes and the generator identity are retained with the figure pack. The latest updater runs did not save validation scores at every training step; a convergence trace cannot be reconstructed from endpoint scores or progress counters. No such curve has been invented here.

<!-- pagebreak -->

## 11. Conclusion and thesis integration

In this simulated non-IID intrusion-classification setting, population-weighted gradient aggregation through a persistent server optimizer substantially improves balanced eight-class performance over the tested FedAvg configuration. Four-bit blockwise residual-feedback compression preserves most of that improvement at a substantially smaller shared-gradient payload. The conclusion held across three training seeds on the extension's reserved confirmation cohort, under unchanged per-seed safeguards.

The strongest claim is a measured trade-off: 11.39 macro-F1 points gained over the paired FedAvg control, 0.36 points lost to uncompressed shared Adam and 86.4% less shared-gradient application payload. The remaining costs are frequent synchronization, a 2.10-point mean gap to centralized-light, and weak rare-category identification. Neither rare-class resolution nor centralized equivalence is established.

The thesis should present this as an optimizer and communication extension to the completed centralized-versus-federated study. A coherent full-study narrative is: establish trustworthy data and baseline comparisons; diagnose class-specific failures; test update designs under explicit budgets; confirm one fixed candidate; and interpret quality, alarm burden and communication together. The abandoned classifier searches need not dominate that narrative; their role is limited to explaining why the final question focused on training updates rather than another prediction layer.

Further client-partition robustness, independent-capture validation and real-network synchronization measurements would strengthen a future study. They require new evaluation designs and, where appropriate, fresh data. They are not reasons to keep tuning against the now-open confirmation cohort or to spend the remaining cloud credits simply because they are available.

## References

1. McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS, PMLR 54, 1273-1282. [Primary paper](https://proceedings.mlr.press/v54/mcmahan17a.html).
2. Kingma and Ba (2015). Adam: A Method for Stochastic Optimization. ICLR. [Primary paper](https://arxiv.org/abs/1412.6980).
3. Karimireddy et al. (2019). Error Feedback Fixes SignSGD and other Gradient Compression Schemes. ICML, PMLR 97, 3252-3261. [Primary paper](https://proceedings.mlr.press/v97/karimireddy19a.html).
4. Tradecraft. Audited reserved-confirmation results and frozen protocol. [Aggregate evidence](../confirmation-results.json), [execution plan](../confirmation-execution-plan.json), [final findings](../confirmation-findings.md).
5. Tradecraft. Earlier official39 final-test synthesis. [Aggregate evidence](../../federated_class_safeguards/study-synthesis-results.json), [completed-study handoff](../../full_data_extension/final-closeout.md).
