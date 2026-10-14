# Centralized and federated learning for IoT intrusion classification

## Tradecraft integrated thesis draft

DSA4266 Group 8 | 14 October 2026

### Abstract

An intrusion classifier must recognize attacks without overwhelming users with false alerts. This study investigates that trade-off for compact and larger neural networks trained centrally or through simulated federation on CICIoT2023. The task distinguishes benign traffic from seven attack categories. Macro-F1, every class's recall, benign false alerts and computational or communication requirements are considered together.

Three separately evaluated studies form the investigation. In the original 46-feature configured-model comparison, a 198,984-parameter heavy network attained mean validation macro-F1 of 0.6747, compared with 0.6583 for a 5,544-parameter centralized-light model. IID and non-IID federated-light models attained 0.6328 and 0.5557. Configuration differences and validation-based checkpoint selection limit causal interpretation. A subsequent official39 experiment used the same light architecture across training settings and two million training rows. Its final-test macro-F1 means were 0.6567 centrally, 0.6117 under IID federation and 0.4049 under controlled non-IID federation. Several rare categories remained poorly recognized.

A final extension changed the update procedure rather than adding another classifier. On 398,502 reserved confirmation rows, compressed shared Adam achieved macro-F1 of 0.6325 against 0.5187 for its paired FedAvg control and 0.6535 centrally. It reduced the uncompressed shared-gradient application payload by 86.4%, while retaining 20 times the synchronization count of FedAvg. Actual benign false alerts remained approximately 5.3%, and rare-class recall remained weak.

The principal finding is that update design can recover much of the centralized performance gap in this simulated setting, but that compactness, low communication volume and low false-alert burden are distinct objectives. Attribution studies identify recurring flow features and sensitive explanations, not causal security rules. The evidence supports a reproducible quality-cost trade-off, not a deployment-ready detector, a general rare-class solution or an independently established new optimization algorithm.

<!-- pagebreak -->

## 1. Introduction and research questions

Network intrusion detection is not adequately summarized by the percentage of correctly classified flows. If common attack categories dominate a collection, a classifier can perform well on those categories while failing the ones for which detection is most needed. Conversely, a system that rarely raises an alarm may appear unobtrusive while classifying attacks as benign. Tradecraft therefore treats attack coverage, benign false alerts and resource requirements as joint outcomes.

The project asks whether a compact, federated-trained model can offer a useful alternative to a larger centralized model. The scope is deliberately narrower than a production security platform: eight-class prediction from numeric network-flow statistics, simulated virtual clients, and analysis of the trained models. The available CSV features do not provide a usable device-identity topology. No per-device graph, physical edge deployment or device-specific vulnerability inference is part of the experiment.

Four questions organize the study. First, what quality and model-size trade-offs emerge among the original heavy, light and federated configurations? Second, how do IID and heterogeneous client allocations differ when the light architecture is held fixed? Third, can shared optimization and compression narrow the non-IID performance gap without concealing communication or false-alert costs? Fourth, what can class-level errors and attribution diagnostics legitimately contribute to security interpretation?

The investigation developed in stages because the initial failures admitted several explanations: model capacity, preprocessing, class imbalance, data scale, client heterogeneity and update dynamics. The thesis does not treat every exploratory attempt as a separate main contribution. Its main evidence consists of the original configured-model comparison, the completed official39 baseline and the prospectively locked optimizer confirmation. Earlier controls explain why those comparisons were chosen.

The contribution is empirical and methodological. A single pipeline supports reproducible data handling, interrupted-run recovery, class-level reporting and explicit evaluation boundaries. The positive result is a confirmed optimizer trade-off under one protocol. The negative results establish where additional data, larger models or favorable aggregate scores did not yield satisfactory rare-category recognition.

<!-- pagebreak -->

## 2. Background and methodological foundations

The [CICIoT2023 release](https://www.unb.ca/cic/datasets/iotdataset-2023.html) describes 33 attacks generated in a topology of 105 devices and groups them into seven categories. Tradecraft retains those categories and adds benign traffic as the eighth class. The task therefore differs from binary attack detection: confusing DoS with DDoS is an error even when both predictions would raise an attack alert.

The project was motivated by [Diaz-Gorrin et al. (2026)](https://doi.org/10.3390/app16031442), who investigated an auxiliary-classifier generative adversarial network on a Jetson Orin Nano. Their focus on TCP-related threats and binary classification is not the same benchmark as this eight-class study. Their physical deployment motivates the resource question; it does not supply measured latency, power or memory for Tradecraft. No direct numerical performance comparison with that paper is made here.

[Federated averaging](https://proceedings.mlr.press/v54/mcmahan17a.html) combines locally computed model updates while reducing the need for synchronized communication at every learning step. That communication advantage creates a relevant tension: clients trained on different class mixtures can take different local trajectories before aggregation. In this project, clients are simulated subsets rather than real devices. Pooled preprocessing and available global class counts are conveniences of the simulation, not proof of a fully decentralized privacy-preserving system.

[Adam](https://arxiv.org/abs/1412.6980) maintains adaptive moment estimates. The final extension asks whether maintaining a shared optimizer over aggregated client gradients is preferable to the tested reset-client Adam configuration. [Error-feedback compression](https://proceedings.mlr.press/v97/karimireddy19a.html) motivates retaining information omitted by quantization for later updates. These are established mechanisms. Combining them here is an application-specific experimental design, not evidence that the underlying optimization ideas are new.

[Integrated gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) and [SHAP expected gradients](https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html) relate model-output differences to input features against reference inputs. Their outputs depend on the model, background and numerical approximation. The original study uses SHAP GradientExplainer; the official39 study uses background-averaged integrated gradients. Their names and diagnostic requirements are retained separately throughout the analysis.

<!-- pagebreak -->

## 3. Study structure and data boundaries

![Figure 1. Three separate study populations](figures/study_map.png)

Figure 1. The panels describe the progression of research questions, not a common leaderboard. Rows, features, checkpoint rules and decision thresholds differ between studies.

The original 46-feature study uses a partial Kaggle mirror supplied as upstream train, validation and test files. Sampling stays within those boundaries. The completed configured-model comparison uses 440,594 training rows and 327,257 validation rows. Its test split had been inspected earlier in the project; it cannot be relabeled as fresh confirmation. The reported comparison is explicitly validation based.

The downloaded official release used by the extension has 39 numeric features. Its preprocessing and models are separate from the legacy pipeline. Duplicate-grouped within-collection splitting keeps identical feature vectors together, but does not establish capture-session independence. The final baseline's primary population additionally excludes exact model-input overlaps with training or validation under any evaluated feature mask.

The shared-optimizer experiment reserves fitting, development and confirmation membership within a historical two-million-row training view. It provides a clearer boundary against direct selection on its confirmation rows, but those rows still belong to a previously investigated collection. Scores from this experiment must be compared with its own controls, not subtracted from the earlier final-test scores as if only the optimizer had changed.

<!-- pagebreak -->

## 4. Common implementation and evaluation principles

All labels follow the repository's canonical mapping: Benign, DDoS, DoS, Recon, Web-based, Brute Force, Spoofing and Mirai. Feature ordering and fitted preprocessing are retained with each experiment. Large files are processed through bounded sampling or disk-backed arrays rather than loading the complete release into memory. Seeds, source identities, data identities, configuration and checkpoint selection are logged so that a score can be traced to a specific run.

### Architectures and training settings

| Setting | Hidden layers | Parameters | Normalization and dropout |
| --- | --- | ---: | --- |
| Legacy heavy | 512, 256, 128, 64 | 198,984 | BatchNorm, 0.3 |
| Legacy central light | 64, 32 | 5,544 | BatchNorm, 0.2 |
| Legacy federated light | 64, 32 | 5,544 | LayerNorm, 0.2 |
| Official39 baseline light | 64, 32 | 5,096 | LayerNorm, 0.2 |
| Shared-optimizer study light | 64, 32 | 5,096 | LayerNorm, 0.0 |

The original configured-model study uses square-root-weighted cross-entropy, Adam learning rate 0.001, batch size 512 and weight decay 0.00001. Each run receives 60 epochs or rounds, with its first strict maximum validation macro-F1 retained. Federation uses 20 clients, full participation, one local epoch per round and a fixed partition seed. Non-IID assignment uses Dirichlet alpha 0.5. Equal epoch and round counts do not imply identical optimizer trajectories.

The official39 baseline retains learning rate 0.001, batch size 512 and weight decay 0.00001, at 20 epochs or full-participation rounds. The later optimizer study uses batch size 500, Adam moments (0.9, 0.99), epsilon 0.001 and no weight decay. Development selection fixes learning rate 0.001 centrally and 0.003 for FedAvg and both shared variants. These are protocol differences, not settings chosen on confirmation. Sources: [official39 recipe](../full_data_extension/ablation-confirmation-plan.json), [optimizer design](../federated_shared_optimizer/performance-plan.json) and [replication settings](../federated_shared_optimizer/replication-plan.json).

### Metrics and measurement scope

For each class, precision measures the fraction of its predictions that are correct, while recall measures the fraction of its true examples correctly identified. F1 is their harmonic mean; macro-F1 averages the eight class F1 values equally. Benign false-alert rate is the fraction of genuinely benign rows labeled as attacks. It is not the fraction of alerts that are false, which also depends on traffic prevalence.

Means weight training seeds equally. Sample standard deviations describe seed variation, not independent devices or datasets. Training-example counts, optimizer calls, synchronization rounds and bytes are reported separately because none is a universal compute-equivalence measure. GPU training times refer to the recorded training-and-validation scope. Model parameter counts are not peak memory, inference latency or edge power. Recovery pilots establish implementation parity, not predictive quality.

<!-- pagebreak -->

## 5. Original heavy and light model comparison

![Figure 2. Original configured-model quality and model size](figures/legacy_quality.png)

Figure 2. Every marker is a validation-selected endpoint. Open circles identify screening seed 0; squares and triangles identify new training seeds 1 and 2. The right axis is logarithmic. Parameter counts are not hardware benchmarks.

| Configured model | Macro-F1 mean and sample SD | New-seed mean | Mean false alerts |
| --- | --- | ---: | ---: |
| Central heavy | 0.6747 +/- 0.0031 | 0.6735 | 21.97% |
| Central light | 0.6583 +/- 0.0029 | 0.6587 | 23.57% |
| IID light | 0.6328 +/- 0.0024 | 0.6322 | 24.23% |
| Non-IID light | 0.5557 +/- 0.0128 | 0.5483 | 4.30% |

The ranking heavy, light, IID and non-IID holds at each seed. Heavy exceeds light by about 1.63 macro-F1 points with approximately 35.9 times as many parameters. This is a useful compactness trade-off, but dropout differs and the configurations were shortlisted using validation. The central-light versus federated-light comparison also changes normalization. Neither gap isolates a single causal factor.

Only seeds 1 and 2 were new after screening. Reporting them separately prevents the more exploratory seed 0 from being presented as fully prospective confirmation. Even the new seeds use the same validation rows and checkpoint-selection rule. The result is training-seed stability, not independent generalization evidence. Source: [configured-model evidence](../confirmation_2026_10_05/results/summary.json).

<!-- pagebreak -->

## 6. Class coverage in the original study

![Figure 3. Original per-class recall](figures/legacy_recall.png)

Figure 3. Mean recalls share a 0-100% color scale. Supports describe the same validation population for every model. The heatmap is not a pooled estimate over independent datasets.

The non-IID model's quiet alert stream is accompanied by weak DoS, Web-based and Brute Force coverage. Both new non-IID seeds have exactly zero Web-based recall. All three non-IID models identify only 14.80% of Brute Force rows, despite high DDoS and Mirai recall. The problem is therefore not visible from common-category performance alone.

The heavy network also remains imperfect. Web-based recall is 39.97% at seed 1 and 19.75% at seed 2, despite comparatively stable macro-F1. Central-light Brute Force recall averages 16.49%. A small aggregate standard deviation should not be interpreted as stable detection of every category.

The central and IID configurations label roughly 22-24% of benign validation traffic as attacks. These empirical proportions are not acceptable merely because their macro-F1 exceeds the non-IID result. The comparison motivates class-aware diagnostics and explicit alarm control, rather than selecting a winner using only one summary statistic.

<!-- pagebreak -->

## 7. Learning trajectories and fixed training budgets

![Figure 4. Original validation trajectories](figures/legacy_curves.png)

Figure 4. All 720 points come from the twelve saved, checksummed result logs. Dots mark each run's selected checkpoint. Lines connect actual successive validation measurements; no smoothing or new training was used. Epochs and federated rounds are different work units.

Several selected endpoints occur near the budget boundary. Central-light seed 2, IID seed 1 and non-IID seed 1 select step 60, while other federated runs select steps 58 or 59. The data establish performance within the fixed budget, not completed convergence. Extending training based on the same validation observations would constitute another development experiment.

These curves belong only to the original 46-feature study. The final shared-optimizer runs did not save validation scores at every update, so comparable optimizer convergence traces are unavailable. Endpoint scores and progress counters cannot reconstruct them.

<!-- pagebreak -->

## 8. Cloud controls and why the investigation changed direction

The later cloud controls give all three light lanes LayerNorm with dropout 0.2, retaining the original partial-release splits. At seeds 7, 8 and 9, centralized-light mean validation macro-F1 is 0.6872, IID is 0.6345 and non-IID is 0.5559. Mean benign false alerts are 20.99%, 25.50% and 4.27%. The central-light advantage therefore persists in this more closely aligned architecture comparison, although the cloud study still uses repeated validation and fixed training budgets.

| Exploratory cloud control | Validation macro-F1 | Benign false alerts | Interpretation |
| --- | ---: | ---: | --- |
| FedAvg reference, seed 7 | 0.5525 | 3.56% | Paired non-IID reference |
| FedAvg at 120 rounds | 0.5769 | 4.22% | Some categories improve, but the DDoS recall guardrail fails |
| Inverse-frequency weighting | 0.5567 | 3.40% | More rare predictions, but low precision and losses elsewhere |
| FedProx coefficient 0.01 | 0.5570 | 4.38% | Insufficient gain under the frozen promotion criteria |
| FedProx coefficient 0.1 | 0.5775 | 12.55% | Higher aggregate score with much larger alarm burden |

These are developmental controls, not additional independent test results. The larger FedProx coefficient improves macro-F1 but raises false alerts by approximately nine percentage points. A higher score alone does not make the intervention useful. Two further client partitions also retain serious minority-class weaknesses; the easiest partition was not selected as the reference.

Feature diagnostics found marked differences between training and validation distributions for quantities such as IAT, Number and Weight. Those patterns motivated checking the release and representation, but do not by themselves prove corruption, leakage or indistinguishability. The upstream split was retained. The subsequent official39 experiment was explicitly separate rather than silently replacing this evidence. Sources: [cloud study](../cloud_study_2026_10_06/README.md) and [FedProx diagnostic findings](../cloud_analysis_2026_10_06/results/README.md).

<!-- pagebreak -->

## 9. Official39 data controls and final baseline

![Figure 5. Official39 final-test baseline](../federated_shared_optimizer/thesis/figures/08_historical_context.png)

Figure 5. This completed study uses two million training rows, the common 5,096-parameter light network and fixed step-20 endpoints. The primary final-test population has 2,040,729 rows. Argmax decisions are not calibrated to the later extension's alarm target.

The official39 comparison improves architectural alignment: centralized, IID and controlled non-IID references use the same inference network. Nested client ownership prevents data-scale changes from silently redrawing the non-IID experiment. Three training seeds cover each final lane. Feature masks are separately trained sensitivity controls, not post-test replacements for the full39 references.

Before final scoring, label-blind preparation excluded 20,135 exact model-input matches to training or validation under any mask. It preserved all 2,437 Web-based and 1,310 Brute Force test examples. This protects against one specific overlap mechanism, not broader scenario or capture dependence. All 27 frozen full39/masked models were evaluated; the full39 references remained primary.

Mean final-test macro-F1 is 0.65673 centrally, 0.61169 under IID and 0.40494 under controlled non-IID. The corresponding false-alert rates are 17.49%, 19.08% and 0.25%. Non-IID Web-based and Brute Force recall is exactly zero at every seed. There is no matched official39 heavy arm, so these results cannot establish that a light network replaces a heavy one. Source: [final study closeout](../full_data_extension/final-closeout.md).

<!-- pagebreak -->

## 10. Diagnosing the official39 performance gap

![Figure 6. Contributions to the official39 macro-F1 gap](figures/official_gap.png)

Figure 6. Each contribution is one eighth of the class F1 difference between centralized-light and controlled non-IID, averaged across seeds. Contributions sum to the total gap; they are an arithmetic decomposition, not estimated causal effects.

Spoofing contributes about 9.30 of the 25.18 macro-F1 percentage-point gap. Recon, Brute Force and DoS also contribute materially. Web-based failures are important but do not explain the entire deficit. This redirects the question from finding a specialist for one rare class toward improving how the shared model is updated across heterogeneous clients.

Scaling from 500,000 to two million training rows did not consistently improve the centralized or IID model at equal processed-example budgets. Gains were consistent at 20 passes, which also quadrupled training exposure. Additional unique data and additional optimization work therefore cannot be credited interchangeably. LayerNorm, class weighting, local-update caps, loss-denominator controls and feature masks each addressed a plausible explanation without resolving the full trade-off.

The primary full39 models all contain 5,096 parameters, or 20,384 bytes of FP32 parameter values. Each final run processes 40 million training examples. Recorded GPU training-plus-validation times average about 300-311 seconds across the three lanes, excluding setup and checkpoint I/O. These are single-machine workloads, not distributed network timings or edge-device measurements. Source: [all-class metrics and cost accounting](../federated_class_safeguards/study-synthesis-tables.md).

<!-- pagebreak -->

## 11. Explainability and security interpretation

![Figure 7. Attribution reconstruction and repeated ranking stability](figures/explanation_quality.png)

Figure 7. Original-study SHAP expected-gradients diagnostics are shown for all twelve checkpoints. Markers represent seeds 0, 1 and 2. Reconstruction residuals and repeat overlaps answer different numerical questions; neither is a predictive performance metric.

The original SHAP study uses a shared 64-case balanced core and a separate 60-case targeted supplement, with 128 training-background rows. Two Monte Carlo repeats use 256 integration samples each. Aggregate relative reconstruction residuals are approximately 5.4-8.8%, even when repeated feature rankings overlap strongly. Some individual residuals are large relative to the decision margin, so stable ranking alone cannot justify a precise signed explanation of an error.

The official39 explanation study instead uses background-averaged integrated gradients on 219 selected validation cases across seven checkpoints. It includes three centralized and three IID seeds, but only one non-IID seed. Numerical reconstruction and resolution checks pass. Number ranks first in every model's balanced-core summary; protocol, flag and size features also recur. Reference-background sensitivity and correlated features limit independent interpretation of those contributions.

The resulting security implications are modest but useful. A benign prediction should not be treated as proof that rare web or authentication activity is safe. Repeated flow-feature associations can motivate inspection of the extractor and independent application telemetry. They do not establish numerical firewall rules, causal attack mechanisms or vulnerable device types. These interpretation limits do not insert a human-review stage into the classifier or claim a performance gain from one. Sources: [original attribution evidence](../explanations_2026_10_05/results/README.md) and [official39 explanation study](../full_data_extension/explanation-study.md).
