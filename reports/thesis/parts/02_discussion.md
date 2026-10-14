## 20. Overall validity and limitations

### Comparability within and between studies

The original configured-model comparison changes normalization and dropout as well as architecture or training location. Its seed-0 recipes were screened before new seeds were run, and all endpoints were selected on reused validation data. Its strengths are explicit configuration records, complete trajectories and honest seed-level reporting; it is not an isolated causal estimate of federation or model size.

The official39 baseline improves architecture matching and fixes final endpoints before scoring. Its duplicate-group and exact-input checks address direct overlap, while its common test population supports paired comparisons. Nevertheless, within-collection splitting is not equivalent to evaluating future traffic, new capture sessions or new devices. The absence of a matched heavy model limits conclusions about model replacement.

The optimizer extension freezes twelve checkpoints and 48 calibrated thresholds before confirmation. Each checkpoint reproduces its saved development predictions, and all final metrics are independently recomputed from saved outputs. However, shared optimization changes update frequency and effective batching along with optimizer placement. FedAvg uses a particular reset-client Adam design. Three training seeds share one client partition and the same examples. These choices delimit the positive finding rather than establishing general superiority over federated averaging.

The earlier default-argmax development screen failed its complete continuation criteria, despite shared Adam's high macro-F1. A separately specified alarm-calibrated diagnostic then asked a different question, followed by seed replication and compression safeguards fixed before confirmation. The final success does not retroactively turn the original failed screen into a pass. This distinction preserves the development history and avoids presenting the final protocol as the only question ever tested. Sources: [original screen](../federated_shared_optimizer/performance-findings.md) and [separate alarm diagnostic](../federated_shared_optimizer/alarm-findings.md).

### Resource and security limits

The clients are simulated on a single machine. No physical communication delay, straggler behavior, energy consumption or edge inference latency was measured. Parameter size and application-byte accounting should not be substituted for those quantities. Gradient sharing is still distributed-data training in the simulated algorithm, but does not itself guarantee privacy; pooled preprocessing and global class counts are explicit simplifying assumptions.

All studies retain substantial class-specific failures. The optimizer extension's actual false alerts exceed its nominal 5% calibration target, and its Web-based recall remains below 10%. The fact that every predeclared research safeguard passes does not make the model operationally safe. Deployment thresholds and incident-response policies would require a separate operational validation design.

### Interpretation and experimental reuse

Attributions explain model behavior relative to selected backgrounds, not causes in the network. Correlated inputs and interpolation through unrealistic feature combinations complicate interpretation. The balanced explanation samples are diagnostic rather than prevalence-weighted performance samples. Earlier use of the underlying collection also limits how independent the final extension's reserved confirmation can be considered. Further tuning on either exposed final population would require a newly labeled development study and fresh evaluation evidence.

<!-- pagebreak -->

## 21. Discussion of the research questions

### Compactness does not establish deployability

The original heavy network's roughly 36-fold parameter increase yields a modest configured-model macro-F1 advantage. This supports retaining a compact model as a serious research reference. It does not show that the light model is equivalent centrally, that the heavier architecture is unnecessary under every protocol, or that either model meets an operational requirement. Their false-alert and minority-class results are central to the trade-off, not incidental caveats.

### Heterogeneity affects more than the rarest categories

The official39 IID model performs much closer to centralized-light than the controlled non-IID model. The class decomposition shows that Spoofing, Recon and DoS contribute substantial losses alongside rare attacks. This makes the update procedure a reasonable target for investigation. The evidence does not isolate label mixture as the only cause: optimizer state, local trajectories, class weights and representation remain coupled to the observed setting.

### Shared optimization offers the strongest confirmed improvement

Against the extension's own FedAvg reference, compressed shared Adam gains 11.39 macro-F1 points and closes approximately 84.4% of the mean gap to centralized-light. Only 0.36 points are lost relative to uncompressed shared Adam. The safeguards pass separately at all three seeds, preventing a favorable aggregate from hiding a failed seed. These are the strongest grounds for treating the extension as a useful contribution.

The costs are equally concrete. Compression reduces bytes but not the 2,160 synchronizations, compared with 108 for FedAvg. Its application payload remains 2.73 times FedAvg's. On high-latency networks, the quality improvement may therefore be expensive. No claim of universal communication efficiency follows from the 86.4% saving relative to the more communication-intensive uncompressed reference.

### Explanations support investigation rather than automatic policy

The explanation work reveals recurring feature dependence and warns against converting a visually compelling ranking into a security rule. Numerical residual checks are important because an approximate explanation can appear stable while failing to reconstruct the decision margin closely. Feature masking and background checks provide sensitivity evidence, not proof that a feature is causal or entirely spurious.

The study's coherent outcome is therefore a measured compromise: compact inference remains possible, a shared update design recovers much of the non-IID quality gap, and compression makes that design smaller in bytes. Rare-class reliability, alarm burden and synchronization remain open constraints. That is a more defensible result than either declaring federated learning unsuccessful in general or claiming that the final model solves intrusion detection.

<!-- pagebreak -->

## 22. Conclusion and future work

Tradecraft investigated centralized-heavy, centralized-light and simulated federated-light intrusion classification using an eight-class CICIoT2023 task. The original configured-model study established a modest aggregate benefit from a much larger network and persistent non-IID class failures. The official39 baseline strengthened architectural matching and final evaluation, but retained large heterogeneous-client losses and weak rare-category recognition.

The final shared-optimizer extension provides a positive result within those limits. A persistent shared Adam update with four-bit blockwise residual-feedback compression reaches mean confirmation macro-F1 of 0.6325, compared with 0.5187 for the paired FedAvg control and 0.6535 centrally, at similar observed primary alarm rates. Compression saves 86.4% of the uncompressed shared-gradient payload while preserving most of its predictive quality. Frequent synchronization and minority-class failures remain important costs.

The project demonstrates the value of treating evaluation design as part of model engineering. Shared feature schemas, work counters, exact recovery checks, frozen candidates, alarm calibration and class-level diagnostics make the final findings more interpretable than a sequence of best-score reports. The algorithmic components are grounded in existing optimization and compression methods; the contribution is their verified comparison and application-specific quality-cost analysis.

The next research priorities are independent capture-session evaluation, repeated non-IID partitions and measured network-delay experiments. Ablations that hold effective batching and synchronization schedules fixed could distinguish optimizer-memory effects from scheduling effects. Secure aggregation, private preprocessing or partial participation would need explicit new designs rather than being inferred from the present simulation. Any future rare-class method should report correct-category recall, missed attacks and false alerts together.

These are future studies, not prerequisites for reporting the completed results and not a reason to spend remaining cloud credits without a hypothesis. The current confirmation set has been opened. It should remain a record of the locked experiment, not become another hidden tuning resource.

<!-- pagebreak -->

## References

1. Canadian Institute for Cybersecurity. CIC IoT Dataset 2023. Dataset description and attack taxonomy. [Official release documentation](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
2. Diaz-Gorrin, J., Caballero-Gil, C., and Brankovic, L. (2026). Enhancing Network Security with Generative AI on Jetson Orin Nano. Applied Sciences, 16(3), 1442. [DOI](https://doi.org/10.3390/app16031442).
3. McMahan, B., Moore, E., Ramage, D., Hampson, S., and Aguera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS, PMLR 54, 1273-1282. [Primary paper](https://proceedings.mlr.press/v54/mcmahan17a.html).
4. Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR. [Primary paper](https://arxiv.org/abs/1412.6980).
5. Karimireddy, S. P., Rebjock, Q., Stich, S., and Jaggi, M. (2019). Error Feedback Fixes SignSGD and other Gradient Compression Schemes. ICML, PMLR 97, 3252-3261. [Primary paper](https://proceedings.mlr.press/v97/karimireddy19a.html).
6. Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML, PMLR 70, 3319-3328. [Primary paper](https://proceedings.mlr.press/v70/sundararajan17a.html).
7. SHAP developers. GradientExplainer documentation. Expected gradients and reference-background semantics. [Official documentation](https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html).
8. Tradecraft. Original configured-model seed-stability study. [Protocol and findings](../confirmation_2026_10_05/README.md), [aggregate evidence](../confirmation_2026_10_05/results/summary.json).
9. Tradecraft. Cloud controls and FedProx analysis. [Cloud study](../cloud_study_2026_10_06/README.md), [diagnostic analysis](../cloud_analysis_2026_10_06/results/README.md).
10. Tradecraft. Official39 final evaluation and explanations. [Final closeout](../full_data_extension/final-closeout.md), [class-level synthesis](../federated_class_safeguards/study-synthesis-tables.md), [explanation findings](../full_data_extension/explanation-study.md).
11. Tradecraft. Shared-optimizer reserved confirmation. [Frozen execution plan](../federated_shared_optimizer/confirmation-execution-plan.json), [audited results](../federated_shared_optimizer/confirmation-results.json), [final findings](../federated_shared_optimizer/confirmation-findings.md).

### Reproducibility note

The editable manuscript is assembled from checked-in context and discussion sections plus the reviewed optimizer chapter. Fifteen figures combine six new aggregate-evidence plots with nine existing optimizer/baseline figures. Original training curves use a portable export of 720 verified saved validation points. The figures do not require model fitting, checkpoint loading or new test inference. Figure and assembly manifests record exact source identities. The PDF is a review copy outside Git; source text and vector figures remain editable.
