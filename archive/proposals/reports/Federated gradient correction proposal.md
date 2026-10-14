# Testing class safeguards in federated aggregation

**Stop the classifier search and return to centralized versus federated learning.** Preserve the abandoned extensions as historical evidence rather than deleting their code or results. One focused hypothesis is worth investigating: can the server adjust a federated update so that it does not locally push rare-attack learning backwards, while also protecting benign traffic? Keep the existing small network and automatic inference unchanged. This is a proposed adaptation of established gradient-projection methods, not a demonstrated new algorithm or a promise of substantial improvement. The first step should be a small training-only diagnostic, not another full training campaign.

## The remaining question concerns aggregation rather than architecture

The completed official39 comparison found test macro-F1 of **0.65673 centralized, 0.61169 IID federated and 0.40494 non-IID federated**, using the same lightweight architecture. Non-IID Web-based and Brute Force recall were zero across three seeds. However, centralized recall was also weak: 8.00% and 26.29%. Better aggregation could reduce the additional federated gap without solving the underlying rare-class discrimination problem. ([Final closeout](/Users/haydn/Documents/DSA4266Proj/reports/full_data_extension/final-closeout.md))

Some individual clients retained useful Brute Force predictions that disappeared after aggregation, but other apparent specialists generated extreme false alerts, and Web-based learning was weak even locally. Therefore, “aggregation destroys otherwise good rare detectors” is a hypothesis to test, not an established explanation. ([Aggregation diagnostic](/Users/haydn/Documents/DSA4266Proj/reports/full_data_extension/aggregation-mechanism-probe.md))

## Protect class learning by minimally correcting the server update

The proposed **class-risk-constrained FedAvg** changes one ingredient: the server’s final parameter update. Clients train exactly as before. If client models move from the shared weights by displacements `delta_i`, ordinary FedAvg proposes their sample-count-weighted average, `delta_avg`.

Before local training, clients additionally compute full-network loss-gradient sums at the same shared weights for their Benign, Web-based and Brute Force training examples. The server combines these sums using the corresponding example counts to obtain one pooled gradient `g_c` per protected class. The initial diagnostic streams all available training examples of these classes; raw rows remain with their assigned clients. Empty class holders contribute zero sums and counts, never invented gradients.

The server then finds the smallest change to the ordinary update:

```text
minimize    ½ ||delta − delta_avg||²
subject to  g_c · delta ≤ 0
            for Benign, Web-based and Brute Force
```

It adds this corrected displacement to the global weights. Intuitively, **keep the ordinary federated step wherever possible, but remove directions predicted to increase the three protected training losses**. When no constraint is violated, the result must equal FedAvg. The model still has eight outputs and the same 5,096 parameters; no ensemble, language model or human review is introduced.

Each class gradient includes all eight softmax outputs and the shared representation—not merely that class’s classifier row. Clients lacking rare examples still contribute their normal updates, including legitimate evidence against incorrectly predicting rare attacks. Simply discarding those clients’ rare-output updates would remove that evidence. Global class weights and the existing loss denominator remain fixed, but the constraints explicitly change the allowable optimization directions.

## The contribution would be a tested adaptation

The projection geometry closely follows Gradient Episodic Memory, which constrains updates using other objectives’ gradients. FedFV already corrects conflicting client updates, while Fed-GraB already balances class gradients in federated long-tailed learning. Our proposed distinction is an explicit pooled-class safeguard on the actual server displacement, with benign and rare-attack objectives considered together. That is a defensible application-level contribution **if it works**, not evidence of global novelty. ([GEM](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf), [FedFV](https://www.ijcai.org/proceedings/2021/0223.pdf), [Fed-GraB](https://proceedings.neurips.cc/paper_files/paper/2023/file/f4b8ddb9b1aa3cb11462d64a70b84db2-Paper-Conference.pdf))

The constraint concerns a **first-order prediction of mean training loss**, not actual finite-step loss, recall, false-alert rate or generalization. Conflicting constraints can shrink the update to zero. Lower benign loss does not guarantee fewer benign alerts. These are reasons to measure the mechanism before training, rather than treating an attractive equation as a solution.

Raw training records stay local, so this remains federated learning in the project’s simulated setting. The trade-off moves to training: three extra float32 gradients cost up to **61,152 bytes per client per round**, plus counts and protocol overhead, and require additional gradient computation. That is three extra model-sized uploads, not free communication. Exchanging gradients provides no automatic privacy guarantee. ([Gradient leakage research](https://www.usenix.org/system/files/usenixsecurity23-yue.pdf))

## A small diagnostic must justify any further experiment

Preselect three existing early, middle and late training checkpoints. At each shared starting model, compare the common-point pooled gradient with the actual displacement produced by the multi-step local optimizer. Measure whether FedAvg predicts and actually causes increased rare-class training loss, whether projection removes that increase, and whether it retains a useful nonzero step. Also inspect benign loss, all eight classes and holder concentration. **Stop if the harmful mechanism is absent, the correction stalls, or actual losses contradict the linear prediction.** Do not rescue the idea with a new sweep.

Only a convincing diagnostic would justify a frozen pilot. The comparison should include same-code FedAvg, the proposed correction and the nearby published FedFV method, plus ordinary centralized-light and a centralized same-constraint control. The latter distinguishes a generic constrained-optimization benefit from a specifically federated benefit. Match data, initialization, loss and local optimizer schedules; separately count extra gradient work and communication. If SCAFFOLD is later selected instead, its documented SGD formulation needs a matched SGD control—not an unqualified comparison against reset Adam. ([SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html))

## Conclusion

The revised thesis is whether a small, explicit change to federated optimization can narrow the centralized gap without unacceptable false alerts or training overhead. That is more focused than discovering another classifier. No training is launched by this proposal. Keep official39 separate from the original46 heavy-model study, and keep the already exposed final test closed to selection. Any pilot remains exploratory until independently evaluated; success must be demonstrated, not assumed from the method’s name.
