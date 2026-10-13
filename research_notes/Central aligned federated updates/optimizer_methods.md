# Central aligned federated optimizers

## Which established method addresses the remaining optimizer mismatch?

### Takeaway

Recommend one bounded **MIME-Adam** experiment as the strongest mechanism-matched comparator. It addresses a question that the existing plain-SGD SCAFFOLD experiment did not: whether global adaptive statistics plus gradient correction recover useful optimization behavior. This is an established-method comparison, with no expected macro-F1 gain asserted.

### Cited Findings

- MIME's published version is *Breaking the centralized barrier for cross-device federated learning*, NeurIPS 2021. For anchor model `x`, form `c = sum_i p_i grad F_i(x)`; use `g = grad loss_i(y; b) - grad loss_i(x; b) + c` locally, then average final client models. Global optimizer state remains fixed within each round and updates from `c`. — [MIME, Algorithm 1](https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf)
- The paper's Adam decomposition uses `U(g;m,v) = ((1-beta1)g + beta1*m)/(sqrt(v)+epsilon)`; server updates use `m <- beta1*m+(1-beta1)c`, `v <- beta2*v+(1-beta2)c²`. This uses lagged `v`, unlike standard Adam. — [MIME appendix, Table 5](https://arxiv.org/pdf/2008.03606)
- Its reduction requires smoothness, bounded gradient variances, constrained step sizes, and a suitable updater. The explicit Adam corollary additionally assumes bounded coordinate gradients and `beta1=0`. The stronger communication-complexity result uses momentum variance reduction and Hessian similarity, not Adam. — [MIME, Theorem I, Corollary III, §4.2](https://proceedings.neurips.cc/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf)
- MimeLite omits correction, but still computes full anchor gradients and freezes server-supplied optimizer state locally. TensorFlow Federated provides an example-weighted implementation, with separate gradient and model-delta aggregators; ordinary state-mutating Keras optimizers cannot directly implement that interface. This is implementation evidence, not a recommendation to add TensorFlow. — [Official MimeLite documentation](https://www.tensorflow.org/federated/api_docs/python/tff/learning/algorithms/build_weighted_mime_lite)

### Inferences

- Choose full MIME for this diagnostic because correction and adaptive preconditioning are both relevant to the severe non-IID setup. MimeLite is a cheaper potential simplification, not an additional initial sweep.
- A faithful implementation needs explicit optimizer equations and round counters; repeatedly calling ordinary `Adam.step()` locally would change the intended method. Record initial `v`, epsilon, moment ordering, and any bias-correction convention. Compare to the same centralized base-optimizer variant before interpreting a difference against existing standard Adam.
- A convergence rate in gradient norm does not establish rare-class recall, benign FAR, macro-F1, or a useful operating threshold. No reviewed theorem establishes that this model will match or exceed the centralized result.

### Gaps

- No direct evidence here establishes MIME-Adam's performance on this CICIoT2023 cohort. The repository's current results do not separate optimizer-memory effects from partition-induced gradient effects. The needed outcome remains an experiment, not a literature-derived numerical prediction.

## How do FedAdam, FedNova, and communication costs compare?

### Takeaway

FedAdam is the inexpensive adaptive baseline; FedNova targets unequal local progress. Neither is a direct replacement for the proposed experiment's central question. Count compute and bytes as well as rounds.

### Cited Findings

- FedOpt uses an aggregated client displacement as a server pseudo-gradient. Standard FedAdam specializes to local SGD and persistent server moments: for `d = sum_i p_i(y_i-x)`, update `m=beta1*m+(1-beta1)d`, `v=beta2*v+(1-beta2)d²`, and `x += eta_server*m/(sqrt(v)+tau)`. It adds no model-sized communication relative to FedAvg. Its analysis assumes smoothness, bounded gradients, bounded local/global gradient variance and small enough local steps; the main exposition uses full participation and `beta1=0`, discussing extensions. — [FedOpt, Algorithms 1–2, §3](https://arxiv.org/html/2003.00295v5)
- FedNova normalizes accumulated local updates to remove unequal-step weighting bias. For plain SGD, `x_next = x + (sum_i p_i K_i) sum_i p_i delta_i/K_i`. Equal `K_i` gives FedAvg. Momentum/proximal solvers require their own accumulation normalizers. Guarantees rely on smoothness, unbiased bounded-variance gradients, bounded dissimilarity and step-size conditions; they address stationarity of the intended objective. — [FedNova, §4–5, Eq. 12](https://papers.neurips.cc/paper/2020/file/564127c03caab942e503ee6f810f54fd-Paper.pdf)
- The repository documents `P=5,096` FP32 parameters, hence `4P=20,384` bytes per model-sized vector. — [Local measured-model record](../../reports/federated_class_safeguards/scaffold-findings.md)

### Inferences

The following are **derived payload counts**, not measured network, hardware memory, latency, or power. Assume 20 participating clients, unchanged FP32 tensors, no compression/protocol overhead, and each selected client performing both gradient and model work. Cache each full anchor gradient upload for server-state updating.

| Method | Down/up vectors per client per round | Total bytes over 20 clients | Backward-example work for one local epoch |
|---|---:|---:|---:|
| FedAvg / FedAdam | 1 / 1 | 815,360 | roughly `N` |
| FedNova, plain SGD | 1 / 1, plus scalar normalization metadata | roughly 815,360 | roughly `N` |
| MimeLite-Adam | 3 / 2: model+m+v / anchor-gradient+model | 2,038,400 | roughly `2N` |
| MIME-Adam | 4 / 2: model+m+v+c / anchor-gradient+model | 2,446,080 | roughly `3N` |

For `K` mini-batches rather than an epoch, MimeLite costs approximately `N + K*B` backward-example evaluations and MIME approximately `N + 2*K*B`, summed across clients. `N` here is the participating clients' full anchor cohort, `B` their mini-batch size; final partial batches alter the exact total. With 2M anchor examples and small `K`, the full anchor pass can dominate. MIME also requires a gradient-aggregation barrier before local training. These counts follow from the tensor dependencies above and the documented model size, not execution measurements.

The paper's separate-cohort experiment has a different accounting boundary (Table 4 lists 6 model units for MimeLite-Adam and 7 for MIME-Adam); do not call the table's accounting identical to the reused-client protocol above. — [MIME appendix, Table 4 and §5.1](https://arxiv.org/pdf/2008.03606)

- FedNova offers little new for the existing equal-`K=50`, plain-SGD diagnostic: its scalar normalization becomes a no-op. Dividing ordinary Adam displacement by step count does not fix coordinate-dependent adaptivity.
- FedAdam adds only `2P` persistent server moment entries, and avoids local Adam state. Its moments summarize round displacements, not the centralized sequence of mini-batch gradients; tune server/client learning rates separately if it is used later.

### Gaps

- Runtime, memory peaks and realized byte serialization are unmeasured for the proposed implementation. The rough backward-work ratios do not imply wall-clock ratios. Exchanging optimizer statistics does not itself supply differential privacy or a secure-aggregation guarantee.

## Why are Adam-state averaging and optimizer resets insufficient, and what should the bounded path test?

### Takeaway

Retaining Adam memory may change performance, but averaging client moments does not reconstruct centralized Adam. The next run should isolate one optimizer-alignment mechanism with a common objective and an internal training-derived evaluation protocol.

### Cited Findings

- Adam tracks exponentially weighted gradients and squared gradients, then uses a nonlinear ratio with bias correction determined by step count. — [Original Adam, Algorithm 1](https://arxiv.org/pdf/1412.6980); [PyTorch Adam equations](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)
- For integer class targets, PyTorch weighted cross-entropy with `reduction='mean'` divides by the sum of target weights within the batch. — [Official cross-entropy formula](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

### Inferences

**Counterexample, derived from Adam's equations:** consider two equally weighted one-dimensional client gradients `g1=1`, `g2=3`, with zero previous state. Averaging local squared gradients gives `5`; squaring their aggregate gradient gives `4`. Averaging local first-step normalized updates gives approximately `1`, but applying the ratio to averaged moments gives `2/sqrt(5)`, approximately `0.894`. Central Adam on the pooled gradient gives approximately `1`. Thus even before multiple local steps, “average moments,” “average Adam updates,” and “Adam on the aggregated gradient” are distinct. For opposite signs, the discrepancy can be larger. Sequential centralized Adam also depends on gradient order and the parameter trajectory, which local branches do not share.

Resetting each client optimizer every round discards moments and restarts the bias-correction clock. Keeping private moments instead retains statistics tied to each client's prior local trajectory, while its model is overwritten by the new server model. Neither option establishes central equivalence. Repeated identical-gradient steps can still agree, so reset harm is a hypothesis, not a universal claim.

**Proposed bounded experiment, not work performed:**

1. Keep the architecture, original 20-client assignments, seed, training provenance and common initialization. Fix a training-derived development/confirmation split before tuning; do not access original validation/test rows.
2. Match the objective: use the existing fixed global denominator for both central and federated comparison arms. With `F_i=(1/n_i)sum a_y*loss` and `p_i=n_i/N`, the weighted sum reproduces the same global per-example loss. Local random weight-sum denominators instead optimize expectations of batch ratios; averaging them does not generally reproduce pooled batches.
3. Implement and verify one MIME-Adam arm against a centralized version of its precise base update. Check the full weighted anchor-gradient identity, state freezing, one-step correspondence, and same-batch subtraction at the anchor; with dropout, pair the random masks or explicitly define the expected-loss estimator. Begin with a small fixed common `K`, and charge anchor work in the budget. Any stochastic anchor approximation must be labeled a variant and its gradient variance recorded.
4. Use a narrow predeclared step-size search on the internal development split. Compare at matched backward-example and communication budgets as well as matched rounds. Confirm only a selected setting on the reserved training-derived confirmation split; report macro-F1, all class recalls and FAR together, including a matched-FAR comparison.
5. Advance only if the result demonstrates a useful aggregate trade-off with class failures plainly reported. An improvement that still leaves Web-based recall zero is evidence about optimization, not successful eight-class detection. Do not promise to beat the central model or treat repeated tuning as independent confirmation.

### Gaps

- Matching the pooled mini-batch gradient and applying one persistent central Adam optimizer per synchronized update would give exact numerical correspondence only with matched batches, stochastic state and loss normalization (up to floating-point reduction order). That is a useful diagnostic reference, but its synchronization cost changes the deployment trade-off; it does not establish that a multi-local-step method will reproduce the archived central trajectory.
- Reset/persistent-state ablations could diagnose memory loss separately, but are not part of the single recommended new baseline path. No training, model changes, new dependencies or dataset access were performed for these notes.
