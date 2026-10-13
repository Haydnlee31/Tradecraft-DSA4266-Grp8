# Adaptive anchors and central alignment

## 1. What is established, and where is the closest prior art?

### Takeaway

Same-weights distributed gradients, shared optimizer statistics, and drift-dependent synchronization are established ideas. The useful Tradecraft contribution would be a carefully audited, bounded adaptation that shows how much centralized performance can be retained per unit of communication; novelty of a new optimization principle is not established.

### Cited Findings

- **Mime/MimeLite (ICML 2021):** both distribute a server optimizer state, keep it fixed during a local block, and collect gradients evaluated at the starting global parameters to update that state. Mime adds the correction `grad f_i(local; batch) − grad f_i(anchor; same batch) + global_anchor_gradient`; MimeLite omits this correction. Algorithm 1 uses full local gradients for the state update, so MimeLite is not “free FedAvg with global Adam.” Its shared statistics and extra gradient computation/communication must be included in the budget. The paper's Adam analysis includes restrictive settings such as zero first-moment coefficient; its convergence claims do not establish exact reproduction of an ordinary central Adam trajectory. — [Mime, Algorithm 1 and Corollary III](https://arxiv.org/html/2008.03606)
- **Wang et al. (JSAC 2019) is especially close:** the controller receives gradients at a common global parameter vector, estimates local/global gradient divergence, and chooses the next local-update interval using loss/smoothness estimates and measured resource costs. It starts with interval one and limits how far the interval can increase. The main theory assumes convex, Lipschitz, smooth objectives; neural-network experiments extend beyond those assumptions. Thus “occasionally probe gradient disagreement, then adapt local work” cannot be claimed as a new principle here. — [Adaptive Federated Learning in Resource Constrained Edge Computing Systems, Algorithms 2–3 and Section V](https://arxiv.org/html/1804.05271)
- **AdaComm (MLSys 2019):** changes the synchronization period to balance runtime and error; its practical schedules use training loss and learning rate. The authors explicitly cap/modify an aggressive schedule after observing divergence at very long periods. It supplies an appropriate simple adaptive-synchronization comparator, but no guarantee for this weighted, non-IID neural-network application. — [Adaptive Communication Strategies, Sections 4.3–4.4](https://mlsys.org/Conferences/2019/doc/2019/124.pdf)
- **SlowMo (ICLR 2020):** combines periodic parameter averaging with an outer momentum update. It distinguishes reset, retained, and averaged inner optimizer buffers; their relative usefulness differs across image and translation experiments. Its one-inner-step SGD case recovers large-minibatch momentum SGD, and it provides stationary-point results under assumptions. It is relevant evidence that optimizer-state policy matters, not a gradient-probe controller or a proof of improved attack detection. — [SlowMo, Algorithm 1 and Appendix B.4](https://arxiv.org/html/1910.00643)
- **FedOpt/FedAdam:** using the negative average client delta as a server pseudo-gradient is already a general framework. The standard adaptive instantiations use local SGD and server adaptive state; example-count-weighted variants are explicitly given. Server Adam after multiple local steps is consequently a known family, not centralized Adam in disguise. — [Adaptive Federated Optimization, Algorithms 1 and 5](https://arxiv.org/html/2003.00295)

### Inferences

- A defensible description is **“central-aligned gradient reference plus a bounded synchronization controller for the Tradecraft workload.”** Use “sparse” to mean infrequent probes, not coordinate sparsification. Do not name the adaptation Mime/MimeLite unless it implements their actual local updates and state rules.
- A novel contribution could be the experimentally supported diagnosis, reproducible equivalence audit, and budget-controlled adaptation on this task. It would be premature to claim a novel algorithm, first drift-based communication controller, or universal improvement over central training.

### Gaps

- This is a bounded primary-source review, not an exhaustive novelty search. A search also surfaced *FedADM: Adaptive Federated Learning via Dissimilarity Measure* as an ICLR-2025-era submission; the full OpenReview page/PDF could not be fetched past a browser challenge, so its status and algorithm were not verified and are not used as evidence. — [Unverified submission](https://openreview.net/forum?id=IsHWcsk4Fz)
- None of the cited experiments establishes that this particular 39-feature, eight-class network will improve, or that a probe score predicts its validation performance.

## 2. What exact central reference should be established first?

### Takeaway

First establish a one-step **aggregate data gradient → persistent server Adam** reference, matched to a central gradient-accumulation implementation on exactly the same rows and coefficients. This removes local drift and client-Adam resets, but its exact comparator is a central large-batch/accumulation step, not the existing sequence of many central minibatch Adam updates.

### Cited Findings

- The current official runner constructs `MLPClassifier(39, 8, config)`, retains central Adam across epochs, constructs a fresh Adam for each client round, and weights client models by assigned row counts. The light architecture has hidden dimensions `(64, 32)` with dropout `0.2`; the runner supports the LayerNorm configuration. — [official_streaming.py](/Users/haydn/Documents/DSA4266Proj/src/models/official_streaming.py:240), [architectures.py](/Users/haydn/Documents/DSA4266Proj/src/models/architectures.py:82)
- The existing fixed-denominator loss implements weighted per-example CE averaged over the batch, divided by the training cohort's mean target weight. Its module explicitly warns that this does not make multistep local Adam plus averaging equal central training. — [official_losses.py](/Users/haydn/Documents/DSA4266Proj/src/models/official_losses.py:1)
- LayerNorm with a one-dimensional hidden width computes normalization across each example's hidden features, whereas dropout draws random zero masks during training. These details affect comparisons between one concatenated forward pass and several microbatch forwards. — [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.LayerNorm.html), [PyTorch Dropout](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Dropout.html)
- The original FedAvg paper already formulates learning from decentralized data while sharing updates. A synchronized aggregate-gradient baseline is within federated/distributed optimization; retaining raw rows on clients is not itself a new algorithm. — [McMahan et al., AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html)

### Inferences

**Exact bridge, derived from linearity of differentiation:** define client weights `p_i = n_i/N`, frozen class weights `a_y`, and training-only constant `mu = sum_training a_y/N`. At global model `theta`, client `i` computes

`g_i = grad_theta [(1 / (|B_i| mu)) sum_{z in B_i} a_y CE(theta; z)]`.

Aggregate `g = sum_i p_i g_i`, then apply exactly one server Adam step. A central implementation that accumulates those same microbatch losses with factors `p_i` produces the same data gradient, up to floating-point reduction order. Adam parameters and first/second moments then match if their initial states, step counters, hyperparameters, regularization, and schedules match. This should be checked for the gradient, parameters, and Adam moments over several deterministic steps, not inferred from final accuracy.

There are several consequential conditions:

- **Rows and coefficients:** this bridge is a weighted concatenation. With unequal client sizes but equal local batch sizes, `p_i` is generally not `|B_i|/sum_j |B_j|`. Therefore an ordinary uniform mean over concatenated rows is a different realized gradient. To match it, use sample-count coefficients; to match the population objective in expectation with equal client batches, use the fixed `p_i` and reproduce those coefficients centrally. Record which comparison is made. Full local gradients at a shared model recover the full pooled objective exactly, but are much more expensive.
- **Loss denominator:** averaging client/batch losses that each divide by their own observed class-weight sum generally differs from dividing the pooled weighted loss by one common denominator. Fixed `mu` provides one explicit objective. Changing to it is an experimental treatment, so the central bridge must change too.
- **Dropout:** identical seeds alone do not guarantee identical masks when batching, execution order, or kernels differ. For an algebraic diagnostic, disable dropout on both sides; then separately test training-mode equivalence by replaying the same microbatches and per-microbatch RNG state, or using prescribed per-example masks. Eval-mode gradients are diagnostic gradients and must not be silently substituted for the training algorithm's stochastic objective.
- **Normalization:** retain LayerNorm. BatchNorm with separate client batches would change activations, gradients, and running statistics; a matching-seed test would not fix that. Feature preprocessing must also be identical and fitted only on the permitted training cohort.
- **Optimizer state:** average gradients before the nonlinear Adam step. Averaging local Adam parameters or moments is generally different. Perform weight decay/L2 and any clipping in exactly the same place on both paths; avoid applying client regularization and then adding it again at the server. A model-only warm start with new moments is distinct from resuming central Adam.
- **Participation:** full participation avoids extra sampling effects in the first reference. Renormalizing over a changing participant set changes the realized objective. Unbiased subsampling corrections may target the same objective in expectation, but do not create exact trajectory equality.

If each of 20 clients contributes `b` rows, this reference consumes approximately `20b` rows per server step. Ordinary centralized minibatch Adam might instead make 20 sequential updates, evaluating later gradients at changed parameters and updating moments 20 times. Gradient accumulation makes one update. Equality to the bridge therefore does **not** imply equality to, or the performance of, the existing central benchmark. Smaller per-client batches can reduce the effective batch, but then introduce a sampling/noise and communication trade-off that must be evaluated.

### Gaps

- No bridge implementation or numerical equivalence test has been run in this research task.
- The performance gap between the central accumulated-gradient reference and ordinary centralized minibatch Adam remains an empirical question; it must be established before attributing a remaining gap to federation.

## 3. What bounded follow-on experiment is justified, and what would invalidate it?

### Takeaway

Only after the reference is validated should infrequent same-weights probes control a short-versus-long local block. The first adaptation should have two allowed block lengths, explicit probe costs, and a fixed-step control; adding gradient correction, persistent client Adam, rollback, and another outer momentum mechanism at once would make the result hard to interpret.

### Cited Findings

- In convex heterogeneous optimization, Woodworth, Patel, and Srebro show regimes where local SGD is worse than minibatch SGD, while advantages require additional restrictions on heterogeneity. These results do not settle the neural-network case, but reject a universal “local updates beat central” argument. — [Minibatch vs Local SGD for Heterogeneous Distributed Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/45713f6ff2041d3fdfae927b82488db8-Paper.pdf)
- Shared gradients can reveal training information; keeping raw samples local does not by itself establish privacy. Secure aggregation protects individual contributions by releasing a sum, but the sum still reveals aggregate information. — [Deep Leakage from Gradients](https://papers.neurips.cc/paper_files/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html), [Bonawitz et al., secure aggregation](https://research.google/pubs/practical-secure-aggregation-for-federated-learning-on-user-held-data/)

### Inferences

**Proposed project adaptation, unvalidated:** retain the existing small inference architecture and eight-class output. After new development-split planning and authorization, freeze the objective, partitions, data preprocessing, and selection rule. Then compare the one-step reference, a small fixed local block, and the adaptive block under a declared resource budget.

1. Use plain local SGD and one persistent server Adam state as the initial follow-on family. For a common local learning rate `eta` and equal block length `K`, define the server pseudo-gradient `q = sum_i p_i (theta − theta_i)/(eta K)`. For `K=1` this is the aggregate data gradient; for `K>1` it averages gradients along different local paths and is an approximation. Normalizing by `K` avoids silently scaling the pseudo-gradient when the controller changes `K`, but it changes the usual unnormalized FedAdam convention and must be named explicitly. Keep regularization placement consistent with the bridge.
2. At predefined probe opportunities, clients evaluate a fixed-size, training-only probe at the same broadcast `theta`. Compute `g = sum_i p_i g_i` and dispersion `D = sqrt(sum_i p_i ||g_i − g||²)/(||g|| + epsilon)`. Start with this unpreconditioned diagnostic; adding Adam-preconditioned norms introduces another design choice. The numerator contains minibatch noise as well as true client disagreement, and the ratio is unstable when the global gradient is tiny.
3. Use a prespecified conservative rule to choose one common `K` from `{1, K_long}`; for example, permit the long block only after repeated low-dispersion observations, and immediately return to one after a high/noisy score. Select `K_long`, threshold, probe cadence, and hysteresis on the new development protocol only. Do not invent numerical “safe” thresholds from the literature. Log how often the controller selects either block; a controller that always chooses one is still useful evidence, but has not bought communication savings.
4. Keep the probes diagnostic in the first version: they do not advance Adam counters or silently contribute extra updates. If probe gradients are later reused as training gradients, define this as a separate variant with explicit row reuse and state transitions. A “probe” at the original shared weights measures disagreement; it does not measure the actual central multistep trajectory, guarantee descent, or certify a long block as safe. Stale probe values cannot certify later models.
5. Save server moments and step counters, current `K`, last probe round, controller history, row manifests, and all RNG/sampling state in checkpoints. Persistent client control variates are not necessary for this minimal controller. If later added, record their anchor/model, age, update rule, and communication cost; do not treat stale correction state as a current global gradient.

**Budget and validation requirements:** log actual examples processed, gradient evaluations (including every probe, retry, rejected candidate and full-gradient pass), server updates, local updates, broadcasts, upload/download bytes, barriers, and wall time. A same-weights probe can require an additional gradient upload and a decision broadcast. A controller benefit must survive comparison with fixed-step alternatives at matched work and matched communication separately; equal rounds alone are not a fair match. Tune the central reference and fixed-step comparator comparably. Twenty virtual clients do not measure physical network speed or edge power.

**Security/metric constraints:** maintain full eight-class confusion matrices, per-class recall, macro-F1, and benign false-alarm rate even if the new decision criterion emphasizes aggregate performance. Better weighted training loss, cosine alignment, or aggregate accuracy cannot certify preserved rare-class recall or benign behavior. The supplied task context reports previous loss/aggregate improvements that failed alarm/rare gates; those outcomes motivate explicit outcome checks, not an assumption that this controller repairs them. This note makes no new empirical claim about those runs.

**Privacy implications:** avoid a privacy guarantee for this simulation. Per-client gradient vectors and disagreement diagnostics reveal additional information. In principle the unpreconditioned variance can be derived from aggregate `sum p_i g_i` and `sum p_i ||g_i||²`, avoiding server access to every individual vector if a suitable secure-aggregation protocol exists; the aggregate norms still leak statistics. Adding differential privacy would require a separate clipping/noise/accounting design covering probes, with costs and utility effects counted.

**When to stop this line:** if the matched one-step bridge itself cannot approach the desired central aggregate result at the feasible budget, an adaptive longer block has no established reason to recover that result. If a fixed short block already matches the controller, use the simpler method. If reported gains disappear after charging probe work or cause unacceptable FAR/recall deterioration, report the trade-off rather than claiming success.

### Gaps

- The original final test is already exposed and closed per the task context. No original-test tuning, new training, or implementation is authorized by this research note; further work needs the separately planned development split and experiment authorization.
- No guarantee follows that this adaptation beats the centralized small model, matches it at a practical communication budget, or exceeds a centralized heavy model. The strongest exact claim available is the restricted gradient/optimizer equivalence in Section 2, to be numerically verified; everything about eventual predictive performance remains an experiment.
