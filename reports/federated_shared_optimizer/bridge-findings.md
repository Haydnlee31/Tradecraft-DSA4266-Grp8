# Shared optimizer numerical checks pass

14 October 2026. **The shared-gradient/server-Adam control reproduces its matched centralized updates within floating-point tolerance.** All four real-training-data cases pass, including active dropout, and saved midpoint states resume to bit-identical endpoints within each arm. Independent execution reproduces every non-timing receipt field.

This completes the first implementation gate in the [shared-optimizer proposal](../Central%20aligned%20federated%20updates.md). A separate full-MIME kernel also passes synthetic mathematical and recovery checks. Neither result establishes improved attack classification: no quality scores or new held-out predictions were computed. The completed study's results remain unchanged.

## What the successful comparison means

Imagine twenty clients calculating slopes at the same point on a hill. The server combines those slopes and makes one step, remembering Adam's past slopes and squared slopes. The centralized control calculates the same weighted objective and makes the same single step. The numerical agreement confirms that the distributed calculation preserves this update when client populations, loss normalization, stochastic masks and optimizer history are aligned.

The reference is **centralized gradient accumulation**, not twenty sequential centralized Adam steps. Sequential training moves the model and updates its memory between minibatches; later gradients are therefore evaluated at different parameters. This control cannot establish that the new path will reproduce the historical centralized model's macro-F1 or converge as quickly.

The clients in this audit do not run local Adam and average weights. They send gradients; only the server owns Adam's persistent moments. It is a synchronous federated-gradient control, not a new algorithm, FedAvg, FedAdam on model differences, or MIME. It retains raw rows within the simulated client interface, but this is not a privacy or physical-distribution guarantee.

## The numerical results

The [plan](bridge-plan.json) and implementation were committed as `ac90e82` before the real-data audit. Each case starts from fresh seeded 39–64–32–8 LayerNorm weights and performs 12 synchronized updates over the fixed 20-client non-IID assignment. Every client samples 512 training rows per update, with replacement. Population sizes range from 20,800 to 198,245, so equal-sized minibatches do **not** justify equal client weights.

The centralized implementation weights each microbatch loss before differentiation. The federated implementation calculates local gradients separately and weights them afterwards. Both use the same square-root class weights with the fixed training-mean denominator. Norm clipping occurs after aggregation, followed by Adam's coupled L2 term. Dropout seeds are matched separately for every client and step, not just once at initialization.

| Numeric precision | Dropout | Largest raw-gradient difference | Largest parameter difference | Midpoint recovery |
| --- | ---: | ---: | ---: | --- |
| float64 | 0 | 1.67e-16 | 1.11e-16 | Exact |
| float64 | 0.2 | 1.80e-16 | 1.11e-16 | Exact |
| float32 | 0 | 1.19e-7 | 1.19e-7 | Exact |
| float32 | 0.2 | 1.34e-7 | 7.45e-8 | Exact |

These are maximum absolute element differences over all twelve updates. Both Adam moments, optimizer counters and clipped gradients were checked at every step as well. Cross-implementation trajectories are numerically equivalent, **not bit-identical**; their small differences arise from floating-point operation order. In contrast, each implementation's serialized recovery exactly matches its own uninterrupted execution. Declared tolerances were not relaxed after observing results.

Negative-control tests demonstrate that uniform weighting, different dropout masks and repeatedly discarding Adam's moments can change the result. An independent scalar calculation checks Adam's bias correction, moments, L2 and clipping order. These checks prevent “both programs run” from being mistaken for optimizer equivalence.

The [public evidence](bridge-results.json) preserves all steps, source and data hashes, client counts, sample fingerprints, work counters, primary timings and independent replay timings. Both original receipt byte hashes can be reconstructed from it without private data.

## MIME needs its own optimizer check

The new [MIME kernel](../../src/federated/mime_adam.py) implements population-weighted full anchors and paired local gradient correction. Its adaptive moments stay fixed during local steps and update once per server round. The published lagged-second-moment update is not ordinary PyTorch Adam; the kernel explicitly omits bias correction, clipping and weight decay. [MIME Algorithm 1 and Table 5](https://arxiv.org/html/2008.03606).

Seven synthetic tests cover the equations, one-local-step agreement with a matching centralized base update, an independent multi-step scalar recursion, unchanged source state, streamed anchors with a shorter final batch, invalid inputs and serialized recovery. These are kernel checks, not a dataset training trial. The caller must still verify exact row coverage, client ownership and preprocessing provenance before using this kernel for a real comparison.

There is one practical issue to settle before training: a zero initial second moment combined with a tiny epsilon can create an excessively large first step. The numerical tests use explicitly stated values; they do not select a performant learning rate, epsilon or initialization. A matched centralized version of this same base updater is required so that changing optimizer conventions is not credited to federation.

## Work and communication

One full shared-Adam audit, including both calculation paths and replay from the midpoint, processes **1,474,560 backward examples**, 2,880 backward batches and 144 optimizer steps. Independent repetition doubles these audit counts. The cases reuse the same sampled-row schedule; this is processed work, not that many unique flows. There are zero quality-evaluation examples. CPU execution uses two intra-op threads and one inter-op thread; no cloud instance or GPU is required.

For the federated path alone, twelve FP32 synchronization steps would exchange an estimated **9,784,320 bytes**: one model broadcast and one gradient upload per client per step. FP64 doubles that payload. These counts exclude audit replay, protocol overhead and raw-data collection, and are not measured network traffic. More frequent synchronization can preserve optimizer behavior while worsening communication cost.

For full MIME, count every full anchor pass as well as both local gradient calculations. With N anchor examples and W local examples per round, the implemented backward work is N + 2W. The same-client protocol also exchanges six model-sized tensors per client per round when the anchor upload is reused. Reporting only local steps or rounds would hide much of its cost. Inference still uses the same small classifier.

## The next performance experiment

The next stage should be a **bounded, from-scratch local comparison**, not cloud scaling or a new classifier search. Its primary question is whether globally shared optimizer information closes the non-IID macro-F1 gap at an acceptable alarm and training-cost budget. Rare-class recovery can remain a secondary outcome, with all eight recalls still reported.

Before fitting, freeze a training-derived fit/development/confirmation split with duplicate groups kept together, and refit preprocessing and class weights using only the fit portion. Previously used training rows are not wholly independent new evidence; keep that limitation explicit. Original validation and final-test arrays remain outside method selection. The reused historical scaler in this numerical audit is not a substitute for that new performance protocol.

Include ordinary centralized Adam, its matched accumulated-gradient reference, reset-client-Adam FedAvg, the precise centralized MIME base updater and fixed-block MIME. Use comparable, small tuning allowances and report matched backward work and modeled communication separately. Define a practically meaningful central-performance gap and false-alert allowance before scores are observed. Do not add the adaptive block-length extension unless fixed MIME first demonstrates a useful trade-off.

This stage therefore supplies a verified calculation path and a tested MIME kernel. The open question is empirical: how much classification quality they retain for the compute and communication spent. No claim of novelty or performance improvement follows from numerical agreement alone.

## Local verification

```bash
.venv/bin/python -m unittest tests.test_shared_adam tests.test_mime_adam tests.test_shared_adam_evidence -q
```

The modules are separate from historical runners and frozen artifacts. Existing comments, completed model weights and unrelated working-tree changes are preserved. Implementation is on the local `experiment/federated-shared-optimizer` branch; no push or cloud launch is required for this check.
