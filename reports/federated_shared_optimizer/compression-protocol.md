# Smaller messages for the shared optimizer

The shared-gradient reference approaches centralized-light quality but uses twenty times FedAvg's modeled training payload. This experiment asks whether smaller messages can preserve that quality. It targets bytes, not synchronization frequency: all candidates still perform 2,160 synchronized updates.

## Mechanism and prior work

Instead of sending every gradient coordinate as a 32-bit float, send a packed 4-bit or 2-bit approximation and a scale per tensor. With error feedback, each client remembers what rounding omitted and adds it to its next gradient message. The server keeps its full-precision Adam moments and target weights. Its outgoing message describes the difference between that target and the last model actually delivered to clients, carrying undelivered updates forward. Every client differentiates the delivered model; evaluation also uses that model, not the hidden server target.

Error feedback is an established idea, not a novelty claim. [Karimireddy et al.](https://proceedings.mlr.press/v97/karimireddy19a.html) show why discarded compression error matters and analyze it for SGD. [Tang et al.](https://proceedings.mlr.press/v139/tang21a.html) explain why Adam's nonlinear dependence on gradients needs special treatment; their 1-bit Adam freezes a variance preconditioner after warmup. [Chen et al.](https://arxiv.org/abs/2004.14180) also study quantized distributed Adam with error feedback. These works motivate the experiment, but their guarantees do not establish that this custom two-way recipe will work on our non-IID clients. We do not reproduce their algorithms or claim their guarantees.

The 2-bit codec is deliberately aggressive: it has three signed levels and one unused symbol, not four equally useful signed levels. Both codecs use deterministic nearest rounding with a maximum-absolute scale for each tensor. An 8-byte packet header carries version, precision, tensor count and sequence number. Every tensor carries a 4-byte element count and 4-byte scale. Values are actually bit-packed into bytes, then decoded before use. This prevents claiming 4-bit savings while passing full-precision tensors underneath.

## Frozen comparison

Run four seed 7 arms: a full-precision identity bridge, 4-bit error feedback, 2-bit error feedback, and a 4-bit no-feedback control. The bridge must exactly reproduce the previous full shared-gradient endpoint, sample chain and optimizer state. The control removes both upstream error retention and downstream catch-up; it tells us whether retaining those errors helps this recipe, without being an additional eligible candidate.

All four arms start fresh and use the same initialization, client samples, data, loss, 0.003 learning rate and 21.6-million backward-example budget. Total training work is 86.4 million examples; replaying the last two steps of each arm adds 80,000 separately charged recovery examples. No early stopping, best-checkpoint selection or settings changes are allowed after scores are seen.

Use the existing grouped calibration/audit split. Freeze each threshold using calibration benign rows, then report every class on the audit rows at the existing 1%, 5%, 10% and 20% targets. The primary target remains 5%. Reserved confirmation and original holdouts remain closed.

## Acceptance and communication accounting

For either error-feedback candidate to warrant further replication, all primary checks must hold:

- Lose no more than 0.01 macro-F1 versus the seed 7 shared-gradient reference.
- Differ in actual audit false-alert rate by no more than 0.005 from that reference.
- Lose no more than 0.05 recall in any class versus that reference.
- Retain at least 0.08 macro-F1 advantage over seed 7 FedAvg at the same calibration target.
- Use at most 15% of the historical shared-gradient payload for 4-bit, or 8% for 2-bit.

If both pass, prefer the smaller serialized payload; break a byte tie by primary macro-F1. Passing allows a later paired-seed replication, not a final success or holdout claim. Failure will be reported without changing the tolerances. The original uncalibrated continuation screen is not revised.

Count both directions, tensor metadata, padding, sequence headers and the initial full-precision broadcast. The historical reference did not include these codec headers or an extra startup charge; comparing our larger accounting scope against its raw tensor estimate is conservative for a savings claim. Delivering the final update is also charged. Shapes and model structure are pre-agreed; transport, encryption, retransmission and setup-schema overhead remain outside the estimate. Store no raw rows or messages in Git.

Byte savings do not establish faster training: this local simulation includes codec CPU overhead but no network latency. Both the server and each virtual client need residual/model state, which must be included in recovery tests. The primary scientific question is a development quality-versus-payload trade-off, not rare-class resolution, privacy or new-session generalization.
