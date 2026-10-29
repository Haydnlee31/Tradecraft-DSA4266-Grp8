# Bounded extensions to the primary study

The primary choice remains **block128 compressed Shared Adam**. These three
extensions explain where improvements come from and what they cost; retaining
that choice does not imply it has the highest macro-F1. No new classifier search
or hyperparameter sweep is part of this integration.

| Extension | Scientific question | Required comparison |
| --- | --- | --- |
| Neural FedLC | Does client-specific loss calibration reduce label skew? | FedAvg ± FedLC and compressed Shared Adam ± FedLC; same initialization, sampled rows, work, architecture and class weights |
| Cyclic XGBoost + FedLC | Does the calibration idea transfer to trees? | Cyclic boosting ± FedLC, plus a centralized tree-capacity control |
| Piecewise-linear encoding (PLE) | Does a better numerical representation help compressed training? | Central PLE, compressed PLE, raw compressed and a raw wider network of almost equal parameter count |

See the generated [results and all class recalls](../../reports/bounded_extensions/results.md)
and [interpretation](../../reports/bounded_extensions/findings.md). The
[committed plan](plan.json) was frozen before the additional controls were run.
Historical source branches and their evidence remain credited there. No source
comments or historical unfavorable results were removed.

## Reproduce

Use the Python environment documented in [primary reproduction](../../docs/reproduction.md).
Exact endpoint checks were tested on the recorded local macOS ARM CPU runtime;
cross-platform synthetic tests do not promise bitwise numerical equivalence.
XGBoost and PyTorch must run in separate processes on this macOS build. Do not
use `KMP_DUPLICATE_LIB_OK` to suppress native runtime conflicts.

From the repository root:

```bash
python -m pip install -r requirements-extensions.txt
python -m pip check
python scripts/study.py setup
python -m unittest tests.test_bounded_extensions tests.test_neural_fedlc tests.test_ple_shared_adam tests.test_federated_xgboost_components tests.test_federated_xgboost_mechanisms -v
```

Then run each command to completion. Use unused output paths:

```bash
python -u -m extensions.fedlc.run --data outputs/shared-optimizer-development-data-v1 --output extensions/fedlc/runs/my-reproduction
python -u scripts/bounded_extensions.py trees --output outputs/my-extension-trees
python -u scripts/bounded_extensions.py ple --output outputs/my-extension-ple
python scripts/bounded_extensions_report.py --neural extensions/fedlc/runs/my-reproduction --trees outputs/my-extension-trees --ple outputs/my-extension-ple --output outputs/my-extension-report
```

The neural runner checkpoints and can resume with the same command. Completed
runs are protected from overwrite. Tree and PLE training currently restart in a
new output directory if interrupted; their bounded runs do not claim resumable
optimizer recovery. The neural synthetic tests check exact recovery.

The tree and PLE adapters retrain existing frozen candidates and verify historical
models, development outputs/thresholds and confirmation endpoints. They also run
the added controls on development only. They never rewrite historical evidence.
Confirmation replay checks reproducibility of already exposed data; it is **not**
a new holdout or permission to tune thresholds on confirmation. Original
validation/test arrays are not read. No cloud instance is needed.

To rebuild the table and plot without training or downloading data:

```bash
python scripts/bounded_extensions_report.py --evidence reports/bounded_extensions/results.json --output outputs/extension-report-rebuild
```

## Fairness and interpretation

All reported development rows share the same fit population, fixed client
partition, calibration/audit split and seeds 7/17/27. All neural arms use 21.6M
backward examples, but PLE has 16,232 parameters versus 5,096, so equal examples
do not imply equal FLOPs or time. The raw wider control has 16,196 parameters.
PLE quantile bins use fitting rows only; fitting them centrally is an explicit
simulation convenience, not a decentralized preprocessing implementation.

The centralized tree control and cyclic models have 200 boosting rounds with
the same depth and learning-rate schedule. Central trees see all 500k fitting
rows each round (100M row-visits); cyclic trees see one client's rows each visit
(5M row-visits over ten complete passes). They are **capacity controls, not
compute-matched controls**. Tree models are much larger than the small MLP.
Historical tree tuning was more extensive than neural FedLC's fixed tau=1;
there is no equal hyperparameter-search-budget claim.

Tree cost is reported two ways: the historical cached-tree estimate and the sum
of full-model UBJSON transfers at each sequential client visit. The original
simulation does not implement a cached-tree transport, so the ~54 MB estimate
must not be presented as measured traffic. Neural payload likewise counts
application messages, not TCP/TLS overhead or measured network latency. Logical
barriers, client participation and model sizes differ.

The neural FedLC adaptation uses floored local class counts and the existing
square-root-weighted loss. It is not an exact reproduction of the original paper;
the combination can correct imbalance twice. Inference does not apply margins.
Tree FedLC begins offsets after the first client initializes the intercept.
PLE here is a fixed clipped-ramp encoding, not a claim to reproduce every learned
embedding variant from its source paper.

All cohorts are within-collection and were previously examined; there is one
fixed client partition. Three training seeds are not three independent datasets.
Rare-class improvement requires examining precision and recall, not macro-F1 alone.
Primary, development and MimeLite tables remain separate.

## Sources

- [FedLC: Zhang et al., ICML 2022](https://proceedings.mlr.press/v162/zhang22p.html).
- [Numerical embeddings / PLE: Gorishniy et al., NeurIPS 2022](https://arxiv.org/abs/2203.05556).
- [Flower's cyclic XGBoost example](https://flower.ai/docs/examples/xgboost-comprehensive.html).

This repository simulates the cyclic mechanism; these extension runs are not
physical Flower client deployments.

The integration executes independent local jobs concurrently. Recorded elapsed
times are provenance only, not a controlled runtime comparison. A timing study
would need isolated repeated runs on the same hardware. Central tree intercepts
are initialized from pooled fitting data, whereas cyclic trees start with the
first client; this is part of the algorithm difference. The ±FedLC tree pair
shares that same first client and subsequent visit order.

Two different operations use the word *calibration*: **FedLC** changes the logits
inside each client's training loss; **alarm calibration** sets a post-training
benign-versus-attack threshold on the development calibration half. Every method,
including controls without FedLC, uses the latter evaluation rule. PLE instead
turns each numerical feature into positions within fitting-data quantile ranges;
it changes the representation supplied to the neural network.

Payload accounting covers training model/gradient messages. It excludes shared
preprocessing setup (scaler, class-weight and PLE-bin distribution), raw-data
upload for centralized training, transport framing and retransmissions. Those
costs are not silently treated as measured zeros. A tree's 200 sequential client
visits are not equivalent to 200 full-participation neural synchronization rounds.

The tree capacity control matches the **maximum depth and boosting-round count**,
not the realized number of leaves or model bytes. With no subsampling, this
centralized tree configuration is deterministic across the requested seeds;
identical repeats are not evidence of zero population uncertainty. Cyclic seeds
change client visit order. Neural seed error bars are descriptive training
variation on one fixed dataset and partition, not confidence intervals.
