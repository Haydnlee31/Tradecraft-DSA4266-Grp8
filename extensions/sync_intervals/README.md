# Synchronization-interval extension

**Completed real-data screen:** [findings and audit](../../reports/extensions/sync_intervals/findings.md).
K=4 and K=20 reduced synchronization but did not retain K=1 macro-F1 or
per-class recall within the predeclared tolerances. All five endpoints were
retained and independently verified.

An isolated CPU experiment: allow clients to take K plain SGD steps before one
persistent server Adam update. Compare K=1,4,20, plus local-SGD FedAvg and reset
client-Adam FedAvg at K=20. All use the same canonical light model, global weighted
loss, dataset, client populations and per-client ordered minibatch streams.

This is a normalized [FedOpt-inspired adaptation](https://arxiv.org/html/2003.00295v5),
not unchanged published FedAdam or a new algorithm. The local message is the
mean gradient along its SGD trajectory, mathematically
`(start_parameters - end_parameters)/(local_lr*K)`. Accumulating path gradients
avoids float32 cancellation from subtracting nearly equal parameter tensors.
At K=1, use the original gradient directly: no redundant local update, and an
exact numerical bridge to the original shared-gradient Adam control.

The server weights these messages by client population and applies one Adam
update. Its moments persist between rounds and through recovery. Local SGD has
no momentum, weight decay or clipping. Local/server rates are frozen at .003,
not selected by this screen. The original server Adam betas (.9,.99), epsilon
.001 and zero weight decay are retained. Compression is disabled.

## Protocol fixed before results

The checked-in plan uses seed 7, 500,000 fit rows, 399,552 development rows, twenty
clients, batch size 500, and 21.6 million backward examples per arm. The five
arms cost 108 million backward examples total, plus separately counted audit
replays. K=1,4,20 yield 2,160 / 540 / 108 synchronization rounds. Server Adam
update counts and effective aggregation batch sizes also change; this is not
a pure communication-cost intervention with an unchanged optimization path.

Each client's microstep s maps to the original sampler's `(s//20, client, s%20)`.
Per-client hash chains must agree across all arms even though inter-client
traversal order differs. This retains historical K=1 shared-Adam and K=20
reset-Adam FedAvg samples, allowing exact bridges for both controls.

Report final-development argmax macro-F1, all eight classes' P/R/F1/support,
confusion matrix, benign false alerts, host training time and counted payload.
No confirmation/test data are opened, no threshold is tuned, and no best-round
selection occurs. The historical 0.6325 confirmation score is a different
protocol and is not the reference score here.

A useful communication trade-off must lose no more than .01 macro-F1, increase
FAR by no more than .005, lose no more than .05 recall in any class, and reduce
barriers by at least fourfold, relative to K=1. These are proposed engineering
gates, not statistical significance tests. All outcomes, including failures,
are retained. One seed/partition cannot establish a general performance gain.

## Run from the original workspace root

Use the already verified environment and downloaded bundle:

```bash
outputs/Tradecraft-reproduce/.venv-fedntd/bin/python -m unittest extensions.sync_intervals.test_sync_intervals -v
outputs/Tradecraft-reproduce/.venv-fedntd/bin/python -m extensions.sync_intervals.run \
  --data outputs/Tradecraft-reproduce/outputs/shared-optimizer-development-data-v1 \
  --output extensions/sync_intervals/runs/seed7-screen-v1
outputs/Tradecraft-reproduce/.venv-fedntd/bin/python -m extensions.sync_intervals.audit \
  --data outputs/Tradecraft-reproduce/outputs/shared-optimizer-development-data-v1 \
  --run extensions/sync_intervals/runs/seed7-screen-v1 \
  --output reports/extensions/sync_intervals/audit.json
```

The audit replays the last synchronization from each `recovery.pt`, comparing
weights, Adam moments, counters and sample hashes bit for bit. It independently
recomputes metrics with scikit-learn, verifies original-reference equality for
K=1 and reset-Adam FedAvg, and checks payload/work/gates. Audit backward examples
are additional verification work, not hidden inside the original arm budget.

Re-run an interrupted training command to resume from its latest completed
synchronization. A contract binds the code, plans, data and runtime. It refuses
changed contracts and completed-run overwrite. `RUNNING.lock` prevents two
writers; remove a stale lock only after verifying its process has stopped.

## Isolation and limits

All new source is under this extension. It imports the previous extension's
read-only data adapter, model/evaluation/atomic-write helpers, and the original
numerical optimizer helpers. It does not modify or monkey-patch any of them.
The existing source-hashing contracts remain valid. No dependency changes,
new Flower registration, compression changes, or original flow edits are needed.
Outputs are confined to this extension's ignored `runs/` directory.

Model-sized messages are counted in both directions per synchronization. Payload
excludes protocol overhead; network speed and edge feasibility are unmeasured.
CPU training-loop time excludes checkpoint I/O and evaluation. Local SGD rate
and server learning rate interact; this fixed-recipe result is not an exhaustive
assessment of local SGD or FedOpt. The data were previously used for research,
so development results remain exploratory.
