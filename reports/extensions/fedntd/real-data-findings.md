# FedNTD real-data development screen

The frozen three-arm seed-7 screen completed. **Neither tested FedNTD setting
improved macro-F1 over the paired FedAvg control.** Both reduced benign false
alerts under ordinary argmax decisions. This is a useful negative result for
these settings, not evidence that FedNTD fails across all configurations.

## Results

- **FedAvg / beta 0:** macro-F1 **0.483683**, benign false-alert rate **1.4250%**.
- **FedNTD / beta 0.1:** macro-F1 **0.478103**, false-alert rate **0.8399%**.
  Macro-F1 decreased by **0.558 percentage points**.
- **FedNTD / beta 1:** macro-F1 **0.470483**, false-alert rate **0.4294%**.
  Macro-F1 decreased by **1.320 percentage points**.

All eight classes' recall percentages, in order **FedAvg / beta 0.1 / beta 1**:

- Benign: **98.575 / 99.160 / 99.571**; support 21,193.
- DDoS: **99.418 / 99.289 / 99.366**; support 235,894.
- DoS: **18.362 / 19.846 / 18.851**; support 72,090.
- Recon: **26.793 / 24.543 / 21.425**; support 13,246.
- Web-based: **0 / 0 / 0**; support 495.
- Brute Force: **19.600 / 17.200 / 22.000**; support 250.
- Spoofing: **15.467 / 14.018 / 10.568**; support 8,696.
- Mirai: **98.960 / 98.941 / 99.023**; support 47,688.

The lower false-alert rates come with mixed attack-category effects. Neither
setting recovers any Web-based examples. Beta 1 improves Brute Force recall but
reduces Recon and Spoofing recall. The broader pattern is consistent with a
more conservative classifier, though this screen alone does not establish the
mechanism. No extra beta, temperature or learning-rate search was launched.

The [audited JSON](real-data-results.json) includes full precision/recall/F1,
supports, confusion matrices, work counts, hashes and runtime information.

## Matched experimental conditions

The same 500,000 fit rows, 399,552 development rows, 20-client assignment,
initialization and sample sequence were used for every arm. Each model has
39 inputs, LayerNorm hidden widths 64/32, dropout zero and eight outputs:
5,096 parameters. Temperature was 1. Each client resets Adam each round, using
the frozen .003 learning rate, (.9,.99) moments, .001 epsilon, no weight decay,
and the fixed-training-mean square-root-weighted CE criterion.

Each arm completed 108 rounds, 20 local steps per client per round, batch size
500, and 21.6 million backward examples. Each nonzero-beta arm additionally
performed 21.6 million teacher-forward examples. All arms count **88,058,880
bytes** of model exchange; this is modeled tensor payload, excluding protocol
overhead, not measured network traffic.

Measured simulation-host training-loop time was **59.10 s / 83.67 s / 120.09 s**
for beta 0 / .1 / 1. These single-run times exclude preparation, checkpoint
writes and final evaluation and are not a repeated runtime benchmark or edge
hardware measurement. Different nonzero beta values use the same number of
teacher forwards; their observed time difference does not establish an
inherent complexity difference.

## Reproduction and audit

Before the extension, the original compressed Shared Adam experiment was
retrained from scratch for seeds 7, 17 and 27 in an isolated checkout of tag
`shared-adam-reproduction-v1` at commit
`9fd3f6b4c685a1ffb3c928eaed35b3311ffe0545`. Every training-state, threshold and
prediction equality check passed. Confirmation macro-F1 was **0.63026742,
0.63040034, 0.63694243**, giving **0.63253673** mean.

That **0.6325 is a separate reproduction result**: it uses the prior confirmation
cohort and fitted 5% target alarm rule. The new FedNTD results use development
argmax decisions and cannot be ranked directly against it.

The new FedAvg control also exactly matched the original seed-7 FedAvg endpoint's
weights, complete metrics and sample chain. The [audit script](audit_run.py)
reloaded all three extension checkpoints, re-inferred every development row,
and independently recomputed metrics with scikit-learn. All metric, source,
data, checkpoint and accounting checks passed. The 11 extension tests passed
in the frozen environment and again inside the reproduction checkout.

No tracked baseline files were edited in either checkout. A SHA-256 comparison
confirmed all 210 previously recorded original source/script/test and selected
configuration files remained unchanged. The extension is on local branch
`angel/fedntd-extension` in `outputs/Tradecraft-reproduce`; nothing was pushed.

The default bootstrap's macOS virtual environment failed to locate its Python
shared library. A separate `.venv-fedntd` was created with uv using Python
**3.13.9**, then the exact pinned dependencies were installed. The unchanged
setup script's supported `--python` option accepted it, and all frozen checks
passed. The incomplete default `.venv-shared-adam` was preserved.

## Artifacts and replay

Paths relative to the original workspace:

- Frozen checkout: `outputs/Tradecraft-reproduce/`.
- Prepared data: `outputs/Tradecraft-reproduce/outputs/shared-optimizer-development-data-v1/`.
- Baseline reproduction: `outputs/Tradecraft-reproduce/outputs/local-reproduction-v1/summary.json`.
- FedNTD results/checkpoints: `outputs/Tradecraft-reproduce/extensions/fedntd/runs/seed7-screen-v1/`.
- Training log: `outputs/Tradecraft-reproduce/outputs/fedntd-seed7-screen-v1.log`.

Replay the independent audit from the original workspace root:

```bash
outputs/Tradecraft-reproduce/.venv-fedntd/bin/python reports/extensions/fedntd/audit_run.py
```

The data, environment, nested checkout and checkpoints are ignored by Git.
The report and compact audit receipt can be versioned independently. Preserve
the local run directory for checkpoint replay; the JSON receipt alone cannot
reconstruct model weights.

## Interpretation

This is one seed and one client partition on an already used development cohort.
FedNTD training and its audit did not score confirmation or test data; only the
separate frozen baseline reproduction replayed its previously published
confirmation predictions. No new external generalization or privacy claim is
supported. The planned screen provides no macro-F1 reason to promote either
tested FedNTD setting over FedAvg. Any further calibration comparison or tuning
would be a separately specified experiment.
