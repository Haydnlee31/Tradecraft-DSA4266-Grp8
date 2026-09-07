# Edge Sentinel — Project Context for Claude Code

This file gives Claude Code the working context for this repository. Read it before
making architectural suggestions, adding dependencies, or re-opening scope decisions
that are already settled below. The full narrative plan (rationale, timeline, decision
log) lives in the published artifact linked at the bottom — this file is the condensed,
implementation-facing version of the same plan.

## What this project is

A semester deep learning project on IoT network security, built on the CICIoT2023
dataset. The task: classify network flows into 8 classes (benign + 7 attack categories)
and compare three training/deployment settings for that same task — centralized-heavy,
centralized-light, and federated-light — then use explainability (SHAP) to translate
the trained model's behavior into concrete security-policy recommendations.

This is deliberately **not** a pure accuracy leaderboard. The point of the project is
the trade-off: is a small, federated-trained model good enough to justify skipping a
heavy centralized one, and can the model's reasoning be turned into something a
security team could act on.

## Locked decisions — do not re-litigate

These were evaluated and decided already. Re-raising them costs a review cycle; only
revisit if new evidence contradicts the stated reason.

- **Classification target: 8-class (benign + 7 CICIoT2023 attack categories)** —
  DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, Mirai, Benign. Not binary
  (too narrow — that's what the reference paper already did), not full 33-attack
  granularity (several subclasses have too few samples for stable evaluation this
  semester; may be a stretch goal, not the baseline).
- **No device-topology GNN.** The public CICIoT2023 flow CSVs (47 columns: 46 flow
  statistics + label) contain no Src/Dst IP, MAC, or device-ID field. Building a real
  device graph would require the raw ~548 GB pcap release plus custom flow-to-device
  correlation — out of scope. Do not suggest "build a graph of the 105 devices" as a
  quick add-on; it isn't one.
- **No physical edge hardware.** All "edge feasibility" numbers (latency, memory,
  power) are *projected*, calibrated against the reference paper's measured NVIDIA
  Jetson Orin Nano benchmarks (~1s/flow inference, ~300 MB RAM, 10–12 W power,
  ~20 epochs to >90% accuracy on-device). Do not present projected numbers as measured.
- **No per-device vulnerability claims.** Because device identity isn't in the data,
  explainability output (Section 07 of the plan) stays at the protocol/feature level
  (e.g. "SYN-flag concentration drives DoS classification") — never "device type X is
  most vulnerable."
- **Federated learning is simulated**, not physically distributed. Multiple virtual
  clients on one machine via Flower, non-IID partitioning by attack-category mix. This
  is a stated scope choice, not a limitation to quietly work around.

## Reference paper (base comparison point)

Díaz-Gorrin, J.; Caballero-Gil, C.; Brankovic, L. *Enhancing Network Security with
Generative AI on Jetson Orin Nano.* Appl. Sci. 2026, 16, 1442. Their scope: binary
TCP DoS/DDoS detection only, using an AC-GAN, deployed on a real Jetson Orin Nano.
Their measured hardware numbers are our calibration anchor (see above). Their dataset
preprocessing (11 hand-picked TCP flow features, 10k balanced samples) is a useful
reference for feature selection but not binding — this project uses the fuller
46-feature set and the full attack taxonomy, not their TCP-only slice.

## Dataset notes (CICIoT2023)

- 105 real IoT devices, 33 attacks across 7 categories, ~46M+ flow records across
  ~169 CSV files, 47 columns (46 numeric flow-statistic features + label).
- Features are flow-level aggregates only: header length, protocol flags
  (fin/syn/rst/psh/ack/ece/cwr counts), rate/Srate/Drate, inter-arrival time stats,
  protocol indicator flags (HTTP/HTTPS/DNS/TCP/UDP/ICMP/etc.), and statistical
  moments (min/max/avg/std/magnitude/radius/covariance/variance/weight).
- **Do not attempt to load the full dataset into memory at once.** Use stratified,
  chunked sampling from the start. Preserve per-file provenance in the sampling
  script so results are reproducible.
- Severe class imbalance: DDoS and Mirai dominate; Brute Force and Web-based are thin
  slices. Never report bare accuracy as the headline metric — macro-F1 and per-class
  recall are the real metrics. A model that ignores rare classes and still scores
  95%+ accuracy is a failure, not a result.
- The Kaggle mirror (`himadri07/ciciot2023`) **is confirmed partial/merged**: it ships
  pre-split into `train.csv`/`validation.csv`/`test.csv` (~7.84M rows total, not the
  original 169-file layout), roughly 17% of the official ~46M-row release. All 34 raw
  labels are present in every split and match `src/data/label_map.py` exactly (47
  columns, same names assumed there). Because the split boundary is already fixed
  upstream, the pipeline respects it (`src/data/sample_dataset.py` samples within each
  given split rather than re-splitting) instead of building its own train/val/test
  split from a pooled sample.

## Repository structure (target)

```
data/
  raw/            # untouched CSVs (or a pointer/download script — do not commit raw data)
  sampled/        # stratified samples used for training, with a manifest of source files
  splits/         # fixed train/val/test splits (seeded, no flow leakage across splits)
src/
  data/           # sampling, label mapping (33 attacks -> 8 classes), preprocessing
  models/         # heavy / light architectures, shared training loop
  federated/      # Flower client/server setup, FedAvg orchestration
  eval/           # metrics (macro-F1, per-class recall), FLOPs/size accounting
  explain/        # SHAP pipeline, policy-recommendation generation
notebooks/        # exploration only — nothing load-bearing lives only in a notebook
reports/          # final write-up, figures, trade-off tables
```

## Tech stack

- Python 3.11+
- Polars or Dask for large-CSV handling (not plain pandas on the full dataset)
- PyTorch for model training
- Flower (`flwr`) for federated simulation
- scikit-learn for baselines and metrics
- SHAP for explainability

## Conventions

- Every training run is seeded and logs which sampled files/rows it used.
- Label mapping (33 attacks → 7 categories + benign) lives in one place
  (`src/data/label_map.py`) — do not hardcode category strings elsewhere.
- Any number presented as "measured" must have been actually run in this repo;
  anything derived from the reference paper's hardware is labeled "projected"
  in code comments, plots, and the report.
- Metrics reporting always includes per-class recall, not just an aggregate score.

## Current phase

Check `reports/` or ask the user — phases are: (1) scoping/literature, (2) data
engineering, (3) centralized baselines, (4) federated simulation, (5) efficiency/
trade-off analysis, (6) explainability & policy synthesis, (7) write-up. Full
week-by-week timeline is in the plan artifact below.

## Full plan

The complete narrative plan — problem statement, decision log with reasoning, dataset
constraints, the three-lane architecture comparison, evaluation plan, illustrative
explainability output format, and risks/limitations — is published here:
https://claude.ai/code/artifact/7d3aa32f-7fba-4f1f-9c01-7aa711009230
