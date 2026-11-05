# Tradecraft

**How closely can federated training approach centralized IoT intrusion classification, and what communication cost is required to close the gap?**

Tradecraft studies centralized versus federated learning using the downloaded CICIoT2023 **39-feature** representation and eight traffic classes. Central Adam and FedAvg establish the performance gap; shared-gradient Adam and its compressed version investigate how much of that gap can be recovered and at what communication cost.

Networked robot fleets motivate the application: sites may benefit from collaborative intrusion detection while facing constraints on pooling operational records. The experiments use public IoT traffic statistics and simulated clients, **not physical robots or a deployed intrusion-detection system**. The contribution is a reproducible empirical comparison, not a claim to a new Adam optimizer or a privacy guarantee.

## Start here

| Need | Guide |
| --- | --- |
| Research scope and claims | [Study scope](docs/study-scope.md) |
| Sources, row selection and split meanings | [Data protocol](docs/data-protocol.md) |
| Set-up and methodology for the thesis | [Methodology](docs/methodology.md) |
| Run the report's experiments | [Reproduction guide](docs/reproduction.md) |
| Main results, class recalls and figures | [Primary results](reports/methodology_comparison/results.md) |
| What the results support | [Findings and limitations](reports/methodology_comparison/findings.md) |
| All studies and their evaluation populations | [Results index](docs/results-index.md) |
| FedLC, cyclic XGBoost and PLE | [Bounded extensions and reproduction](studies/bounded-extensions/README.md) |
| Additional MimeLite comparison | [Separate supplement](studies/mimelite-supplement/README.md) |
| Earlier Kaggle/cloud/SHAP studies | [Historical index](studies/historical/README.md) |
| Source integrity and verification | [Repository maintenance](docs/repository-maintenance.md) |

## Main findings

These are **three-seed mean confirmation results**, using thresholds fitted to a 5% benign false-alert target on development data. Actual confirmation false-alert rates are reported separately.

| Method | Macro-F1 (mean ± SD) | Actual false alerts | Training payload (MB) | Synchronizations |
| --- | ---: | ---: | ---: | ---: |
| Central Adam | 0.6535 ± 0.0076 | 5.30% | — | — |
| FedAvg-Adam | 0.5187 ± 0.0033 | 5.27% | 88.06 | 108 |
| Shared-gradient Adam | 0.6361 ± 0.0033 | 5.33% | 1,761.18 | 2,160 |
| **Compressed Shared Adam** | **0.6325 ± 0.0038** | **5.36%** | **240.26** | **2,160** |

Shared Adam aggregates client gradients at a common model and maintains one persistent server optimizer. The compressed version uses **four-bit quantization in blocks of up to 128 values, with residual feedback in both communication directions**. Evaluation uses the model delivered to clients.

- Compression reduces payload by **86.36%** relative to uncompressed Shared Adam, with a **0.0036 macro-F1 decrease**.
- Compressed Shared Adam remains **0.0210 macro-F1 below Central Adam**. It uses approximately **2.73 times FedAvg's payload and 20 times its synchronization count**.
- Rare-class detection remains unresolved: compressed Shared Adam has **9.87% Web-based recall and 26.28% Brute Force recall**. **0.6325 is macro-F1, not recall or accuracy.**

Payload describes application-level training messages, not measured network traffic or latency. Centralized raw-data collection, shared preprocessing setup and transport overhead are outside this accounting. Compression reduces bytes, not synchronization frequency. These results support a quality–communication trade-off, not universal superiority or deployment readiness.

The [full primary table](reports/methodology_comparison/results.md) retains **11 configurations × 3 seeds**, including both FedProx coefficients, SCAFFOLD, full MIME and the SGD/MIME-base optimizer controls. See the [interpretation](reports/methodology_comparison/findings.md) before comparing methods with different base optimizers.

## Data and comparison protocol

| Component | Primary protocol |
| --- | --- |
| Inputs | All 39 numeric features supplied by the downloaded official source; separate from the historical 46-feature Kaggle representation |
| Targets | Benign, DDoS, DoS, Recon, Web-based, Brute Force, Spoofing and Mirai |
| Fitting data | **500,000 rows**, selected from a historical 2M-row official training view |
| Development data | **399,552 rows**; includes 200,022 calibration and 199,530 audit rows |
| Confirmation data | **398,502 rows**; already evaluated, retrospective within-collection evidence |
| Classifier | One 39 → 64 → 32 → 8 LayerNorm MLP, **5,096 parameters** |
| Federation | **20 fixed non-IID virtual clients**, full participation and fitting-population aggregation weights |
| Training variation | Seeds **7, 17 and 27** on the same client partition |
| Objective | Square-root inverse-frequency weighted cross-entropy, with a fixed fitting-derived denominator |
| Work budget | **21.6M backward examples per run**, including anchor/correction gradients where applicable |

The models **do not train on all 2M rows**. Backward examples count repeated use of fitting rows during gradient calculations, not additional unique observations. Equal backward work does not mean equal FLOPs, optimizer calls, communication, wall time or hyperparameter-search effort.

Preparation streams the source CSVs, excludes malformed/nonfinite records, groups exact feature duplicates and removes groups with contradictory eight-class targets. Fixed hashes determine selection and cohort membership. The scaler and class weights use fitting rows only; an additional float32 collision check removes exact model-input overlap across cohorts. The resulting population represents cleaned unique vectors, not original repeated-flow frequencies. See the [data protocol](docs/data-protocol.md) for the complete provenance chain and [detailed methodology](reports/methodology_comparison/methodology-draft.md) for design choices.

**Evaluation boundary:** development was used during model selection, so its audit subset is not an independent final test. Confirmation was reserved for the original optimizer extension, but its rows had appeared in historical training and have since been evaluated. The expanded comparison is retrospective; it does not establish generalization to unseen devices or capture sessions. Thresholds are saved before confirmation scoring and are not refitted on confirmation.

## Quick start

Use `scripts/study.py` as the supported primary entry point. No cloud instance is needed.

### Set up the recorded environment

Exact all-method replay uses **Python 3.13.9, PyTorch 2.14.0, NumPy 2.5.3 and Flower 1.38.0 on macOS ARM CPU**. The primary runner enforces the recorded runtime; installing packages alone does not make exact training replay portable.

For a fresh checkout with Python 3.13.9 installed:

```bash
git clone --branch main https://github.com/Haydnlee31/Tradecraft-DSA4266-Grp8.git
cd Tradecraft-DSA4266-Grp8
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-methodology.txt
python -m pip check
python scripts/study.py --help
```

If you already have a working checkout and environment, activate it instead of recreating it. Linux/WSL users should consult the separate [compressed-model portable reproduction guide](reports/federated_shared_optimizer/reproduction-guide.md); that route reports cross-platform differences and does not validate exact replay of every method on CUDA or another OS.

### Audit and rebuild the published report without training

```bash
python scripts/study.py audit
python scripts/study.py report --output outputs/report-rebuild
```

This checks metrics against committed confusion-matrix evidence and rebuilds tables and figures. It needs no dataset download or checkpoints and **does not demonstrate fresh model training**. Choose an output directory that does not already exist.

### Retrain and compare all 33 primary endpoints

```bash
python scripts/study.py reproduce --output outputs/primary-reproduction
```

This downloads checksum-verified prepared inputs from the pinned release asset, checks runtime/source contracts, trains from initialization, calibrates on development, scores confirmation using saved thresholds, compares endpoints and builds a report. It does not search hyperparameters, select favourable seeds or tune on confirmation.

The output directory must be new. Inspect `reproduction-checks.json` for endpoint comparisons and `report/` for the generated results. The primary trainer has recovery verification tests but **no user-facing interrupted-training resume**; preserve partial outputs and start a new training directory if interrupted.

For manual stages, local archive inputs, tests and optional raw-CSV reconstruction, follow [docs/reproduction.md](docs/reproduction.md). Rebuilding recorded evidence, fresh training from prepared inputs and reconstructing inputs from raw CSVs are distinct workflows.

## Supporting studies

| Study | Question | How to interpret it |
| --- | --- | --- |
| Neural FedLC | Does client-specific loss calibration help label skew? | Paired FedAvg and compressed Shared Adam controls; modest FedAvg gains do not imply gains for every optimizer |
| Cyclic XGBoost with FedLC | Does calibration help sequential client boosting? | Paired tree control plus a central capacity control; row-work and transport assumptions differ from the neural comparison |
| Piecewise-linear encoding (PLE) | Does the numeric representation improve compressed training? | Includes centralized PLE and a similarly sized raw-input control; better scores come with a larger model and payload |
| MimeLite | What is the trade-off with fewer local correction costs? | Separate **23.1M-budget, development-only** supplement; not part of the primary confirmation table |

Read the [bounded-extension findings](reports/bounded_extensions/findings.md) and [run guide](studies/bounded-extensions/README.md) for FedLC, cyclic XGBoost and PLE. Compressed Shared Adam remains the primary choice for the study narrative, **not a claim to the highest accuracy across all representations and model families**. Existing candidate confirmation replays verify reproducibility; they do not create new independent evidence. The [MimeLite guide](studies/mimelite-supplement/README.md) keeps that protocol separate.

## What this study does not establish

- **Physical federation or robotics performance:** clients execute in one-machine simulation using custom PyTorch updates and some Flower helpers, not a physically distributed Flower deployment. The features do not identify real devices.
- **Privacy or security guarantees:** raw rows remain client-local during federated gradient calculations, but shared preprocessing is pooled in the simulation. Updates are not demonstrated private, and compression is not encryption.
- **Deployment latency or energy savings:** communication is application-payload accounting; no physical-network, edge-power or real-time performance claim is made.
- **A new optimizer theorem or a solution to rare attacks:** the evidence concerns the tested combinations and settings. Frequent synchronization, weak rare-class recall and one fixed client partition remain limitations.

Primary methodology runs used **local macOS ARM CPU**. Earlier **NVIDIA Tesla T4 cloud runs** belong to a separate experimental phase. Similar scores do not establish equivalent numerical trajectories, runtime or energy use across hardware.

## Repository boundaries

| Location | Purpose |
| --- | --- |
| `scripts/study.py` | Supported primary audit, training, comparison and reporting workflow |
| `src/` | Active implementations and compatibility dependencies |
| `studies/` | Primary, bounded-extension, MimeLite and historical study guides |
| `docs/` | Scope, methodology, data boundaries and reproduction instructions |
| `reports/` | Frozen evidence, results, figures and historical reports |
| `extensions/` | Supporting extension implementations and run interfaces |
| `tests/` | Implementation, protocol and regression checks |
| `archive/` | Retired work with checksums and restoration support |
| `data/`, `outputs/` | Local inputs, derived arrays, checkpoints and scratch runs; excluded from Git as appropriate |

Earlier Kaggle scores, official39 cloud scores, primary confirmation scores and MimeLite development scores use different protocols. **Do not combine them into one leaderboard.** Historical SHAP work remains supporting feature-level interpretation rather than validated security-policy automation; no dashboard is required for the current research scope.

Contributors: read [AGENTS.md](AGENTS.md) before changing the protocol. Preserve explanatory comments, hash-pinned implementations, plans and results, including unfavourable findings. Raw inputs and checkpoints stay out of Git; use the verified release asset for default reproduction. See [repository maintenance](docs/repository-maintenance.md) for archive and verification rules.
