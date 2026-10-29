# Tradecraft

**How closely can federated training approach centralized IoT intrusion classification, and what communication cost is required to close the gap?**

The primary study uses the downloaded CICIoT2023 **39-feature** representation, eight classes, one small classifier and 20 simulated non-IID clients. Central Adam and FedAvg establish the gap; shared-gradient Adam and its blockwise compressed version investigate the quality/communication trade-off. FedProx, SCAFFOLD and full MIME provide supporting comparisons with optimizer controls.

## Start here

| Need | Guide |
| --- | --- |
| Research scope and claims | [Study scope](docs/study-scope.md) |
| Sources, row selection and split meanings | [Data protocol](docs/data-protocol.md) |
| Run the report's experiments | [Reproduction](docs/reproduction.md) |
| Main results and figures | [Primary study](studies/primary/README.md) |
| Additional MimeLite comparison | [Separate supplement](studies/mimelite-supplement/README.md) |
| Earlier Kaggle/cloud/SHAP studies | [Historical index](studies/historical/README.md) |
| Cleanup and verification | [Repository maintenance](docs/repository-maintenance.md) |

The primary report contains **11 configurations x 3 seeds**, each with 21.6M backward examples, including anchor/correction gradients. Confirmation macro-F1 is 0.6535 Central Adam, 0.5187 FedAvg, 0.6361 Shared Adam and 0.6325 compressed Shared Adam. These are within-collection results on an already evaluated confirmation cohort, not a new external test. Rare-class recognition remains weak.

## One supported entry point

From a configured environment:

```bash
python scripts/study.py --help
python scripts/study.py audit
python scripts/study.py report --output outputs/report-rebuild
```

Those commands do not train or score traffic. To download verified inputs and retrain all 33 primary endpoints, follow the runtime setup in the reproduction guide, then use a **new output directory**:

```bash
python scripts/study.py reproduce --output outputs/primary-reproduction
```

This performs setup, preflight, training, frozen-threshold confirmation scoring, exact endpoint comparison and report generation. It does not tune on confirmation. The primary runner retains its strict macOS ARM CPU runtime checks. Portable compressed-model reproduction is a separate route; it does not promise bitwise equality or portability for all methods. No cloud instance is needed.

## Repository boundaries

`src/` contains active implementations and compatibility dependencies; `studies/` provides experiment entry points; `reports/` preserves immutable evidence and figures; `archive/` preserves retired work with checksums. Raw/derived data, checkpoints and scratch runs stay outside Git in `data/` and `outputs/`. Prepared inputs are a verified release asset. Raw-CSV reconstruction is an optional, heavier route.

Earlier Kaggle scores, official39 cloud scores, primary confirmation scores and MimeLite development scores use different protocols. Do not combine them into one leaderboard. Current methodology runs used local CPU, not Tesla T4. Communication is application-payload accounting, not physical-network timing.

Contributors: read [AGENTS.md](AGENTS.md). Comments, source hashes and unfavorable results are preserved; cleanup does not erase contrary evidence.
