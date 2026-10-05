# Bounded cloud-pilot handoff

The next stage is a small environment/runtime pilot, not a full sweep. Cloud
provisioning, provider selection and spending require the user's separate approval.
Do not treat the branch name as deployment approval.

1. Clone `pre-cloud-deployment` and record its exact commit. Use a fresh Linux
   Python 3.12 environment and the shared `requirements.txt` (or
   `requirements-baselines.txt` if running classical baselines). Do not install the
   macOS/Python-3.13 lock on Linux. Select the appropriate CUDA-enabled PyTorch
   installation using the [official installation selector](https://pytorch.org/get-started/locally/),
   then run `python -m pip check` and `python -m unittest discover -v`.
   Record the resolved package versions, GPU, driver and CUDA runtime; broad shared
   requirements alone do not reproduce the historical local environment.
2. Transfer train/validation samples and required artifacts privately, preserving
   relative paths. Verify sizes and SHA-256 against `checks/summary.json` before
   using them. Git contains code and reports, **not** the ignored data or weights.
   Keep the test split out of tuning. Load only trusted checkpoint/scaler files.
3. Confirm `torch.cuda.is_available()` and inspect the runner's `--help` and
   [research instructions](../../src/models/RESEARCH.md). Run fresh two-step heavy
   and non-IID LayerNorm jobs with explicit `--device cuda`, then repeat with
   pause/resume inside that same cloud environment. Inspect device manifests and
   compare final tensors and metrics against uninterrupted jobs. Never infer GPU
   use from an instance label or merely successful imports. If deterministic parity
   fails, investigate and document it before a longer run.
4. Use new output directories. Historical CPU recovery states cannot be assumed
   resumable under changed source, packages or devices; the runner intentionally
   checks provenance. Preserve original environments/source for historical recovery.
   Do not bypass those checks to make a GPU resume appear successful.
5. Benchmark measured wall time and peak memory for the pilot, and estimate the
   proposed total from the actual hourly rate, planned runs, setup/retries and storage.
   A $180 credit balance is a ceiling, not evidence that an arbitrary sweep fits.
   Verify provider billing increments, persistent storage, shutdown behavior and
   budget alerts. Establish an explicit cost cap and stop procedure before training.
6. Verify outputs survive stopping the instance and can be downloaded with hashes.
   Keep recovery states and manifests on durable storage; save metrics and logs
   frequently. Do not put credentials, datasets or checkpoints into Git to achieve this.
7. Only after those checks pass, approve and freeze the next bounded experiment
   plan. Record run count, seeds, partition seeds, selection metric, class-recall/FPR
   guardrails and budget. Retain the completed local study unchanged as the reference.

No CUDA parity, cloud persistence, provider price, remote CI result or cost cap has
been verified by the local checkpoint. These are explicit pilot acceptance checks,
not completed work. Federation remains a single-machine simulation even on a GPU.
