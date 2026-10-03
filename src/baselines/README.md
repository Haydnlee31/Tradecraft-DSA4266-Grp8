# Classical baseline experiments

These supplementary baselines do not replace or modify the original MLP lanes:

- `logistic`: centralized eight-class softmax regression (46 inputs, 376 parameters).
- `logistic-iid`: the same model with 20 full-participation IID clients.
- `logistic-dirichlet`: the same model with self-balancing Dirichlet alpha 0.5 clients.
- `xgboost`: centralized multiclass histogram gradient boosting. No federated XGBoost.

PyTorch is only the optimization library for logistic regression: a single linear
layer with softmax probabilities is not a deep model. Flower performs actual
sample-count-weighted parameter aggregation, sequentially on one CPU host. Ray
and an online logging account are not required.

XGBoost runs in a fresh subprocess, deliberately without importing PyTorch.
This avoids a native OpenMP runtime conflict reproduced with the installed
macOS wheels. Temporary array-transfer files are cleaned up after the worker
finishes; their time is included in XGBoost's host elapsed time.

## Installation and execution

Use the project's Python 3.11+ virtual environment on macOS, Linux or Windows.
Logistic regression uses existing core dependencies. For XGBoost, install:

```sh
python -m pip install -r requirements-baselines.txt
```

Alternatively use `python -m pip install -e ".[baselines]"`. An XGBoost import
failure on macOS may require an OpenMP runtime; see the official installation
guide: https://xgboost.readthedocs.io/en/stable/install.html.

Run a validation-only screen (all four lanes, one model seed):

```sh
python -m src.baselines.run --output outputs/baselines/screen-01 --seeds 0
```

Run just the logistic lanes without installing XGBoost:

```sh
python -m src.baselines.run --output outputs/baselines/logistic-01 --lanes logistic logistic-iid logistic-dirichlet --seeds 0 1 2
```

Every output directory must be new. Existing results are never overwritten or
silently resumed. Defaults are 30 epochs/rounds, patience 5 (0 disables stopping),
batch 512, Adam learning rate 0.001, weight decay 0.00001, square-root weighted CE.
All examples are retained, including singleton batches. Central Adam persists;
client Adam resets every round. Each client trains one local epoch.

XGBoost defaults: 200 boosting rounds, depth 6, learning rate 0.05, lambda 1,
full row/column sampling, CPU histogram trees, two threads. It uses the same
train-derived class weights as per-row sample weights, not the same optimizer or
minibatch loss normalization as PyTorch. Best checkpoint selection maximizes
unweighted validation macro-F1, not accuracy or weighted validation loss.
The saved ensemble is physically sliced to the selected boosting round.

Only after freezing a recipe, add `--evaluate-test` with a new output directory.
This reruns training deterministically and evaluates the selected checkpoint.
Without that flag, test.parquet is neither loaded nor required. The existing test
set has already been inspected, so further test scores remain descriptive.

## Reproducibility and comparison

Both model families reuse the canonical labels, feature order and pooled,
train-only scaler. Existing upstream split boundaries are never reshuffled.
Only sampled splits are loaded into RAM; do not point this at the raw release.
Global scaler/class statistics and full labeled server validation are simulation
assumptions, not privacy guarantees. XGBoost also receives these standardized
features for consistency; scaling is not claimed necessary for trees.

`--partition-seed 0` is independent of `--seeds`. IID follows the original row-
modulo assignment; non-IID calls the original partition builder (minimum 10 rows
per client). Exact mappings are saved and hashed. For direct comparison to an
older non-IID MLP run, use its partition seed; the old study tied both seeds.

Each experiment records source hashes, Git commit/status, package versions,
split hashes, scaler, class counts, feature order, settings and partition hashes.
Each run saves a model, checkpoint hash, validation P/R/F1 for all eight classes,
benign false alerts, confusion matrix and host elapsed time. Logistic runs also
record actual example exposure and optimizer updates. XGBoost does not have a
directly equivalent neural parameter count or minibatch update count.

Model tensor bytes, serialized checkpoint bytes and runtime RAM are different
quantities. No physical edge latency, power, network or privacy is measured.
These reports live separately and are not fed into the MLP-only decision selector.

## Follow-up neural-network experiments (not implemented here)

1. Verify matched one-client MLP update parity, including RNG, BN and optimizer
   reset semantics. Standardize provenance and validation-loss reporting first.
2. Repeat four MLP lanes up to 60 epochs/rounds without early stopping; compare
   best through 30 versus 60 and retrospectively apply patience 5.
3. Screen one factor at a time: CE versus sqrt-weighted CE; BN versus LayerNorm
   across matched light lanes; heavy dropout 0.3 versus 0.2 to isolate capacity.
4. If optimization evidence warrants it, compare FL learning rates 0.001 and
   0.0003, or reset versus persistent Adam moments—not both simultaneously.
5. Freeze a shortlisted recipe and confirm across three training seeds with
   fixed partitions. Use validation macro-F1, every class's recall/precision and
   benign false alerts. Do not choose settings on test scores or assume more
   layers will solve the federated gap.

Preserve original baselines and use equal validation search budgets when claiming
best-tuned model comparisons. One central epoch is not one FL round in optimizer
trajectory or compute, even at equal example exposure.
