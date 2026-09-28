#!/usr/bin/env bash
# Non-IID sweep: Dirichlet alpha in {0.1, 0.5, 1.0} plus the IID baseline, 3 seeds each.
# Builds each partition file first (skipped if it exists), then runs ./run_fed.sh.
# Usage: NUM_NODES=20 ./run_sweep.sh [extra run-config pairs, e.g. "strategy='fedavg'"]
set -euo pipefail
cd "$(dirname "$0")"
export NUM_NODES=${NUM_NODES:-20}
for seed in 0 1 2; do
  for alpha in 0.1 0.5 1.0; do
    if [ ! -f "data/partitions/dirichlet_a${alpha}_n${NUM_NODES}_s${seed}.parquet" ]; then
      python -m src.federated.partition --alpha "$alpha" --num-partitions "$NUM_NODES" --seed "$seed"
    fi
    SEED=$seed PARTITIONER=dirichlet ALPHA=$alpha ./run_fed.sh "$@"
  done
  SEED=$seed PARTITIONER=iid ./run_fed.sh "$@"
done
