#!/usr/bin/env bash
# Usage: [NUM_NODES=20] [SEED=0] [PARTITIONER=dirichlet|iid] [ALPHA=0.5] ./run_fed.sh [extra run-config pairs]
#   e.g. SEED=1 ALPHA=0.1 ./run_fed.sh "strategy='fedavg' num-server-rounds=1"
# NUM_NODES: number of simulated clients (Flower supernodes), default 105. Each client
#   gets one partition, so with partitioner=dirichlet it must equal the N the partition
#   file was built with (python -m src.federated.partition --num-partitions N). Clients
#   are arbitrary shards of the data, not the 105 real CICIoT2023 devices.
# SEED / PARTITIONER / ALPHA: passed through as seed / partitioner / dirichlet-alpha;
#   unset = pyproject.toml's value. Pairs given as arguments override them.
# Concurrency = INIT_CPUS / CLIENT_CPUS (default 4 / 2 = 2 clients at a time).
# Needs Flower's Ray runtime: on Windows with Smart App Control, run under WSL2 (see README).
cd "$(dirname "$0")"
PASS="${SEED:+seed=$SEED }${PARTITIONER:+partitioner='$PARTITIONER' }${ALPHA:+dirichlet-alpha=$ALPHA }"
flwr run . local --stream \
  --run-config "splits-dir='$PWD/data/splits' output-dir='$PWD/outputs/federated' reports-dir='$PWD/reports' $PASS$*" \
  --federation-config "num-supernodes=${NUM_NODES:-105} init-args-num-cpus=${INIT_CPUS:-4} client-resources-num-cpus=${CLIENT_CPUS:-2}"
