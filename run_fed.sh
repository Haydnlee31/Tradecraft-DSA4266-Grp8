#!/usr/bin/env bash
# Usage: ./run_fed.sh [extra run-config pairs, e.g. "num-server-rounds=1"]
# Concurrency = INIT_CPUS / CLIENT_CPUS (default 4 / 2 = 2 clients at a time).
# NUM_NODES means arbitrary simulated client shards, not the 105 physical devices:
# device identity is absent from the public flow CSVs (see CLAUDE.md).
# SEED / PARTITIONER / ALPHA are forwarded to Flower; explicit argument pairs
# override them. NUM_NODES must match the persisted partition's client count.
# Needs Flower's Ray runtime: on Windows with Smart App Control, use WSL2.
set -euo pipefail
cd "$(dirname "$0")"
PASS="${SEED:+seed=$SEED }${PARTITIONER:+partitioner='$PARTITIONER' }${ALPHA:+dirichlet-alpha=$ALPHA }"
flwr run . local --stream \
  --run-config "splits-dir='$PWD/data/splits' output-dir='$PWD/outputs/federated' reports-dir='$PWD/reports' $PASS$*" \
  --federation-config "num-supernodes=${NUM_NODES:-20} init-args-num-cpus=${INIT_CPUS:-4} client-resources-num-cpus=${CLIENT_CPUS:-2}"
