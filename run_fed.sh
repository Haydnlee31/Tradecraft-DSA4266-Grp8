#!/usr/bin/env bash
# Usage: ./run_fed.sh [extra run-config pairs, e.g. "num-server-rounds=1"]
# Concurrency = INIT_CPUS / CLIENT_CPUS (default 4 / 2 = 2 clients at a time).
cd "$(dirname "$0")"
flwr run . local --stream \
  --run-config "splits-dir='$PWD/data/splits' output-dir='$PWD/src/federated' reports-dir='$PWD/reports' $*" \
  --federation-config "num-supernodes=${NUM_NODES:-105} init-args-num-cpus=${INIT_CPUS:-4} client-resources-num-cpus=${CLIENT_CPUS:-2}"
