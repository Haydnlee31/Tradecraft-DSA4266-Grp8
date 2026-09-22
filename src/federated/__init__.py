"""Federated-light lane: Flower-simulated FedAvg over non-IID virtual clients.

Modules:
    partition     load the sampled splits, standardize, carve non-IID client shards
    client_app    Flower ClientApp — local training/evaluation on one shard
    server_app    Flower ServerApp — FedAvg aggregation + centralized val scoring
    run_simulation  CLI entrypoint that wires the two together

Intentionally free of import-time side effects: the simulation backend
re-imports these modules inside Ray worker processes, so anything expensive or
stateful must happen inside a function, not at module scope.
"""
