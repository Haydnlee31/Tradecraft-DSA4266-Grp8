"""Federated-light lane: Flower-simulated FedAvg over virtual clients.

Modules:
    partition       persist reproducible IID/non-IID client assignments
    task            shared model, preprocessing, loss, and train/evaluate code
    client_app      Flower ``ClientApp`` adapters around the shared task code
    server_app      Flower ``ServerApp``, validation, checkpoints, and reports
    custom_strategy FedAvg with validation-based checkpointing/early stopping

The package is intentionally free of import-time side effects. Flower can
re-import these modules inside worker processes, so expensive or stateful work
belongs inside functions rather than at package import time.
"""
