"""Sequential in-process FedAvg compatibility runner.

Flower's normal Simulation Runtime uses Ray workers. This small driver is
useful for teaching/smoke tests on machines where Ray cannot launch (notably
some Windows Smart App Control setups). It calls the exact same pure client
helpers as ``client_app.py`` and Flower's own aggregation helpers, so it does
not maintain a second model, loss, scaler, or FedAvg implementation.

What it does *not* reproduce is process isolation, concurrent clients, message
serialization, or Grid-based sampling. Those affect throughput and systems
realism. Wall-clock timings here are simulation timings and must never be
reported as measured edge-hardware latency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
)

from src.federated.client_app import evaluate_partition, train_partition

StateDict = dict[str, torch.Tensor]


def run_local_federation(
    initial_state: StateDict,
    num_clients: int,
    num_rounds: int,
    run_config: Mapping[str, Any],
    evaluate_fn: Callable[[int, StateDict], dict[str, float] | None] | None = None,
    fraction_evaluate: float = 0.0,
) -> tuple[StateDict, list[dict[str, float]]]:
    """Run deterministic full-participation FedAvg in one process."""
    if num_clients < 1 or num_rounds < 1:
        raise ValueError("num_clients and num_rounds must both be >= 1")
    global_state = {
        key: value.detach().cpu().clone() for key, value in initial_state.items()
    }
    history: list[dict[str, float]] = []

    for server_round in range(1, num_rounds + 1):
        replies: list[RecordDict] = []
        for partition_id in range(num_clients):
            state, metrics = train_partition(
                global_state=global_state,
                partition_id=partition_id,
                num_partitions=num_clients,
                run_config=run_config,
                server_round=server_round,
                learning_rate=float(run_config["learning-rate"]),
            )
            replies.append(
                RecordDict(
                    {
                        "arrays": ArrayRecord(state),
                        "metrics": MetricRecord(metrics),
                    }
                )
            )

        global_state = aggregate_arrayrecords(
            replies, "num-examples"
        ).to_torch_state_dict()
        train_metrics = aggregate_metricrecords(replies, "num-examples")
        print(
            f"[local] round {server_round:>3}  "
            f"train_loss={float(train_metrics['train_loss']):.4f}  "
            f"clients={num_clients}"
        )

        if fraction_evaluate > 0.0:
            eval_replies = []
            for partition_id in range(num_clients):
                metrics = evaluate_partition(
                    global_state=global_state,
                    partition_id=partition_id,
                    num_partitions=num_clients,
                    run_config=run_config,
                )
                eval_replies.append(RecordDict({"metrics": MetricRecord(metrics)}))
            client_metrics = aggregate_metricrecords(eval_replies, "num-examples")
            print(
                f"[local] round {server_round:>3}  client-val "
                f"macro_f1={float(client_metrics['eval_macro_f1']):.4f}  "
                f"acc={float(client_metrics['eval_accuracy']):.4f}"
            )

        if evaluate_fn is not None:
            centralized = evaluate_fn(server_round, global_state)
            if centralized is not None:
                history.append({"round": server_round, **centralized})

    return global_state, history
