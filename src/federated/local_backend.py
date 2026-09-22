"""In-process FedAvg driver — the federated lane without Ray.

Why this exists: Flower's simulation runtime spawns Ray, and Ray launches
native helper binaries (raylet.exe, gcs_server.exe) as subprocesses. Under
Windows Smart App Control those launches fail with
`OSError: [WinError 4551] An Application Control policy has blocked this file`,
so the Ray backend cannot start on such a machine. Disabling Smart App Control
is irreversible without reinstalling Windows, which is not a reasonable price
for running a coursework simulation.

This driver runs the identical client code (`local_train` / `local_evaluate`
from client_app) sequentially in one process, and aggregates with Flower's own
`aggregate_arrayrecords`, so the FedAvg arithmetic is Flower's rather than a
second implementation that could silently drift from it.

What it does NOT reproduce: process isolation, concurrent client execution,
message serialization, and client sampling via a Grid. Those matter for
throughput and for realism about network cost — not for the learned parameters.
For the report: results from this backend are the same computation as the Ray
backend; timing figures from it are not comparable to a distributed setting,
and neither backend produces *measured* edge numbers (see CLAUDE.md).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict

# Flower's internal aggregation helper. Imported deliberately rather than
# reimplemented: if a future Flower release moves it, this fails loudly at
# import instead of quietly averaging differently from the Ray backend.
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
)

from src.eval.metrics import format_summary_line
from src.federated.client_app import local_evaluate, local_train, shard_for

StateDict = dict[str, torch.Tensor]


def run_local_federation(
    initial_state: StateDict,
    num_clients: int,
    num_rounds: int,
    config: Mapping[str, Any],
    evaluate_fn: Callable[[int, StateDict], dict[str, float] | None] | None = None,
    fraction_evaluate: float = 1.0,
) -> tuple[StateDict, list[dict[str, float]]]:
    """Run `num_rounds` of FedAvg across `num_clients` sequential virtual clients.

    Returns the final global state dict and the per-round centralized metrics.
    """
    global_state = {key: value.clone() for key, value in initial_state.items()}
    history: list[dict[str, float]] = []

    # Shards are built once and reused across rounds — client_shard is cached,
    # but resolving them up front also surfaces a bad partition immediately.
    shards = [
        shard_for(client_id, num_clients, config) for client_id in range(num_clients)
    ]

    for server_round in range(1, num_rounds + 1):
        round_config = dict(config)
        round_config["server-round"] = server_round

        replies: list[RecordDict] = []
        for client_id, shard in enumerate(shards):
            state, metrics = local_train(global_state, shard, round_config, client_id)
            replies.append(
                RecordDict(
                    {
                        "arrays": ArrayRecord(state),
                        "metrics": MetricRecord(metrics),
                    }
                )
            )

        aggregated = aggregate_arrayrecords(replies, "num-examples")
        train_metrics = aggregate_metricrecords(replies, "num-examples")
        global_state = aggregated.to_torch_state_dict()

        print(
            f"[local] round {server_round:>3}  "
            f"train_loss={float(train_metrics['train_loss']):.4f}  "
            f"clients={num_clients}"
        )

        if fraction_evaluate > 0.0:
            eval_replies = [
                RecordDict(
                    {"metrics": MetricRecord(local_evaluate(global_state, shard, round_config))}
                )
                for shard in shards
            ]
            client_metrics = aggregate_metricrecords(eval_replies, "num-examples")
            print(
                f"[local] round {server_round:>3}  federated (per-shard) "
                f"{format_summary_line({k: float(v) for k, v in client_metrics.items()})}"
            )

        if evaluate_fn is not None:
            centralized = evaluate_fn(server_round, global_state)
            if centralized is not None:
                history.append({"round": server_round, **centralized})

    return global_state, history
