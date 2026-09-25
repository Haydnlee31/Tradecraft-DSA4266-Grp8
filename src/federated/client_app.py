"""Flower ``ClientApp`` for the simulated federated-light lane.

The pure :func:`train_partition` and :func:`evaluate_partition` helpers keep
the actual learning independent of Flower messages. The decorated handlers
only unpack a message/context and pack the reply. This makes the computation
easier to test and lets the optional in-process compatibility runner reuse the
same model, loss, scaler, seeds, and client shards.

Every parameter a client needs comes from ``Context.run_config`` or the
server's ``ConfigRecord``. Flower workers are separate processes, so relying on
driver-only module state would make the simulation silently inconsistent.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.eval.metrics import summarize
from src.federated import task
from src.utils.seed import client_seed, set_seed

app = ClientApp()
StateDict = dict[str, torch.Tensor]


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_partition(
    global_state: Mapping[str, torch.Tensor],
    partition_id: int,
    num_partitions: int,
    run_config: Mapping[str, Any],
    server_round: int,
    learning_rate: float,
) -> tuple[StateDict, dict[str, float]]:
    """Train one client from the received global weights.

    The seed depends only on ``(run seed, client, round)``. It therefore stays
    reproducible even when Flower schedules a client on a different worker or
    executes clients in a different order.
    """
    task.configure(run_config)
    started_at = time.perf_counter()
    seed = client_seed(int(run_config["seed"]), partition_id, server_round)
    set_seed(seed)

    model = task.load_model()
    model.load_state_dict(dict(global_state))
    device = _device()
    model.to(device)

    trainloader, _ = task.load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=int(run_config["batch-size"]),
        seed=seed,
    )
    train_loss = task.train(
        model=model,
        trainloader=trainloader,
        epochs=int(run_config["local-epochs"]),
        lr=float(learning_rate),
        device=device,
        loss_name=str(run_config["loss"]),
        class_weights=str(run_config["class-weights"]),
    )

    # Flower serializes these tensors for aggregation. Moving them back to CPU
    # avoids backend/device-specific serialization and frees client GPU memory.
    model.to("cpu")
    metrics = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
        # This is measured simulation wall time, not edge-device latency. The
        # distinction is required by CLAUDE.md and the final report.
        "training_time_seconds": time.perf_counter() - started_at,
    }
    return model.state_dict(), metrics


def evaluate_partition(
    global_state: Mapping[str, torch.Tensor],
    partition_id: int,
    num_partitions: int,
    run_config: Mapping[str, Any],
) -> dict[str, float]:
    """Evaluate the global model on one client's validation slice.

    These site-level numbers describe behavior under each client's traffic
    mix. They are not the cross-lane benchmark; that comes from the server's
    full shared validation split.
    """
    task.configure(run_config)
    model = task.load_model()
    model.load_state_dict(dict(global_state))
    device = _device()
    model.to(device)

    valloader = task.load_client_val(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=int(run_config["batch-size"]),
    )
    eval_loss, y_true, y_pred = task.predict(model, valloader, device)
    metrics = summarize(y_true, y_pred)
    flat = {
        "eval_loss": float(eval_loss),
        "eval_accuracy": metrics["accuracy"],
        "eval_macro_f1": metrics["macro_f1"],
        "num-examples": len(valloader.dataset),
    }
    for class_name in task.CLASSES:
        flat[f"eval_recall/{class_name}"] = metrics[f"recall/{class_name}"]
    return flat


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Unpack one Flower training message and return the local update."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    server_round = int(msg.content["config"]["server-round"])

    state, metrics = train_partition(
        global_state=msg.content["arrays"].to_torch_state_dict(),
        partition_id=partition_id,
        num_partitions=num_partitions,
        run_config=context.run_config,
        server_round=server_round,
        learning_rate=float(msg.content["config"]["lr"]),
    )
    content = RecordDict(
        {
            "arrays": ArrayRecord(state),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Unpack one Flower evaluation message and return scalar metrics."""
    metrics = evaluate_partition(
        global_state=msg.content["arrays"].to_torch_state_dict(),
        partition_id=int(context.node_config["partition-id"]),
        num_partitions=int(context.node_config["num-partitions"]),
        run_config=context.run_config,
    )
    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )
