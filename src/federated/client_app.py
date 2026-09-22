"""Flower ClientApp — one simulated IoT site training on its own shard.

Port of the notebook's `train_one_client`: same idea (copy the global weights,
run local SGD epochs, hand the weights back with a row count so the server can
weight the average), expressed against Flower's message API instead of a
hand-rolled loop.

The actual work lives in `local_train` / `local_evaluate`, which take plain
tensors and a config dict. The `@app.train()` / `@app.evaluate()` handlers are
thin adapters that unwrap a Flower Message and call them. That split exists so
the same code path serves both execution backends — Flower's Ray simulation
runtime and the in-process driver in local_backend.py (needed because Ray's
native helper binaries cannot launch under Windows Smart App Control).

Every parameter a client needs arrives inside the message's ConfigRecord rather
than from module state or environment: the Ray backend runs clients in separate
worker processes, so anything set on the driver after import is invisible here.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.eval.metrics import summarize
from src.federated.partition import SplitData, client_shard
from src.models.mlp import make_light_mlp

app = ClientApp()

StateDict = dict[str, torch.Tensor]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(input_features: int, config: Mapping[str, Any]) -> nn.Module:
    hidden = tuple(int(width) for width in config["hidden-dims"])
    return make_light_mlp(input_features=input_features, hidden_dims=hidden)


def shard_for(partition_id: int, num_partitions: int, config: Mapping[str, Any]) -> SplitData:
    """Resolve one client's shard from its partition index and the round config."""
    return client_shard(
        partition_id=partition_id,
        num_partitions=num_partitions,
        max_rows=int(config["max-rows"]) or None,
        alpha=float(config["alpha"]),
        seed=int(config["seed"]),
        splits_dir_str=str(config["splits-dir"]),
    )


def local_train(
    global_state: StateDict,
    shard: SplitData,
    config: Mapping[str, Any],
    partition_id: int,
) -> tuple[StateDict, dict[str, float]]:
    """Run local epochs from the received global weights; return the update.

    Backend-agnostic: no Message, no Context, so the in-process driver and the
    Flower runtime execute byte-identical training.
    """
    device = _device()
    model = build_model(shard.x.shape[1], config)
    model.load_state_dict(global_state)
    model.to(device)
    model.train()

    # Seeded per (round, client) so shuffling is reproducible without every
    # client drawing the identical batch order.
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + 1000 * int(config["server-round"]) + partition_id
    )
    loader = DataLoader(
        TensorDataset(shard.x, shard.y),
        batch_size=int(config["batch-size"]),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(config["lr"]), momentum=float(config["momentum"])
    )
    loss_fn = nn.CrossEntropyLoss()

    total_loss, num_batches = 0.0, 0
    for _ in range(int(config["local-epochs"])):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1

    model.to("cpu")
    metrics = {
        # FedAvg's weighting key — this is what makes aggregation weighted by
        # shard size, matching the notebook's sum(w_i * n_i) / sum(n_i).
        "num-examples": len(shard),
        "train_loss": total_loss / max(num_batches, 1),
    }
    return model.state_dict(), metrics


def local_evaluate(
    global_state: StateDict, shard: SplitData, config: Mapping[str, Any]
) -> dict[str, float]:
    """Score the incoming global model on one client's own shard.

    This is *local* evaluation on a non-IID shard: it says how the global model
    performs on that site's traffic mix. The headline project numbers come from
    the server's centralized pass over the held-out val split (server_app.py).
    """
    device = _device()
    model = build_model(shard.x.shape[1], config)
    model.load_state_dict(global_state)
    model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    predictions: list[torch.Tensor] = []
    total_loss, num_batches = 0.0, 0
    loader = DataLoader(TensorDataset(shard.x, shard.y), batch_size=4096, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            total_loss += float(loss_fn(logits, batch_y.to(device)).item())
            num_batches += 1
            predictions.append(logits.argmax(dim=1).cpu())

    y_pred = torch.cat(predictions).numpy()
    metrics = summarize(shard.y.numpy(), y_pred)
    metrics["num-examples"] = len(shard)
    metrics["eval_loss"] = total_loss / max(num_batches, 1)
    return metrics


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Flower adapter: unwrap the Message, delegate to local_train."""
    config = msg.content["config"]
    partition_id = int(context.node_config["partition-id"])
    shard = shard_for(partition_id, int(context.node_config["num-partitions"]), config)
    state, metrics = local_train(
        msg.content["arrays"].to_torch_state_dict(), shard, config, partition_id
    )
    content = RecordDict(
        {"arrays": ArrayRecord(state), "metrics": MetricRecord(metrics)}
    )
    return Message(content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Flower adapter: unwrap the Message, delegate to local_evaluate."""
    config = msg.content["config"]
    partition_id = int(context.node_config["partition-id"])
    shard = shard_for(partition_id, int(context.node_config["num-partitions"]), config)
    metrics = local_evaluate(msg.content["arrays"].to_torch_state_dict(), shard, config)
    return Message(RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
