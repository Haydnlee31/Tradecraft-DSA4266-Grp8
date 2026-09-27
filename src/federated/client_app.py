"""fed_ciciot: Flower ClientApp for CICIoT2023 federated learning simulation."""

import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.eval.metrics import compute_metrics
from src.federated import task
from src.federated.task import load_model, load_train_partition, load_val_partition
from src.federated.task import test as test_fn
from src.federated.task import train as train_fn
from src.utils.seed import client_seed, set_seed

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    task.configure(context.run_config)
    start_time = time.time()

    # Seed from (run seed, partition, round): distinct per client, independent of
    # which worker runs it or in what order
    partition_id = context.node_config["partition-id"]
    server_round = int(msg.content["config"]["server-round"])
    seed = client_seed(int(context.run_config["seed"]), partition_id, server_round)
    set_seed(seed)

    # Load the model and initialize it with the received weights
    model = load_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader = load_train_partition(partition_id, num_partitions, batch_size, seed=seed)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        loss_name=context.run_config["loss"],
        class_weights=context.run_config["class-weights"],
    )

    # Measured (simulation host): wall-clock seconds on the machine running the
    # simulation. Not an edge-device figure -- edge latency/memory/power stay "projected".
    training_time_sim_host_s = time.time() - start_time

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "training_time_sim_host_s": training_time_sim_host_s,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the global model on this client's val slice.

    Reports this client's own macro-F1 (with its partition id) so the strategy can show
    how evenly clients are served. Not a substitute for the server's full-val macro-F1:
    a weighted average of per-client macro-F1 is a different number.
    """

    task.configure(context.run_config)

    # Load the model and initialize it with the received weights
    model = load_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data (val only; train isn't needed here)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    valloader = load_val_partition(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, y_true, y_pred = test_fn(model, valloader, device)
    m = compute_metrics(y_true, y_pred)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": m["accuracy"],
        "client_macro_f1": m["macro_f1"],
        "partition-id": partition_id,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
