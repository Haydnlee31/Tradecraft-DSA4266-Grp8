"""fed_ciciot: Flower ClientApp for CICIoT2023 federated learning simulation."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.federated import task
from src.federated.task import load_model, load_data
from src.federated.task import test as test_fn
from src.federated.task import train as train_fn
from src.utils.seed import client_seed, set_seed

import time

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
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, seed=seed)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    end_time = time.time()
    training_time = end_time - start_time
    
    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "training_time": training_time,  # New metric
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    task.configure(context.run_config)

    # Load the model and initialize it with the received weights
    model = load_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc, _ = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
