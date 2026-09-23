"""fed_ciciot: Flower ServerApp for CICIoT2023 federated learning simulation."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdagrad
from src.federated.custom_strategy import FEDERATED_DIR, CustomFedAdagrad

from src.federated import task
from src.federated.task import load_model, load_centralized_dataset, test
from src.utils.seed import set_seed

from datetime import datetime

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    task.configure(context.run_config)

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    seed: int = int(context.run_config["seed"])

    # Seed before building the model: fixes the initial weights and Flower's
    # per-round client sampling
    set_seed(seed)

    # Load global model
    global_model = load_model()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg/FedAdagrad/CustomFedAdagrad strategy
    strategy = CustomFedAdagrad(fraction_evaluate=fraction_evaluate)
    
    # Get the current date and time
    current_time = datetime.now()
    # Seed in the folder name, so runs with different seeds started in the same
    # second don't collide
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S") + f"_seed{seed}"
    # Save path is based on the current directory
    out_base = Path(context.run_config.get("output-dir") or FEDERATED_DIR)
    save_path = out_base / "outputs" / run_dir
    save_path.mkdir(parents=True, exist_ok=False)

    # Log what this run used, per the repo's reproducibility convention
    run_info = {
        "seed": seed,
        "run_config": dict(context.run_config),
        # No partition file yet: clients take every num_partitions-th row of these
        # splits (task._load_slice). Record the Dirichlet partition file here once
        # Step 4 lands.
        "partition": {
            "scheme": "row-modulo",
            "train_file": task.TRAIN_PATH.name,
            "val_file": task.VAL_PATH.name,
            "splits_dir": str(task.SPLITS_DIR),
        },
    }
    (save_path / "run_info.json").write_text(json.dumps(run_info, indent=2))

    # Set the path where results and model checkpoints will be saved
    strategy.set_save_path(save_path)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, save_path / "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = load_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
