import io
import json
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

PROJECT_NAME = "fed-ciciot"
FEDERATED_DIR = Path(__file__).resolve().parent


class CustomFedAdagrad(FedAdagrad):

    def __init__(self, *args, patience: int = 0, **kwargs):
        """`patience`: stop after this many rounds without a val macro-F1 improvement,
        mirroring Trainer.fit. 0 disables early stopping."""
        super().__init__(*args, **kwargs)
        self.patience = patience

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            print("LR decreased to:", config["lr"])
        # Clients derive their per-round seed from this; set it ourselves rather than
        # rely on the parent strategy adding it
        config["server-round"] = server_round
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)

    def set_save_path(self, path: Path):
        """Set the path where wandb logs and model checkpoints will be saved."""
        self.save_path = path

    def _update_best_f1(
        self, current_round: int, macro_f1: float, arrays: ArrayRecord
    ) -> bool:
        """Save a checkpoint if this round has the best val macro-F1 so far.
        Returns whether it improved.

        Same selection rule as the centralized Trainer.fit, so the lanes compare
        like for like. Overwrites best_model.pt / best_model.json each time.
        """
        if macro_f1 > self.best_f1_so_far:
            self.best_f1_so_far = macro_f1
            logger.log(INFO, "💡 New best global model found: val_macro_f1=%f", macro_f1)
            torch.save(arrays.to_torch_state_dict(), self.save_path / "best_model.pt")
            (self.save_path / "best_model.json").write_text(
                json.dumps({"round": current_round, "val_macro_f1": macro_f1}, indent=2)
            )
            logger.log(INFO, "💾 Best model saved to disk (round %d)", current_round)
            return True
        return False

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        """Execute the federated learning strategy logging results to W&B and saving
        them to disk."""

        # Init W&B
        name = f"{str(self.save_path.parent.name)}/{str(self.save_path.name)}-ServerApp"
        wandb.init(project=PROJECT_NAME, name=name, dir=str(self.save_path.parents[2]))

        # Keep track of best val macro-F1
        self.best_f1_so_far = -1.0
        rounds_without_improvement = 0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                # Log to W&B
                wandb.log(dict(agg_train_metrics), step=current_round)

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                # Log to W&B
                wandb.log(dict(agg_evaluate_metrics), step=current_round)
            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    # Maybe save to disk if new best is found
                    improved = self._update_best_f1(current_round, res["val_macro_f1"], arrays)
                    # Log to W&B
                    wandb.log(dict(res), step=current_round)
                    rounds_without_improvement = 0 if improved else rounds_without_improvement + 1
                    if self.patience and rounds_without_improvement >= self.patience:
                        log(
                            INFO,
                            "Early stopping at round %s (best val_macro_f1=%f)",
                            current_round,
                            self.best_f1_so_far,
                        )
                        break

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result