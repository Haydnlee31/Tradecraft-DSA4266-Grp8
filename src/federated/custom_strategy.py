"""Server-side strategies for the federated-light lane.

`CustomFedAvg` (plain FedAvg baseline) and `CustomFedAdagrad` (adaptive server
optimizer) share one training loop via `_TradecraftStrategy`:

- model selection and early stopping on the server's full-val macro-F1, the same rule
  as the centralized Trainer.fit; the best weights go to best_model.pt/.json;
- a halving of the client learning rate every 5 rounds;
- per-round metrics always written to history.json, and to W&B only if enabled;
- client-side eval summarized as the spread (min/median/max) of per-client macro-F1,
  never as a weighted average -- that average is not the global macro-F1.
"""

import io
import json
import os
import statistics
import time
from logging import INFO, WARNING
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

PROJECT_NAME = "fed-ciciot"


class _TradecraftStrategy:
    """Mixin holding the shared loop; combine with a Flower strategy class."""

    def __init__(self, *args, patience: int = 0, use_wandb: bool = False, **kwargs):
        """`patience`: stop after this many rounds without a val macro-F1 improvement,
        mirroring Trainer.fit; 0 disables early stopping. `use_wandb`: also log to W&B."""
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.use_wandb = use_wandb
        self._wandb = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        if server_round % 5 == 0:
            config["lr"] *= 0.5
            log(INFO, "LR decreased to: %s", config["lr"])
        # Clients derive their per-round seed from this; set it ourselves rather than
        # rely on the parent strategy adding it
        config["server-round"] = server_round
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> Optional[MetricRecord]:
        """Summarize client-side eval as the spread of per-client macro-F1.

        Keeps each client's value (by partition id) for history.json; the returned
        record has weighted eval loss/accuracy plus min/median/max client macro-F1.
        """
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        if not valid_replies:
            return None
        per_client = []
        for msg in valid_replies:
            m = next(iter(msg.content.metric_records.values()))
            per_client.append(m)
        n_total = sum(m["num-examples"] for m in per_client)
        f1s = [m["client_macro_f1"] for m in per_client]
        self._client_eval[server_round] = {
            int(m["partition-id"]): m["client_macro_f1"] for m in per_client
        }
        return MetricRecord(
            {
                "client_eval_loss": sum(m["eval_loss"] * m["num-examples"] for m in per_client) / n_total,
                "client_eval_acc": sum(m["eval_acc"] * m["num-examples"] for m in per_client) / n_total,
                "client_macro_f1_min": min(f1s),
                "client_macro_f1_median": statistics.median(f1s),
                "client_macro_f1_max": max(f1s),
            }
        )

    def set_output(self, save_path: Path, run_config: dict) -> None:
        """Where checkpoints and history.json go, and the config recorded with W&B."""
        self.save_path = save_path
        self.run_config = run_config

    def _init_wandb(self) -> None:
        if not self.use_wandb:
            return
        import wandb  # Optional dependency: only needed when use-wandb = true

        if not os.environ.get("WANDB_API_KEY"):
            try:
                logged_in = bool(wandb.api.api_key)
            except Exception:
                logged_in = False
            if not logged_in:
                os.environ["WANDB_MODE"] = "offline"
                log(WARNING, "Not logged in to W&B; logging offline (sync later with `wandb sync`)")
        name = f"{self.save_path.parent.name}/{self.save_path.name}-ServerApp"
        wandb.init(
            project=PROJECT_NAME, name=name, dir=str(self.save_path.parents[1]), config=self.run_config
        )
        self._wandb = wandb

    def _log_round(self, current_round: int, key: str, metrics: Optional[MetricRecord]) -> None:
        """Record a round's metrics in history (always) and W&B (if enabled)."""
        if metrics is None:
            return
        self._history.setdefault(current_round, {"round": current_round})[key] = dict(metrics)
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=current_round)

    def _write_history(self) -> None:
        rounds = []
        for rnd in sorted(self._history):
            entry = dict(self._history[rnd])
            if rnd in self._client_eval:
                entry["client_macro_f1_by_partition"] = self._client_eval[rnd]
            rounds.append(entry)
        (self.save_path / "history.json").write_text(json.dumps(rounds, indent=2))

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
            self.best_round = current_round
            log(INFO, "💡 New best global model found: val_macro_f1=%f", macro_f1)
            torch.save(arrays.to_torch_state_dict(), self.save_path / "best_model.pt")
            (self.save_path / "best_model.json").write_text(
                json.dumps({"round": current_round, "val_macro_f1": macro_f1}, indent=2)
            )
            log(INFO, "💾 Best model saved to disk (round %d)", current_round)
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
        """Execute the federated learning strategy, saving results to disk (and W&B
        if enabled)."""

        self._init_wandb()
        self._history: dict[int, dict] = {}
        self._client_eval: dict[int, dict[int, float]] = {}

        # Keep track of best val macro-F1
        self.best_f1_so_far = -1.0
        rounds_without_improvement = 0
        self.best_round = None
        self.rounds_run = 0
        self.early_stopped = False

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
                self._log_round(0, "server_eval", res)

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)
            self.rounds_run = current_round

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
                self._log_round(current_round, "client_train", agg_train_metrics)

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

            # Log evaluation metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                self._log_round(current_round, "client_eval", agg_evaluate_metrics)

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            stop = False
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._log_round(current_round, "server_eval", res)
                    # Maybe save to disk if new best is found
                    improved = self._update_best_f1(current_round, res["val_macro_f1"], arrays)
                    rounds_without_improvement = 0 if improved else rounds_without_improvement + 1
                    if self.patience and rounds_without_improvement >= self.patience:
                        log(
                            INFO,
                            "Early stopping at round %s (best val_macro_f1=%f)",
                            current_round,
                            self.best_f1_so_far,
                        )
                        self.early_stopped = True
                        stop = True

            # Rewrite every round, so a crashed run still leaves its history
            self._write_history()
            if stop:
                break

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        if self._wandb is not None:
            self._wandb.finish()
        return result


class CustomFedAvg(_TradecraftStrategy, FedAvg):
    """Plain FedAvg baseline."""


class CustomFedAdagrad(_TradecraftStrategy, FedAdagrad):
    """FedAdagrad (adaptive server optimizer over the averaged client update)."""


STRATEGIES = {"fedavg": CustomFedAvg, "fedadagrad": CustomFedAdagrad}
