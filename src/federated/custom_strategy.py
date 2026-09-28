"""Small FedAvg extension for reproducible checkpointing and early stopping.

Flower's built-in ``FedAvg`` already performs the sample-count-weighted model
aggregation required by this project. The only custom behavior here is:

* place the round number in the client config for deterministic client seeds;
* optionally decay the client learning rate (disabled by default);
* select/checkpoint the best global model by validation macro-F1; and
* stop after a configurable number of non-improving rounds.

The previous branch inherited from ``FedAdagrad`` even though the experiment,
documentation, and report all called it FedAvg. That changes the algorithm and
makes the three-lane comparison invalid, so this integrated version inherits
from ``FedAvg`` explicitly.

Joel's optional FedAdagrad now shares this control loop without changing the
FedAvg baseline. Client fairness summaries and history.json are always local;
W&B remains opt-in. Incoming explanatory notes are preserved in the merge report.
"""

from __future__ import annotations

import io
import json
import os
import statistics
import time
from collections.abc import Callable, Iterable
from logging import INFO, WARNING
from pathlib import Path
from typing import Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import logger, log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

FEDERATED_DIR = Path(__file__).resolve().parent
PROJECT_NAME = "fed-ciciot"


class _TradecraftStrategy:
    """FedAvg with validation-based best-model selection and early stopping."""

    def __init__(
        self,
        *args,
        patience: int = 0,
        lr_decay_every: int = 0,
        lr_decay_factor: float = 0.5,
        use_wandb: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.lr_decay_every = lr_decay_every
        self.lr_decay_factor = lr_decay_factor
        self.save_path: Path | None = None
        self.best_f1_so_far = -1.0
        self.best_round: int | None = None
        self.rounds_run = 0
        self.early_stopped = False
        self.use_wandb = use_wandb
        self._wandb = None
        self.run_config = {}
        self._history = {}
        self._client_eval = {}

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure a training round and expose its number to every client."""
        if (
            self.lr_decay_every > 0
            and server_round > 1
            and (server_round - 1) % self.lr_decay_every == 0
        ):
            config["lr"] *= self.lr_decay_factor
            logger.log(INFO, "Client learning rate decreased to %s", config["lr"])

        # Clients derive their per-round seed from this value. Setting it here
        # makes that contract explicit even if Flower's default changes.
        config["server-round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def set_save_path(self, path: Path) -> None:
        """Set the directory used for best-model and run metadata files."""
        path.mkdir(parents=True, exist_ok=True)
        self.save_path = path

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
        self.set_save_path(save_path)
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
        # Rewrite every round, so a crashed run still leaves its history.
        self._write_history()

    def _write_history(self) -> None:
        rounds = []
        for rnd in sorted(self._history):
            entry = dict(self._history[rnd])
            if rnd in self._client_eval:
                entry["client_macro_f1_by_partition"] = self._client_eval[rnd]
            rounds.append(entry)
        (self.save_path / "history.json").write_text(json.dumps(rounds, indent=2))

    def _update_best_f1(
        self,
        current_round: int,
        macro_f1: float,
        arrays: ArrayRecord,
    ) -> bool:
        """Checkpoint ``arrays`` when validation macro-F1 improves."""
        if self.save_path is None:
            raise RuntimeError("set_save_path() must be called before strategy.start()")
        if macro_f1 <= self.best_f1_so_far:
            return False

        self.best_f1_so_far = macro_f1
        self.best_round = current_round
        logger.log(
            INFO,
            "New best global model: round=%d val_macro_f1=%f",
            current_round,
            macro_f1,
        )
        torch.save(arrays.to_torch_state_dict(), self.save_path / "best_model.pt")
        (self.save_path / "best_model.json").write_text(
            json.dumps({"round": current_round, "val_macro_f1": macro_f1}, indent=2),
            encoding="utf-8",
        )
        return True

    def start(self, *args, **kwargs) -> Result:
        """Always close optional logging, including when a worker fails."""
        if self.save_path is None:
            raise RuntimeError("set_save_path() must be called before strategy.start()")
        self._history, self._client_eval = {}, {}
        try:
            self._init_wandb()
            return self._start(*args, **kwargs)
        finally:
            self._write_history()
            if self._wandb is not None:
                self._wandb.finish()
                self._wandb = None

    def _start(
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
        """Run FedAvg while retaining the best validation checkpoint.

        This closely follows Flower's ``Strategy.start`` control flow. The
        aggregation itself still comes from ``FedAvg.aggregate_train``;
        duplicating that arithmetic here would risk silently changing the
        experiment.
        """
        if self.save_path is None:
            raise RuntimeError("set_save_path() must be called before strategy.start()")

        self.best_f1_so_far = -1.0
        self.best_round = None
        self.rounds_run = 0
        self.early_stopped = False
        rounds_without_improvement = 0

        logger.log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        logger.log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()
        arrays = initial_arrays
        result.arrays = arrays
        started_at = time.time()

        # Evaluating and checkpointing round 0 means final test evaluation still
        # has a valid, val-selected model even if the first round gets worse.
        if evaluate_fn:
            initial_metrics = evaluate_fn(0, arrays)
            logger.log(INFO, "Initial global evaluation: %s", initial_metrics)
            if initial_metrics is not None:
                result.evaluate_metrics_serverapp[0] = initial_metrics
                self._log_round(0, "server_eval", initial_metrics)
                self._update_best_f1(0, float(initial_metrics["val_macro_f1"]), arrays)

        for current_round in range(1, num_rounds + 1):
            logger.log(INFO, "")
            logger.log(INFO, "[ROUND %s/%s]", current_round, num_rounds)
            self.rounds_run = current_round

            train_messages = list(self.configure_train(current_round, arrays, train_config, grid))
            train_replies = list(grid.send_and_receive(
                messages=train_messages,
                timeout=timeout,
            ))
            # A failed simulation must not publish an apparently complete
            # benchmark of unchanged initial weights or a partial client set.
            # Network-fault tolerance would be a separate experimental setting.
            if (not train_replies or len(train_replies) != len(train_messages)
                    or any(reply.has_error() for reply in train_replies)):
                raise RuntimeError("Client training failed; refusing to report an incomplete FedAvg round")
            aggregated_arrays, train_metrics = self.aggregate_train(
                current_round, train_replies
            )
            if aggregated_arrays is None or train_metrics is None:
                raise RuntimeError("FedAvg produced no update; refusing to continue")
            if aggregated_arrays is not None:
                arrays = aggregated_arrays
                result.arrays = arrays
            if train_metrics is not None:
                result.train_metrics_clientapp[current_round] = train_metrics
                self._log_round(current_round, "client_train", train_metrics)
                logger.log(INFO, "Aggregated train metrics: %s", train_metrics)

            # Client-side evaluation is optional. The server-side full-val pass
            # below remains the only cross-lane comparable validation result.
            evaluation_messages = []
            if self.fraction_evaluate > 0.0:
                evaluation_messages = list(
                    self.configure_evaluate(
                        current_round, arrays, evaluate_config, grid
                    )
                )
            if evaluation_messages:
                evaluate_replies = grid.send_and_receive(
                    messages=evaluation_messages, timeout=timeout
                )
                client_metrics = self.aggregate_evaluate(
                    current_round, evaluate_replies
                )
                if client_metrics is not None:
                    result.evaluate_metrics_clientapp[current_round] = client_metrics
                    self._log_round(current_round, "client_eval", client_metrics)
                    logger.log(INFO, "Aggregated client evaluation: %s", client_metrics)

            if evaluate_fn:
                server_metrics = evaluate_fn(current_round, arrays)
                logger.log(INFO, "Global validation: %s", server_metrics)
                if server_metrics is not None:
                    result.evaluate_metrics_serverapp[current_round] = server_metrics
                    self._log_round(current_round, "server_eval", server_metrics)
                    improved = self._update_best_f1(
                        current_round,
                        float(server_metrics["val_macro_f1"]),
                        arrays,
                    )
                    rounds_without_improvement = (
                        0 if improved else rounds_without_improvement + 1
                    )
                    if (
                        self.patience > 0
                        and rounds_without_improvement >= self.patience
                    ):
                        logger.log(
                            INFO,
                            "Early stopping at round %s (best val_macro_f1=%f)",
                            current_round,
                            self.best_f1_so_far,
                        )
                        self.early_stopped = True
                        break

        logger.log(INFO, "")
        logger.log(INFO, "Strategy finished in %.2fs", time.time() - started_at)
        logger.log(INFO, "Final results:")
        for line in io.StringIO(str(result)):
            logger.log(INFO, "\t%s", line.rstrip("\n"))
        return result


# Temporary import compatibility for teammates who used the old class name.
# It now has FedAvg semantics; new code should import ``CustomFedAvg``.
# Historical note above applied to the previous integration. As of Joel's merge,
# CustomFedAdagrad is a real adaptive strategy, never a misleading FedAvg alias.
class CustomFedAvg(_TradecraftStrategy, FedAvg):
    """Plain FedAvg baseline, with shared checkpointing and local history."""


class CustomFedAdagrad(_TradecraftStrategy, FedAdagrad):
    """FedAdagrad (adaptive server optimizer over the averaged client update)."""


STRATEGIES = {"fedavg": CustomFedAvg, "fedadagrad": CustomFedAdagrad}
