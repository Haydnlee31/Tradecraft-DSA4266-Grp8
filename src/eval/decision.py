"""Build an auditable three-lane deployment decision from JSON reports.

The decision layer does not invent a single opaque score. It first selects one
configuration per lane using validation macro-F1, then checks explicit gates on
test macro-F1, every class's recall, model size, report completeness, and seed
count. Among eligible models within an allowed macro-F1 drop from the best, it
prefers the smaller model; a caller can use ``--prefer-federated`` to break an
equal-size tie in favor of the federated training setting.

Test metrics are used only for the final cross-lane comparison. They are never
used to select a loss/configuration within a lane, which would leak test
performance into model selection. The default also requires the same
``sqrt_weighted_ce`` loss across lanes so an old incompatible experiment is not
silently mixed into the table.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.label_map import CLASSES

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
EXPECTED_LANES = ("centralized-heavy", "centralized-light", "federated-light")


@dataclass(frozen=True)
class Criteria:
    """Explicit gates used by the deployment decision."""

    loss: str | None = "sqrt_weighted_ce"
    min_macro_f1: float = 0.60
    min_class_recall: float = 0.20
    max_model_mib: float = 1.0
    max_macro_f1_drop: float = 0.02
    min_seeds: int = 3
    require_federated: bool = False
    prefer_federated: bool = False


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _lane_name(report: dict[str, Any]) -> str:
    variant = str(report.get("variant", "")).lower()
    lane = str(report.get("lane", "centralized")).lower()
    if lane == "federated":
        if variant != "light":
            raise ValueError("the project only defines a federated-light lane")
        return "federated-light"
    if variant in {"heavy", "light"}:
        return f"centralized-{variant}"
    raise ValueError(f"cannot infer lane from lane={lane!r}, variant={variant!r}")


def _seed(report: dict[str, Any]) -> int:
    args = report.get("args", {})
    if "seed" not in args:
        raise ValueError("report args do not contain a seed")
    return int(args["seed"])


def _best_validation_macro_f1(report: dict[str, Any]) -> float:
    values = [
        float(entry["val_macro_f1"])
        for entry in report.get("history", [])
        if entry.get("val_macro_f1") is not None
    ]
    if not values:
        raise ValueError("report history contains no validation macro-F1")
    return max(values)


def _validate_report(report: dict[str, Any], source: Path) -> None:
    required = ("variant", "loss", "num_parameters", "history", "test_metrics", "args")
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"{source.name} is missing keys: {missing}")
    test_metrics = report["test_metrics"]
    for key in ("accuracy", "macro_f1", "per_class_recall"):
        if key not in test_metrics:
            raise ValueError(f"{source.name} test_metrics is missing {key!r}")
    missing_classes = set(CLASSES) - set(test_metrics["per_class_recall"])
    if missing_classes:
        raise ValueError(f"{source.name} has no recall for: {sorted(missing_classes)}")
    if int(report["num_parameters"]) <= 0:
        raise ValueError(f"{source.name} has a non-positive parameter count")
    _lane_name(report)
    _seed(report)
    _best_validation_macro_f1(report)


def load_reports(reports_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load training reports and return non-fatal file warnings separately."""
    reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in sorted(Path(reports_dir).glob("*.json")):
        if path.name.startswith("decision_summary"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Non-training JSON files can coexist in reports/. Ignore them with
            # a visible warning instead of mistaking them for a failed lane.
            if "test_metrics" not in payload:
                warnings.append(f"ignored non-training JSON: {path.name}")
                continue
            _validate_report(payload, path)
            payload = dict(payload)
            payload["_source"] = str(path)
            reports.append(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            warnings.append(f"ignored invalid report {path.name}: {error}")
    return reports, warnings


def _configuration_key(report: dict[str, Any]) -> str:
    """Group repeated seeds without folding different FL experiments together."""
    lane = _lane_name(report)
    key: dict[str, Any] = {
        "lane": lane,
        "loss": report["loss"],
        "config": report.get("config", {}),
    }
    if lane == "federated-light":
        key.update(
            {
                "partitioner": report.get("partitioner"),
                "alpha": report.get("alpha"),
                "num_clients": report.get("num_clients"),
                "local_epochs": report.get("local_epochs"),
                "fraction_train": report.get("fraction_train"),
                "class_weights": report.get("class_weights"),
                "strategy": report.get("strategy"),
            }
        )
    else:
        args = report.get("args", {})
        # These affect optimization/model selection and therefore define a
        # configuration. Paths and seed intentionally do not.
        key["training"] = {
            name: args.get(name)
            for name in (
                "epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "l1_lambda",
                "patience",
            )
            if name in args
        }
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def aggregate_candidates(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated seeds for each comparable configuration."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[_configuration_key(report)].append(report)

    candidates: list[dict[str, Any]] = []
    for key, group in grouped.items():
        config_key = json.loads(key)
        seeds = [_seed(report) for report in group]
        duplicate_seeds = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
        parameter_counts = {int(report["num_parameters"]) for report in group}
        if len(parameter_counts) != 1:
            raise ValueError(
                f"configuration {config_key} has inconsistent parameter counts: "
                f"{sorted(parameter_counts)}"
            )
        parameter_count = parameter_counts.pop()
        parameter_bytes = {
            int(report.get("parameter_bytes", parameter_count * 4)) for report in group
        }
        if len(parameter_bytes) != 1:
            raise ValueError(
                f"configuration {config_key} has inconsistent parameter byte counts"
            )

        validation = [_best_validation_macro_f1(report) for report in group]
        test_macro = [float(report["test_metrics"]["macro_f1"]) for report in group]
        test_accuracy = [float(report["test_metrics"]["accuracy"]) for report in group]
        recalls = {
            class_name: [
                float(report["test_metrics"]["per_class_recall"][class_name])
                for report in group
            ]
            for class_name in CLASSES
        }
        exact_parameter_bytes = parameter_bytes.pop()
        candidates.append(
            {
                **config_key,
                "seeds": sorted(set(seeds)),
                "num_seeds": len(set(seeds)),
                "duplicate_seeds": duplicate_seeds,
                "sources": [report["_source"] for report in group],
                "num_parameters": parameter_count,
                "parameter_bytes": exact_parameter_bytes,
                "model_mib": exact_parameter_bytes / (1024**2),
                "validation_macro_f1_mean": _mean(validation),
                "validation_macro_f1_std": _sample_std(validation),
                "test_macro_f1_mean": _mean(test_macro),
                "test_macro_f1_std": _sample_std(test_macro),
                "test_accuracy_mean": _mean(test_accuracy),
                "test_accuracy_std": _sample_std(test_accuracy),
                "per_class_recall_mean": {
                    class_name: _mean(values) for class_name, values in recalls.items()
                },
                "per_class_recall_std": {
                    class_name: _sample_std(values)
                    for class_name, values in recalls.items()
                },
                "worst_class_recall_mean": min(
                    _mean(values) for values in recalls.values()
                ),
            }
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["lane"],
            -candidate["validation_macro_f1_mean"],
        ),
    )


def _select_by_validation(
    candidates: list[dict[str, Any]], loss: str | None
) -> dict[str, dict[str, Any]]:
    """Select the best validation configuration inside each lane."""
    selected: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if loss is not None and candidate["loss"] != loss:
            continue
        lane = candidate["lane"]
        current = selected.get(lane)
        if (
            current is None
            or candidate["validation_macro_f1_mean"]
            > current["validation_macro_f1_mean"]
        ):
            selected[lane] = candidate
    return selected


def build_decision(reports: list[dict[str, Any]], criteria: Criteria) -> dict[str, Any]:
    """Return a JSON-serializable decision with blockers and reasoning."""
    candidates = aggregate_candidates(reports)
    selected = _select_by_validation(candidates, criteria.loss)
    blockers: list[str] = []
    warnings: list[str] = []
    available_losses: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        available_losses[candidate["lane"]].add(candidate["loss"])

    for lane in EXPECTED_LANES:
        if lane not in selected:
            loss_note = f" with loss={criteria.loss}" if criteria.loss else ""
            blockers.append(f"missing {lane} report{loss_note}")
            if available_losses.get(lane):
                warnings.append(
                    f"{lane} has only these excluded losses: "
                    f"{sorted(available_losses[lane])}"
                )
            continue
        candidate = selected[lane]
        if candidate["num_seeds"] < criteria.min_seeds:
            blockers.append(
                f"{lane} has {candidate['num_seeds']} seed(s); "
                f"need {criteria.min_seeds}"
            )
        if candidate["duplicate_seeds"]:
            blockers.append(
                f"{lane} has duplicate reports for seeds {candidate['duplicate_seeds']}"
            )

    evaluated: list[dict[str, Any]] = []
    for lane, candidate in selected.items():
        failures = []
        if candidate["test_macro_f1_mean"] < criteria.min_macro_f1:
            failures.append(
                f"macro-F1 {candidate['test_macro_f1_mean']:.4f} "
                f"< {criteria.min_macro_f1:.4f}"
            )
        weak_classes = [
            class_name
            for class_name, recall in candidate["per_class_recall_mean"].items()
            if recall < criteria.min_class_recall
        ]
        if weak_classes:
            failures.append("recall below threshold for " + ", ".join(weak_classes))
        if candidate["model_mib"] > criteria.max_model_mib:
            failures.append(
                f"model {candidate['model_mib']:.4f} MiB "
                f"> {criteria.max_model_mib:.4f} MiB"
            )
        evaluated.append(
            {**candidate, "eligible": not failures, "constraint_failures": failures}
        )

    eligible = [candidate for candidate in evaluated if candidate["eligible"]]
    if criteria.require_federated:
        eligible = [
            candidate
            for candidate in eligible
            if candidate["lane"] == "federated-light"
        ]
        if not eligible:
            blockers.append(
                "federated training is required but no federated candidate passes"
            )

    recommendation = None
    if eligible:
        best_macro = max(candidate["test_macro_f1_mean"] for candidate in eligible)
        near_best = [
            candidate
            for candidate in eligible
            if best_macro - candidate["test_macro_f1_mean"]
            <= criteria.max_macro_f1_drop
        ]
        near_best.sort(
            key=lambda candidate: (
                candidate["parameter_bytes"],
                0
                if criteria.prefer_federated and candidate["lane"] == "federated-light"
                else 1,
                -candidate["test_macro_f1_mean"],
            )
        )
        recommendation = near_best[0]["lane"]
    else:
        blockers.append("no candidate passes all metric and size constraints")

    # A recommendation can still be useful as a provisional debugging signal,
    # but it is not final while completeness/reliability blockers remain.
    status = "ready" if recommendation is not None and not blockers else "incomplete"
    if criteria.loss is None:
        chosen_losses = {candidate["loss"] for candidate in selected.values()}
        if len(chosen_losses) > 1:
            warnings.append(
                "selected lanes use different losses; prefer --loss for a controlled comparison"
            )

    return {
        "status": status,
        "recommended_lane": recommendation if status == "ready" else None,
        "provisional_recommendation": recommendation,
        "criteria": {
            "loss": criteria.loss,
            "min_macro_f1": criteria.min_macro_f1,
            "min_class_recall": criteria.min_class_recall,
            "max_model_mib": criteria.max_model_mib,
            "max_macro_f1_drop": criteria.max_macro_f1_drop,
            "min_seeds": criteria.min_seeds,
            "require_federated": criteria.require_federated,
            "prefer_federated": criteria.prefer_federated,
        },
        "selection_metric": "mean best validation macro-F1 across seeds",
        "comparison_metrics": "test metrics of validation-selected checkpoints",
        "blocking_issues": blockers,
        "warnings": warnings,
        "selected_candidates": {
            candidate["lane"]: candidate for candidate in evaluated
        },
        "all_candidates": candidates,
    }


def render_markdown(decision: dict[str, Any]) -> str:
    """Render the decision in a beginner-readable audit table."""
    lines = [
        "# Tradecraft decision summary",
        "",
        f"Status: **{decision['status']}**",
        "",
    ]
    if decision["recommended_lane"]:
        lines.append(f"Recommendation: **{decision['recommended_lane']}**")
    elif decision["provisional_recommendation"]:
        lines.append(
            f"Provisional only: **{decision['provisional_recommendation']}**; "
            "resolve the blockers below before reporting it as the project decision."
        )
    else:
        lines.append("Recommendation: none yet.")

    lines.extend(
        [
            "",
            "Configurations are selected by validation macro-F1. Test metrics are used "
            "only for the final comparison.",
            "",
            "Current gates: loss={loss}, macro-F1 ≥ {macro:.3f}, every class recall "
            "≥ {recall:.3f}, model ≤ {size:.3f} MiB, at least {seeds} seeds, and "
            "macro-F1 drop from the best ≤ {drop:.3f}.".format(
                loss=decision["criteria"]["loss"] or "any",
                macro=decision["criteria"]["min_macro_f1"],
                recall=decision["criteria"]["min_class_recall"],
                size=decision["criteria"]["max_model_mib"],
                seeds=decision["criteria"]["min_seeds"],
                drop=decision["criteria"]["max_macro_f1_drop"],
            ),
            "",
            "| lane | seeds | val macro-F1 | test macro-F1 | worst recall | model MiB | eligible |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for lane in EXPECTED_LANES:
        candidate = decision["selected_candidates"].get(lane)
        if candidate is None:
            lines.append(f"| {lane} | — | — | — | — | — | no report |")
            continue
        lines.append(
            "| {lane} | {num_seeds} | {val:.4f} ± {val_std:.4f} | "
            "{test:.4f} ± {test_std:.4f} | {worst:.4f} | {size:.4f} | {eligible} |".format(
                lane=lane,
                num_seeds=candidate["num_seeds"],
                val=candidate["validation_macro_f1_mean"],
                val_std=candidate["validation_macro_f1_std"],
                test=candidate["test_macro_f1_mean"],
                test_std=candidate["test_macro_f1_std"],
                worst=candidate["worst_class_recall_mean"],
                size=candidate["model_mib"],
                eligible="yes" if candidate["eligible"] else "no",
            )
        )

    if decision["blocking_issues"]:
        lines.extend(["", "## Blocking issues", ""])
        lines.extend(f"- {issue}" for issue in decision["blocking_issues"])
    if decision["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in decision["warnings"])

    lines.extend(
        [
            "",
            "Model MiB is derived from parameter tensors (float32 unless a report "
            "provides an exact byte count). It is not measured runtime RAM. No latency, "
            "power, or edge-hardware measurement is inferred here.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--loss",
        default="sqrt_weighted_ce",
        help="Require one shared loss across lanes; use 'any' to disable.",
    )
    parser.add_argument("--min-macro-f1", type=float, default=0.60)
    parser.add_argument("--min-class-recall", type=float, default=0.20)
    parser.add_argument("--max-model-mib", type=float, default=1.0)
    parser.add_argument("--max-macro-f1-drop", type=float, default=0.02)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--require-federated", action="store_true")
    parser.add_argument("--prefer-federated", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the decision is incomplete (useful in CI).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "min_macro_f1",
        "min_class_recall",
        "max_macro_f1_drop",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1 or math.isnan(value):
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.max_model_mib <= 0 or args.min_seeds < 1:
        raise SystemExit("--max-model-mib and --min-seeds must be positive")

    reports, load_warnings = load_reports(args.reports_dir)
    criteria = Criteria(
        loss=None if args.loss.lower() == "any" else args.loss,
        min_macro_f1=args.min_macro_f1,
        min_class_recall=args.min_class_recall,
        max_model_mib=args.max_model_mib,
        max_macro_f1_drop=args.max_macro_f1_drop,
        min_seeds=args.min_seeds,
        require_federated=args.require_federated,
        prefer_federated=args.prefer_federated,
    )
    decision = build_decision(reports, criteria)
    decision["warnings"] = [*load_warnings, *decision["warnings"]]

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / "decision_summary.json"
    markdown_path = args.reports_dir / "decision_summary.md"
    json_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(decision), encoding="utf-8")

    print(render_markdown(decision))
    print(f"JSON -> {json_path}")
    print(f"Markdown -> {markdown_path}")
    if args.strict and decision["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
