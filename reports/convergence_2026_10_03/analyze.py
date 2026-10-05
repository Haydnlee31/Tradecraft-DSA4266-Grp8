"""Summarize completed validation-only convergence controls without test access."""

import argparse
import hashlib
import json
from pathlib import Path


def best(history):
    return max(history, key=lambda row: row["validation_metrics"]["macro_f1"])


def patience_selection(history, patience=5):
    selected, stale = None, 0
    for row in history:
        if selected is None or row["validation_metrics"]["macro_f1"] > selected["validation_metrics"]["macro_f1"]:
            selected, stale = row, 0
        else:
            stale += 1
        if stale >= patience:
            return selected, row["step"]
    return selected, history[-1]["step"]


def compare(before, after, plan):
    a, b = before["validation_metrics"], after["validation_metrics"]
    recall_delta = {c: b["per_class_recall"][c] - a["per_class_recall"][c] for c in a["per_class_recall"]}
    gain = b["macro_f1"] - a["macro_f1"]
    false_alert_change = b["benign_false_alert_rate"] - a["benign_false_alert_rate"]
    ga, gb = plan["promotion"]["option_a"], plan["promotion"]["option_b"]
    pass_a = (gain >= ga["macro_f1_gain_min"] and false_alert_change <= ga["benign_false_alert_increase_max"]
              and min(recall_delta.values()) >= -ga["any_class_recall_drop_max"])
    pass_b = (-false_alert_change >= gb["benign_false_alert_reduction_min"] and gain >= -gb["macro_f1_drop_max"]
              and min(recall_delta[c] for c in ("Web-based", "Brute Force")) >= -gb["web_and_brute_force_recall_drop_max"])
    return {"macro_f1_gain": gain, "benign_false_alert_change": false_alert_change,
            "per_class_recall_change": recall_delta, "promotion_gate_a": pass_a, "promotion_gate_b": pass_b}


def summarize(root, plan):
    summary, histories, manifest_reference = {}, {}, None
    for lane in ("heavy", "light", "iid", "dirichlet"):
        directory = root / lane
        report = json.loads((directory / "result.json").read_text())
        manifest = json.loads((directory / "environment.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        assert status["status"] == "complete" and report["steps"] == 60
        assert report["test_metrics"] is None and set(manifest["split_sha256"]) == {"train", "val"}
        assert manifest["settings"]["seed"] == 0 and manifest["settings"]["partition_seed"] == 0
        assert manifest["settings"]["patience"] == 0
        for key, value in plan["common"].items():
            assert manifest["settings"][key] == value, f"Plan mismatch: {lane}/{key}"
        if manifest_reference is None:
            manifest_reference = manifest
        else:
            for key in ("source_sha256", "split_sha256", "packages", "git_commit"):
                assert manifest[key] == manifest_reference[key], f"Unmatched {key} for {lane}"
        history = report["history"]
        assert [h["step"] for h in history] == list(range(1, 61))
        histories[lane] = history
        b30, b60 = best(history[:30]), best(history)
        old, stopped_at = patience_selection(history[:30])
        assert b60["step"] == report["best_step"]
        assert b60["validation_metrics"] == report["validation_metrics"]
        summary[lane] = {
            "best_through_30": b30, "best_through_60": b60,
            "retrospective_patience_5_cap_30": {"checkpoint": old, "stop_step": stopped_at},
            "change_30_to_60": compare(b30, b60, plan),
            "change_old_policy_to_60": compare(old, b60, plan),
            "final_step": history[-1], "step_seconds_sum": report["step_seconds_sum"],
            "examples_processed": report["examples_processed"], "optimizer_steps": report["optimizer_steps"],
            "process_peak_rss_bytes": report["process_peak_rss_bytes"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "model_config": manifest["model_config"],
            "last_ten_best_gain_over_first_fifty": best(history[-10:])["validation_metrics"]["macro_f1"] - best(history[:50])["validation_metrics"]["macro_f1"],
        }
    return {"stage": "convergence", "seed": 0, "partition_seed": 0,
            "source_commit": manifest_reference["git_commit"],
            "source_sha256": manifest_reference["source_sha256"],
            "split_sha256": manifest_reference["split_sha256"],
            "packages": manifest_reference["packages"], "platform": manifest_reference["platform"],
            "classes": manifest_reference["classes"], "lanes": summary,
            "test_evaluated": False,
            "limitations": ["Single seed, exploratory validation checkpoint search; not a significance test.",
                            "Best-of-60 has more validation selection opportunities than best-of-30.",
                            "Equal examples do not imply equal optimizer steps, compute or trajectories.",
                            "CPU host timing includes training/validation, not setup/checkpoint IO or physical edge measurements."]}, histories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("outputs/mlp-tuning/convergence"))
    p.add_argument("--plan", type=Path, default=Path("configs/local_tuning_plan.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    report, histories = summarize(args.runs, plan)
    report["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    report["screen_runs_completed"] = 4
    report["screen_runs_remaining"] = plan["max_screen_runs"] - 4
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    labels = {"heavy": "Central heavy", "light": "Central light", "iid": "FL IID", "dirichlet": "FL non-IID"}
    for lane, history in histories.items():
        x = [h["step"] for h in history]
        axes[0].plot(x, [h["validation_metrics"]["macro_f1"] for h in history], label=labels[lane])
        axes[1].plot(x, [100*h["validation_metrics"]["benign_false_alert_rate"] for h in history], label=labels[lane])
    for ax in axes:
        ax.axvline(30, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch (central) / round (FL); not equivalent compute")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Validation macro-F1")
    axes[1].set_ylabel("Benign false alerts (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Convergence controls — seed 0, fixed partitions, no early stopping")
    fig.savefig(args.output / "curves.png", dpi=160)
    plt.close(fig)
    lines = ["# Convergence controls — 60 epochs/rounds", "",
             "Validation only; four configurations, model seed 0 and partition seed 0. No test evaluation or cloud resources.", "",
             f"Training source commit: `{report['source_commit']}`. All four runs share source, data and package hashes.", "",
             "| Lane | Old patience-5 selected / stopped | Old macro-F1 | Best ≤30 (step) | Best ≤60 (step) | Gain 30→60 (pp) | Gate A / B |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for lane, data in report["lanes"].items():
        old = data["retrospective_patience_5_cap_30"]
        b30, b60, change = data["best_through_30"], data["best_through_60"], data["change_30_to_60"]
        lines.append(f"| {labels[lane]} | {old['checkpoint']['step']} / {old['stop_step']} | {old['checkpoint']['validation_metrics']['macro_f1']:.4f} | {b30['validation_metrics']['macro_f1']:.4f} ({b30['step']}) | {b60['validation_metrics']['macro_f1']:.4f} ({b60['step']}) | {100*change['macro_f1_gain']:+.2f} | {change['promotion_gate_a']} / {change['promotion_gate_b']} |")
    lines += ["", "Gates are the predeclared project preferences, not significance or deployment tests. A requires +1 pp macro-F1 with at most +2 pp false alerts and at most 2 pp loss in any class recall. B requires −5 pp false alerts with at most 1 pp macro-F1 loss and at most 2 pp Web/Brute Force recall loss.", "",
              "## Validation metrics at selected checkpoints", "", "| Lane | Budget | Benign false alerts | Web recall | Brute Force recall | DoS recall |", "|---|---:|---:|---:|---:|---:|"]
    for lane, data in report["lanes"].items():
        for cap in (30, 60):
            m = data[f"best_through_{cap}"]["validation_metrics"]
            r = m["per_class_recall"]
            lines.append(f"| {labels[lane]} | {cap} | {100*m['benign_false_alert_rate']:.2f}% | {100*r['Web-based']:.2f}% | {100*r['Brute Force']:.2f}% | {100*r['DoS']:.2f}% |")
    lines += ["", "## Per-class recall at the best-through-60 checkpoint", "", "| Class | Heavy | Light | FL IID | FL non-IID |", "|---|---:|---:|---:|---:|"]
    for name in report["classes"]:
        values = [f"{100*report['lanes'][lane]['best_through_60']['validation_metrics']['per_class_recall'][name]:.2f}%" for lane in labels]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines += ["", "Full per-class precision/recall/F1 and confusion matrices are in [summary.json](summary.json).", "",
              "## Measured local compute", "", "| Lane | Training + validation seconds | Training examples processed | Optimizer updates | Process peak RSS (MiB) |", "|---|---:|---:|---:|---:|"]
    for lane, d in report["lanes"].items():
        lines.append(f"| {labels[lane]} | {d['step_seconds_sum']:.1f} | {d['examples_processed']:,} | {d['optimizer_steps']:,} | {d['process_peak_rss_bytes']/1024**2:.1f} |")
    lines += ["", "RSS includes data and libraries; it is not inference memory. Time excludes data setup/checkpoint IO. FL update totals span separate client trajectories.", "",
              "![Validation trajectories](curves.png)", "", "## Interpretation limits", ""]
    lines += ["- " + item for item in report["limitations"]]
    lines += ["", f"Full checkpoints and histories remain under `{args.runs}` (Git-ignored). Original baseline reports are unchanged.", ""]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({lane: {"best30": d["best_through_30"]["validation_metrics"]["macro_f1"],
                             "best60": d["best_through_60"]["validation_metrics"]["macro_f1"],
                             "step60": d["best_through_60"]["step"], "change": d["change_30_to_60"]}
                      for lane,d in report["lanes"].items()}, indent=2))


if __name__ == "__main__":
    main()
