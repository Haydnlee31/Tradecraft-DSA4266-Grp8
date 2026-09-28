"""Compile all study configurations without hiding the IID control.

Standard deviations describe variation across training seeds, not confidence
intervals over independent datasets. Binary alert metrics below are diagnostic
views of the eight-class confusion matrices; the task remains eight-class.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from src.data.label_map import CLASSES
from src.eval.decision import aggregate_candidates, load_reports


def name(candidate):
    if candidate["lane"] == "federated-light":
        base = "FL IID control" if candidate["partitioner"] == "iid" else f"FL Dirichlet α={candidate.get('alpha')}"
        strategy = str(candidate.get("strategy", "fedavg")).lower()
        return base if strategy == "fedavg" else f"{base} / {strategy}"
    return candidate["lane"]


def compile_study(folder):
    reports, warnings = load_reports(folder)
    candidates = aggregate_candidates(reports)
    order = {"centralized-heavy": 0, "centralized-light": 1, "FL IID control": 2, "FL Dirichlet α=0.5": 3}
    # Alternative optimizers must not overwrite each other's alert diagnostics
    # or receive the same legend label when a report directory includes both.
    candidates.sort(key=lambda c: (
        (2 if c["partitioner"] == "iid" else 3) if c["lane"] == "federated-light" else order[c["lane"]],
        str(c.get("strategy", "")),
    ))
    lookup = {r["_source"]: r for r in reports}
    lines = ["# Full-run measured results", "",
        "Fixed sampled splits; three seeds per configuration. Maximum 30 epochs/rounds, "
        "patience 5, batch 512, Adam 0.001, weight decay 0.00001, sqrt-weighted CE. "
        "FL uses 20 clients, full participation, one local epoch and global train class weights. "
        "Checkpoint selection uses validation macro-F1 only.", "",
        "FL runs use the actual client training helpers and Flower's sample-weighted aggregation, "
        "executed sequentially on this Mac. This measures learning, not distributed-system throughput, "
        "communication overhead, privacy or edge-hardware performance.", "",
        "## Aggregate metrics", "",
        "± denotes sample standard deviation across seeds (not a confidence interval).", "",
        "| Configuration | Seeds | Validation macro-F1 | Test macro-F1 | Test accuracy | Parameters | Parameter KiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in candidates:
        lines.append(f"| {name(c)} | {c['num_seeds']} | {c['validation_macro_f1_mean']:.4f} ± {c['validation_macro_f1_std']:.4f} | "
                     f"{c['test_macro_f1_mean']:.4f} ± {c['test_macro_f1_std']:.4f} | {c['test_accuracy_mean']:.4f} | "
                     f"{c['num_parameters']:,} | {c['parameter_bytes']/1024:.2f} |")
    lines.extend(["", "## Test recall by class", "", "| Class | " + " | ".join(name(c) for c in candidates) + " |",
                  "| --- | " + " | ".join("---:" for _ in candidates) + " |"])
    for label in CLASSES:
        lines.append("| " + label + " | " + " | ".join(
            f"{c['per_class_recall_mean'][label]:.3f} ± {c['per_class_recall_std'][label]:.3f}" for c in candidates) + " |")
    lines.extend(["", "## Alert interpretation and confusions", "",
        "Treating any non-Benign prediction as an alert is only a diagnostic view. "
        "An attack assigned the wrong attack category is still detected by that binary rule.", ""])
    extras = {}
    for c in candidates:
        matrices = np.asarray([lookup[path]["test_metrics"]["confusion_matrix"] for path in c["sources"]])
        mean = matrices.mean(axis=0)
        attack_recall = 1 - mean[1:, 0].sum() / mean[1:, :].sum()
        benign_fpr = 1 - mean[0, 0] / mean[0, :].sum()
        normalized = mean / mean.sum(axis=1, keepdims=True)
        mistaken = normalized.copy()
        np.fill_diagonal(mistaken, 0)
        pairs = np.dstack(np.unravel_index(np.argsort(mistaken.ravel())[::-1][:4], mistaken.shape))[0]
        confusions = [f"{CLASSES[i]} → {CLASSES[j]}: {mistaken[i,j]:.1%}" for i, j in pairs]
        extras[name(c)] = {"attack_detection_recall": float(attack_recall), "benign_false_alert_rate": float(benign_fpr),
                           "largest_row_normalized_confusions": confusions}
        lines.extend([f"- {name(c)}: attack-detection recall {attack_recall:.2%}; benign false-alert rate {benign_fpr:.2%}. "
                      + "; ".join(confusions) + "."])
    lines.extend(["", "## Run completion", "", "| Report | Selected epoch/round | Executed | Budget reached? |",
                  "| --- | ---: | ---: | --- |"])
    for r in reports:
        best = max(r["history"], key=lambda h: h.get("val_macro_f1", -1))
        step = best.get("round", best.get("epoch"))
        lines.append(f"| {Path(r['_source']).name} | {step} | {len(r['history'])} | {'yes' if len(r['history']) == 30 else 'no'} |")
    lines.extend(["", "## Limits on interpretation", "",
        "- This is the existing partial Kaggle mirror/sample, not all official CICIoT2023 traffic. "
        "Within-split duplicates and unknown upstream session/device relationships remain; an exact-overlap screen is not proof of independence.",
        "- Only one loss and one non-IID alpha are confirmed here; do not claim a universal architecture winner. "
        "Training seeds also change Dirichlet assignments, so FL variance combines both sources.",
        "- Heavy/light differ in width, depth AND dropout (0.3 versus 0.2). This compares configured model families, not a pure parameter-count intervention.",
        "- Each FL client resets Adam each round. Centralized Adam retains moments. "
        "Equal row-pass budgets do not make their optimization trajectories identical.",
        "- Global scaling, class counts and server validation are simulation conveniences; no privacy guarantee follows.",
        "- Parameter size excludes buffers, activations, framework memory and serialization. No edge latency/power was measured.",
        "- Decision thresholds are illustrative research gates; passing them is not deployment approval. "
        "Future architecture/loss changes should be selected on validation; these already-inspected test results are exploratory for subsequent iterations.", ""])
    payload = {"candidates": candidates, "alert_diagnostics": extras, "warnings": warnings}
    (folder / "study_summary.json").write_text(json.dumps(payload, indent=2))
    (folder / "RESULTS.md").write_text("\n".join(lines))
    # Standard scientific figure: every class on the same fixed [0, 1] scale.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tradecraft-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = np.array([[c["per_class_recall_mean"][k] for k in CLASSES] for c in candidates])
    fig, ax = plt.subplots(figsize=(12, 4.4), layout="constrained")
    im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(len(candidates)), [name(c) for c in candidates])
    ax.set_title("Test recall by class — mean of three training seeds")
    for i in range(len(candidates)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", color="white" if values[i,j] > .65 else "black")
    fig.colorbar(im, ax=ax, label="Recall")
    fig.savefig(folder / "recall_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Compiled {len(reports)} runs -> {folder / 'RESULTS.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/full_run"))
    args = parser.parse_args()
    compile_study(args.reports_dir)


if __name__ == "__main__":
    main()
