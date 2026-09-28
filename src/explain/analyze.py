"""Small, reproducible SHAP expected-gradients audit of saved MLPs.

Use training rows as background and a class-balanced validation sample for
explanations. Test data never drives policy hypotheses. Logit attributions
describe model behavior, not causal security rules or feature thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import shap
import torch

from src.data.label_map import CLASSES
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.dataset import load_or_fit_scaler, to_arrays


def explain(checkpoint: Path, splits: Path, output: Path, per_class=16, nsamples=256):
    """Explain one predeclared seed-0 checkpoint, including misclassified rows."""
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    scaler, features = load_or_fit_scaler(splits)
    if saved["feature_columns"] != features or saved["classes"] != CLASSES:
        raise ValueError("checkpoint/preprocessing schema mismatch")
    config = MLPConfig("explain", tuple(saved["config"]["hidden_dims"]), saved["config"]["dropout"])
    model = MLPClassifier(len(features), len(CLASSES), config)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    # Natural training prevalence defines the reference distribution. Balanced
    # validation sampling is deliberately NOT a population-weighted explanation.
    background_frame = pl.read_parquet(splits / "train.parquet").sample(n=128, seed=0)
    validation = pl.read_parquet(splits / "val.parquet")
    selected = pl.concat([
        validation.filter(pl.col("class") == name).sample(
            n=min(per_class, validation.filter(pl.col("class") == name).height), seed=0
        ) for name in CLASSES
    ])
    background, _ = to_arrays(background_frame, scaler, features)
    x, y = to_arrays(selected, scaler, features)
    tensor = torch.from_numpy(x)
    explainer = shap.GradientExplainer(model, torch.from_numpy(background))
    values = np.asarray(explainer.shap_values(tensor, nsamples=nsamples, rseed=0))
    if values.shape != (len(y), len(features), len(CLASSES)) or not np.isfinite(values).all():
        raise ValueError(f"unexpected SHAP values: {values.shape}")
    with torch.no_grad():
        logits = model(tensor).numpy()
        reference = model(torch.from_numpy(background)).mean(0).numpy()
    # Monte Carlo approximation is not exactly additive. Record its error,
    # rather than silently treating approximate attributions as exact SHAP.
    residual = logits - reference - values.sum(axis=1)
    classes = {}
    for index, name in enumerate(CLASSES):
        mask = y == index
        mean_abs = np.abs(values[mask, :, index]).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:8]
        classes[name] = {
            "sample_count": int(mask.sum()),
            "correct_predictions": int((logits[mask].argmax(1) == index).sum()),
            "top_features": [
                {"feature": features[i], "mean_absolute_logit_attribution": float(mean_abs[i])}
                for i in order
            ],
        }
    payload = {
        "checkpoint": str(checkpoint), "method": "SHAP GradientExplainer / expected gradients",
        "background": "128 seeded natural-prevalence training rows",
        "explanation_sample": "seeded class-balanced validation rows; includes mistakes",
        "seed": 0, "integration_samples": nsamples,
        "mean_absolute_additivity_residual_logits": float(np.abs(residual).mean()),
        "p95_absolute_additivity_residual_logits": float(np.quantile(np.abs(residual), .95)),
        "mean_absolute_logit_difference_from_background": float(np.abs(logits-reference).mean()),
        "classes": classes,
        "limitations": [
            "One predeclared seed per configuration; not seed-stable policy evidence.",
            "Small background/sample; correlated flow statistics can share attribution.",
            "Absolute importance does not establish feature direction or a threshold.",
            "No device identities, causal interpretation, automatic blocking or deployment claims.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{checkpoint.stem}.json").write_text(json.dumps(payload, indent=2))
    lines = [f"# Explanation audit: {checkpoint.stem}", "", 
        "Exploratory validation evidence only. These are class-logit attributions, not probabilities.", "",
        "| True class | Correct / sampled | Leading features for this class's logit |",
        "| --- | ---: | --- |"]
    for name, item in classes.items():
        names = ", ".join(x["feature"] for x in item["top_features"][:5])
        lines.append(f"| {name} | {item['correct_predictions']} / {item['sample_count']} | {names} |")
    lines.extend(["", "## Policy interpretation", "",
        "Use these feature groups to prioritize telemetry and analyst investigation. Inspect the corresponding "
        "flow counters alongside packet/application logs; compare flagged traffic with normal-service baselines. "
        "Do not turn a top-feature list into a deny rule: direction, thresholds, false-positive cost and stability "
        "have not been established. Rare-class misses require independent application/authentication monitoring.", "",
        f"Mean absolute additivity residual: {np.abs(residual).mean():.4f} logits; "
        f"95th percentile: {np.quantile(np.abs(residual), .95):.4f} logits.", "",
        "Method: https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html", ""])
    (output / f"{checkpoint.stem}.md").write_text("\n".join(lines))
    print(f"SHAP audit -> {output / checkpoint.stem}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("reports/full_run/checkpoints"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--output", type=Path, default=Path("reports/full_run/explanations"))
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--nsamples", type=int, default=256)
    args = parser.parse_args()
    checkpoints = sorted(args.checkpoint_dir.glob("*seed0.pt"))
    if not checkpoints:
        raise SystemExit("No seed-0 checkpoints found")
    for checkpoint in checkpoints:
        explain(checkpoint, args.splits, args.output, args.per_class, args.nsamples)


if __name__ == "__main__":
    main()
