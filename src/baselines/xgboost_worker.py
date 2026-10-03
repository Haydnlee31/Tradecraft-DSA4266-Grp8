"""Isolated XGBoost worker: deliberately never imports PyTorch.

Some macOS wheels bundle incompatible OpenMP runtimes. A fresh process avoids
loading Torch and XGBoost native runtimes together; no unsafe runtime overrides
are needed. The parent owns data preparation/provenance, this worker owns trees.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.data.label_map import CLASSES
from src.eval.metrics import compute_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("request", type=Path)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    data = np.load(request["arrays"])
    params = request["params"]
    threads = params["nthread"]
    dtrain = xgb.DMatrix(data["train_x"], label=data["train_y"],
                        weight=data["weights"][data["train_y"]], nthread=threads)
    dval = xgb.DMatrix(data["val_x"], label=data["val_y"], nthread=threads)

    def macro_f1(predictions, matrix):
        return "macro_f1", float(f1_score(matrix.get_label(), predictions.argmax(1),
                                          labels=range(len(CLASSES)), average="macro", zero_division=0))

    history = {}
    booster = xgb.train(params, dtrain, num_boost_round=request["rounds"],
                        evals=[(dval, "validation")], custom_metric=macro_f1, maximize=True,
                        early_stopping_rounds=request["patience"] or None,
                        evals_result=history, verbose_eval=False)
    best_step = int(np.argmax(history["validation"]["macro_f1"])) + 1
    # Physically discard trees after the chosen validation checkpoint.
    booster = booster[:best_step]
    booster.save_model(request["checkpoint"])
    # Score the reloaded artifact too: reported metrics describe the saved model.
    restored = xgb.Booster(params={"nthread": threads})
    restored.load_model(request["checkpoint"])

    def score(name):
        truth = data[f"{name}_y"]
        prediction = restored.predict(xgb.DMatrix(data[f"{name}_x"], nthread=threads)).argmax(1)
        result = compute_metrics(truth, prediction)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, prediction, labels=range(len(CLASSES)), zero_division=0)
        result["per_class"] = {
            c: dict(precision=float(p), recall=float(r), f1=float(f), support=int(n))
            for c,p,r,f,n in zip(CLASSES, precision, recall, f1, support)}
        mask = truth == CLASSES.index("Benign")
        result["benign_false_alert_rate"] = float(np.mean(prediction[mask] != CLASSES.index("Benign"))) if mask.any() else None
        return result

    report = {"history": history, "best_step": best_step, "xgboost_params": params,
              "num_trees": len(restored.get_dump()), "saved_boost_rounds": restored.num_boosted_rounds(),
              "backend": "cpu-hist-isolated-process", "num_parameters": None, "parameter_bytes": None,
              "training_sample_weight": "global class weight per row; not PyTorch batch normalization",
              "validation_metrics": score("val"),
              "test_metrics": score("test") if "test_y" in data else None}
    Path(request["result"]).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
