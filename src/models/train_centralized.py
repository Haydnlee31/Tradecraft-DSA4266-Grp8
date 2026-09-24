"""Train the centralized-heavy or centralized-light MLP lane end-to-end.

Loads the shared standardized splits (dataset.py), trains the chosen architecture
(architectures.py) with the shared loop (train.py), evaluates on the held-out test
split with macro-F1 / per-class recall (src/eval/metrics.py), and writes a JSON
result to reports/ -- one file per (variant, loss, seed) so Heavy vs. Light and
loss-function ablations can be compared without re-running everything.

Usage:
    python -m src.models.train_centralized --variant heavy
    python -m src.models.train_centralized --variant light --loss focal --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.label_map import CLASSES
from src.eval.metrics import compute_metrics
from src.models.architectures import HEAVY_CONFIG, LIGHT_CONFIG, MLPClassifier
from src.models.dataset import build_dataloaders
from src.models.losses import build_criterion
from src.models.train import Trainer
from src.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["heavy", "light"], required=True)
    parser.add_argument(
        "--loss",
        choices=[
            "ce",
            "weighted_ce",
            "sqrt_weighted_ce",
            "focal",
    ],
    default="weighted_ce",
)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Mini-batch size. Per Lecture 1's 'Gradient Descent: Batch size' slide: batch=1 "
        "(pure SGD) gives noisy per-step gradient estimates, while a full-dataset batch gives "
        "the best gradient estimate but doesn't fit in memory/isn't practical on ~440k train "
        "rows -- 512 is a mid-sized mini-batch trading off update-noise against throughput.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--l1-lambda", type=float, default=0.0,
        help="Optional L1 weight penalty (Lecture 2 'Weight Regularisation'), on top of "
        "Adam's built-in L2 weight_decay. See train.py's Trainer docstring for why L2 is the "
        "default here and L1 is opt-in.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    data = build_dataloaders(batch_size=args.batch_size)
    config = HEAVY_CONFIG if args.variant == "heavy" else LIGHT_CONFIG
    model = MLPClassifier(data["num_features"], data["num_classes"], config)
    criterion = build_criterion(args.loss, data["class_counts"])

    trainer = Trainer(model, criterion, lr=args.lr, l1_lambda=args.l1_lambda, patience=args.patience)
    history = trainer.fit(data["loaders"]["train"], data["loaders"]["val"], epochs=args.epochs)
    
    # save best restored model
    CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / (
        f"{config.name}_{args.loss}_seed{args.seed}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dims": config.hidden_dims,
                "dropout": config.dropout,
            },
            "num_features": data["num_features"],
            "num_classes": data["num_classes"],
            "feature_columns": data["feature_columns"],
            "classes": CLASSES,
            "loss": args.loss,
            "seed": args.seed,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to {checkpoint_path}")

    test_logits, y_true = trainer.predict(data["loaders"]["test"])
    y_pred = test_logits.argmax(axis=1)
    test_metrics = compute_metrics(y_true, y_pred)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{config.name}_{args.loss}_seed{args.seed}.json"
    result = {
        "variant": args.variant,
        "config": {"hidden_dims": config.hidden_dims, "dropout": config.dropout},
        "loss": args.loss,
        "num_parameters": model.num_parameters(),
        "history": history,
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\nSaved results to {out_path}")
    print(f"Params: {model.num_parameters():,}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}  macro-F1: {test_metrics['macro_f1']:.4f}")
    for c, r in test_metrics["per_class_recall"].items():
        print(f"  recall[{c}]: {r:.4f}")


if __name__ == "__main__":
    main()
