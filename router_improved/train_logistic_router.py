"""Train a regularized linear router, selecting regularization on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def load(data: Path, embeddings: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(data)
    labels = frame["binary_label"].to_numpy() if "binary_label" in frame else (frame["label"] == "wins").astype(int).to_numpy()
    vectors = torch.load(embeddings, map_location="cpu", weights_only=False).float().numpy()
    if len(vectors) != len(labels):
        raise ValueError("Data and embeddings have different row counts")
    return vectors, labels.astype(np.int64)


def curve(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    weak, strong = (1 - labels).mean(), labels.mean()
    output = {}
    for level in range(10, 101, 10):
        selected = np.zeros(len(labels), dtype=bool)
        selected[np.argsort(probabilities)[::-1][:round(len(labels) * level / 100)]] = True
        quality = np.where(selected, labels, 1 - labels).mean()
        output[str(level)] = {
            "strong_usage": float(selected.mean()),
            "quality": float(quality),
            "pgr": float((quality - weak) / (strong - weak)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-data", type=Path, required=True)
        parser.add_argument(f"--{split}-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mpnet_logistic_true_test_results.json"))
    args = parser.parse_args()
    train_x, train_y = load(args.train_data, args.train_embeddings)
    val_x, val_y = load(args.val_data, args.val_embeddings)
    test_x, test_y = load(args.test_data, args.test_embeddings)
    candidates = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    best = None
    for c_value in candidates:
        model = LogisticRegression(C=c_value, max_iter=500, solver="lbfgs", random_state=42)
        model.fit(train_x, train_y)
        val_probabilities = model.predict_proba(val_x)[:, 1]
        value = log_loss(val_y, val_probabilities)
        print(f"C={c_value}: validation log loss={value:.6f}", flush=True)
        if best is None or value < best[0]:
            best = (value, c_value, model)
    assert best is not None
    _, c_value, model = best
    probabilities = model.predict_proba(test_x)[:, 1]
    result = {
        "selected_c": c_value,
        "validation_log_loss": best[0],
        "n_examples": int(len(test_y)),
        "strong_quality": float(test_y.mean()),
        "weak_quality": float((1 - test_y).mean()),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "pr_auc": float(average_precision_score(test_y, probabilities)),
        "accuracy_at_0_5": float(((probabilities >= 0.5).astype(int) == test_y).mean()),
        "cost_levels": curve(probabilities, test_y),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
