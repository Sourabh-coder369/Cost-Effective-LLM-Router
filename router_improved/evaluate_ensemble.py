"""Validation-select a Domain-MF + logistic-router ensemble, then test once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402
from domain_specific.domain_model import DomainMFRouter  # noqa: E402


def labels_and_vectors(data: Path, embeddings: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(data)
    labels = frame["binary_label"].to_numpy() if "binary_label" in frame else (frame["label"] == "wins").astype(int).to_numpy()
    return labels.astype(np.int64), torch.load(embeddings, map_location="cpu", weights_only=False).float().numpy()


def domain_probabilities(data: Path, embeddings: Path, checkpoint_path: Path) -> np.ndarray:
    dataset = DomainRouterDataset(str(data), str(embeddings))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DomainMFRouter(query_embedding_dim=checkpoint["embedding_dim"], model_embedding_dim=64)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output = []
    with torch.inference_mode():
        for batch, _, domains in DataLoader(dataset, batch_size=512, shuffle=False):
            output.append(model(batch, domains).squeeze(1).numpy())
    return np.concatenate(output)


def curve(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    weak, strong = (1 - labels).mean(), labels.mean()
    result = {}
    for level in range(10, 101, 10):
        use_strong = np.zeros(len(labels), dtype=bool)
        use_strong[np.argsort(probabilities)[::-1][:round(len(labels) * level / 100)]] = True
        quality = np.where(use_strong, labels, 1 - labels).mean()
        result[str(level)] = {"strong_usage": float(use_strong.mean()), "quality": float(quality), "pgr": float((quality - weak) / (strong - weak))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-data", type=Path, required=True)
        parser.add_argument(f"--{split}-embeddings", type=Path, required=True)
    parser.add_argument("--domain-checkpoint", type=Path, default=Path("artifacts/mpnet_domain_reproduction/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mpnet_ensemble_true_test_results.json"))
    args = parser.parse_args()
    train_y, train_x = labels_and_vectors(args.train_data, args.train_embeddings)
    val_y, val_x = labels_and_vectors(args.val_data, args.val_embeddings)
    test_y, test_x = labels_and_vectors(args.test_data, args.test_embeddings)
    logistic = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42).fit(train_x, train_y)
    logistic_val, logistic_test = logistic.predict_proba(val_x)[:, 1], logistic.predict_proba(test_x)[:, 1]
    domain_val = domain_probabilities(args.val_data, args.val_embeddings, args.domain_checkpoint)
    domain_test = domain_probabilities(args.test_data, args.test_embeddings, args.domain_checkpoint)
    choices = [(alpha, log_loss(val_y, alpha * domain_val + (1 - alpha) * logistic_val)) for alpha in np.linspace(0, 1, 21)]
    alpha, validation_loss = min(choices, key=lambda item: item[1])
    probabilities = alpha * domain_test + (1 - alpha) * logistic_test
    result = {
        "domain_mf_weight": float(alpha),
        "logistic_weight": float(1 - alpha),
        "validation_log_loss": float(validation_loss),
        "n_examples": int(len(test_y)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "pr_auc": float(average_precision_score(test_y, probabilities)),
        "accuracy_at_0_5": float(((probabilities >= 0.5).astype(int) == test_y).mean()),
        "cost_levels": curve(probabilities, test_y),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
