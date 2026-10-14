"""Evaluate the existing MPNet Domain-MF checkpoint on the true held-out test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
sys.path.insert(0, str(ROOT / "evaluation"))
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402
from domain_specific.domain_model import DomainMFRouter  # noqa: E402
from routellm_evaluation import compute_apgr, compute_cpt  # noqa: E402


def cost_quality(probabilities: np.ndarray, labels: np.ndarray, fraction: float) -> dict[str, float]:
    selected = np.zeros(len(probabilities), dtype=bool)
    selected[np.argsort(probabilities)[::-1][: max(1, round(len(probabilities) * fraction))]] = True
    quality = np.where(selected, labels, 1 - labels).mean()
    weak, strong = (1 - labels).mean(), labels.mean()
    return {
        "strong_usage": float(selected.mean()),
        "quality": float(quality),
        "pgr": float((quality - weak) / (strong - weak)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--test-embeddings", required=True, type=Path)
    parser.add_argument("--checkpoint", default=Path("artifacts/mpnet_domain_reproduction/best_model.pt"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/mpnet_domain_true_test_results.json"), type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DomainRouterDataset(str(args.test_data), str(args.test_embeddings))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DomainMFRouter(query_embedding_dim=checkpoint["embedding_dim"], model_embedding_dim=64).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probabilities, labels = [], []
    with torch.inference_mode():
        for embeddings, batch_labels, domains in DataLoader(dataset, batch_size=512, shuffle=False):
            probabilities.append(model(embeddings.to(device), domains.to(device)).squeeze(1).cpu().numpy())
            labels.append(batch_labels.numpy().astype(np.int64))
    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    from sklearn.metrics import average_precision_score, roc_auc_score
    result = {
        "n_examples": int(len(labels)),
        "strong_quality": float(labels.mean()),
        "weak_quality": float((1 - labels).mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy_at_0_5": float(((probabilities >= 0.5).astype(int) == labels).mean()),
        "apgr": float(compute_apgr(probabilities, labels, (1 - labels).mean(), labels.mean())),
        "cpt_50": compute_cpt(probabilities, labels, (1 - labels).mean(), labels.mean(), 0.5),
        "cpt_80": compute_cpt(probabilities, labels, (1 - labels).mean(), labels.mean(), 0.8),
        "cost_levels": {str(level): cost_quality(probabilities, labels, level / 100) for level in range(10, 101, 10)},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
