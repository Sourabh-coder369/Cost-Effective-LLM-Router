"""Evaluate BGE baseline routers using thresholds tuned on validation accuracy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from baseline.model import MatrixFactorizationRouter

DATA_DIR = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced"
EMBEDDING_DIR = ROOT / "data/original_gpt4_llama7b/bge_embeddings"
CHECKPOINTS = {
    32: ROOT / "router/baseline/checkpoints_bge_modeldim_32/best_model.pt",
    64: ROOT / "router/baseline/checkpoints_bge_large_current/best_model.pt",
    128: ROOT / "router/baseline/checkpoints_bge_modeldim_128/best_model.pt",
    256: ROOT / "router/baseline/checkpoints_bge_modeldim_256/best_model.pt",
    512: ROOT / "router/baseline/checkpoints_bge_modeldim_512/best_model.pt",
}
OUTPUT = ROOT / "router/baseline/bge_tuned_threshold_metrics.json"


def load_split(name: str) -> tuple[torch.Tensor, np.ndarray]:
    labels = pd.read_parquet(DATA_DIR / f"router_{name}.parquet")["binary_label"].to_numpy(dtype=np.int64)
    embeddings = torch.load(EMBEDDING_DIR / f"router_{name}_bge_large_embeddings.pt", map_location="cpu", weights_only=True)
    if len(embeddings) != len(labels):
        raise ValueError(f"{name}: embeddings and labels have different lengths")
    return embeddings, labels


@torch.no_grad()
def probabilities(checkpoint_path: Path, embeddings: torch.Tensor, device: torch.device) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    model = MatrixFactorizationRouter(
        query_embedding_dim=state["text_proj.0.weight"].shape[1],
        model_embedding_dim=checkpoint["config"]["model_embedding_dim"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return torch.cat([model(batch.to(device)) for batch in embeddings.split(512)]).cpu().numpy().reshape(-1)


def tune_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.0, 1.0, 1001)
    accuracies = np.array([((probs >= threshold).astype(int) == labels).mean() for threshold in thresholds])
    best = int(accuracies.argmax())
    return float(thresholds[best]), float(accuracies[best])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_embeddings, val_labels = load_split("val")
    test_embeddings, test_labels = load_split("test")
    results = {"selection_rule": "Threshold maximizes validation routing accuracy", "device": str(device), "models": {}}

    for dimension, checkpoint_path in CHECKPOINTS.items():
        val_probs = probabilities(checkpoint_path, val_embeddings, device)
        threshold, val_accuracy = tune_threshold(val_probs, val_labels)
        test_probs = probabilities(checkpoint_path, test_embeddings, device)
        predictions = (test_probs >= threshold).astype(int)
        strong_usage = float(predictions.mean())
        results["models"][str(dimension)] = {
            "threshold": threshold,
            "validation_accuracy": val_accuracy,
            "test_accuracy": float((predictions == test_labels).mean()),
            "test_f1": float(f1_score(test_labels, predictions, zero_division=0)),
            "test_auc": float(roc_auc_score(test_labels, test_probs)),
            "test_strong_usage": strong_usage,
            "test_cost_saving": 1.0 - strong_usage,
        }

    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
