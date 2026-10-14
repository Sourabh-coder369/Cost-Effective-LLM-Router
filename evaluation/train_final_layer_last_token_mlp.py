"""Train a failure router from the final Qwen layer's last-prompt-token embedding.

The layer-embedding dataset stores one [layers, hidden_size] tensor per
question.  This baseline deliberately uses only the final layer and the last
prompt token, leaving a single 896-dimensional vector per question.  A small
MLP then predicts ``strong_model_wins`` (equivalently, Qwen failed).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FinalLayerMLP(nn.Module):
    """Small regularized classifier for a final-layer hidden state."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_final_layer_vectors(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    shard_paths = sorted(data_dir.glob("embeddings_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No embedding shards found in {data_dir}")

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        # [records, layers, hidden_size] -> final layer [records, hidden_size]
        vectors.append(shard["layer_last_token_embeddings"][:, -1, :].float())
        labels.append(shard["strong_model_wins"].to(torch.float32))

    return torch.cat(vectors).numpy(), torch.cat(labels).numpy()


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).squeeze(1).cpu().numpy())
            labels.append(y.numpy())
    return 1.0 / (1.0 + np.exp(-np.concatenate(logits))), np.concatenate(labels).astype(np.int64)


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | dict[str, int]]:
    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision_strong": float(precision_score(y_true, predicted, zero_division=0)),
        "failure_recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "strong_route_rate": float(predicted.mean()),
        "avoidable_escalation_rate": float(fp / len(y_true)),
        "missed_failures_rate": float(fn / len(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/qwen_math_layers_full_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_layer_last_token_mlp"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y = load_final_layer_vectors(args.data_dir)
    train_x, validation_x, train_y, validation_y = train_test_split(
        x, y, test_size=0.20, random_state=args.seed, stratify=y
    )

    # Fit normalization only on training rows, avoiding validation leakage.
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_x = (train_x - mean) / std
    validation_x = (validation_x - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y[:, None])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y[:, None])),
        batch_size=args.batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalLayerMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_loader.dataset)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                validation_loss += loss_fn(model(batch_x.to(device)), batch_y.to(device)).item() * len(batch_x)
        validation_loss /= len(validation_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"Epoch {epoch:02d}: train={train_loss:.4f} validation={validation_loss:.4f}", flush=True)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping after epoch {epoch}", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    probabilities, labels = predict(model, validation_loader, device)
    sweep = [threshold_metrics(labels, probabilities, threshold) for threshold in np.arange(0.05, 0.951, 0.01)]
    best_f1 = max(sweep, key=lambda row: float(row["f1"]))
    results = {
        "experiment": "final Qwen layer, last prompt token, small MLP",
        "data_dir": str(args.data_dir.resolve()),
        "n_total": int(len(x)),
        "n_train": int(len(train_x)),
        "n_validation": int(len(validation_x)),
        "hidden_size": int(x.shape[1]),
        "device": str(device),
        "best_validation_loss": float(best_loss),
        "label_definition": "1 = Qwen answer is incorrect; route to stronger model",
        "metrics": {
            "failure_rate": float(labels.mean()),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "pr_auc": float(average_precision_score(labels, probabilities)),
            "threshold_0_50": threshold_metrics(labels, probabilities, 0.50),
            "best_f1_threshold": best_f1,
        },
        "history": history,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": best_state, "mean": mean, "std": std}, args.output_dir / "best_model.pt")
    print(json.dumps(results["metrics"], indent=2), flush=True)
    print(f"Saved results to {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
