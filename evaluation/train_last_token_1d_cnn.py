"""Train a 1D CNN on Qwen's last-prompt-token representation across layers.

Expected data directory (for example, a Kaggle input directory)::

    qwen_math_layers_full_v2/
      embeddings_00000.pt
      embeddings_00001.pt
      ...

Each shard must contain ``layer_last_token_embeddings`` with shape
``[records, layers, hidden_size]`` and ``strong_model_wins`` where 1 means
Qwen answered incorrectly and the query should be routed to a stronger model.
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


class LayerTrajectoryCNN(nn.Module):
    """Classify an ordered sequence of per-layer hidden states."""

    def __init__(self, hidden_size: int, projected_size: int = 64) -> None:
        super().__init__()
        # Applied independently to each of the 25 layer vectors.
        self.normalize = nn.LayerNorm(hidden_size)
        self.project = nn.Sequential(
            nn.Linear(hidden_size, projected_size),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        # Conv1d expects [batch, channels, sequence_length].  Here channels
        # are projected embedding features and the sequence is model layers.
        self.convolution = nn.Sequential(
            nn.Conv1d(projected_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),  # 32 avg-pool values + 32 max-pool values
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, layers, hidden_size], e.g. [B, 25, 896]
        x = self.normalize(x)
        x = self.project(x)              # [B, 25, projected_size]
        x = x.transpose(1, 2)            # [B, projected_size, 25]
        x = self.convolution(x)          # [B, 32, 25]
        average = x.mean(dim=2)          # [B, 32]
        maximum = x.max(dim=2).values    # [B, 32]
        return self.classifier(torch.cat([average, maximum], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = [], []
    shard_paths = sorted(data_dir.glob("embeddings_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No embeddings_*.pt files found in {data_dir}")
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        sequences.append(shard["layer_last_token_embeddings"].to(torch.float16))
        labels.append(shard["strong_model_wins"].to(torch.float32))
    x = torch.cat(sequences, dim=0)
    y = torch.cat(labels, dim=0)
    print(f"Loaded {len(x)} records with shape {tuple(x.shape[1:])}; failure rate={y.mean().item():.3f}")
    return x, y


def collect_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probabilities, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x.to(device, non_blocking=True).float()).squeeze(1)
            all_probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_probabilities), np.concatenate(all_labels).astype(np.int64)


def metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_strong": float(precision_score(labels, predictions, zero_division=0)),
        "failure_recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "strong_route_rate": float(predictions.mean()),
        "avoidable_escalation_rate": float(fp / len(labels)),
        "missed_failures_rate": float(fn / len(labels)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/last_token_1d_cnn"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y = load_dataset(args.data_dir)
    indices = np.arange(len(y))
    train_indices, validation_indices = train_test_split(
        indices, test_size=0.20, random_state=args.seed, stratify=y.numpy()
    )
    train_data = TensorDataset(x[train_indices], y[train_indices, None])
    validation_data = TensorDataset(x[validation_indices], y[validation_indices, None])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {"num_workers": 2, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, **loader_options)

    model = LayerTrajectoryCNN(hidden_size=x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss, best_state, waiting = float("inf"), None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch_x.to(device, non_blocking=True).float())
                loss = loss_function(logits, batch_y.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_loss += loss.item() * len(batch_x)
        training_loss /= len(train_data)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(batch_x.to(device, non_blocking=True).float())
                    loss = loss_function(logits, batch_y.to(device, non_blocking=True))
                validation_loss += loss.item() * len(batch_x)
        validation_loss /= len(validation_data)
        history.append({"epoch": epoch, "train_loss": training_loss, "validation_loss": validation_loss})
        print(f"Epoch {epoch:02d}: train={training_loss:.4f}, validation={validation_loss:.4f}", flush=True)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= args.patience:
                print("Early stopping.", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    probabilities, labels = collect_probabilities(model, validation_loader, device)
    threshold_sweep = [metrics_at_threshold(labels, probabilities, float(t)) for t in np.arange(0.05, 0.951, 0.01)]
    best_f1 = max(threshold_sweep, key=lambda row: row["f1"])
    results = {
        "experiment": "last prompt token across all Qwen layers -> 1D CNN",
        "dataset": str(args.data_dir.resolve()),
        "device": str(device),
        "n_total": int(len(x)),
        "n_train": int(len(train_data)),
        "n_validation": int(len(validation_data)),
        "input_shape_per_example": list(x.shape[1:]),
        "label_definition": "1 = Qwen answer is incorrect, so route to the stronger model",
        "best_validation_loss": float(best_loss),
        "metrics": {
            "failure_rate": float(labels.mean()),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "pr_auc": float(average_precision_score(labels, probabilities)),
            "threshold_0_50": metrics_at_threshold(labels, probabilities, 0.50),
            "best_f1_threshold": best_f1,
        },
        "history": history,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": best_state, "hidden_size": int(x.shape[2])}, args.output_dir / "best_model.pt")
    print(json.dumps(results["metrics"], indent=2), flush=True)
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
