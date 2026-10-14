"""Train the existing matrix-factorization router on Qwen trajectory signals.

The source trajectory project defines ``is_correct=False`` as a Qwen-0.5B
failure.  For routing, that becomes the positive label: ``1`` means route to
the stronger model and ``0`` means keep the query on Qwen-0.5B.

The query representation is the 23 adjacent-layer cosine similarities plus
the final-token entropy (24 features).  The model side is still represented
by the learned weak/strong MF embeddings from ``router/baseline/model.py``.

Example:
    python evaluation/train_trajectory_mf.py \
      --trajectory-data C:/.../trajectory_data_f32.jsonl \
      --output-dir artifacts/trajectory_mf
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "router" / "baseline"))
from model import MatrixFactorizationRouter  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_trajectory_data(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")

    features = []
    labels = []
    indices = []
    feature_width = None
    for record in records:
        cosine_sims = record.get("cosine_sims")
        entropy = record.get("final_entropy")
        if not isinstance(cosine_sims, list) or entropy is None:
            raise ValueError("Each record must contain cosine_sims and final_entropy")
        row = [float(value) for value in cosine_sims] + [float(entropy)]
        feature_width = feature_width or len(row)
        if len(row) != feature_width:
            raise ValueError("Trajectory records have inconsistent feature widths")

        # Qwen failure => the stronger model is the preferred route.
        if "is_correct" not in record:
            raise ValueError("This script expects the trajectory is_correct field")
        labels.append(0 if bool(record["is_correct"]) else 1)
        features.append(row)
        indices.append(int(record.get("idx", len(indices))))

    x = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(labels, dtype=np.float32)
    return x, y, indices


def standardize_train_only(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_other - mean) / std, mean, std


def probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            probs = model(batch_x.to(device)).squeeze(1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch_y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    # Under the user's assumption, every Qwen failure is fixed if routed to
    # the stronger model; every missed failure remains incorrect.
    assumed_quality = 1.0 - (fn / len(y_true))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_strong": float(precision_score(y_true, pred, zero_division=0)),
        "failure_recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "strong_route_rate": float(pred.mean()),
        "avoidable_escalation_rate": float(fp / len(y_true)),
        "missed_failures_rate": float(fn / len(y_true)),
        "assumed_routed_quality": float(assumed_quality),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(y_true: np.ndarray, probs: np.ndarray) -> dict:
    thresholds = np.linspace(0.05, 0.95, 91)
    by_threshold = [threshold_metrics(y_true, probs, float(t)) for t in thresholds]
    best_f1 = max(by_threshold, key=lambda item: item["f1"])
    best_quality_at_budget = max(by_threshold, key=lambda item: item["assumed_routed_quality"])
    result = {
        "n": int(len(y_true)),
        "failure_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "threshold_0_50": threshold_metrics(y_true, probs, 0.50),
        "best_f1_threshold": best_f1,
        "threshold_sweep": by_threshold,
    }
    # This is included explicitly because quality can be traded against the
    # percentage of queries escalated to the stronger model.
    result["best_assumed_quality"] = best_quality_at_budget
    return result


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, indices = load_trajectory_data(Path(args.trajectory_data))
    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(
        x, y, indices, test_size=args.val_fraction, random_state=args.seed, stratify=y
    )
    x_train, x_val, feature_mean, feature_std = standardize_train_only(x_train, x_val)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train[:, None]))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val[:, None]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MatrixFactorizationRouter(
        query_embedding_dim=x.shape[1],
        model_embedding_dim=args.model_embedding_dim,
        num_models=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model.forward_logits(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                loss = criterion(model.forward_logits(batch_x.to(device)), batch_y.to(device))
                val_loss += loss.item() * len(batch_x)
        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    val_probs, val_labels = probabilities(model, val_loader, device)
    metrics = evaluate(val_labels.astype(np.int64), val_probs)

    torch.save(
        {
            "model_state_dict": best_state,
            "query_embedding_dim": int(x.shape[1]),
            "model_embedding_dim": int(args.model_embedding_dim),
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "label_definition": "1 = Qwen-0.5B failure; route to stronger model",
        },
        output_dir / "best_model.pt",
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    results = {
        "dataset": str(Path(args.trajectory_data).resolve()),
        "n_total": int(len(x)),
        "n_train": int(len(x_train)),
        "n_validation": int(len(x_val)),
        "train_indices": [int(i) for i in idx_train],
        "validation_indices": [int(i) for i in idx_val],
        "device": str(device),
        "best_val_loss": float(best_val_loss),
        "metrics": metrics,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-data", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts/trajectory_mf"), type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--model-embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    results = train(args)
    metrics = results["metrics"]
    fixed = metrics["threshold_0_50"]
    best = metrics["best_f1_threshold"]
    print(f"Loaded {results['n_total']} trajectory records")
    print(f"Train/validation: {results['n_train']}/{results['n_validation']}")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f} | PR AUC: {metrics['pr_auc']:.4f}")
    print(f"@0.50: accuracy={fixed['accuracy']:.4f}, failure_recall={fixed['failure_recall']:.4f}, strong_rate={fixed['strong_route_rate']:.4f}")
    print(f"Best F1: threshold={best['threshold']:.2f}, F1={best['f1']:.4f}, failure_recall={best['failure_recall']:.4f}")
    print(f"Results written to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
