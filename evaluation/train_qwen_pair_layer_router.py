"""Train a weak-versus-strong router from paired Qwen layer embedding shards.

Input comes from generate_qwen_pair_math_layers.py. Features are adjacent-layer
cosine similarities from both models, for both mean-pooled and last-token
representations. The target is route_to_strong.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/kaggle/working/qwen_pair_remaining_math_layers"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/qwen_pair_layer_router"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def adjacent_cosines(layers: torch.Tensor) -> np.ndarray:
    layers = layers.float()
    return functional.cosine_similarity(layers[:, :-1], layers[:, 1:], dim=-1).numpy()


def load_features(input_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    shard_paths = sorted(input_dir.glob("pair_embeddings_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No pair embedding shards found in {input_dir}")
    feature_parts, label_parts = [], []
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        features = np.concatenate(
            (
                adjacent_cosines(shard["weak_layer_last_token_embeddings"]),
                adjacent_cosines(shard["strong_layer_last_token_embeddings"]),
                adjacent_cosines(shard["weak_layer_embeddings"]),
                adjacent_cosines(shard["strong_layer_embeddings"]),
            ),
            axis=1,
        )
        feature_parts.append(features.astype(np.float32))
        label_parts.append(shard["route_to_strong"].numpy().astype(np.float32))
    return np.concatenate(feature_parts), np.concatenate(label_parts)


def standardize(train: np.ndarray, *other: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, [(value - mean) / std for value in other], mean, std


class PairLayerRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels[:, None]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predictions(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for batch, _ in loader(features, np.zeros(len(features), dtype=np.float32), batch_size, False):
            values.append(torch.sigmoid(model(batch.to(device))).squeeze(1).cpu().numpy())
    return np.concatenate(values)


def report(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.astype(np.int64),
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "n": int(len(labels)),
        "strong_route_rate": float(labels.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_strong": float(precision),
        "recall_strong": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    features, labels = load_features(args.input_dir)
    if len(np.unique(labels)) != 2:
        raise ValueError("The pair dataset must contain both strong-route and weak-route examples.")

    indices = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=args.seed,
    )
    validation_idx, test_idx = train_test_split(
        rest_idx,
        test_size=0.50,
        stratify=labels[rest_idx],
        random_state=args.seed,
    )
    train_x, transformed, feature_mean, feature_std = standardize(
        features[train_idx],
        features[validation_idx],
        features[test_idx],
    )
    validation_x, test_x = transformed
    train_y, validation_y, test_y = labels[train_idx], labels[validation_idx], labels[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairLayerRouter(train_x.shape[1], args.hidden_dim).to(device)
    negatives = float((train_y == 0).sum())
    positives = float((train_y == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_loss, best_state, wait = float("inf"), None, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in loader(train_x, train_y, args.batch_size, True):
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_y)

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for batch_x, batch_y in loader(validation_x, validation_y, args.batch_size, False):
                validation_loss += loss_fn(model(batch_x.to(device)), batch_y.to(device)).item() * len(batch_x)
        validation_loss /= len(validation_y)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_probabilities = predictions(model, validation_x, device, args.batch_size)
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = float(max(
        thresholds,
        key=lambda value: report(validation_y, validation_probabilities, float(value))["f1"],
    ))
    test_probabilities = predictions(model, test_x, device, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "feature_mean": feature_mean.squeeze(0),
            "feature_std": feature_std.squeeze(0),
            "threshold": threshold,
            "feature_definition": "Adjacent-layer cosine similarities for weak and strong Qwen, using last-token and mean-pooled vectors.",
            "label_definition": "1 means Qwen 1.7B correct and Qwen 0.5B incorrect.",
        },
        args.output_dir / "router.pt",
    )
    results = {
        "input_dir": str(args.input_dir),
        "n_examples": int(len(labels)),
        "n_features": int(train_x.shape[1]),
        "positive_rate": float(labels.mean()),
        "split_sizes": {"train": int(len(train_y)), "validation": int(len(validation_y)), "test": int(len(test_y))},
        "best_validation_loss": best_loss,
        "selected_threshold": threshold,
        "test_metrics": report(test_y, test_probabilities, threshold),
        "history": history,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["test_metrics"], indent=2))
    print(f"Saved router checkpoint and results to {args.output_dir}")


if __name__ == "__main__":
    main()
