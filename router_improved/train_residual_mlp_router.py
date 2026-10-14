"""Train a validation-selected residual MLP router with semantic and length features."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


class ResidualMLPRouter(nn.Module):
    """Embedding encoder plus a compact branch for explicit routing features."""

    def __init__(self, embedding_dim: int, structured_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.embedding_encoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = ResidualBlock(256, dropout)
        self.structured_encoder = nn.Sequential(
            nn.LayerNorm(structured_dim),
            nn.Linear(structured_dim, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(288),
            nn.Linear(288, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward_logits(self, embeddings: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        semantic = torch.nn.functional.gelu(self.residual(self.embedding_encoder(embeddings)))
        features = self.structured_encoder(structured)
        return self.head(torch.cat((semantic, features), dim=1))

    def forward(self, embeddings: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(embeddings, structured))


def apgr(probabilities: np.ndarray, labels: np.ndarray) -> float:
    strong, weak = labels.mean(), (1 - labels).mean()
    ordering = np.argsort(probabilities)[::-1]
    values = []
    for pct in range(10, 101, 10):
        chosen = np.zeros(len(labels), dtype=bool)
        chosen[ordering[:round(len(labels) * pct / 100)]] = True
        values.append((np.where(chosen, labels, 1 - labels).mean() - weak) / (strong - weak))
    return float(np.mean(values))


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    strong, weak = labels.mean(), (1 - labels).mean()
    ordering = np.argsort(probabilities)[::-1]
    levels = {}
    for pct in range(10, 101, 10):
        chosen = np.zeros(len(labels), dtype=bool)
        chosen[ordering[:round(len(labels) * pct / 100)]] = True
        quality = np.where(chosen, labels, 1 - labels).mean()
        levels[str(pct)] = {"strong_usage": float(chosen.mean()), "quality": float(quality), "pgr": float((quality - weak) / (strong - weak))}
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy_at_0_5": float(((probabilities >= 0.5).astype(int) == labels).mean()),
        "apgr": float(np.mean([row["pgr"] for row in levels.values()])),
        "cost_levels": levels,
    }


def predict(model: ResidualMLPRouter, embeddings: np.ndarray, structured: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(embeddings), torch.from_numpy(structured)), batch_size=512)
    values = []
    model.eval()
    with torch.inference_mode():
        for batch_embedding, batch_structured in loader:
            values.append(model(batch_embedding.to(device), batch_structured.to(device)).squeeze(1).cpu().numpy())
    return np.concatenate(values)


def log_word_count(dataset: DomainRouterDataset) -> np.ndarray:
    return np.log1p(dataset.df["prompt"].map(lambda value: len(str(value).split())).to_numpy(dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-data", required=True, type=Path)
        parser.add_argument(f"--{split}-embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/residual_mlp_router"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        name: DomainRouterDataset(str(getattr(args, f"{name}_data")), str(getattr(args, f"{name}_embeddings")))
        for name in ("train", "val", "test")
    }
    embeddings = {name: data.embeddings.numpy().astype(np.float32) for name, data in datasets.items()}
    labels = {name: data.df["binary_label"].to_numpy(dtype=np.int64) for name, data in datasets.items()}
    math = {name: data.domain_ids.numpy().astype(np.int64) for name, data in datasets.items()}
    log_length = {name: log_word_count(data) for name, data in datasets.items()}
    length_mean, length_std = float(log_length["train"].mean()), float(log_length["train"].std())
    z_length = {name: ((value - length_mean) / max(length_std, 1e-6)).astype(np.float32) for name, value in log_length.items()}
    edges = np.quantile(log_length["train"], np.linspace(0, 1, 6))
    internal_edges = np.unique(edges[1:-1])
    length_bucket = {name: np.digitize(value, internal_edges, right=True) for name, value in log_length.items()}
    bucket_count = len(internal_edges) + 1
    one_hot = {name: np.eye(bucket_count, dtype=np.float32)[bucket] for name, bucket in length_bucket.items()}

    # Gate uses training inputs only and is a feature rather than a target at
    # inference; no validation/test labels enter the gate or feature fitting.
    detector = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=args.seed)
    detector.fit(embeddings["train"], math["train"])
    math_gate = {name: detector.predict_proba(value)[:, 1].astype(np.float32) for name, value in embeddings.items()}
    structured = {
        name: np.column_stack((
            z_length[name],
            math_gate[name],
            one_hot[name],
            one_hot[name] * math_gate[name][:, None],
        )).astype(np.float32)
        for name in embeddings
    }
    detector_metrics = {
        "validation_math_log_loss": float(log_loss(math["val"], math_gate["val"])),
        "validation_math_roc_auc": float(roc_auc_score(math["val"], math_gate["val"])),
    }

    model = ResidualMLPRouter(embeddings["train"].shape[1], structured["train"].shape[1]).to(device)
    pos_weight = torch.tensor([(len(labels["train"]) - labels["train"].sum()) / labels["train"].sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(embeddings["train"]), torch.from_numpy(structured["train"]), torch.from_numpy(labels["train"].astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_apgr, best_state, best_epoch, stale, history = -float("inf"), None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_embedding, batch_structured, batch_label in loader:
            optimiser.zero_grad()
            loss = criterion(model.forward_logits(batch_embedding.to(device), batch_structured.to(device)).squeeze(1), batch_label.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            loss_sum += loss.item() * len(batch_label)
        val_apgr = apgr(predict(model, embeddings["val"], structured["val"], device), labels["val"])
        history.append({"epoch": epoch, "train_loss": loss_sum / len(labels["train"]), "validation_apgr": val_apgr})
        if val_apgr > best_apgr + 1e-6:
            best_apgr, best_epoch, stale = val_apgr, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    test_probabilities = predict(model, embeddings["test"], structured["test"], device)
    result = {
        "best_validation_apgr": best_apgr,
        "best_epoch": best_epoch,
        "architecture": {"embedding_branch": "LayerNorm -> 256 -> residual block", "structured_features": "log prompt length, math probability, five length buckets, math-by-bucket interactions"},
        "math_detector": detector_metrics,
        "history": history,
        "test": metrics(test_probabilities, labels["test"]),
    }
    torch.save({"model_state_dict": best_state, "detector": detector, "embedding_dim": embeddings["train"].shape[1], "structured_dim": structured["train"].shape[1], "length_mean": length_mean, "length_std": length_std, "length_edges": edges}, args.output_dir / "best_model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
