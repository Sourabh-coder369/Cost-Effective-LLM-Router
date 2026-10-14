"""Train the domain-aware router on explicit Parquet/embedding splits.

Unlike the original experiment script, this entry point has no hard-coded data
paths, so baseline and domain-aware runs can use exactly the same split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from domain_dataset import DomainRouterDataset
from domain_model import DomainMFRouter


def collate(batch):
    embeddings, labels, domains = zip(*batch)
    return torch.stack(embeddings), torch.stack(labels), torch.stack(domains)


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for embeddings, labels, domains in loader:
        embeddings, labels, domains = embeddings.to(device), labels.to(device).unsqueeze(1), domains.to(device)
        optimizer.zero_grad()
        loss = criterion(model.forward_logits(embeddings, domains), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


@torch.no_grad()
def predictions(model, loader, device):
    model.eval()
    probabilities, labels, domains = [], [], []
    for embeddings, batch_labels, batch_domains in loader:
        probabilities.append(model(embeddings.to(device), batch_domains.to(device)).cpu().numpy().reshape(-1))
        labels.append(batch_labels.numpy())
        domains.append(batch_domains.numpy())
    return np.concatenate(probabilities), np.concatenate(labels), np.concatenate(domains)


def metrics_at_cost(probabilities, labels, fraction):
    count = max(1, round(fraction * len(probabilities)))
    order = np.argsort(probabilities)[::-1]
    chosen = np.zeros(len(probabilities), dtype=bool)
    chosen[order[:count]] = True
    quality = np.where(chosen, labels, 1 - labels).mean()
    weak_quality = (1 - labels).mean()
    strong_quality = labels.mean()
    pgr = (quality - weak_quality) / (strong_quality - weak_quality)
    return {"strong_usage": float(chosen.mean()), "quality": float(quality), "pgr": float(pgr)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--val-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--model-embedding-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DomainRouterDataset(str(args.train_data), str(args.train_embeddings))
    val_ds = DomainRouterDataset(str(args.val_data), str(args.val_embeddings))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = DomainMFRouter(
        query_embedding_dim=train_ds.embeddings.shape[1],
        model_embedding_dim=args.model_embedding_dim,
    ).to(device)
    positives = int((train_ds.df["binary_label"] == 1).sum())
    negatives = int((train_ds.df["binary_label"] == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    best_loss = float("inf")
    history = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        val_probs, val_labels, _ = predictions(model, val_loader, device)
        val_logits = torch.logit(torch.tensor(val_probs).clamp(1e-6, 1 - 1e-6)).unsqueeze(1)
        val_loss = criterion(val_logits.to(device), torch.tensor(val_labels, device=device).unsqueeze(1)).item()
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "embedding_dim": train_ds.embeddings.shape[1],
                "model_embedding_dim": args.model_embedding_dim,
            }, args.output_dir / "best_model.pt")

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_probs, val_labels, val_domains = predictions(model, val_loader, device)
    result = {
        "num_validation_examples": int(len(val_labels)),
        "best_validation_loss": best_loss,
        "strong_quality": float(val_labels.mean()),
        "weak_quality": float((1 - val_labels).mean()),
        "accuracy_at_0_5": float(((val_probs >= 0.5).astype(int) == val_labels).mean()),
        "math_examples": int((val_domains == 1).sum()),
        "cost_levels": {str(int(level * 100)): metrics_at_cost(val_probs, val_labels, level) for level in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)},
        "history": history,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
