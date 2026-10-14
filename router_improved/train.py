"""Train and evaluate the improved soft-domain GPT-4/Llama router."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import PreferenceDataset
from model import SoftDomainRouter


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: PreferenceDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def get_logits(model: SoftDomainRouter, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, labels, weights = [], [], []
    with torch.inference_mode():
        for embeddings, batch_labels in loader:
            expert_logits, batch_weights = model.expert_logits_and_weights(embeddings.to(device))
            batch_logits = (expert_logits * batch_weights).sum(dim=-1)
            logits.append(batch_logits.cpu().numpy())
            labels.append(batch_labels.numpy())
            weights.append(batch_weights.cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels).astype(np.int64), np.concatenate(weights)


def fit_temperature(validation_logits: np.ndarray, validation_labels: np.ndarray, device: torch.device) -> float:
    """Fit one scalar temperature on validation data only."""
    logits = torch.tensor(validation_logits, dtype=torch.float32, device=device)
    labels = torch.tensor(validation_labels, dtype=torch.float32, device=device)
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = loss_fn(logits / log_temperature.exp().clamp(0.05, 10.0), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 10.0).cpu())


def cost_quality(probabilities: np.ndarray, labels: np.ndarray, fraction: float) -> dict[str, float]:
    count = max(1, int(round(fraction * len(probabilities))))
    route_strong = np.zeros(len(probabilities), dtype=bool)
    route_strong[np.argsort(probabilities)[::-1][:count]] = True
    quality = np.where(route_strong, labels, 1 - labels).mean()
    return {"strong_usage": float(route_strong.mean()), "quality": float(quality)}


def evaluate(probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray, temperature: float) -> dict:
    order = np.argsort(probabilities)[::-1]
    strong = labels.mean()
    weak = 1 - strong
    cost_levels = {}
    for percentage in range(10, 101, 10):
        item = cost_quality(probabilities, labels, percentage / 100)
        denominator = strong - weak
        item["pgr"] = float((item["quality"] - weak) / denominator) if abs(denominator) > 1e-8 else None
        cost_levels[str(percentage)] = item
    return {
        "n_examples": int(len(labels)),
        "strong_quality": float(strong),
        "weak_quality": float(weak),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy_at_0_5": float(((probabilities >= 0.5).astype(int) == labels).mean()),
        "temperature": temperature,
        "mean_expert_weight": weights.mean(axis=0).tolist(),
        "cost_levels": cost_levels,
        "ranked_probability_summary": {"p10": float(np.quantile(probabilities, 0.10)), "p50": float(np.median(probabilities)), "p90": float(np.quantile(probabilities, 0.90))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-data", type=Path, required=True)
        parser.add_argument(f"--{split}-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {name: PreferenceDataset(getattr(args, f"{name}_data"), getattr(args, f"{name}_embeddings")) for name in ("train", "val", "test")}
    if len({dataset.embedding_dim for dataset in datasets.values()}) != 1:
        raise ValueError("All embedding files must have the same embedding width")
    loaders = {name: make_loader(dataset, args.batch_size, name == "train") for name, dataset in datasets.items()}

    model = SoftDomainRouter(datasets["train"].embedding_dim, args.hidden_dim, args.num_experts).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, stale_epochs, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for embeddings, labels in loaders["train"]:
            optimiser.zero_grad()
            logits = model.forward_logits(embeddings.to(device)).squeeze(1)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item() * len(labels)
        val_logits, val_labels, _ = get_logits(model, loaders["val"], device)
        val_loss = float(nn.BCEWithLogitsLoss()(torch.tensor(val_logits), torch.tensor(val_labels, dtype=torch.float32)))
        record = {"epoch": epoch, "train_loss": total_loss / len(datasets["train"]), "val_loss": val_loss}
        history.append(record)
        print(f"Epoch {epoch:02d}: train_loss={record['train_loss']:.5f} val_loss={val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, stale_epochs = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    val_logits, val_labels, _ = get_logits(model, loaders["val"], device)
    temperature = fit_temperature(val_logits, val_labels, device)
    test_logits, test_labels, test_weights = get_logits(model, loaders["test"], device)
    test_probabilities = 1 / (1 + np.exp(-test_logits / temperature))
    results = evaluate(test_probabilities, test_labels, test_weights, temperature)
    results.update({"best_validation_loss": best_loss, "history": history, "device": str(device), "config": vars(args) | {"output_dir": str(args.output_dir)}})
    torch.save({"model_state_dict": best_state, "embedding_dim": datasets["train"].embedding_dim, "hidden_dim": args.hidden_dim, "num_experts": args.num_experts, "temperature": temperature}, args.output_dir / "best_model.pt")
    np.savez_compressed(args.output_dir / "test_predictions.npz", probabilities=test_probabilities, labels=test_labels, expert_weights=test_weights)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
