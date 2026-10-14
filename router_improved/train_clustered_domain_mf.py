"""Train a clustered-domain MF router and select it by validation APGR.

This extends the two hard-coded domains used by the original Domain-MF model.
Semantic clusters are fitted only on training embeddings; the existing math flag
is retained as a second capability signal.  Candidate cluster counts are chosen
on validation APGR, then the selected checkpoint is evaluated once on test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402
from domain_specific.domain_model import DomainMFRouter  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apgr(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean PGR at 10%, ..., 100% strong-model use."""
    strong, weak = labels.mean(), (1 - labels).mean()
    gap = strong - weak
    order = np.argsort(probabilities)[::-1]
    values = []
    for pct in range(10, 101, 10):
        strong_route = np.zeros(len(labels), dtype=bool)
        strong_route[order[:round(len(labels) * pct / 100)]] = True
        quality = np.where(strong_route, labels, 1 - labels).mean()
        values.append((quality - weak) / gap)
    return float(np.mean(values))


def evaluate(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    strong, weak = labels.mean(), (1 - labels).mean()
    gap, order = strong - weak, np.argsort(probabilities)[::-1]
    levels = {}
    for pct in range(10, 101, 10):
        chosen = np.zeros(len(labels), dtype=bool)
        chosen[order[:round(len(labels) * pct / 100)]] = True
        quality = np.where(chosen, labels, 1 - labels).mean()
        levels[str(pct)] = {
            "strong_usage": float(chosen.mean()),
            "quality": float(quality),
            "pgr": float((quality - weak) / gap),
        }
    return {
        "n_examples": int(len(labels)),
        "strong_quality": float(strong),
        "weak_quality": float(weak),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy_at_0_5": float(((probabilities >= 0.5).astype(int) == labels).mean()),
        "apgr": float(np.mean([item["pgr"] for item in levels.values()])),
        "cost_levels": levels,
    }


def predict(model: DomainMFRouter, features: np.ndarray, domains: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(domains)), batch_size=512, shuffle=False)
    values = []
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_d in loader:
            values.append(model(batch_x.to(device), batch_d.to(device)).squeeze(1).cpu().numpy())
    return np.concatenate(values)


def train_one(
    train_x: np.ndarray, train_y: np.ndarray, train_domains: np.ndarray,
    val_x: np.ndarray, val_y: np.ndarray, val_domains: np.ndarray,
    num_domains: int, args: argparse.Namespace, device: torch.device,
) -> tuple[dict, dict]:
    model = DomainMFRouter(
        query_embedding_dim=train_x.shape[1], model_embedding_dim=args.model_embedding_dim,
        num_models=2, num_domains=num_domains,
    ).to(device)
    pos_weight = torch.tensor([(len(train_y) - train_y.sum()) / train_y.sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)), torch.from_numpy(train_domains)),
        batch_size=args.batch_size, shuffle=True,
    )
    best = {"apgr": -float("inf"), "state": None, "epoch": 0}
    history = []
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y, batch_d in loader:
            optimiser.zero_grad()
            loss = criterion(model.forward_logits(batch_x.to(device), batch_d.to(device)).squeeze(1), batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item() * len(batch_y)
        val_probs = predict(model, val_x, val_domains, device)
        val_apgr = apgr(val_probs, val_y)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y), "validation_apgr": val_apgr})
        if val_apgr > best["apgr"] + 1e-6:
            best = {"apgr": val_apgr, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    assert best["state"] is not None
    model.load_state_dict(best["state"])
    return best, {"history": history, "model": model}


def math_flags(dataset: DomainRouterDataset) -> np.ndarray:
    return dataset.domain_ids.numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-data", required=True, type=Path)
        parser.add_argument(f"--{split}-embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/clustered_domain_mf"))
    parser.add_argument("--clusters", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--model-embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        name: DomainRouterDataset(str(getattr(args, f"{name}_data")), str(getattr(args, f"{name}_embeddings")))
        for name in ("train", "val", "test")
    }
    train_x = datasets["train"].embeddings.numpy().astype(np.float32)
    val_x = datasets["val"].embeddings.numpy().astype(np.float32)
    test_x = datasets["test"].embeddings.numpy().astype(np.float32)
    train_y = datasets["train"].df["binary_label"].to_numpy(dtype=np.int64)
    val_y = datasets["val"].df["binary_label"].to_numpy(dtype=np.int64)
    test_y = datasets["test"].df["binary_label"].to_numpy(dtype=np.int64)
    candidates, selected = {}, None
    for clusters in args.clusters:
        kmeans = MiniBatchKMeans(n_clusters=clusters, batch_size=2048, n_init=5, random_state=args.seed)
        train_cluster = kmeans.fit_predict(train_x).astype(np.int64)
        val_cluster = kmeans.predict(val_x).astype(np.int64)
        test_cluster = kmeans.predict(test_x).astype(np.int64)
        # Cluster × math-flag: retains the original domain signal while adding
        # broad semantic capability groups discovered from training prompts.
        train_domains = train_cluster * 2 + math_flags(datasets["train"])
        val_domains = val_cluster * 2 + math_flags(datasets["val"])
        test_domains = test_cluster * 2 + math_flags(datasets["test"])
        best, run = train_one(train_x, train_y, train_domains, val_x, val_y, val_domains, clusters * 2, args, device)
        key = f"clusters_{clusters}_with_math"
        candidates[key] = {"validation_apgr": best["apgr"], "best_epoch": best["epoch"], "history": run["history"]}
        print(f"{key}: validation APGR={best['apgr']:.4f} at epoch {best['epoch']}", flush=True)
        if selected is None or best["apgr"] > selected["best"]["apgr"]:
            selected = {"name": key, "clusters": clusters, "best": best, "model": run["model"], "test_domains": test_domains}
    assert selected is not None
    test_probs = predict(selected["model"], test_x, selected["test_domains"], device)
    result = {
        "selected_candidate": selected["name"],
        "selection_metric": "validation_apgr",
        "candidates": candidates,
        "test": evaluate(test_probs, test_y),
    }
    torch.save(
        {"model_state_dict": selected["best"]["state"], "num_domains": selected["clusters"] * 2,
         "embedding_dim": int(train_x.shape[1]), "selected_candidate": selected["name"]},
        args.output_dir / "best_model.pt",
    )
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
