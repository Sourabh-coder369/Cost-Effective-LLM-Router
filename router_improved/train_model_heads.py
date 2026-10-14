"""Train small GPT/Llama scoring heads on frozen BGE prompt embeddings.

Uses winner-only labels, so only the score difference is supervised. Chooses
the checkpoint by validation APGR, then evaluates the held-out test set once.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_heads import ModelHeadRouter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced"
EMBEDDINGS = ROOT / "data/original_gpt4_llama7b/bge_embeddings"


def load_split(name: str) -> tuple[torch.Tensor, np.ndarray]:
    frame = pd.read_parquet(DATA / f"router_{name}.parquet")
    vectors = torch.load(EMBEDDINGS / f"router_{name}_bge_large_embeddings.pt", map_location="cpu", weights_only=True)
    labels = frame["binary_label"].to_numpy(dtype=np.int64)
    if vectors.ndim != 2 or len(vectors) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{name}: embeddings and binary labels are incompatible")
    return vectors.float(), labels


def budget_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    n = len(labels)
    order = np.argsort(probabilities)[::-1]
    weak_quality = float((1 - labels).mean())
    strong_quality = float(labels.mean())
    gap = strong_quality - weak_quality
    levels = {}
    for percent in range(10, 101, 10):
        count = round(percent * n / 100)
        chosen = np.zeros(n, dtype=bool)
        chosen[order[:count]] = True
        routed_quality = float(np.where(chosen, labels, 1 - labels).mean())
        levels[str(percent)] = {
            "strong_call_fraction": count / n,
            "pairwise_quality": routed_quality,
            "pgr": (routed_quality - weak_quality) / gap,
        }
    return {
        "n": n,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "apgr": float(np.mean([value["pgr"] for value in levels.values()])),
        "quality_at_50_percent_strong_calls": levels["50"]["pairwise_quality"],
        "cost_levels": levels,
    }


@torch.inference_mode()
def predict(model: ModelHeadRouter, vectors: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    batches = [model(batch.to(device)).cpu().numpy().ravel() for batch in vectors.split(512)]
    return np.concatenate(batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/model_heads_bge_256")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Train and validation embeddings have different dimensions")

    model = ModelHeadRouter(embedding_dim=train_x.shape[1]).to(device)
    # Match the class weighting used by the saved baseline MF training run.
    pos_weight = torch.tensor([(len(train_y) - train_y.sum()) / train_y.sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(train_x, torch.from_numpy(train_y.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_apgr = -float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model.forward_logits(batch_x.to(device)).ravel()
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        val_metrics = budget_metrics(predict(model, val_x, device), val_y)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y), "validation_apgr": val_metrics["apgr"]})
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.4f} val_apgr={val_metrics['apgr']:.4f}", flush=True)
        if val_metrics["apgr"] > best_apgr + 1e-6:
            best_apgr = val_metrics["apgr"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    validation = budget_metrics(predict(model, val_x, device), val_y)

    # Test data are read only after all training and checkpoint selection end.
    test_x, test_y = load_split("test")
    if test_x.shape[1] != train_x.shape[1]:
        raise ValueError("Test embeddings have a different dimension")
    test = budget_metrics(predict(model, test_x, device), test_y)
    result = {
        "architecture": "BGE embedding -> shared 1024-to-256 projection -> separate 256-to-32-to-1 heads -> strong minus weak",
        "label_definition": "1 = GPT-4 preferred to Llama-2-7B; two head scores are not independently calibrated",
        "selection_metric": "validation_apgr",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_apgr": best_apgr,
        "train_examples": len(train_y),
        "validation": validation,
        "test": test,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "embedding_dim": train_x.shape[1], "projection_dim": 256, "head_dim": 32}, args.output_dir / "best_model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "validation": {k: validation[k] for k in ("roc_auc", "apgr", "quality_at_50_percent_strong_calls")}, "test": {k: test[k] for k in ("roc_auc", "apgr", "quality_at_50_percent_strong_calls")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
