"""Sweep the widths of the two model-specific neural heads.

The shared BGE input and 256-wide query projection stay fixed. Each head
width is trained with three seeds, using validation APGR for checkpoint and
architecture selection. Only the selected configuration is tested.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_heads import ModelHeadRouter
from train_model_heads import budget_metrics, load_split, predict

ROOT = Path(__file__).resolve().parents[1]


def train_run(
    width: int,
    seed: int,
    train_x: torch.Tensor,
    train_y: np.ndarray,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict[str, torch.Tensor]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ModelHeadRouter(embedding_dim=train_x.shape[1], projection_dim=256, head_dim=width).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    head_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("weak_head") or name.startswith("strong_head")
    )
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
        loss_sum = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model.forward_logits(batch_x.to(device)).ravel()
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_y)
        validation = budget_metrics(predict(model, val_x, device), val_y)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_y), "validation_apgr": validation["apgr"]})
        if validation["apgr"] > best_apgr + 1e-6:
            best_apgr = validation["apgr"]
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
    summary = {
        "head_width": width,
        "seed": seed,
        "total_parameters": parameters,
        "two_head_parameters": head_parameters,
        "best_epoch": best_epoch,
        "validation_apgr": validation["apgr"],
        "validation_roc_auc": validation["roc_auc"],
        "validation_quality_at_50_percent_calls": validation["quality_at_50_percent_strong_calls"],
        "history": history,
    }
    return summary, best_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-widths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/model_head_size_sweep")
    args = parser.parse_args()
    if any(width < 1 for width in args.head_widths) or len(set(args.head_widths)) != len(args.head_widths):
        parser.error("head widths must be positive and distinct")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds or args.epochs < 1 or args.patience < 1:
        parser.error("seeds must be distinct and epochs/patience must be positive")

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, train_y = load_split("train")
    val_x, val_y = load_split("val")
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Train and validation embeddings have different dimensions")

    runs = []
    states = {}
    for width in args.head_widths:
        for seed in args.seeds:
            summary, state = train_run(width, seed, train_x, train_y, val_x, val_y, args, device)
            runs.append(summary)
            states[(width, seed)] = state
            print(
                f"head={width:>3} seed={seed} best_epoch={summary['best_epoch']:>2} "
                f"val_apgr={summary['validation_apgr']:.5f} "
                f"val_auc={summary['validation_roc_auc']:.5f}",
                flush=True,
            )

    widths = []
    for width in args.head_widths:
        group = [run for run in runs if run["head_width"] == width]
        widths.append({
            "head_width": width,
            "two_head_parameters": group[0]["two_head_parameters"],
            "total_parameters": group[0]["total_parameters"],
            "mean_validation_apgr": float(np.mean([run["validation_apgr"] for run in group])),
            "std_validation_apgr": float(np.std([run["validation_apgr"] for run in group], ddof=1)) if len(group) > 1 else None,
            "mean_validation_roc_auc": float(np.mean([run["validation_roc_auc"] for run in group])),
            "mean_validation_quality_at_50_percent_calls": float(np.mean([run["validation_quality_at_50_percent_calls"] for run in group])),
        })
    selected_width = sorted(widths, key=lambda row: (-row["mean_validation_apgr"], row["head_width"]))[0]["head_width"]
    selected_runs = sorted(
        (run for run in runs if run["head_width"] == selected_width),
        key=lambda row: (row["validation_apgr"], row["seed"]),
    )
    selected_run = selected_runs[len(selected_runs) // 2]  # representative median seed, not the luckiest seed
    selected_seed = selected_run["seed"]
    selected_model = ModelHeadRouter(embedding_dim=train_x.shape[1], projection_dim=256, head_dim=selected_width).to(device)
    selected_model.load_state_dict(states[(selected_width, selected_seed)])

    # Test is loaded only after the width and representative run are selected.
    test_x, test_y = load_split("test")
    if test_x.shape[1] != train_x.shape[1]:
        raise ValueError("Test embeddings have a different dimension")
    test = budget_metrics(predict(selected_model, test_x, device), test_y)

    result = {
        "fixed_projection_dim": 256,
        "head_widths": args.head_widths,
        "seeds": args.seeds,
        "training": {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "learning_rate": args.learning_rate},
        "selection_rule": "highest mean validation APGR across seeds; then median-validation-APGR seed",
        "width_summary": widths,
        "runs": runs,
        "selected": {
            "head_width": selected_width,
            "seed": selected_seed,
            "validation_apgr": selected_run["validation_apgr"],
            "validation_roc_auc": selected_run["validation_roc_auc"],
            "test": test,
        },
        "caveat": "The test split has been used in prior experiments; this is an exploratory architecture comparison.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": states[(selected_width, selected_seed)], "embedding_dim": train_x.shape[1], "projection_dim": 256, "head_dim": selected_width},
        args.output_dir / "selected_model.pt",
    )
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("WIDTH SUMMARY", flush=True)
    for row in widths:
        print(
            f"head={row['head_width']:>3} mean_val_apgr={row['mean_validation_apgr']:.5f} "
            f"std={row['std_validation_apgr']:.5f} params={row['total_parameters']}",
            flush=True,
        )
    print(
        f"SELECTED head={selected_width} seed={selected_seed} "
        f"test_apgr={test['apgr']:.5f} test_auc={test['roc_auc']:.5f} "
        f"test_quality_50={test['quality_at_50_percent_strong_calls']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
