"""Train/evaluate baseline and Qwen-domain MF using BGE Large embeddings."""

from __future__ import annotations

import json
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from router.baseline.model import MatrixFactorizationRouter
from router.domain_specific.domain_model import DomainMFRouter

DATA = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced"
EMBEDDINGS = ROOT / "data/original_gpt4_llama7b/bge_embeddings"
LABELS = ROOT / "artifacts/kaggle_qwen_labels/extracted/qwen_domain_labels"
OUT = ROOT / "artifacts/qwen_domain_mf_bge_large"
SEED = 42

SPLITS = {
    "train": "router_train_bge_large_embeddings.pt",
    "val": "router_val_bge_large_embeddings.pt",
    "test": "router_test_bge_large_embeddings.pt",
}
LABEL_DIRS = {"train": "00_router_train", "val": "01_router_val", "test": "02_router_test"}


def apgr(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean fraction of the always-strong/always-weak quality gap recovered at 10..100% strong calls."""
    ordering = np.argsort(probabilities)[::-1]
    strong_quality, weak_quality = labels.mean(), (1 - labels).mean()
    if np.isclose(strong_quality, weak_quality):
        return 0.0
    scores = []
    for percent in range(10, 101, 10):
        selected = np.zeros(len(labels), dtype=bool)
        selected[ordering[: max(1, round(len(labels) * percent / 100))]] = True
        quality = np.where(selected, labels, 1 - labels).mean()
        scores.append((quality - weak_quality) / (strong_quality - weak_quality))
    return float(np.mean(scores))


def score(model, embeddings: torch.Tensor, domains: torch.Tensor, device: torch.device,
          batch_size: int = 512) -> np.ndarray:
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            x = embeddings[start:start + batch_size].to(device)
            d = domains[start:start + batch_size].to(device)
            if isinstance(model, DomainMFRouter):
                probabilities = model(x, d)
            else:
                probabilities = model(x)
            result.extend(probabilities.flatten().cpu().numpy())
    return np.asarray(result)


def budget_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    ordering = np.argsort(probabilities)[::-1]
    weak_quality, strong_quality = (1 - labels).mean(), labels.mean()
    rows = {}
    for percent in range(10, 101, 10):
        selected = np.zeros(len(labels), dtype=bool)
        selected[ordering[: max(1, round(len(labels) * percent / 100))]] = True
        quality = float(np.where(selected, labels, 1 - labels).mean())
        recovery = ((quality - weak_quality) / (strong_quality - weak_quality)
                    if not np.isclose(strong_quality, weak_quality) else 0.0)
        rows[str(percent)] = {
            "strong_call_budget_pct": percent, "actual_strong_usage_pct": float(selected.mean() * 100),
            "routing_quality_pct": float(quality * 100),
            "strong_always_quality_pct": float(strong_quality * 100),
            "weak_always_quality_pct": float(weak_quality * 100),
            "performance_gap_recovered_pct": float(recovery * 100),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-embedding-dim", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.model_embedding_dim <= 0:
        parser.error("--model-embedding-dim must be positive")
    output_dir = args.output_dir.resolve()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    original: dict[str, pd.DataFrame] = {}
    labeled: dict[str, pd.DataFrame] = {}
    embeddings: dict[str, torch.Tensor] = {}
    labels: dict[str, np.ndarray] = {}
    domains: dict[str, torch.Tensor] = {}
    selection = {}
    source_rows: dict[str, np.ndarray] = {}

    for split, filename in SPLITS.items():
        frame = pd.read_parquet(DATA / f"router_{split}.parquet")
        annotations = pd.read_parquet(LABELS / LABEL_DIRS[split] / "labeled.parquet")
        if len(frame) != len(annotations) or frame["prompt"].fillna("").tolist() != annotations["prompt"].fillna("").tolist():
            raise ValueError(f"Qwen output is not row-aligned with {split} original data")
        vector = torch.load(EMBEDDINGS / filename, map_location="cpu", weights_only=True)
        if len(vector) != len(frame) or vector.ndim != 2 or vector.shape[1] != 1024:
            raise ValueError(f"Expected {split} BGE Large embeddings [N,1024], got {tuple(vector.shape)}")
        mask = annotations["qwen_label_status"].eq("ok").to_numpy()
        if not annotations.loc[mask, "qwen_domain"].isin(["math", "general"]).all():
            raise ValueError(f"Non-binary Qwen label marked ok in {split}")
        original[split] = frame.loc[mask].reset_index(drop=True)
        labeled[split] = annotations.loc[mask].reset_index(drop=True)
        source_rows[split] = np.flatnonzero(mask)
        embeddings[split] = vector[torch.tensor(mask.copy(), dtype=torch.bool)].float().contiguous()
        labels[split] = original[split]["binary_label"].to_numpy(dtype=np.float32)
        domains[split] = torch.tensor(labeled[split]["qwen_domain_id"].to_numpy(), dtype=torch.long)
        selection[split] = {
            "original_rows": len(frame), "accepted_qwen_rows": int(mask.sum()),
            "excluded_unresolved_rows": int((~mask).sum()),
            "math_rows": int((domains[split] == 1).sum()), "general_rows": int((domains[split] == 0).sum()),
            "strong_win_rate": float(labels[split].mean()),
        }
        print(f"{split}: {len(frame):,} original, {mask.sum():,} accepted; "
              f"{selection[split]['math_rows']:,} math / {selection[split]['general_rows']:,} general", flush=True)

    dim = int(embeddings["train"].shape[1])
    model_dim = args.model_embedding_dim
    models = {
        "baseline_mf": MatrixFactorizationRouter(query_embedding_dim=dim, model_embedding_dim=model_dim).to(device),
        "qwen_domain_mf": DomainMFRouter(query_embedding_dim=dim, model_embedding_dim=model_dim,
                                          num_models=2, num_domains=2).to(device),
    }
    generator = torch.Generator().manual_seed(SEED)
    histories, selected_epochs = {}, {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        is_domain = isinstance(model, DomainMFRouter)
        tensors = (embeddings["train"], torch.from_numpy(labels["train"]), domains["train"]) if is_domain else (
            embeddings["train"], torch.from_numpy(labels["train"]))
        train_loader = DataLoader(TensorDataset(*tensors), batch_size=128, shuffle=True,
                                  generator=generator, num_workers=0)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        best_apgr, best_epoch, stale, best_state, history = -float("inf"), 0, 0, None, []
        print(f"Training {model_name} with BGE Large (up to 30 epochs)...", flush=True)
        for epoch in range(1, 31):
            model.train()
            loss_sum = 0.0
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                d = batch[2].to(device) if is_domain else None
                optimizer.zero_grad(set_to_none=True)
                logits = model.forward_logits(x, d) if is_domain else model.forward_logits(x)
                loss = criterion(logits.squeeze(1), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(y)
            val_prob = score(model, embeddings["val"], domains["val"], device)
            val_metric = apgr(val_prob, labels["val"])
            row = {"epoch": epoch, "train_bce": loss_sum / len(labels["train"]),
                   "validation_apgr": val_metric,
                   "validation_roc_auc": float(roc_auc_score(labels["val"], val_prob))}
            history.append(row)
            if val_metric > best_apgr + 1e-6:
                best_apgr, best_epoch, stale = val_metric, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({"epoch": epoch, "model_state_dict": best_state, "model": model_name,
                            "query_embedding_dim": dim, "model_embedding_dim": model_dim, "seed": SEED,
                            "embedding_model": "BAAI/bge-large-en-v1.5", "selection": "validation APGR",
                            "qwen_model": "Qwen3-1.7B",
                            "qwen_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"},
                           output_dir / f"{model_name}_best_model.pt")
            else:
                stale += 1
            print(f"{model_name} epoch {epoch:02d}: train BCE {row['train_bce']:.4f}, "
                  f"val APGR {val_metric:.4f}, val AUC {row['validation_roc_auc']:.4f}"
                  f"{' *' if epoch == best_epoch else ''}", flush=True)
            if stale >= 5:
                print(f"Early stopping {model_name} after {epoch} epochs; selected epoch {best_epoch}.", flush=True)
                break
        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {model_name}")
        model.load_state_dict(best_state)
        model.eval()
        histories[model_name] = history
        selected_epochs[model_name] = {"best_epoch": best_epoch, "best_validation_apgr": best_apgr,
                                       "epochs_run": len(history)}

    results = {
        "models": ["MatrixFactorizationRouter baseline", "Qwen-labeled DomainMFRouter"],
        "device": str(device), "seed": SEED,
        "training": {"selected_models": selected_epochs, "history": histories,
                     "embedding_model": "BAAI/bge-large-en-v1.5", "embedding_dim": dim,
                     "model_embedding_dim": model_dim,
                     "batch_size": 128, "learning_rate": 3e-4, "early_stopping_patience": 5},
        "data_selection": selection,
        "test": {},
    }
    predictions = {}
    for name, router in models.items():
        probability = score(router, embeddings["test"], domains["test"], device)
        predictions[name] = probability
        pred = (probability >= 0.5).astype(np.int64)
        results["test"][name] = {
            "roc_auc": float(roc_auc_score(labels["test"], probability)),
            "accuracy_at_0_5": float((pred == labels["test"]).mean()),
            "strong_model_win_rate": float(labels["test"].mean()),
            "apgr": apgr(probability, labels["test"]),
            "routing_quality_at_0_5_pct": float(np.where(pred, labels["test"], 1 - labels["test"]).mean() * 100),
            "strong_model_usage_at_0_5_pct": float(pred.mean() * 100),
            "strong_call_budgets": budget_metrics(probability, labels["test"]),
        }
        results["test"][name]["per_qwen_domain"] = {}
        for domain_id, domain_name in ((1, "math"), (0, "general")):
            mask = domains["test"].numpy() == domain_id
            domain_pred = (probability[mask] >= 0.5).astype(np.int64)
            results["test"][name]["per_qwen_domain"][domain_name] = {
                "rows": int(mask.sum()),
                "strong_win_rate": float(labels["test"][mask].mean()),
                "accuracy_at_0_5": float((domain_pred == labels["test"][mask]).mean()),
                "routing_quality_at_0_5_pct": float(np.where(domain_pred, labels["test"][mask],
                                                               1 - labels["test"][mask]).mean() * 100),
                "strong_model_usage_at_0_5_pct": float(domain_pred.mean() * 100),
            }
    pd.DataFrame({"source_row": source_rows["test"],
                      "prompt": original["test"]["prompt"], "qwen_domain": labeled["test"]["qwen_domain"],
                      "binary_label": labels["test"], "baseline_probability": predictions["baseline_mf"],
                      "domain_mf_probability": predictions["qwen_domain_mf"],
                      "qwen_domain_id": domains["test"].numpy()}).to_parquet(output_dir / "test_predictions.parquet", index=False)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nTest-set comparison (same accepted Qwen-labeled rows):")
    print(json.dumps({name: {key: value for key, value in metrics.items() if key != "strong_call_budgets"}
                      for name, metrics in results["test"].items()}, indent=2))
    print(f"\nSaved checkpoint, metrics and test predictions in {output_dir}")


if __name__ == "__main__":
    main()
