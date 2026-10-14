"""Train a matrix-factorization failure router from saved Qwen layer shards.

The dataset generator stores a representation for every Qwen hidden-state
layer. This script converts those vectors to adjacent-layer cosine similarities
and trains the repository's MatrixFactorizationRouter with the routing label:

    strong_model_wins = 1  if Qwen answered incorrectly
    strong_model_wins = 0  if Qwen answered correctly

It intentionally reads labels from the binary ``.pt`` shards, not the JSONL
metadata, so a run interrupted while writing a metadata line can still be used.

Example:
    python evaluation/train_qwen_layer_cosine_mf.py \
      --input-dir data/qwen_math_layers_full_v2 \
      --input-dir data/qwen_math_other \
      --output-dir artifacts/qwen_layer_cosine_mf
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "router" / "baseline"))
from model import MatrixFactorizationRouter  # noqa: E402


@dataclass
class CosineDataset:
    last: np.ndarray
    mean: np.ndarray
    labels: np.ndarray
    sources: np.ndarray
    record_ids: list[str]
    skipped_duplicates: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_from_id(record_id: str) -> str:
    return record_id.split(":", 1)[0] if ":" in record_id else "unknown"


def adjacent_cosines(layer_embeddings: torch.Tensor) -> torch.Tensor:
    """Return [batch, num_layers - 1] adjacent layer cosine similarities."""
    vectors = layer_embeddings.float()
    return functional.cosine_similarity(vectors[:, :-1, :], vectors[:, 1:, :], dim=-1)


def load_dataset(input_dirs: Iterable[Path]) -> CosineDataset:
    last_features: list[np.ndarray] = []
    mean_features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sources: list[str] = []
    record_ids: list[str] = []
    seen_ids: set[str] = set()
    skipped_duplicates = 0

    for input_dir in input_dirs:
        shards = sorted(input_dir.glob("embeddings_*.pt"))
        if not shards:
            raise FileNotFoundError(f"No embedding shards found in {input_dir}")
        print(f"Loading {len(shards)} shards from {input_dir}", flush=True)
        for shard_path in shards:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            ids = [str(item) for item in payload["record_ids"]]
            shard_labels = payload["strong_model_wins"].cpu().numpy().astype(np.int64)
            if len(ids) != len(shard_labels):
                raise ValueError(f"Mismatched record/label count in {shard_path}")

            keep = np.asarray([record_id not in seen_ids for record_id in ids], dtype=bool)
            skipped_duplicates += int((~keep).sum())
            for record_id, include in zip(ids, keep):
                if include:
                    seen_ids.add(record_id)
            if not keep.any():
                continue

            last = adjacent_cosines(payload["layer_last_token_embeddings"])[keep].numpy().astype(np.float32)
            mean = adjacent_cosines(payload["layer_embeddings"])[keep].numpy().astype(np.float32)
            last_features.append(last)
            mean_features.append(mean)
            labels.append(shard_labels[keep])
            record_ids.extend(record_id for record_id, include in zip(ids, keep) if include)
            sources.extend(source_from_id(record_id) for record_id, include in zip(ids, keep) if include)

    if not labels:
        raise ValueError("No complete examples were loaded")
    return CosineDataset(
        last=np.concatenate(last_features),
        mean=np.concatenate(mean_features),
        labels=np.concatenate(labels),
        sources=np.asarray(sources),
        record_ids=record_ids,
        skipped_duplicates=skipped_duplicates,
    )


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, [(item - mean) / std for item in others], mean, std


def create_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels.astype(np.float32)[:, None]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, features: np.ndarray, labels: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    loader = create_loader(features, labels, batch_size=batch_size, shuffle=False)
    model.eval()
    probs, targets = [], []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            probs.append(model(batch_x.to(device)).squeeze(1).cpu().numpy())
            targets.append(batch_y.squeeze(1).numpy())
    return np.concatenate(probs), np.concatenate(targets).astype(np.int64)


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    prediction = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "failure_rate": float(labels.mean()),
        "accuracy": float((prediction == labels).mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "precision_strong": float(precision_score(labels, prediction, zero_division=0)),
        "failure_recall": float(recall_score(labels, prediction, zero_division=0)),
        "strong_route_rate": float(prediction.mean()),
        "avoidable_escalation_rate": float(fp / len(labels)),
        "missed_failure_rate": float(fn / len(labels)),
        # This is explicitly conditional on the user's routing assumption:
        # escalation fixes every Qwen failure.
        "assumed_routed_quality": float(1.0 - fn / len(labels)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    return float(max(candidates, key=lambda threshold: f1_score(labels, probabilities >= threshold, zero_division=0)))


def fit_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[MatrixFactorizationRouter, float, list[dict]]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MatrixFactorizationRouter(
        query_embedding_dim=train_x.shape[1],
        model_embedding_dim=args.model_embedding_dim,
        num_models=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = create_loader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = create_loader(val_x, val_y, args.batch_size, shuffle=False)
    best_loss = float("inf")
    best_state = None
    wait = 0
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
        train_loss /= len(train_y)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch_x, batch_y in val_loader:
                val_loss += criterion(model.forward_logits(batch_x.to(device)), batch_y.to(device)).item() * len(batch_x)
        val_loss /= len(val_y)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_loss, history


def run_iid(features: np.ndarray, labels: np.ndarray, sources: np.ndarray, args: argparse.Namespace) -> dict:
    indices = np.arange(len(labels))
    train_indices, temp_indices = train_test_split(indices, test_size=0.30, random_state=args.seed, stratify=labels)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=args.seed,
        stratify=labels[temp_indices],
    )
    train_x, other_x, feature_mean, feature_std = standardize(features[train_indices], features[val_indices], features[test_indices])
    val_x, test_x = other_x
    model, best_loss, history = fit_model(train_x, labels[train_indices], val_x, labels[val_indices], args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    val_probs, val_labels = predict(model, val_x, labels[val_indices], device, args.batch_size)
    test_probs, test_labels = predict(model, test_x, labels[test_indices], device, args.batch_size)
    threshold = best_f1_threshold(val_labels, val_probs)
    by_source = {}
    for source in sorted(set(sources[test_indices])):
        mask = sources[test_indices] == source
        by_source[source] = metrics(test_labels[mask], test_probs[mask], threshold)
    return {
        "best_validation_loss": best_loss,
        "best_threshold_from_validation": threshold,
        "test_at_threshold_0_5": metrics(test_labels, test_probs, 0.5),
        "test_at_validation_f1_threshold": metrics(test_labels, test_probs, threshold),
        "test_by_source": by_source,
        "split_sizes": {"train": int(len(train_indices)), "validation": int(len(val_indices)), "test": int(len(test_indices))},
        "history": history,
        "feature_mean": feature_mean.squeeze(0).tolist(),
        "feature_std": feature_std.squeeze(0).tolist(),
        "checkpoint": {"state_dict": model.state_dict(), "query_embedding_dim": int(features.shape[1])},
    }


def run_gsm8k_ood(features: np.ndarray, labels: np.ndarray, sources: np.ndarray, args: argparse.Namespace) -> dict:
    train_pool = np.where(sources == "gsm8k")[0]
    ood_indices = np.where(sources != "gsm8k")[0]
    train_indices, val_indices = train_test_split(train_pool, test_size=0.20, random_state=args.seed, stratify=labels[train_pool])
    train_x, other_x, _, _ = standardize(features[train_indices], features[val_indices], features[ood_indices])
    val_x, ood_x = other_x
    model, best_loss, history = fit_model(train_x, labels[train_indices], val_x, labels[val_indices], args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    val_probs, val_labels = predict(model, val_x, labels[val_indices], device, args.batch_size)
    ood_probs, ood_labels = predict(model, ood_x, labels[ood_indices], device, args.batch_size)
    threshold = best_f1_threshold(val_labels, val_probs)
    by_source = {}
    for source in sorted(set(sources[ood_indices])):
        mask = sources[ood_indices] == source
        by_source[source] = metrics(ood_labels[mask], ood_probs[mask], threshold)
    return {
        "best_validation_loss": best_loss,
        "best_threshold_from_gsm8k_validation": threshold,
        "ood_at_threshold_0_5": metrics(ood_labels, ood_probs, 0.5),
        "ood_at_gsm8k_f1_threshold": metrics(ood_labels, ood_probs, threshold),
        "ood_by_source": by_source,
        "split_sizes": {"gsm8k_train": int(len(train_indices)), "gsm8k_validation": int(len(val_indices)), "ood_test": int(len(ood_indices))},
        "history": history,
    }


def compact_results(result: dict) -> dict:
    """Remove checkpoint tensors before serializing JSON."""
    result = dict(result)
    result.pop("checkpoint", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen_layer_cosine_mf"))
    parser.add_argument("--model-embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.input_dir)
    source_counts = {source: int((dataset.sources == source).sum()) for source in sorted(set(dataset.sources))}
    print(f"Loaded {len(dataset.labels)} complete examples: {source_counts}")
    print(f"Qwen failure / strong-route label rate: {dataset.labels.mean():.2%}")

    all_results = {
        "input_dirs": [str(path.resolve()) for path in args.input_dir],
        "n_examples": int(len(dataset.labels)),
        "source_counts": source_counts,
        "strong_model_wins_rate": float(dataset.labels.mean()),
        "skipped_duplicate_record_ids": int(dataset.skipped_duplicates),
        "feature_definition": "cosine similarity between adjacent Qwen hidden-state layers",
        "label_definition": "1 = Qwen answer was incorrect; route to stronger model",
        "representations": {},
    }

    checkpoints = {}
    for name, features in (("last_prompt_token", dataset.last), ("mean_pooled_prompt", dataset.mean)):
        print(f"\nTraining {name} MF router with {features.shape[1]} cosine features", flush=True)
        iid_result = run_iid(features, dataset.labels, dataset.sources, args)
        checkpoints[name] = iid_result.pop("checkpoint")
        ood_result = run_gsm8k_ood(features, dataset.labels, dataset.sources, args)
        all_results["representations"][name] = {
            "iid": compact_results(iid_result),
            "gsm8k_to_other_ood": compact_results(ood_result),
        }
        iid_metric = all_results["representations"][name]["iid"]["test_at_validation_f1_threshold"]
        print(
            f"{name}: IID F1={iid_metric['f1']:.4f}, recall={iid_metric['failure_recall']:.4f}, "
            f"assumed quality={iid_metric['assumed_routed_quality']:.4f}",
            flush=True,
        )

    torch.save(
        {
            "checkpoints": checkpoints,
            "feature_definition": all_results["feature_definition"],
            "label_definition": all_results["label_definition"],
        },
        output_dir / "models.pt",
    )
    np.savez_compressed(
        output_dir / "cosine_features.npz",
        last_prompt_token=dataset.last,
        mean_pooled_prompt=dataset.mean,
        labels=dataset.labels,
        sources=dataset.sources,
        record_ids=np.asarray(dataset.record_ids),
    )
    (output_dir / "results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved cosine features, models, and results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
