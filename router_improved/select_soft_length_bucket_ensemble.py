"""Select a soft/bucketed-router probability ensemble by validation APGR."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402
from train_soft_math_mf import SoftMathMF, apgr, metrics  # noqa: E402
from train_soft_math_length_bucket_mf import SoftMathLengthBucketMF  # noqa: E402


def predict_soft(model: SoftMathMF, embeddings: np.ndarray, gate: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(embeddings), torch.from_numpy(gate)).squeeze(1).numpy()


def predict_bucket(model: SoftMathLengthBucketMF, embeddings: np.ndarray, gate: np.ndarray, buckets: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(embeddings), torch.from_numpy(gate), torch.from_numpy(buckets)).squeeze(1).numpy()


def log_word_count(dataset: DomainRouterDataset) -> np.ndarray:
    return np.log1p(dataset.df["prompt"].map(lambda value: len(str(value).split())).to_numpy(dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("val", "test"):
        parser.add_argument(f"--{split}-data", required=True, type=Path)
        parser.add_argument(f"--{split}-embeddings", required=True, type=Path)
    parser.add_argument("--soft-checkpoint", default=Path("artifacts/soft_math_mf/best_model.pt"), type=Path)
    parser.add_argument("--bucket-checkpoint", default=Path("artifacts/soft_math_length_bucket_mf/best_model.pt"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/soft_length_bucket_ensemble/results.json"), type=Path)
    args = parser.parse_args()
    datasets = {name: DomainRouterDataset(str(getattr(args, f"{name}_data")), str(getattr(args, f"{name}_embeddings"))) for name in ("val", "test")}
    embeddings = {name: data.embeddings.numpy().astype(np.float32) for name, data in datasets.items()}
    labels = {name: data.df["binary_label"].to_numpy(dtype=np.int64) for name, data in datasets.items()}

    soft_checkpoint = torch.load(args.soft_checkpoint, map_location="cpu", weights_only=False)
    soft_model = SoftMathMF(query_embedding_dim=soft_checkpoint["embedding_dim"], model_embedding_dim=64, num_models=2, num_domains=2)
    soft_model.load_state_dict(soft_checkpoint["model_state_dict"])
    detector = soft_checkpoint["detector"]
    gate = {name: detector.predict_proba(embeddings[name])[:, 1].astype(np.float32) for name in embeddings}

    bucket_checkpoint = torch.load(args.bucket_checkpoint, map_location="cpu", weights_only=False)
    edges = np.asarray(bucket_checkpoint["length_edges"])
    internal_edges = np.unique(edges[1:-1])
    bucket_model = SoftMathLengthBucketMF(query_embedding_dim=bucket_checkpoint["embedding_dim"], model_embedding_dim=64, num_models=2, num_domains=2, num_length_buckets=len(internal_edges) + 1)
    bucket_model.load_state_dict(bucket_checkpoint["model_state_dict"])
    buckets = {name: np.digitize(log_word_count(data), internal_edges, right=True).astype(np.int64) for name, data in datasets.items()}

    soft_probs = {name: predict_soft(soft_model, embeddings[name], gate[name]) for name in embeddings}
    bucket_probs = {name: predict_bucket(bucket_model, embeddings[name], gate[name], buckets[name]) for name in embeddings}
    grid = np.linspace(0, 1, 101)
    candidates = [(float(alpha), apgr((1 - alpha) * soft_probs["val"] + alpha * bucket_probs["val"], labels["val"])) for alpha in grid]
    alpha, best_val_apgr = max(candidates, key=lambda item: item[1])
    test_probs = (1 - alpha) * soft_probs["test"] + alpha * bucket_probs["test"]
    output = {
        "selection": {"bucket_probability_weight": alpha, "validation_apgr": best_val_apgr},
        "validation": {"soft": metrics(soft_probs["val"], labels["val"]), "bucket": metrics(bucket_probs["val"], labels["val"]), "ensemble": metrics((1 - alpha) * soft_probs["val"] + alpha * bucket_probs["val"], labels["val"])},
        "test": metrics(test_probs, labels["test"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
