"""Compare regex-labelled math prompts with unsupervised MPNet embedding domains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, normalized_mutual_info_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from domain_specific.domain_dataset import _domain_id  # noqa: E402


def load(data_path: Path, embedding_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(data_path)
    labels = frame["binary_label"].to_numpy(dtype=np.int64)
    math = frame["prompt"].map(_domain_id).to_numpy(dtype=np.int64)
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=False).float().numpy()
    if len(frame) != len(embeddings):
        raise ValueError("Embedding count does not match data")
    return embeddings, labels, math


def sampled_cosine_mean(left: np.ndarray, right: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    left = left[rng.choice(len(left), size=min(256, len(left)), replace=False)]
    right = right[rng.choice(len(right), size=min(256, len(right)), replace=False)]
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return float((left @ right.T).mean())


def clusters_summary(train_x: np.ndarray, val_x: np.ndarray, math: np.ndarray, strong: np.ndarray, k: int, seed: int) -> dict:
    model = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=5, random_state=seed)
    model.fit(train_x)
    clusters = model.predict(val_x)
    base_math = math.mean()
    groups = []
    for cluster_id in range(k):
        mask = clusters == cluster_id
        math_share = float(math[mask].mean())
        groups.append({
            "cluster": int(cluster_id), "size": int(mask.sum()), "share": float(mask.mean()),
            "math_share": math_share, "math_enrichment": float(math_share / base_math),
            "strong_win_rate": float(strong[mask].mean()),
        })
    groups.sort(key=lambda item: item["math_share"], reverse=True)
    top = groups[0]
    return {
        "normalized_mutual_information": float(normalized_mutual_info_score(math, clusters)),
        "most_math_enriched_cluster": top,
        "clusters": groups,
        "assignments": clusters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--val-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/domain_embedding_analysis.json"))
    args = parser.parse_args()
    train_x, _, train_math = load(args.train_data, args.train_embeddings)
    val_x, val_strong, val_math = load(args.val_data, args.val_embeddings)

    classifier = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", class_weight="balanced", random_state=42)
    classifier.fit(train_x, train_math)
    math_probability = classifier.predict_proba(val_x)[:, 1]
    predicted = (math_probability >= 0.5).astype(np.int64)
    summaries = {str(k): clusters_summary(train_x, val_x, val_math, val_strong, k, 42) for k in (2, 4, 8)}
    cluster2 = summaries["2"].pop("assignments")
    for key in ("4", "8"):
        summaries[key].pop("assignments")

    math_vectors, general_vectors = val_x[val_math == 1], val_x[val_math == 0]
    rng = np.random.default_rng(42)
    math_idx = np.flatnonzero(val_math == 1)
    general_idx = rng.choice(np.flatnonzero(val_math == 0), size=min(650, (val_math == 0).sum()), replace=False)
    plot_idx = np.concatenate([math_idx, general_idx])
    coords = PCA(n_components=2, random_state=42).fit_transform(val_x[plot_idx])
    result = {
        "validation_examples": int(len(val_math)),
        "math_examples": int(val_math.sum()),
        "math_share": float(val_math.mean()),
        "embedding_difference": {
            "math_strong_win_rate": float(val_strong[val_math == 1].mean()),
            "general_strong_win_rate": float(val_strong[val_math == 0].mean()),
            "strong_win_rate_difference": float(val_strong[val_math == 1].mean() - val_strong[val_math == 0].mean()),
            "math_centroid_general_centroid_cosine": float(
                np.dot(math_vectors.mean(axis=0), general_vectors.mean(axis=0)) /
                (np.linalg.norm(math_vectors.mean(axis=0)) * np.linalg.norm(general_vectors.mean(axis=0)))
            ),
            "mean_math_to_math_cosine": sampled_cosine_mean(math_vectors, math_vectors, 42),
            "mean_general_to_general_cosine": sampled_cosine_mean(general_vectors, general_vectors, 43),
            "mean_math_to_general_cosine": sampled_cosine_mean(math_vectors, general_vectors, 44),
            "mean_math_embedding_norm": float(np.linalg.norm(math_vectors, axis=1).mean()),
            "mean_general_embedding_norm": float(np.linalg.norm(general_vectors, axis=1).mean()),
        },
        "linear_math_detector": {
            "roc_auc": float(roc_auc_score(val_math, math_probability)),
            "pr_auc": float(average_precision_score(val_math, math_probability)),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(val_math, predicted)),
        },
        "unsupervised_clusters": summaries,
        "pca_scatter": [
            {"x": round(float(x), 4), "y": round(float(y), 4), "math": int(is_math), "cluster2": int(cluster)}
            for (x, y), is_math, cluster in zip(coords, val_math[plot_idx], cluster2[plot_idx])
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
