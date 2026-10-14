"""Test whether existing BGE text embeddings improve the Laya preference router."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss
from sklearn.preprocessing import normalize

if __package__:
    from .compare_rlaif_routers import summarize
    from .laya_rlaif_mf_local import DEFAULT_DATASET, DEFAULT_FEATURES, load_data
else:
    from compare_rlaif_routers import summarize
    from laya_rlaif_mf_local import DEFAULT_DATASET, DEFAULT_FEATURES, load_data


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "original_gpt4_llama7b"
DEFAULT_BGE = DATA_ROOT / "bge_embeddings"
DEFAULT_OUT = Path(__file__).resolve().parent / "work" / "rlaif_router_comparison"


def retained_indices(feature_zip: Path):
    indices = {split: [] for split in ("train", "val", "test")}
    with zipfile.ZipFile(feature_zip) as archive:
        with archive.open("rlaif_features.jsonl") as rows:
            for line in rows:
                row = json.loads(line)
                if row["status"] == "featured":
                    indices[row["split"]].append(int(row["row_index"]))
    for split, values in indices.items():
        if values != sorted(values) or len(values) != len(set(values)):
            raise ValueError(f"Retained {split} row indices are not unique and ordered")
    return indices


def load_bge(bge_dir: Path, indices: dict):
    matrices = {}
    for split, retained in indices.items():
        full = torch.load(bge_dir / f"router_{split}_bge_large_embeddings.pt", map_location="cpu", weights_only=True)
        if full.ndim != 2 or full.shape[1] != 1024 or full.shape[0] <= max(retained):
            raise ValueError(f"Unexpected BGE shape for {split}: {tuple(full.shape)}")
        selected = full[retained].numpy()
        del full
        if not np.isfinite(selected).all():
            raise ValueError(f"Nonfinite BGE values in {split}")
        normalize(selected, norm="l2", copy=False)
        matrices[split] = selected
        print(f"Loaded BGE {split}: {selected.shape}", flush=True)
    return matrices


def fit_catboost(train_x, train_y, val_x, val_y, seed=42):
    model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=5,
        loss_function="Logloss", eval_metric="Logloss", od_type="Iter", od_wait=50,
        random_seed=seed, thread_count=4, verbose=False, allow_writing_files=False)
    model.fit(train_x, train_y, eval_set=(val_x, val_y), use_best_model=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-zip", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bge-dir", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    laya, labels, retained, discarded = load_data(args.feature_zip, args.dataset_dir)
    indices = retained_indices(args.feature_zip)
    for split in indices:
        if len(indices[split]) != retained[split]:
            raise ValueError(f"Laya/BGE retained row count differs in {split}")
    bge = load_bge(args.bge_dir, indices)
    svd = TruncatedSVD(n_components=64, n_iter=3, random_state=42)
    reduced = {"train": svd.fit_transform(bge["train"]).astype(np.float32)}
    for split in ("val", "test"):
        reduced[split] = svd.transform(bge[split]).astype(np.float32)
    joblib.dump(svd, args.out_dir / "bge_svd64.joblib")
    variance = float(np.sum(svd.explained_variance_ratio_))
    print(f"BGE SVD 64 components explain {variance:.3%} of train variance", flush=True)

    report = {"dataset": "original_gpt4_llama7b", "models": ["gpt-4", "llama-2-7b-chat"],
              "retained": retained, "discarded": discarded,
              "bge_model": "BAAI/bge-large-en-v1.5", "bge_preprocessing": "L2 normalization; train-fitted 64-component TruncatedSVD",
              "svd_explained_train_variance": variance,
              "note": "BGE row order is taken from the existing row-aligned embedding files and their generation script",
              "results": {}}
    for name, matrices in (
        ("bge_only_catboost", reduced),
        ("laya_plus_bge_catboost", {split: np.concatenate([laya[split], reduced[split]], axis=1)
                                     for split in ("train", "val", "test")}),
    ):
        model = fit_catboost(matrices["train"], labels["train"], matrices["val"], labels["val"])
        p_val = model.predict_proba(matrices["val"])[:, 1]
        p_test = model.predict_proba(matrices["test"])[:, 1]
        path = args.out_dir / f"{name}.cbm"
        model.save_model(str(path))
        report["results"][name] = summarize(labels["val"], p_val, labels["test"], p_test,
            {"depth": 6, "best_iteration": int(model.get_best_iteration()),
             "validation_log_loss": float(log_loss(labels["val"], p_val)),
             "feature_count": int(matrices["train"].shape[1]), "checkpoint": str(path)})
        print(f"{name}: val log loss={report['results'][name]['validation']['log_loss']:.5f}, "
              f"test accuracy={report['results'][name]['test']['accuracy']:.5f}, "
              f"test AUC={report['results'][name]['test']['roc_auc']:.5f}", flush=True)
    output = args.out_dir / "text_embedding_comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report saved to {output}", flush=True)


if __name__ == "__main__":
    main()
