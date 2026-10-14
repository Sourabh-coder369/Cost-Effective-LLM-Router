"""Evaluate saved domain MF checkpoints at a fixed 0.5 threshold on the test split."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "router"))
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402
from domain_specific.domain_model import DomainMFRouter  # noqa: E402


TEST_DATA = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/router_test.parquet"
VAL_DATA = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/router_val.parquet"
MPNET = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/.embedding_cache/router_test.parquet_mpnet_base_v2.pt"
BGE = ROOT / "data/original_gpt4_llama7b/bge_embeddings/router_test_bge_large_embeddings.pt"
MPNET_VAL = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/.embedding_cache/router_val.parquet_884f7401a8bb0b9d.pt"
BGE_VAL = ROOT / "data/original_gpt4_llama7b/bge_embeddings/router_val_bge_large_embeddings.pt"
FEATURE_ZIP = ROOT / "data/laya_features/laya_features.zip"
OUTPUT = ROOT / "router/domain_specific/domain_mf_test_summary.json"
VARIANTS = {
    "mpnet_64": (MPNET, MPNET_VAL, "checkpoints_domain_mf"),
    "bge_64": (BGE, BGE_VAL, "checkpoints_domain_mf_bge_large"),
    "bge_128": (BGE, BGE_VAL, "checkpoints_domain_mf_bge_modeldim_128"),
    "bge_256": (BGE, BGE_VAL, "checkpoints_domain_mf_bge_modeldim_256"),
}


def metrics(labels, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(np.int64)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "selected_gpt4_fraction": float(predicted.mean()),
        "confusion_matrix_rows_true_0_1_cols_pred_0_1": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "always_gpt4_accuracy": float(labels.mean()),
    }


def retained_test_rows():
    indices = []
    with zipfile.ZipFile(FEATURE_ZIP) as archive:
        with archive.open("rlaif_features.jsonl") as rows:
            for line in rows:
                row = json.loads(line)
                if row["split"] == "test" and row["status"] == "featured":
                    indices.append(int(row["row_index"]))
    if indices != sorted(set(indices)):
        raise ValueError("Laya retained test row indices are not sorted and unique")
    return np.asarray(indices, dtype=np.int64)


def predict(model, dataset):
    batches = []
    with torch.no_grad():
        for start in range(0, len(dataset), 256):
            batch = model(dataset.embeddings[start:start + 256], dataset.domain_ids[start:start + 256])
            batches.append(batch.flatten().numpy())
    return np.concatenate(batches)


def select_accuracy_threshold(labels, probabilities):
    values = np.sort(np.unique(probabilities))
    thresholds = np.r_[0.0, (values[:-1] + values[1:]) / 2.0, 1.0]
    accuracies = np.asarray([accuracy_score(labels, probabilities >= threshold) for threshold in thresholds])
    best = np.flatnonzero(accuracies == accuracies.max())
    index = best[np.argmin(np.abs(thresholds[best] - 0.5))]
    return float(thresholds[index]), float(accuracies[index])


def main():
    subset = retained_test_rows()
    report = {
        "test_data": str(TEST_DATA),
        "threshold": 0.5,
        "label": "binary_label=1 means GPT-4 wins; 0 means Llama-2-7B-Chat wins",
        "laya_feature_subset_rows": int(len(subset)),
        "variants": {},
    }
    datasets = {}
    val_datasets = {}
    for name, (embedding_path, val_embedding_path, checkpoint_dir) in VARIANTS.items():
        key = str(embedding_path)
        if key not in datasets:
            datasets[key] = DomainRouterDataset(str(TEST_DATA), key)
        dataset = datasets[key]
        val_key = str(val_embedding_path)
        if val_key not in val_datasets:
            val_datasets[val_key] = DomainRouterDataset(str(VAL_DATA), val_key)
        val_dataset = val_datasets[val_key]
        checkpoint_path = ROOT / "router/domain_specific" / checkpoint_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        model_dim = state["model_domain_embeddings.weight"].shape[1]
        model = DomainMFRouter(query_embedding_dim=dataset.embeddings.shape[1], model_embedding_dim=model_dim)
        model.load_state_dict(state)
        model.eval()
        probabilities = predict(model, dataset)
        labels = dataset.df["binary_label"].to_numpy(dtype=np.int64)
        val_probabilities = predict(model, val_dataset)
        val_labels = val_dataset.df["binary_label"].to_numpy(dtype=np.int64)
        selected_threshold, selected_val_accuracy = select_accuracy_threshold(val_labels, val_probabilities)
        report["variants"][name] = {
            "checkpoint": str(checkpoint_path),
            "embedding": str(embedding_path),
            "model_vector_dimension": int(model_dim),
            "validation_at_0_5": metrics(val_labels, val_probabilities),
            "validation_selected_threshold": selected_threshold,
            "validation_accuracy_at_selected_threshold": selected_val_accuracy,
            "all_test": metrics(labels, probabilities),
            "all_test_at_validation_selected_threshold": metrics(labels, probabilities, selected_threshold),
            "same_rows_as_laya_tests": metrics(labels[subset], probabilities[subset]),
            "same_rows_at_validation_selected_threshold": metrics(labels[subset], probabilities[subset], selected_threshold),
        }
        print(name, "test accuracy", f"{report['variants'][name]['all_test']['accuracy']:.5f}", flush=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
