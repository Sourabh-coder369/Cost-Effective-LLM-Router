"""Train/evaluate the 40-D Laya matrix-factorization router on GPT-4/Llama-2 RLAIF data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_FEATURES = DEFAULT_DATA / "laya_features" / "laya_features.zip"
DEFAULT_DATASET = DEFAULT_DATA / "original_gpt4_llama7b" / "gpt4_llama7b_data_unbalanced"
DEFAULT_OUT = Path(__file__).resolve().parent / "work" / "rlaif_mf"
MODEL_NAMES = ("gpt-4", "llama-2-7b-chat")
FEATURE_DIM = 40


def load_data(feature_zip: Path, dataset_dir: Path):
    """Join cached vectors to labels by source split and row index; verify prompts match."""
    split_files = {split: dataset_dir / f"router_{split}.parquet" for split in ("train", "val", "test")}
    split_rows = {}
    for split in ("train", "val", "test"):
        table = pq.read_table(
            split_files[split],
            columns=["prompt", "strong_model", "weak_model", "binary_label"],
        )
        split_rows[split] = table.to_pydict()

    matrices = {"train": [], "val": [], "test": []}
    labels = {"train": [], "val": [], "test": []}
    retained = {"train": 0, "val": 0, "test": 0}
    discarded = {"train": 0, "val": 0, "test": 0}
    seen = {"train": set(), "val": set(), "test": set()}

    with zipfile.ZipFile(feature_zip) as archive:
        metadata = json.loads(archive.read("extract_metadata.json"))
        if metadata.get("source_dataset") != "original_gpt4_llama7b":
            raise ValueError("Feature archive is not for original_gpt4_llama7b")
        if metadata.get("max_input_tokens") != 1024:
            raise ValueError(f"Expected 1024-token Laya features, got {metadata.get('max_input_tokens')}")
        expected_files = {split: path.name for split, path in split_files.items()}
        if metadata.get("source_files") != expected_files:
            raise ValueError(f"Feature archive source files do not match {expected_files}")
        source_digest = hashlib.sha256()
        for split, path in split_files.items():
            source_digest.update(split.encode("utf-8") + b"\0" + path.name.encode("utf-8") + b"\0")
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    source_digest.update(chunk)
        if metadata.get("source_sha256") != source_digest.hexdigest():
            raise ValueError("Feature archive source hash does not match the local RLAIF Parquet files")
        for split, rows in split_rows.items():
            if metadata.get("rows_by_split", {}).get(split) != len(rows["prompt"]):
                raise ValueError(f"Feature archive row count does not match local {split} data")
        feature_names = metadata.get("feature_names")
        if not isinstance(feature_names, list) or len(feature_names) != FEATURE_DIM or len(set(feature_names)) != FEATURE_DIM:
            raise ValueError("Feature archive must contain 40 unique feature names")
        names = [n for n in archive.namelist() if n.endswith(".jsonl") and n.startswith("rlaif_features")]
        if len(names) != 1:
            raise ValueError(f"Expected one rlaif_features.jsonl, found {names}")
        with archive.open(names[0]) as src:
            for raw in src:
                row = json.loads(raw)
                split = row["split"]
                index = int(row["row_index"])
                if split not in split_rows:
                    raise ValueError(f"Unknown split {split}")
                if index in seen[split]:
                    raise ValueError(f"Duplicate feature row: {split}[{index}]")
                seen[split].add(index)
                source = split_rows[split]
                if index >= len(source["prompt"]):
                    raise ValueError(f"Feature row index outside source: {split}[{index}]")
                if row["prompt"] != source["prompt"][index]:
                    raise ValueError(f"Feature/source prompt mismatch: {split}[{index}]")
                if source["strong_model"][index] != MODEL_NAMES[0] or source["weak_model"][index] != MODEL_NAMES[1]:
                    raise ValueError("Unexpected strong/weak model ordering")
                label = source["binary_label"][index]
                if label not in (0, 1):
                    raise ValueError(f"Expected binary preference label 0/1 at {split}[{index}]")
                values = row.get("features")
                status = row.get("status")
                if status not in ("featured", "discarded_over_token_limit"):
                    raise ValueError(f"Unexpected feature status at {split}[{index}]: {status}")
                if status == "discarded_over_token_limit":
                    if values is not None:
                        raise ValueError(f"Discarded row unexpectedly has features: {split}[{index}]")
                    discarded[split] += 1
                    continue
                if values is None or len(values) != FEATURE_DIM or not np.isfinite(values).all():
                    raise ValueError(f"Invalid 40-D feature vector at {split}[{index}]")
                if np.any(np.asarray(values) < 0) or np.any(np.asarray(values) > 1):
                    raise ValueError(f"Laya features must lie in [0, 1] at {split}[{index}]")
                matrices[split].append(values)
                labels[split].append(int(label))
                retained[split] += 1

    expected = {"train": 39892, "val": 4986, "test": 4987}
    for split in expected:
        if len(seen[split]) != expected[split]:
            raise ValueError(f"Feature archive has {len(seen[split])} {split} records; expected {expected[split]}")
        matrices[split] = np.asarray(matrices[split], dtype=np.float32)
        labels[split] = np.asarray(labels[split], dtype=np.int64)
    return matrices, labels, retained, discarded


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    yhat = (prob >= 0.5).astype(np.int64)
    tp = int(((y == 1) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    eps = 1e-7
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, prob))
    except ImportError:
        auc = None
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    class_recall = [tn / max(tn + fp, 1), recall]
    return {
        "n": int(len(y)),
        "accuracy": float((yhat == y).mean()),
        "balanced_accuracy": float(sum(class_recall) / 2),
        "macro_f1": float((2 * tn / max(2 * tn + fp + fn, 1) + f1) / 2),
        "roc_auc": auc,
        "log_loss": float(-np.mean(y * np.log(np.clip(prob, eps, 1-eps)) + (1-y) * np.log(np.clip(1-prob, eps, 1-eps)))),
        "gpt4_win_rate": float(y.mean()),
        "selected_gpt4_rate": float(yhat.mean()),
        "confusion_matrix_rows_true_0_1_cols_pred_0_1": [[tn, fp], [fn, tp]],
        "selected_winner_accuracy": float((yhat == y).mean()),
        "mean_raw_model_scores": {
            "gpt-4": float(pred[:, 0].mean()),
            "llama-2-7b-chat": float(pred[:, 1].mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-zip", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    x, y, retained, discarded = load_data(args.feature_zip, args.dataset_dir)
    with zipfile.ZipFile(args.feature_zip) as archive:
        feature_names = json.loads(archive.read("extract_metadata.json"))["feature_names"]
    train_x = torch.tensor(x["train"], device=device)
    train_y = torch.tensor(y["train"], dtype=torch.float32, device=device)
    val_x = torch.tensor(x["val"], device=device)
    val_y = torch.tensor(y["val"], dtype=torch.float32, device=device)

    # Fixed Laya query vectors (40-D) dot learned model vectors, plus per-model bias.
    model_vectors = torch.nn.Parameter(torch.randn(2, FEATURE_DIM, device=device) * 0.01)
    model_biases = torch.nn.Parameter(torch.zeros(2, device=device))
    optimizer = torch.optim.Adam([model_vectors, model_biases], lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = train_x @ model_vectors.T + model_biases
        # Label 1 means GPT-4 wins; pairwise logistic loss ranks its score above Llama's.
        loss = F.binary_cross_entropy_with_logits(scores[:, 0] - scores[:, 1], train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_scores = val_x @ model_vectors.T + model_biases
            val_loss = F.binary_cross_entropy_with_logits(val_scores[:, 0] - val_scores[:, 1], val_y).item()
        history.append({"epoch": epoch, "train_pairwise_bce": float(loss.item()), "validation_pairwise_bce": val_loss})
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = model_vectors.detach().cpu().clone()
            best_biases = model_biases.detach().cpu().clone()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch} train_bce={loss.item():.5f} val_bce={val_loss:.5f}", flush=True)

    assert best_weights is not None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_vectors": best_weights,
        "model_biases": best_biases,
        "model_names": list(MODEL_NAMES),
        "feature_names": feature_names,
        "feature_dim": FEATURE_DIM,
        "objective": "pairwise BCE; label 1=gpt-4 preferred, label 0=llama-2-7b-chat preferred",
        "best_epoch": best_epoch,
        "best_validation_bce": best_val_loss,
        "seed": args.seed,
        "laya_input_token_limit": 1024,
    }
    torch.save(checkpoint, args.out_dir / "rlaif_mf_40.pt")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    with torch.no_grad():
        test_scores = x["test"] @ best_weights.numpy().T + best_biases.numpy()
        logits = test_scores[:, 0] - test_scores[:, 1]
        probs = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    test_metrics = metrics(y["test"], test_scores, probs)
    train_majority = int(y["train"].mean() >= 0.5)
    baseline_metrics = {
        "always_gpt4_accuracy": float(y["test"].mean()),
        "always_llama_accuracy": float(1 - y["test"].mean()),
        "majority_train_class": MODEL_NAMES[0] if train_majority == 1 else MODEL_NAMES[1],
        "majority_train_accuracy": float((y["test"] == train_majority).mean()),
    }
    report = {
        "dataset": "original_gpt4_llama7b",
        "candidate_models": list(MODEL_NAMES),
        "split_rows": {s: int(len(y[s])) + discarded[s] for s in y},
        "retained_feature_rows": retained,
        "discarded_over_1024_token_limit": discarded,
        "feature_dimension": FEATURE_DIM,
        "feature_names": feature_names,
        "model": "fixed Laya query vector dot learned model vector plus model bias; pairwise ranking",
        "selected_epoch": best_epoch,
        "best_validation_pairwise_bce": best_val_loss,
        "test": test_metrics,
        "baselines": baseline_metrics,
        "outputs": ["rlaif_mf_40.pt", "history.json", "test_metrics.json"],
    }
    (args.out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
