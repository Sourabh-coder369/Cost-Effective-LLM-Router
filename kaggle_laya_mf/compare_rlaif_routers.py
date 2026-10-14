"""Compare linear, tree, neural, and nonlinear MF routers on the same Laya features."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

if __package__:
    from .laya_rlaif_mf_local import DEFAULT_DATASET, DEFAULT_FEATURES, MODEL_NAMES, load_data
else:
    from laya_rlaif_mf_local import DEFAULT_DATASET, DEFAULT_FEATURES, MODEL_NAMES, load_data


DEFAULT_OUT = Path(__file__).resolve().parent / "work" / "rlaif_router_comparison"
DEFAULT_MF = Path(__file__).resolve().parent / "work" / "rlaif_mf" / "rlaif_mf_40.pt"


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


class NonlinearMF(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 40))
        self.model_vectors = nn.Parameter(torch.randn(2, 40) * 0.02)
        self.model_biases = nn.Parameter(torch.zeros(2))

    def forward(self, features):
        query_vector = self.query_projection(features)
        scores = query_vector @ self.model_vectors.T + self.model_biases
        return scores[:, 0] - scores[:, 1]


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def evaluate(labels, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(np.int64)
    confusion = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "selected_gpt4_fraction": float(predictions.mean()),
        "confusion_matrix_rows_true_0_1_cols_pred_0_1": confusion.tolist(),
    }


def best_validation_accuracy_threshold(labels, probabilities):
    """Find a threshold using validation data only; prefer the one closest to 0.5 on ties."""
    values = np.sort(np.unique(probabilities))
    candidates = np.r_[0.0, (values[:-1] + values[1:]) / 2.0, 1.0]
    accuracies = np.asarray([accuracy_score(labels, probabilities >= value) for value in candidates])
    best = np.flatnonzero(accuracies == accuracies.max())
    index = best[np.argmin(np.abs(candidates[best] - 0.5))]
    return float(candidates[index]), float(accuracies[index])


def summarize(y_val, p_val, y_test, p_test, config):
    threshold, tuned_val_accuracy = best_validation_accuracy_threshold(y_val, p_val)
    return {
        "config": config,
        "validation": evaluate(y_val, p_val),
        "test": evaluate(y_test, p_test),
        "validation_accuracy_threshold": threshold,
        "validation_accuracy_at_tuned_threshold": tuned_val_accuracy,
        "test_at_validation_tuned_threshold": evaluate(y_test, p_test, threshold),
    }


def train_torch(model, x_train, y_train, x_val, y_val, out_path, scaler, feature_names, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    train_features = torch.from_numpy(x_train)
    train_labels = torch.from_numpy(y_train.astype(np.float32))
    val_features = torch.from_numpy(x_val)
    val_labels = torch.from_numpy(y_val.astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, 121):
        model.train()
        indices = torch.randperm(len(train_labels))
        for batch in indices.split(2048):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_features[batch]), train_labels[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(val_features), val_labels).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 15:
                break
        if epoch == 1 or epoch % 10 == 0:
            print(f"{model.__class__.__name__}: epoch {epoch}, validation log loss {validation_loss:.5f}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "best_epoch": best_epoch, "validation_bce": best_loss,
                "seed": seed, "model_names": list(MODEL_NAMES), "feature_names": feature_names,
                "preprocess_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
                "preprocess_scale": torch.from_numpy(scaler.scale_.astype(np.float32))}, out_path)
    return model, {"best_epoch": best_epoch, "validation_bce": best_loss, "seed": seed}


def torch_probabilities(model, features):
    model.eval()
    with torch.no_grad():
        return sigmoid(model(torch.from_numpy(features)).numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-zip", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mf-checkpoint", type=Path, default=DEFAULT_MF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features, labels, retained, discarded = load_data(args.feature_zip, args.dataset_dir)
    with zipfile.ZipFile(args.feature_zip) as archive:
        feature_names = json.loads(archive.read("extract_metadata.json"))["feature_names"]
    print(f"Loaded {retained} retained features; discarded {discarded}", flush=True)
    x_train, x_val, x_test = (features[s] for s in ("train", "val", "test"))
    y_train, y_val, y_test = (labels[s] for s in ("train", "val", "test"))
    scaler = StandardScaler().fit(x_train)
    x_train_std, x_val_std, x_test_std = (scaler.transform(x).astype(np.float32) for x in (x_train, x_val, x_test))
    report = {"dataset": "original_gpt4_llama7b", "models": list(MODEL_NAMES), "feature_names": feature_names,
              "retained": retained,
              "discarded": discarded, "selection_metric": "validation log loss", "results": {}}

    checkpoint = torch.load(args.mf_checkpoint, map_location="cpu", weights_only=True)
    vectors = checkpoint["model_vectors"].numpy()
    biases = checkpoint["model_biases"].numpy()
    def mf_probabilities(x):
        scores = x @ vectors.T + biases
        return sigmoid(scores[:, 0] - scores[:, 1])
    report["results"]["existing_mf"] = summarize(y_val, mf_probabilities(x_val), y_test, mf_probabilities(x_test),
        {"checkpoint": str(args.mf_checkpoint), "feature_transform": "raw Laya features"})
    print("Existing MF evaluated", flush=True)

    best_linear = None
    for strength in (0.01, 0.1, 1.0, 10.0):
        candidate = LogisticRegression(C=strength, solver="lbfgs", max_iter=1000, random_state=42)
        candidate.fit(x_train_std, y_train)
        val_prob = candidate.predict_proba(x_val_std)[:, 1]
        val_loss = log_loss(y_val, val_prob, labels=[0, 1])
        print(f"Logistic C={strength}: validation log loss {val_loss:.5f}", flush=True)
        if best_linear is None or val_loss < best_linear[0]:
            best_linear = (val_loss, strength, candidate)
    _, strength, linear = best_linear
    joblib.dump({"model": linear, "scaler": scaler, "model_names": MODEL_NAMES,
                 "feature_names": feature_names}, args.out_dir / "linear_baseline.joblib")
    report["results"]["regularized_linear"] = summarize(y_val, linear.predict_proba(x_val_std)[:, 1],
        y_test, linear.predict_proba(x_test_std)[:, 1], {"C": strength, "feature_transform": "train-fitted standardization"})

    best_catboost = None
    for depth in (4, 6):
        candidate = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=depth, l2_leaf_reg=5,
            loss_function="Logloss", eval_metric="Logloss", od_type="Iter", od_wait=50,
            random_seed=42, thread_count=4, verbose=False, allow_writing_files=False)
        candidate.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
        val_prob = candidate.predict_proba(x_val)[:, 1]
        val_loss = log_loss(y_val, val_prob, labels=[0, 1])
        print(f"CatBoost depth={depth}, best iteration={candidate.get_best_iteration()}: validation log loss {val_loss:.5f}", flush=True)
        if best_catboost is None or val_loss < best_catboost[0]:
            best_catboost = (val_loss, depth, candidate)
    _, depth, catboost = best_catboost
    catboost.save_model(str(args.out_dir / "catboost.cbm"))
    report["results"]["catboost"] = summarize(y_val, catboost.predict_proba(x_val)[:, 1],
        y_test, catboost.predict_proba(x_test)[:, 1],
        {"depth": depth, "best_iteration": int(catboost.get_best_iteration()), "feature_transform": "raw Laya features"})

    for name, constructor, architecture in (
        ("small_mlp", SmallMLP, "40-64-ReLU-32-ReLU-1"),
        ("nonlinear_mf", NonlinearMF, "40-64-ReLU-40 dot two 40-D model vectors"),
    ):
        candidates = []
        for seed in (7, 42, 123):
            torch.manual_seed(seed)
            path = args.out_dir / f"{name}_seed{seed}.pt"
            model, config = train_torch(constructor(), x_train_std, y_train, x_val_std, y_val,
                path, scaler, feature_names, seed=seed)
            val_prob = torch_probabilities(model, x_val_std)
            val_loss = log_loss(y_val, val_prob, labels=[0, 1])
            candidates.append((val_loss, seed, model, config, path))
        _, selected_seed, selected_model, selected_config, selected_path = min(candidates, key=lambda row: row[0])
        shutil.copyfile(selected_path, args.out_dir / f"{name}.pt")
        report["results"][name] = summarize(y_val, torch_probabilities(selected_model, x_val_std),
            y_test, torch_probabilities(selected_model, x_test_std),
            {**selected_config, "architecture": architecture, "feature_transform": "train-fitted standardization",
             "selected_seed": selected_seed,
             "seed_validation_log_loss": {str(seed): float(loss) for loss, seed, _, _, _ in candidates},
             "checkpoint": str(args.out_dir / f"{name}.pt")})

    report["baseline"] = {"always_gpt4_test_accuracy": float(y_test.mean()),
                          "always_llama_test_accuracy": float(1 - y_test.mean())}
    (args.out_dir / "comparison_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"validation_log_loss": row["validation"]["log_loss"],
                             "test_accuracy": row["test"]["accuracy"],
                             "test_auc": row["test"]["roc_auc"],
                             "selected_gpt4": row["test"]["selected_gpt4_fraction"]}
                      for name, row in report["results"].items()}, indent=2), flush=True)
    print(f"Report saved to {args.out_dir / 'comparison_results.json'}", flush=True)


if __name__ == "__main__":
    main()
