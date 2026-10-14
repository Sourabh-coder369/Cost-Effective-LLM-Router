"""Train a separate soft-math MF router with normalized log prompt length."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402
from domain_specific.domain_model import DomainMFRouter  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SoftMathLengthMF(DomainMFRouter):
    """Soft math routing with prompt length included in the MF query features."""

    def forward_logits(self, query_features: torch.Tensor, math_probability: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        general = torch.zeros(len(query_features), device=query_features.device, dtype=torch.long)
        math = torch.ones(len(query_features), device=query_features.device, dtype=torch.long)
        general_logit = self.compute_score(query_features, general, 1) - self.compute_score(query_features, general, 0)
        math_logit = self.compute_score(query_features, math, 1) - self.compute_score(query_features, math, 0)
        gate = math_probability.reshape(-1, 1)
        return (1 - gate) * general_logit + gate * math_logit

    def forward(self, query_features: torch.Tensor, math_probability: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.forward_logits(query_features, math_probability))


def apgr(probabilities: np.ndarray, labels: np.ndarray) -> float:
    strong, weak = labels.mean(), (1 - labels).mean()
    ordering = np.argsort(probabilities)[::-1]
    values = []
    for pct in range(10, 101, 10):
        chosen = np.zeros(len(labels), dtype=bool)
        chosen[ordering[:round(len(labels) * pct / 100)]] = True
        values.append((np.where(chosen, labels, 1 - labels).mean() - weak) / (strong - weak))
    return float(np.mean(values))


def predict(model: SoftMathLengthMF, features: np.ndarray, gate: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(gate.astype(np.float32))), batch_size=512)
    output = []
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_gate in loader:
            output.append(model(batch_x.to(device), batch_gate.to(device)).squeeze(1).cpu().numpy())
    return np.concatenate(output)


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    strong, weak = labels.mean(), (1 - labels).mean()
    ordering = np.argsort(probabilities)[::-1]
    levels = {}
    for pct in range(10, 101, 10):
        chosen = np.zeros(len(labels), dtype=bool)
        chosen[ordering[:round(len(labels) * pct / 100)]] = True
        quality = np.where(chosen, labels, 1 - labels).mean()
        levels[str(pct)] = {"strong_usage": float(chosen.mean()), "quality": float(quality), "pgr": float((quality - weak) / (strong - weak))}
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy_at_0_5": float(((probabilities >= 0.5).astype(int) == labels).mean()),
        "apgr": float(np.mean([row["pgr"] for row in levels.values()])),
        "cost_levels": levels,
    }


def log_word_counts(dataset: DomainRouterDataset) -> np.ndarray:
    return np.log1p(dataset.df["prompt"].map(lambda value: len(str(value).split())).to_numpy(dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-data", required=True, type=Path)
        parser.add_argument(f"--{split}-embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/soft_math_length_mf"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        name: DomainRouterDataset(str(getattr(args, f"{name}_data")), str(getattr(args, f"{name}_embeddings")))
        for name in ("train", "val", "test")
    }
    embeddings = {name: data.embeddings.numpy().astype(np.float32) for name, data in datasets.items()}
    y = {name: data.df["binary_label"].to_numpy(dtype=np.int64) for name, data in datasets.items()}
    math = {name: data.domain_ids.numpy().astype(np.int64) for name, data in datasets.items()}

    raw_length = {name: log_word_counts(data) for name, data in datasets.items()}
    length_mean = float(raw_length["train"].mean())
    length_std = float(raw_length["train"].std())
    normalized_length = {name: ((values - length_mean) / max(length_std, 1e-6)).astype(np.float32) for name, values in raw_length.items()}
    features = {name: np.column_stack((embeddings[name], normalized_length[name])).astype(np.float32) for name in embeddings}

    # Both the math detector and MF router receive length.  All statistics and
    # detector weights are fitted on training data only.
    detector = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=args.seed)
    detector.fit(features["train"], math["train"])
    gate = {name: detector.predict_proba(features[name])[:, 1].astype(np.float32) for name in features}
    detector_metrics = {
        "validation_math_log_loss": float(log_loss(math["val"], gate["val"])),
        "validation_math_roc_auc": float(roc_auc_score(math["val"], gate["val"])),
        "validation_gate_mean": float(gate["val"].mean()),
    }

    model = SoftMathLengthMF(query_embedding_dim=features["train"].shape[1], model_embedding_dim=64, num_models=2, num_domains=2).to(device)
    weight = torch.tensor([(len(y["train"]) - y["train"].sum()) / y["train"].sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features["train"]), torch.from_numpy(y["train"].astype(np.float32)), torch.from_numpy(gate["train"])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_apgr, best_state, best_epoch, stale, history = -float("inf"), None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y, batch_gate in loader:
            optimiser.zero_grad()
            loss = criterion(model.forward_logits(batch_x.to(device), batch_gate.to(device)).squeeze(1), batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            loss_sum += loss.item() * len(batch_y)
        val_apgr = apgr(predict(model, features["val"], gate["val"], device), y["val"])
        history.append({"epoch": epoch, "train_loss": loss_sum / len(y["train"]), "validation_apgr": val_apgr})
        if val_apgr > best_apgr + 1e-6:
            best_apgr, best_epoch, stale = val_apgr, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    test_probs = predict(model, features["test"], gate["test"], device)
    result = {
        "best_validation_apgr": best_apgr,
        "best_epoch": best_epoch,
        "length_feature": {"definition": "z-score(log1p(whitespace_word_count))", "train_mean": length_mean, "train_std": length_std, "test_mean_words": float(np.expm1(raw_length["test"]).mean())},
        "math_detector": detector_metrics,
        "history": history,
        "test": metrics(test_probs, y["test"]),
    }
    torch.save({"model_state_dict": best_state, "detector": detector, "embedding_dim": embeddings["train"].shape[1], "length_mean": length_mean, "length_std": length_std}, args.output_dir / "best_model.pt")
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
