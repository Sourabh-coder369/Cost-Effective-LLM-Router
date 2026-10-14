"""Evaluate the saved soft-math MF router on hard math/general test slices."""

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
sys.path.insert(0, str(ROOT / "evaluation"))
from baseline.dataset import RouterDataset  # noqa: E402
from baseline.model import MatrixFactorizationRouter  # noqa: E402
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402
from routellm_evaluation import compute_apgr, compute_cpt  # noqa: E402


def load_soft_math_class():
    module_path = ROOT / "router_improved" / "train_soft_math_mf.py"
    spec = importlib.util.spec_from_file_location("soft_math_training", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SoftMathMF


def cost_savings_at_drop(probabilities: np.ndarray, labels: np.ndarray, allowed_drop: float) -> tuple[float, float]:
    target_quality = labels.mean() - allowed_drop / 100
    candidates: list[tuple[float, float]] = []
    for threshold in np.linspace(0, 1, 1000):
        use_strong = probabilities >= threshold
        quality = np.where(use_strong, labels, 1 - labels).mean()
        if quality >= target_quality:
            candidates.append((float(use_strong.mean()), float(quality)))
    if not candidates:
        raise RuntimeError(f"No threshold meets a {allowed_drop}% quality drop")
    cost, quality = min(candidates, key=lambda item: item[0])
    return (1 - cost) * 100, quality * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--baseline", default=Path("artifacts/mpnet_reproduction/best_model.pt"), type=Path)
    parser.add_argument("--soft-model", default=Path("artifacts/soft_math_mf/best_model.pt"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/soft_math_mf/test_slice_comparison.json"), type=Path)
    args = parser.parse_args()

    domain_data = DomainRouterDataset(str(args.data), str(args.embeddings))
    baseline_data = RouterDataset(str(args.data), precomputed_embeddings_path=str(args.embeddings))
    labels = domain_data.df["binary_label"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels, baseline_data.df["binary_label"].to_numpy(dtype=np.int64)):
        raise ValueError("Baseline and domain data labels do not match")

    baseline_checkpoint = torch.load(args.baseline, map_location="cpu", weights_only=False)
    baseline_model = MatrixFactorizationRouter(query_embedding_dim=768, model_embedding_dim=64)
    baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])

    soft_checkpoint = torch.load(args.soft_model, map_location="cpu", weights_only=False)
    SoftMathMF = load_soft_math_class()
    soft_model = SoftMathMF(query_embedding_dim=768, model_embedding_dim=64, num_models=2, num_domains=2)
    soft_model.load_state_dict(soft_checkpoint["model_state_dict"])
    detector = soft_checkpoint["detector"]

    baseline_model.eval()
    soft_model.eval()
    features = domain_data.embeddings.numpy().astype(np.float32)
    gate = detector.predict_proba(features)[:, 1].astype(np.float32)
    with torch.inference_mode():
        baseline_probs = baseline_model(baseline_data.embeddings).squeeze(1).numpy()
        soft_probs = soft_model(domain_data.embeddings, torch.from_numpy(gate)).squeeze(1).numpy()

    from sklearn.metrics import average_precision_score, roc_auc_score
    weak_quality, strong_quality = (1 - labels).mean(), labels.mean()
    output: dict[str, object] = {
        "data": str(args.data),
        "math_gate": {"mean": float(gate.mean())},
        "baseline_overall": {
            "roc_auc": float(roc_auc_score(labels, baseline_probs)),
            "pr_auc": float(average_precision_score(labels, baseline_probs)),
            "accuracy_at_0_5": float(((baseline_probs >= 0.5).astype(int) == labels).mean()),
            "apgr": float(compute_apgr(baseline_probs, labels, weak_quality, strong_quality)),
            "cpt_50": float(compute_cpt(baseline_probs, labels, weak_quality, strong_quality, 0.5)),
            "cpt_80": float(compute_cpt(baseline_probs, labels, weak_quality, strong_quality, 0.8)),
        },
        "overall": {
            "roc_auc": float(roc_auc_score(labels, soft_probs)),
            "pr_auc": float(average_precision_score(labels, soft_probs)),
            "accuracy_at_0_5": float(((soft_probs >= 0.5).astype(int) == labels).mean()),
            "apgr": float(compute_apgr(soft_probs, labels, weak_quality, strong_quality)),
            "cpt_50": float(compute_cpt(soft_probs, labels, weak_quality, strong_quality, 0.5)),
            "cpt_80": float(compute_cpt(soft_probs, labels, weak_quality, strong_quality, 0.8)),
        },
        "slices": {},
    }
    masks = {"math": domain_data.domain_ids.numpy() == 1, "general": domain_data.domain_ids.numpy() == 0}
    for name, mask in masks.items():
        rows = []
        for drop in (0.0, 0.5, 1.0, 2.0):
            baseline_savings, baseline_quality = cost_savings_at_drop(baseline_probs[mask], labels[mask], drop)
            soft_savings, soft_quality = cost_savings_at_drop(soft_probs[mask], labels[mask], drop)
            rows.append({
                "allowed_quality_drop": drop,
                "baseline_cost_savings": baseline_savings,
                "soft_math_mf_cost_savings": soft_savings,
                "delta": soft_savings - baseline_savings,
                "baseline_quality": baseline_quality,
                "soft_math_mf_quality": soft_quality,
            })
        output["slices"][name] = {"n_examples": int(mask.sum()), "gate_mean": float(gate[mask].mean()), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
