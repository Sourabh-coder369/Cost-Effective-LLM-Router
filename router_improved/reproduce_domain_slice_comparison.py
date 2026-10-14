"""Reproduce math-only and general-only cost-savings tables on matching MPNet models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from baseline.dataset import RouterDataset  # noqa: E402
from baseline.model import MatrixFactorizationRouter  # noqa: E402
from domain_specific.domain_dataset import DomainRouterDataset  # noqa: E402
from domain_specific.domain_model import DomainMFRouter  # noqa: E402


def cost_savings_at_drop(probabilities: np.ndarray, labels: np.ndarray, allowed_drop: float) -> tuple[float, float]:
    target_quality = labels.mean() - allowed_drop / 100
    candidates = []
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
    parser.add_argument("--domain", default=Path("artifacts/mpnet_domain_reproduction/best_model.pt"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/current_mpnet_slice_comparison.json"), type=Path)
    args = parser.parse_args()
    domain_data = DomainRouterDataset(str(args.data), str(args.embeddings))
    baseline_data = RouterDataset(str(args.data), precomputed_embeddings_path=str(args.embeddings))
    labels = domain_data.df["binary_label"].to_numpy(dtype=np.int64)
    if not np.array_equal(labels, baseline_data.df["binary_label"].to_numpy(dtype=np.int64)):
        raise ValueError("Baseline and domain data labels do not match")

    domain_checkpoint = torch.load(args.domain, map_location="cpu", weights_only=False)
    domain_model = DomainMFRouter(query_embedding_dim=768, model_embedding_dim=64)
    domain_model.load_state_dict(domain_checkpoint["model_state_dict"])
    baseline_checkpoint = torch.load(args.baseline, map_location="cpu", weights_only=False)
    baseline_model = MatrixFactorizationRouter(query_embedding_dim=768, model_embedding_dim=64)
    baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
    domain_model.eval(); baseline_model.eval()
    with torch.inference_mode():
        domain_probs = domain_model(domain_data.embeddings, domain_data.domain_ids).squeeze(1).numpy()
        baseline_probs = baseline_model(baseline_data.embeddings).squeeze(1).numpy()

    output = {"data": str(args.data), "slices": {}}
    for name, mask in {"math": domain_data.domain_ids.numpy() == 1, "general": domain_data.domain_ids.numpy() == 0}.items():
        rows = []
        for drop in (0.0, 0.5, 1.0, 2.0):
            baseline_savings, baseline_quality = cost_savings_at_drop(baseline_probs[mask], labels[mask], drop)
            domain_savings, domain_quality = cost_savings_at_drop(domain_probs[mask], labels[mask], drop)
            rows.append({
                "allowed_quality_drop": drop,
                "baseline_cost_savings": baseline_savings,
                "domain_mf_cost_savings": domain_savings,
                "delta": domain_savings - baseline_savings,
                "baseline_quality": baseline_quality,
                "domain_quality": domain_quality,
            })
        output["slices"][name] = {"n_examples": int(mask.sum()), "rows": rows}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
