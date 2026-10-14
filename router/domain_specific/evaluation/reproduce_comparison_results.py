"""Reproduce the overall baseline-vs-domain cost/quality comparison.

Uses the current checkpoints and the 768-dimensional MPNet embeddings available
for all explicit train/validation/test splits. The result is deliberately saved
under a new name, preserving the historical comparison_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.dataset import RouterDataset
from baseline.model import MatrixFactorizationRouter
from domain_specific.domain_dataset import DomainRouterDataset
from domain_specific.domain_model import DomainMFRouter


BASE = Path(__file__).resolve().parents[3]
DATA_DIR = BASE / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced"
TEST_DATA = DATA_DIR / "router_test.parquet"
TEST_EMBEDDINGS = DATA_DIR / ".embedding_cache/router_test.parquet_mpnet_base_v2.pt"
BASELINE_CHECKPOINT = BASE / "router/baseline/checkpoints_bge_large_unbalanced/best_model.pt"
DOMAIN_CHECKPOINT = BASE / "router/domain_specific/checkpoints_domain_mf/best_model.pt"
OUTPUT = BASE / "router/domain_specific/checkpoints_domain_mf/comparison_results_reproduced.json"
DROPS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)


def predict_baseline(model: MatrixFactorizationRouter, dataset: RouterDataset, device: torch.device) -> np.ndarray:
    batches = [embeddings for embeddings, _ in DataLoader(dataset, batch_size=256, shuffle=False)]
    with torch.no_grad():
        return model(torch.cat(batches).to(device)).cpu().numpy().reshape(-1)


def predict_domain(model: DomainMFRouter, dataset: DomainRouterDataset, device: torch.device) -> np.ndarray:
    batches, domains = [], []
    for embeddings, _, domain_ids in DataLoader(dataset, batch_size=256, shuffle=False):
        batches.append(embeddings)
        domains.append(domain_ids)
    with torch.no_grad():
        return model(torch.cat(batches).to(device), torch.cat(domains).to(device)).cpu().numpy().reshape(-1)


def cost_at_drop(probabilities: np.ndarray, labels: np.ndarray, drop: float) -> tuple[float, float, float]:
    target_quality = labels.mean() - drop / 100.0
    thresholds = np.linspace(0, 1, 1000)
    best: tuple[float, float, float] | None = None
    for threshold in thresholds:
        route_strong = probabilities >= threshold
        quality = np.where(route_strong, labels, 1 - labels).mean()
        cost = route_strong.mean()
        if quality >= target_quality and (best is None or cost < best[0]):
            best = (float(cost), float(quality), float(threshold))
    if best is None:
        raise RuntimeError(f"No threshold met a {drop}% quality-drop target")
    return best


def comparison_rows(baseline_probabilities: np.ndarray, domain_probabilities: np.ndarray,
                    labels: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for drop in DROPS:
        baseline_cost, baseline_quality, baseline_threshold = cost_at_drop(baseline_probabilities, labels, drop)
        domain_cost, domain_quality, domain_threshold = cost_at_drop(domain_probabilities, labels, drop)
        baseline_advantage = (1 - baseline_cost) * 100
        domain_advantage = (1 - domain_cost) * 100
        rows.append({
            "drop": drop,
            "baseline_quality": baseline_quality,
            "baseline_cost_adv": baseline_advantage,
            "baseline_threshold": baseline_threshold,
            "domain_quality": domain_quality,
            "domain_cost_adv": domain_advantage,
            "domain_threshold": domain_threshold,
            "delta_cost_adv": domain_advantage - baseline_advantage,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--test-embeddings", type=Path, default=TEST_EMBEDDINGS)
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument("--domain-checkpoint", type=Path, default=DOMAIN_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_dataset = DomainRouterDataset(str(args.test_data), str(args.test_embeddings))
    baseline_dataset = RouterDataset(str(args.test_data), precomputed_embeddings_path=str(args.test_embeddings))
    labels = domain_dataset.df["binary_label"].to_numpy(dtype=np.int64)

    domain_checkpoint = torch.load(args.domain_checkpoint, map_location=device, weights_only=False)
    domain_model_dim = domain_checkpoint["model_state_dict"]["model_domain_embeddings.weight"].shape[1]
    domain_model = DomainMFRouter(
        query_embedding_dim=domain_dataset.embeddings.shape[1],
        model_embedding_dim=domain_model_dim,
    ).to(device)
    domain_model.load_state_dict(domain_checkpoint["model_state_dict"])
    domain_model.eval()

    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline_dim = baseline_checkpoint["model_state_dict"]["text_proj.0.weight"].shape[1]
    baseline_model = MatrixFactorizationRouter(
        query_embedding_dim=baseline_dim,
        model_embedding_dim=baseline_checkpoint["config"]["model_embedding_dim"],
    ).to(device)
    baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
    baseline_model.eval()

    domain_probabilities = predict_domain(domain_model, domain_dataset, device)
    baseline_probabilities = predict_baseline(baseline_model, baseline_dataset, device)
    rows = comparison_rows(baseline_probabilities, domain_probabilities, labels)
    domain_ids = domain_dataset.domain_ids.numpy()
    math_mask = domain_ids == DomainMFRouter.DOMAIN_MATH
    general_mask = domain_ids == DomainMFRouter.DOMAIN_GENERAL

    result = {
        "baseline_always_strong_quality": float(labels.mean()),
        "comparison": rows,
        "per_domain": {
            "math": {
                "examples": int(math_mask.sum()),
                "always_strong_quality": float(labels[math_mask].mean()),
                "comparison": comparison_rows(baseline_probabilities[math_mask], domain_probabilities[math_mask], labels[math_mask]),
            },
            "general": {
                "examples": int(general_mask.sum()),
                "always_strong_quality": float(labels[general_mask].mean()),
                "comparison": comparison_rows(baseline_probabilities[general_mask], domain_probabilities[general_mask], labels[general_mask]),
            },
        },
        "config": {
            "embedding_model": args.embedding_model,
            "query_embedding_dim": int(domain_dataset.embeddings.shape[1]),
            "baseline_model_embedding_dim": baseline_checkpoint["config"]["model_embedding_dim"],
            "domain_model_embedding_dim": domain_model_dim,
            "test_examples": int(len(labels)),
            "device": str(device),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved reproduced comparison to {args.output}")
    print(f"\nOverall ({len(labels)} examples):")
    for row in rows:
        print(f"drop={row['drop']:>3.1f}% baseline_adv={row['baseline_cost_adv']:>6.2f}% "
              f"domain_adv={row['domain_cost_adv']:>6.2f}% delta={row['delta_cost_adv']:+.2f}%")
    for name in ("math", "general"):
        print(f"\n{name} ({result['per_domain'][name]['examples']} examples):")
        for row in result["per_domain"][name]["comparison"]:
            print(f"  drop={row['drop']:>3.1f}% baseline_adv={row['baseline_cost_adv']:>6.2f}% "
                  f"domain_adv={row['domain_cost_adv']:>6.2f}% delta={row['delta_cost_adv']:+.2f}%")


if __name__ == "__main__":
    main()
