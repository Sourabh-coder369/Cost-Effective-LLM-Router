"""Compare saved BGE routers with random routing at a fixed strong-call budget.

The fixed budget is applied independently within each reported slice. Model
predictions and labels must refer to the same ordered test rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
sys.path.insert(0, str(ROOT / "router_improved"))
from baseline.model import MatrixFactorizationRouter  # noqa: E402
from domain_specific.domain_dataset import _domain_id  # noqa: E402
from domain_specific.domain_model import DomainMFRouter  # noqa: E402
from model_heads import ModelHeadRouter  # noqa: E402


@torch.no_grad()
def predict_baseline(path: Path, embeddings: torch.Tensor, device: torch.device) -> np.ndarray:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    model = MatrixFactorizationRouter(
        query_embedding_dim=state["text_proj.0.weight"].shape[1],
        model_embedding_dim=state["model_embeddings.weight"].shape[1],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in embeddings.split(512)]).numpy().ravel()


@torch.no_grad()
def predict_domain(
    path: Path, embeddings: torch.Tensor, domains: torch.Tensor, device: torch.device
) -> np.ndarray:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    model = DomainMFRouter(
        query_embedding_dim=state["text_proj.0.weight"].shape[1],
        model_embedding_dim=state["model_domain_embeddings.weight"].shape[1],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return torch.cat(
        [model(x.to(device), d.to(device)).cpu() for x, d in zip(embeddings.split(512), domains.split(512))]
    ).numpy().ravel()


@torch.no_grad()
def predict_model_heads(path: Path, embeddings: torch.Tensor, device: torch.device) -> np.ndarray:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ModelHeadRouter(
        embedding_dim=checkpoint["embedding_dim"],
        projection_dim=checkpoint["projection_dim"],
        head_dim=checkpoint["head_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in embeddings.split(512)]).numpy().ravel()


def quality(labels: np.ndarray, chosen: np.ndarray) -> float:
    return float(np.where(chosen, labels, 1 - labels).mean())


def top_budget_quality(labels: np.ndarray, scores: np.ndarray, budget: float) -> float:
    count = round(budget * len(labels))
    chosen = np.zeros(len(labels), dtype=bool)
    chosen[np.argsort(scores, kind="stable")[-count:]] = True
    return quality(labels, chosen)


def paired_bootstrap_difference(
    labels: np.ndarray, candidate: np.ndarray, reference: np.ndarray, budget: float, repetitions: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for iteration in range(repetitions):
        indices = rng.integers(0, len(labels), size=len(labels))
        differences[iteration] = (
            top_budget_quality(labels[indices], candidate[indices], budget)
            - top_budget_quality(labels[indices], reference[indices], budget)
        )
    return {
        "observed_quality_difference": top_budget_quality(labels, candidate, budget)
        - top_budget_quality(labels, reference, budget),
        "paired_bootstrap_95_percent_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        "bootstrap_repetitions": repetitions,
    }


def compare_slice(
    labels: np.ndarray, predictions: dict[str, np.ndarray], budget: float, repetitions: int, seed: int
) -> dict:
    n = len(labels)
    k = round(budget * n)
    rng = np.random.default_rng(seed)
    random_qualities = np.empty(repetitions)
    for iteration in range(repetitions):
        chosen = np.zeros(n, dtype=bool)
        chosen[rng.choice(n, size=k, replace=False)] = True
        random_qualities[iteration] = quality(labels, chosen)

    result = {
        "n": n,
        "strong_call_count": k,
        "actual_strong_call_fraction": k / n,
        "always_weak_quality": float((1 - labels).mean()),
        "always_strong_quality": float(labels.mean()),
        "random": {
            "mean_quality": float(random_qualities.mean()),
            "expected_quality_exact": float((1 - labels).mean() + k / n * (2 * labels.mean() - 1)),
            "quality_95_percent_interval": np.quantile(random_qualities, [0.025, 0.975]).tolist(),
            "repetitions": repetitions,
        },
        "routers": {},
    }
    for name, scores in predictions.items():
        observed = top_budget_quality(labels, scores, budget)
        result["routers"][name] = {
            "quality": observed,
            "quality_gain_vs_random_expectation": observed - result["random"]["expected_quality_exact"],
            "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            "random_assignments_at_least_as_good": int(np.count_nonzero(random_qualities >= observed)),
            "monte_carlo_p_value": float((1 + np.count_nonzero(random_qualities >= observed)) / (repetitions + 1)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-heads-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/random_router_50pct/results.json")
    args = parser.parse_args()
    if not 0 < args.budget < 1 or args.repetitions < 1:
        parser.error("Budget must be between 0 and 1 and repetitions must be positive.")

    frame = pd.read_parquet(ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/router_test.parquet")
    labels = frame["binary_label"].to_numpy(dtype=np.int64)
    embeddings = torch.load(
        ROOT / "data/original_gpt4_llama7b/bge_embeddings/router_test_bge_large_embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )
    if len(embeddings) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Embedding row count or binary labels do not match the test data.")
    domain_ids = np.array([_domain_id(prompt) for prompt in frame["prompt"]], dtype=np.int64)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = {
        "baseline_256": predict_baseline(
            ROOT / "router/baseline/checkpoints_bge_modeldim_256/best_model.pt", embeddings, device
        ),
        "domain_256": predict_domain(
            ROOT / "router/domain_specific/checkpoints_domain_mf_bge_modeldim_256/best_model.pt",
            embeddings,
            torch.from_numpy(domain_ids),
            device,
        ),
    }
    model_heads_name = None
    if args.model_heads_checkpoint is not None:
        head_width = torch.load(args.model_heads_checkpoint, map_location="cpu", weights_only=False)["head_dim"]
        model_heads_name = f"model_heads_width_{head_width}"
        scores[model_heads_name] = predict_model_heads(args.model_heads_checkpoint, embeddings, device)
    masks = {"overall": np.ones(len(labels), dtype=bool), "math": domain_ids == 1, "general": domain_ids == 0}
    result = {
        "budget_set_before_test_run": args.budget,
        "random_seed": args.seed,
        "note": "Quality is agreement with binary pairwise preference, not task correctness or monetary cost.",
        "slices": {
            name: compare_slice(labels[mask], {key: value[mask] for key, value in scores.items()}, args.budget, args.repetitions, args.seed)
            for name, mask in masks.items()
        },
    }
    if model_heads_name is not None:
        result["paired_differences"] = {
            name: paired_bootstrap_difference(
                labels,
                scores[model_heads_name],
                scores[name],
                args.budget,
                2000,
                args.seed + 1,
            )
            for name in ("baseline_256", "domain_256")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
