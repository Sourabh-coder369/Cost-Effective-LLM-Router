"""Compute RouteLLM Call-Performance Threshold (CPT) metrics for BGE routers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
from baseline.model import MatrixFactorizationRouter
from domain_specific.domain_dataset import _domain_id
from domain_specific.domain_model import DomainMFRouter

DATA = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/router_test.parquet"
EMBEDDINGS = ROOT / "data/original_gpt4_llama7b/bge_embeddings/router_test_bge_large_embeddings.pt"
RUNS = {
    "128": {
        "baseline": ROOT / "router/baseline/checkpoints_bge_modeldim_128/best_model.pt",
        "domain": ROOT / "router/domain_specific/checkpoints_domain_mf_bge_modeldim_128/best_model.pt",
    },
    "256": {
        "baseline": ROOT / "router/baseline/checkpoints_bge_modeldim_256/best_model.pt",
        "domain": ROOT / "router/domain_specific/checkpoints_domain_mf_bge_modeldim_256/best_model.pt",
    },
}
OUTPUT = ROOT / "router/routellm_cpt_results_bge.json"


@torch.no_grad()
def predict_baseline(path: Path, embeddings: torch.Tensor, device: torch.device) -> np.ndarray:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    model = MatrixFactorizationRouter(
        query_embedding_dim=state["text_proj.0.weight"].shape[1],
        model_embedding_dim=ckpt["config"]["model_embedding_dim"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return torch.cat([model(batch.to(device)) for batch in embeddings.split(512)]).cpu().numpy().reshape(-1)


@torch.no_grad()
def predict_domain(path: Path, embeddings: torch.Tensor, domains: torch.Tensor, device: torch.device) -> np.ndarray:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    model = DomainMFRouter(
        query_embedding_dim=state["text_proj.0.weight"].shape[1],
        model_embedding_dim=state["model_domain_embeddings.weight"].shape[1],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return torch.cat([
        model(batch.to(device), domain_batch.to(device))
        for batch, domain_batch in zip(embeddings.split(512), domains.split(512))
    ]).cpu().numpy().reshape(-1)


def cpt_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Exact prefix sweep: route the top-k strongest scores to the strong model."""
    count = len(labels)
    strong_quality = float(labels.mean())
    weak_quality = float((1 - labels).mean())
    gap = strong_quality - weak_quality
    if gap <= 0:
        raise ValueError("CPT is undefined when the strong model is not better than the weak model.")

    order = np.argsort(probabilities)[::-1]
    gains = 2 * labels[order] - 1
    qualities = weak_quality + np.concatenate(([0.0], np.cumsum(gains) / count))
    pgr = (qualities - weak_quality) / gap
    call_rates = np.arange(count + 1) / count

    def cpt(target: float) -> dict:
        index = int(np.flatnonzero(pgr >= target)[0])
        threshold = float("inf") if index == 0 else float(probabilities[order[index - 1]])
        return {
            "strong_call_pct": float(call_rates[index] * 100),
            "threshold": threshold,
            "achieved_pgr": float(pgr[index]),
            "achieved_quality": float(qualities[index]),
        }

    apgr = float(np.mean([pgr[max(1, round(count * rate))] for rate in np.linspace(0.1, 1.0, 10)]))
    return {
        "examples": count,
        "strong_quality": strong_quality,
        "weak_quality": weak_quality,
        "cpt_50": cpt(0.50),
        "cpt_80": cpt(0.80),
        "apgr": apgr,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_parquet(DATA)
    labels = frame["binary_label"].to_numpy(dtype=np.int64)
    domain_ids = np.array([_domain_id(prompt) for prompt in frame["prompt"]], dtype=np.int64)
    embeddings = torch.load(EMBEDDINGS, map_location="cpu", weights_only=True)
    domains = torch.tensor(domain_ids, dtype=torch.long)
    masks = {"overall": np.ones(len(labels), dtype=bool), "math": domain_ids == 1, "general": domain_ids == 0}

    result = {"definition": "CPT(x) is the minimum strong-model call percentage achieving PGR >= x.", "device": str(device), "runs": {}}
    for dimension, paths in RUNS.items():
        baseline = predict_baseline(paths["baseline"], embeddings, device)
        domain = predict_domain(paths["domain"], embeddings, domains, device)
        result["runs"][dimension] = {
            subset: {"baseline": cpt_metrics(baseline[mask], labels[mask]), "domain": cpt_metrics(domain[mask], labels[mask])}
            for subset, mask in masks.items()
        }

    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
