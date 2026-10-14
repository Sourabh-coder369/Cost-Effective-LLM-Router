"""Compare non-cosine Qwen trajectory features with the cosine baseline.

All variants use the same saved Qwen layer representations, the same
train/validation/test split, and the same MatrixFactorizationRouter. The
positive label is ``strong_model_wins``: Qwen failed and should be escalated.

Feature sets:
  - cosine_baseline: adjacent-layer cosine similarities (reference)
  - layer_norms: L2 norm at every layer
  - adjacent_l2_change: L2 distance between adjacent layers
  - input_drift: L2 distance from the input embedding to each later layer
  - noncosine_combined: norms, L2 changes, drift, mean, and standard deviation
  - mean_pooled_noncosine_combined: same statistics for mean-pooled prompts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "evaluation"))
from train_qwen_layer_cosine_mf import (  # noqa: E402
    adjacent_cosines,
    run_gsm8k_ood,
    run_iid,
    set_seed,
)


def source_from_id(record_id: str) -> str:
    return record_id.split(":", 1)[0] if ":" in record_id else "unknown"


def noncosine_features(layer_vectors: torch.Tensor) -> dict[str, np.ndarray]:
    """Create trajectory features from [batch, layers, hidden] vectors."""
    x = layer_vectors.float()
    norms = torch.linalg.vector_norm(x, dim=-1)
    adjacent_l2 = torch.linalg.vector_norm(x[:, 1:] - x[:, :-1], dim=-1)
    input_drift = torch.linalg.vector_norm(x[:, 1:] - x[:, :1], dim=-1)
    component_mean = x.mean(dim=-1)
    component_std = x.std(dim=-1)
    combined = torch.cat((norms, adjacent_l2, input_drift, component_mean, component_std), dim=1)
    return {
        "layer_norms": norms.numpy().astype(np.float32),
        "adjacent_l2_change": adjacent_l2.numpy().astype(np.float32),
        "input_drift": input_drift.numpy().astype(np.float32),
        "noncosine_combined": combined.numpy().astype(np.float32),
    }


def load_feature_sets(input_dirs: list[Path]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str], int]:
    parts: dict[str, list[np.ndarray]] = {
        "last_cosine_baseline": [],
        "last_layer_norms": [],
        "last_adjacent_l2_change": [],
        "last_input_drift": [],
        "last_noncosine_combined": [],
        "mean_pooled_noncosine_combined": [],
    }
    labels, sources, record_ids = [], [], []
    seen = set()
    skipped_duplicates = 0

    for input_dir in input_dirs:
        shards = sorted(input_dir.glob("embeddings_*.pt"))
        if not shards:
            raise FileNotFoundError(f"No embedding shards found in {input_dir}")
        print(f"Extracting features from {len(shards)} shards in {input_dir}", flush=True)
        for shard_path in shards:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            ids = [str(item) for item in payload["record_ids"]]
            keep = np.asarray([item not in seen for item in ids], dtype=bool)
            skipped_duplicates += int((~keep).sum())
            for item, include in zip(ids, keep):
                if include:
                    seen.add(item)
            if not keep.any():
                continue

            last = payload["layer_last_token_embeddings"]
            mean = payload["layer_embeddings"]
            last_noncosine = noncosine_features(last)
            mean_noncosine = noncosine_features(mean)
            parts["last_cosine_baseline"].append(adjacent_cosines(last)[keep].numpy().astype(np.float32))
            parts["last_layer_norms"].append(last_noncosine["layer_norms"][keep])
            parts["last_adjacent_l2_change"].append(last_noncosine["adjacent_l2_change"][keep])
            parts["last_input_drift"].append(last_noncosine["input_drift"][keep])
            parts["last_noncosine_combined"].append(last_noncosine["noncosine_combined"][keep])
            parts["mean_pooled_noncosine_combined"].append(mean_noncosine["noncosine_combined"][keep])
            labels.append(payload["strong_model_wins"].cpu().numpy().astype(np.int64)[keep])
            record_ids.extend(item for item, include in zip(ids, keep) if include)
            sources.extend(source_from_id(item) for item, include in zip(ids, keep) if include)

    return (
        {name: np.concatenate(values) for name, values in parts.items()},
        np.concatenate(labels),
        np.asarray(sources),
        record_ids,
        skipped_duplicates,
    )


def remove_checkpoint(result: dict) -> dict:
    result = dict(result)
    result.pop("checkpoint", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen_trajectory_ablation"))
    parser.add_argument("--model-embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets, labels, sources, record_ids, skipped = load_feature_sets(args.input_dir)
    source_counts = {source: int((sources == source).sum()) for source in sorted(set(sources))}
    results = {
        "n_examples": int(len(labels)),
        "source_counts": source_counts,
        "failure_rate": float(labels.mean()),
        "skipped_duplicate_record_ids": skipped,
        "features": {},
    }

    for name, features in feature_sets.items():
        print(f"\nTraining {name} ({features.shape[1]} features)", flush=True)
        iid = run_iid(features, labels, sources, args)
        iid.pop("checkpoint")
        ood = run_gsm8k_ood(features, labels, sources, args)
        iid_metric = iid["test_at_validation_f1_threshold"]
        ood_metric = ood["ood_at_gsm8k_f1_threshold"]
        results["features"][name] = {"feature_dim": int(features.shape[1]), "iid": iid, "gsm8k_to_other_ood": ood}
        print(
            f"{name}: IID AUC={iid_metric['roc_auc']:.4f}, F1={iid_metric['f1']:.4f}; "
            f"OOD AUC={ood_metric['roc_auc']:.4f}, F1={ood_metric['f1']:.4f}",
            flush=True,
        )

    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "feature_sets.npz", labels=labels, sources=sources, record_ids=np.asarray(record_ids), **feature_sets)
    print(f"\nSaved ablation results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
