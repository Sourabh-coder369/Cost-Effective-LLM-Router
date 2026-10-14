"""Prepare and evaluate binary router experiments from long-form model outcomes.

The script intentionally accepts a small canonical schema instead of depending on
one particular LLMRouterBench release.  One input row represents one model's
outcome on one prompt.  Required fields are ``query_id``, ``prompt``, ``model``
and ``score``; ``cost`` is strongly recommended.  Column names are configurable.

Examples
--------
Prepare leakage-safe splits for a pair of models::

    python evaluation/llmrouterbench_pairwise.py prepare \
      --input outcomes.jsonl --strong-model strong --weak-model weak \
      --output-dir data/routerbench_pair

Evaluate any router probability file (``query_id``, ``probability``)::

    python evaluation/llmrouterbench_pairwise.py evaluate \
      --pair-data data/routerbench_pair/test_iid.parquet \
      --probabilities predictions.csv --output evaluation/pair_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("query_id", "prompt", "model", "score")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient="records", lines=True, force_ascii=False)


def canonicalize(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    mapping = {
        args.query_id_column: "query_id",
        args.prompt_column: "prompt",
        args.model_column: "model",
        args.score_column: "score",
    }
    if args.cost_column and args.cost_column in df.columns:
        mapping[args.cost_column] = "cost"
    if args.domain_column and args.domain_column in df.columns:
        mapping[args.domain_column] = "domain"
    if args.dataset_column and args.dataset_column in df.columns:
        mapping[args.dataset_column] = "dataset"
    if args.response_column and args.response_column in df.columns:
        mapping[args.response_column] = "response"

    missing = [source for source in mapping if source not in df.columns]
    if missing:
        raise ValueError(f"Input is missing configured columns: {missing}")

    result = df.rename(columns=mapping)[list(mapping.values())].copy()
    if "cost" not in result:
        result["cost"] = np.nan
    if "domain" not in result:
        result["domain"] = "unknown"
    if "dataset" not in result:
        result["dataset"] = "unknown"
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    result["cost"] = pd.to_numeric(result["cost"], errors="coerce")
    result = result.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    result["query_id"] = result["query_id"].astype(str)
    result["model"] = result["model"].astype(str)
    return result


def make_pairwise(outcomes: pd.DataFrame, strong_model: str, weak_model: str) -> tuple[pd.DataFrame, dict]:
    chosen = outcomes[outcomes["model"].isin([strong_model, weak_model])].copy()
    if chosen.empty:
        raise ValueError("Neither selected model appears in the input.")

    duplicate_count = int(chosen.duplicated(["query_id", "model"]).sum())
    # Retaining the first avoids silently using outcomes from different retries.
    chosen = chosen.drop_duplicates(["query_id", "model"], keep="first")
    index = ["query_id", "prompt", "dataset", "domain"]
    scores = chosen.pivot(index=index, columns="model", values="score")
    costs = chosen.pivot(index=index, columns="model", values="cost")
    responses = chosen.pivot(index=index, columns="model", values="response") if "response" in chosen else None

    if strong_model not in scores or weak_model not in scores:
        raise ValueError("At least one selected model has no valid score rows.")
    pair = scores[[strong_model, weak_model]].dropna().reset_index()
    pair = pair.rename(columns={strong_model: "strong_score", weak_model: "weak_score"})
    pair["strong_cost"] = costs.reindex(scores.index)[strong_model].reindex(
        pair.set_index(index).index
    ).to_numpy()
    pair["weak_cost"] = costs.reindex(scores.index)[weak_model].reindex(
        pair.set_index(index).index
    ).to_numpy()
    if responses is not None:
        response_index = pair.set_index(index).index
        pair["strong_response"] = responses.reindex(response_index)[strong_model].to_numpy()
        pair["weak_response"] = responses.reindex(response_index)[weak_model].to_numpy()

    # On a quality tie, the cheaper option is the desired route.  NaN costs are
    # treated as weak-preferred, matching the original project convention.
    pair["binary_label"] = (pair["strong_score"] > pair["weak_score"]).astype(int)
    pair["quality_gap"] = pair["strong_score"] - pair["weak_score"]
    pair["tie"] = np.isclose(pair["quality_gap"], 0.0)
    meta = {
        "rows_in_input": int(len(outcomes)),
        "rows_for_pair": int(len(chosen)),
        "complete_pair_rows": int(len(pair)),
        "duplicate_query_model_rows_discarded": duplicate_count,
        "strong_model": strong_model,
        "weak_model": weak_model,
        "strong_wins": int(pair["binary_label"].sum()),
        "weak_or_tie_wins": int((1 - pair["binary_label"]).sum()),
        "ties": int(pair["tie"].sum()),
    }
    return pair, meta


def stable_bucket(value: str, seed: int) -> int:
    encoded = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16) % 10_000


def split_pairwise(pair: pd.DataFrame, seed: int, ood_datasets: set[str]) -> dict[str, pd.DataFrame]:
    is_ood = pair["dataset"].astype(str).isin(ood_datasets)
    remaining = pair.loc[~is_ood].copy()
    buckets = remaining["query_id"].map(lambda value: stable_bucket(value, seed))
    # Deterministic group split: all outcomes for an ID would stay together even
    # if this schema is later extended to multiple rows per query.
    split = pd.Series("train", index=remaining.index)
    split.loc[buckets >= 7000] = "validation"
    split.loc[buckets >= 8000] = "test_iid"
    return {
        "train": remaining.loc[split == "train"].reset_index(drop=True),
        "validation": remaining.loc[split == "validation"].reset_index(drop=True),
        "test_iid": remaining.loc[split == "test_iid"].reset_index(drop=True),
        "test_ood": pair.loc[is_ood].reset_index(drop=True),
    }


def prepare(args: argparse.Namespace) -> None:
    outcomes = canonicalize(read_table(args.input), args)
    pair, metadata = make_pairwise(outcomes, args.strong_model, args.weak_model)
    splits = split_pairwise(pair, args.seed, set(args.ood_dataset))
    output_dir = args.output_dir
    for name, split in splits.items():
        if split.empty and name == "test_ood":
            continue
        write_table(split, output_dir / f"{name}.parquet")
        metadata[f"{name}_rows"] = int(len(split))
        metadata[f"{name}_strong_win_rate"] = float(split["binary_label"].mean()) if len(split) else None
    metadata["seed"] = args.seed
    metadata["ood_datasets"] = sorted(args.ood_dataset)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def evaluate_threshold(pair: pd.DataFrame, probabilities: pd.Series, threshold: float) -> dict:
    routed_strong = probabilities.to_numpy() >= threshold
    strong_score = pair["strong_score"].to_numpy(float)
    weak_score = pair["weak_score"].to_numpy(float)
    strong_cost = pair["strong_cost"].to_numpy(float)
    weak_cost = pair["weak_cost"].to_numpy(float)
    selected_quality = np.where(routed_strong, strong_score, weak_score)
    selected_cost = np.where(routed_strong, strong_cost, weak_cost)
    expected = pair["binary_label"].to_numpy(int)
    finite_cost = np.isfinite(selected_cost)
    return {
        "threshold": float(threshold),
        "strong_usage": float(routed_strong.mean()),
        "routing_accuracy": float((routed_strong.astype(int) == expected).mean()),
        "selected_quality": float(selected_quality.mean()),
        "oracle_quality": float(np.maximum(strong_score, weak_score).mean()),
        "quality_regret": float((np.maximum(strong_score, weak_score) - selected_quality).mean()),
        "missed_strong_rate": float(((~routed_strong) & (strong_score > weak_score)).mean()),
        "mean_selected_cost": float(selected_cost[finite_cost].mean()) if finite_cost.any() else None,
    }


def read_probabilities(path: Path, probability_column: str) -> pd.DataFrame:
    result = read_table(path)
    if "query_id" not in result or probability_column not in result:
        raise ValueError("Probability file must contain query_id and the configured probability column.")
    result = result[["query_id", probability_column]].rename(columns={probability_column: "probability"})
    result["query_id"] = result["query_id"].astype(str)
    result["probability"] = pd.to_numeric(result["probability"], errors="coerce")
    return result.dropna().drop_duplicates("query_id", keep="first")


def evaluate(args: argparse.Namespace) -> None:
    pair = read_table(args.pair_data)
    probabilities = read_probabilities(args.probabilities, args.probability_column)
    pair["query_id"] = pair["query_id"].astype(str)
    pair = pair.merge(probabilities, on="query_id", how="inner", validate="one_to_one")
    if pair.empty:
        raise ValueError("No query_id values overlap between pair data and probabilities.")
    missing = [column for column in ("strong_score", "weak_score", "strong_cost", "weak_cost", "binary_label") if column not in pair]
    if missing:
        raise ValueError(f"Pair data lacks required prepared columns: {missing}")

    curve = [evaluate_threshold(pair, pair["probability"], threshold) for threshold in np.linspace(0, 1, args.num_thresholds)]
    chosen = evaluate_threshold(pair, pair["probability"], args.threshold)
    strong_cost = pair["strong_cost"].to_numpy(float)
    finite_strong_cost = np.isfinite(strong_cost)
    result = {
        "num_evaluated_queries": int(len(pair)),
        "selected_threshold": chosen,
        "always_strong": {
            "quality": float(pair["strong_score"].mean()),
            "mean_cost": float(strong_cost[finite_strong_cost].mean()) if finite_strong_cost.any() else None,
        },
        "always_weak": {
            "quality": float(pair["weak_score"].mean()),
            "mean_cost": float(pair["weak_cost"].dropna().mean()) if pair["weak_cost"].notna().any() else None,
        },
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["selected_threshold"], indent=2))
    print(f"Saved complete curve to {args.output}")


def common_columns(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-id-column", default="query_id")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--model-column", default="model")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--cost-column", default="cost")
    parser.add_argument("--domain-column", default="domain")
    parser.add_argument("--dataset-column", default="dataset")
    parser.add_argument("--response-column", default="response")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="convert long-form outcomes to router pair splits")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--strong-model", required=True)
    prepare_parser.add_argument("--weak-model", required=True)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--ood-dataset", action="append", default=[])
    common_columns(prepare_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="score a router's probabilities against stored outcomes")
    evaluate_parser.add_argument("--pair-data", type=Path, required=True)
    evaluate_parser.add_argument("--probabilities", type=Path, required=True)
    evaluate_parser.add_argument("--probability-column", default="probability")
    evaluate_parser.add_argument("--threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--num-thresholds", type=int, default=101)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
