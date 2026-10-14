"""Train a CatBoost quality predictor from cached Laya features and benchmark scores."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from laya_mf_local import FEATURE_NAMES, MAX_INPUT_TOKENS, MODEL_POOL, load_features


def build_frame(query_features: np.ndarray, query_ids: list[int], model_ids: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame(query_features[query_ids], columns=FEATURE_NAMES, copy=False)
    frame["model"] = pd.Categorical(
        [MODEL_POOL[index] for index in model_ids], categories=MODEL_POOL
    )
    return frame


def evaluate_predictions(test_rows: list[tuple], query_model_scores: np.ndarray,
                         query_ordinals: dict[int, int]) -> dict:
    by_sample = defaultdict(list)
    for query_index, model_index, score, dataset, sample_key, cost in test_rows:
        by_sample[sample_key].append((query_index, model_index, float(score), dataset, cost))

    chosen_scores = []
    random_scores = []
    oracle_scores = []
    selected_costs = []
    global_covered = 0
    by_dataset = defaultdict(list)
    for available in by_sample.values():
        query_index, _, _, dataset, _ = available[0]
        predictions = query_model_scores[query_ordinals[query_index]]
        global_choice = int(np.argmax(predictions))
        global_covered += any(row[1] == global_choice for row in available)
        selected = max(available, key=lambda row: predictions[row[1]])
        chosen_scores.append(selected[2])
        random_scores.append(float(np.mean([row[2] for row in available])))
        oracle_scores.append(max(row[2] for row in available))
        if selected[4] is not None:
            selected_costs.append(float(selected[4]))
        by_dataset[dataset].append(selected[2])

    return {
        "evaluated_question_instances": len(by_sample),
        "observed_candidate_mean_score": float(np.mean(chosen_scores)),
        "observed_candidate_macro_dataset_score": float(np.mean(
            [np.mean(values) for values in by_dataset.values()])),
        "random_observed_candidate_mean_score": float(np.mean(random_scores)),
        "observed_oracle_mean_score": float(np.mean(oracle_scores)),
        "global_argmax_outcome_coverage": global_covered / len(by_sample),
        "selected_recorded_cost_mean": float(np.mean(selected_costs)) if selected_costs else None,
        "per_dataset": {
            dataset: {"questions": len(values), "mean_score": float(np.mean(values))}
            for dataset, values in sorted(by_dataset.items())
        },
    }


def train(args: argparse.Namespace) -> None:
    db_path = args.work_dir / "benchmark.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"Benchmark database not found: {db_path}; run prepare first")

    feature_map, discarded = load_features(args.work_dir)
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    query_hashes = [row[0] for row in db.execute("SELECT query_hash FROM queries ORDER BY query_hash")]
    known_hashes = set(query_hashes)
    if known_hashes - set(feature_map) - discarded or (set(feature_map) | discarded) - known_hashes:
        raise ValueError("Laya feature/discard lists do not match the prepared question export")

    kept_hashes = [query_hash for query_hash in query_hashes if query_hash in feature_map]
    query_index = {query_hash: index for index, query_hash in enumerate(kept_hashes)}
    if not query_index:
        raise ValueError(f"The {MAX_INPUT_TOKENS}-token limit discarded every question")
    query_features = np.asarray([feature_map[key] for key in kept_hashes], dtype=np.float32)
    model_index = {name: index for index, name in enumerate(MODEL_POOL)}
    partitions = {name: [] for name in ("train", "validation", "test")}
    dataset_counts = Counter()
    discarded_outcomes = 0
    sql = """SELECT s.sample_key, s.query_hash, s.dataset, q.partition, o.model, o.score, o.cost
             FROM outcomes o JOIN samples s USING(sample_key) JOIN queries q USING(query_hash)"""
    for sample_key, query_hash, dataset, part, model, score, cost in db.execute(sql):
        if query_hash in discarded:
            discarded_outcomes += 1
            continue
        row = (query_index[query_hash], model_index[model], float(score), dataset, sample_key, cost)
        partitions[part].append(row)
        if part == "train":
            dataset_counts[dataset] += 1
    db.close()
    if not all(partitions.values()):
        raise ValueError("Train, validation, and test partitions must all contain outcomes")

    train_rows = partitions["train"]
    train_weights = np.asarray([1.0 / dataset_counts[row[3]] for row in train_rows], dtype=np.float32)
    train_weights *= len(train_weights) / train_weights.sum()
    train_x = build_frame(query_features,
                          [row[0] for row in train_rows],
                          [row[1] for row in train_rows])
    train_y = np.asarray([row[2] for row in train_rows], dtype=np.float32)
    train_pool = Pool(train_x, label=train_y, cat_features=["model"], weight=train_weights)

    validation_rows = partitions["validation"]
    validation_x = build_frame(query_features,
                               [row[0] for row in validation_rows],
                               [row[1] for row in validation_rows])
    validation_y = np.asarray([row[2] for row in validation_rows], dtype=np.float32)
    validation_pool = Pool(validation_x, label=validation_y, cat_features=["model"])

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=args.seed,
        thread_count=args.thread_count,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=validation_pool,
              early_stopping_rounds=args.early_stopping_rounds,
              use_best_model=True)

    # Score every model for each held-out query once. Evaluation then chooses among
    # the models that actually have a recorded benchmark score for each sample.
    test_query_ids = sorted({row[0] for row in partitions["test"]})
    query_ordinals = {query_id: ordinal for ordinal, query_id in enumerate(test_query_ids)}
    n_models = len(MODEL_POOL)
    all_test_queries = np.repeat(test_query_ids, n_models).tolist()
    all_test_models = list(range(n_models)) * len(test_query_ids)
    test_all_x = build_frame(query_features, all_test_queries, all_test_models)
    all_predictions = model.predict(test_all_x).reshape(len(test_query_ids), n_models)
    metrics = evaluate_predictions(partitions["test"], all_predictions, query_ordinals)
    metrics.update({
        "best_iteration": int(model.get_best_iteration()),
        "best_validation_rmse": float(model.get_best_score()["validation"]["RMSE"]),
        "train_outcomes": len(partitions["train"]),
        "validation_outcomes": len(validation_rows),
        "test_outcomes": len(partitions["test"]),
        "token_limit": MAX_INPUT_TOKENS,
        "discarded_questions_over_token_limit": len(discarded),
        "discarded_outcomes": discarded_outcomes,
        "catboost_depth": args.depth,
        "catboost_iterations_requested": args.iterations,
    })

    model_path = args.work_dir / "laya_catboost.cbm"
    metrics_path = args.work_dir / "catboost_test_metrics.json"
    model.save_model(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved CatBoost model: {model_path}", flush=True)
    print(f"Saved held-out metrics: {metrics_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).with_name("work"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=75)
    parser.add_argument("--thread-count", type=int, default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
