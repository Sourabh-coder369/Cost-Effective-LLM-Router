"""Regression test for the generic pairwise routing-data adapter."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from llmrouterbench_pairwise import evaluate_threshold, make_pairwise, split_pairwise


class PairwiseAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outcomes = pd.DataFrame(
            [
                {"query_id": "q1", "prompt": "one", "model": "strong", "score": 1, "cost": 10, "dataset": "math", "domain": "math"},
                {"query_id": "q1", "prompt": "one", "model": "weak", "score": 0, "cost": 1, "dataset": "math", "domain": "math"},
                {"query_id": "q2", "prompt": "two", "model": "strong", "score": 0, "cost": 10, "dataset": "code", "domain": "code"},
                {"query_id": "q2", "prompt": "two", "model": "weak", "score": 1, "cost": 1, "dataset": "code", "domain": "code"},
                {"query_id": "q3", "prompt": "three", "model": "strong", "score": 1, "cost": 10, "dataset": "math", "domain": "math"},
                {"query_id": "q3", "prompt": "three", "model": "weak", "score": 1, "cost": 1, "dataset": "math", "domain": "math"},
            ]
        )

    def test_pair_label_and_quality_cost_metrics(self) -> None:
        pair, metadata = make_pairwise(self.outcomes, "strong", "weak")
        self.assertEqual(metadata["complete_pair_rows"], 3)
        self.assertEqual(pair["binary_label"].tolist(), [1, 0, 0])
        metrics = evaluate_threshold(pair, pd.Series([0.9, 0.1, 0.1]), threshold=0.5)
        self.assertEqual(metrics["routing_accuracy"], 1.0)
        self.assertEqual(metrics["selected_quality"], 1.0)
        self.assertEqual(metrics["mean_selected_cost"], 4.0)
        self.assertEqual(metrics["quality_regret"], 0.0)

    def test_ood_dataset_is_never_in_training(self) -> None:
        pair, _ = make_pairwise(self.outcomes, "strong", "weak")
        splits = split_pairwise(pair, seed=42, ood_datasets={"code"})
        self.assertEqual(splits["test_ood"]["query_id"].tolist(), ["q2"])
        self.assertNotIn("q2", set(splits["train"]["query_id"]))


if __name__ == "__main__":
    unittest.main()
