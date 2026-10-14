"""CPU checks for annotation output handling, alignment, and resume checkpoints."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from label_math_general_qwen_kaggle import atomic_json, export_frame, extract_request, load_checkpoint_records, parse_label


class DomainLabelingTests(unittest.TestCase):
    def test_only_complete_labels_are_accepted(self):
        self.assertEqual(parse_label(" Math\n"), ("math", "ok"))
        self.assertEqual(parse_label("general"), ("general", "ok"))
        self.assertEqual(parse_label("uncertain"), (None, "uncertain"))
        for response in ("not math", "math or general", "math. Because...", "", "<think>math</think>general"):
            self.assertEqual(parse_label(response), (None, "invalid_output"))

    def test_extracts_last_user_turn_without_assistant_answer(self):
        raw = "Human: first question Assistant: answer Human: solve x + 2 = 3 Assistant: x = 1"
        self.assertEqual(extract_request(raw), "solve x + 2 = 3")
        self.assertEqual(extract_request(None), "")
        self.assertEqual(extract_request(float('nan')), "")

    def test_resume_and_export_preserve_row_alignment_and_routing_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = pd.DataFrame({"prompt": ["first", "second", "third"], "binary_label": [0, 1, 0]})
            # GPU completion/checkpoint order can differ from the source order.
            atomic_json(directory / "batch_0000000.json", [
                {"row": 2, "label": None, "status": "uncertain", "raw": "uncertain", "input_tokens": 3},
                {"row": 0, "label": "math", "status": "ok", "raw": "math", "input_tokens": 4},
            ])
            # A partially written temporary checkpoint must not count as completed.
            (directory / "batch_0000001.tmp").write_text("[", encoding="utf-8")
            records = load_checkpoint_records(directory, len(frame))
            self.assertEqual(set(records), {0, 2})
            atomic_json(directory / "batch_0000001.json", [
                {"row": 1, "label": "general", "status": "ok", "raw": "general", "input_tokens": 5},
            ])
            records = load_checkpoint_records(directory, len(frame))
            summary = export_frame(frame, records, directory, "test-commit")
            result = pd.read_parquet(directory / "labeled.parquet")
            self.assertEqual(result["prompt"].tolist(), frame["prompt"].tolist())
            self.assertEqual(result["binary_label"].tolist(), [0, 1, 0])
            self.assertEqual(result["qwen_domain_id"].iloc[:2].tolist(), [1, 0])
            self.assertTrue(pd.isna(result["qwen_domain_id"].iloc[2]))
            self.assertEqual(pd.read_csv(directory / "needs_review.csv")["prompt"].tolist(), ["third"])
            self.assertEqual(summary["statuses"], {"ok": 2, "uncertain": 1})
            self.assertEqual(json.loads((directory / "summary.json").read_text())["rows"], 3)

    def test_resume_rejects_duplicate_or_out_of_range_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            atomic_json(directory / "batch_0000000.json", [{"row": 0}, {"row": 0}])
            with self.assertRaises(ValueError):
                load_checkpoint_records(directory, 2)
            atomic_json(directory / "batch_0000000.json", [{"row": 2}])
            with self.assertRaises(ValueError):
                load_checkpoint_records(directory, 2)


if __name__ == "__main__":
    unittest.main()
