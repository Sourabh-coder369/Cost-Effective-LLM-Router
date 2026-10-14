"""Compare downloaded Qwen annotations against this project's existing regex rules."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def regex_functions(root: Path) -> dict:
    """Load the actual regex definitions without importing the GPU dataset loader."""
    path = root / "router/domain_specific/domain_dataset.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"MATH_PATTERNS", "MIN_MATH_SCORE"}
            for target in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in {"_extract_text", "_math_score", "_domain_id"}:
            selected.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/kaggle_qwen_labels/comparison"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    regex = regex_functions(root)
    manifest = json.loads((args.labels_dir / "manifest.json").read_text(encoding="utf-8"))
    tables, validation = [], []
    for split in ("train", "val", "test"):
        original_path = args.original_dir / f"router_{split}.parquet"
        original = pd.read_parquet(original_path)
        matches = list(args.labels_dir.glob(f"*_router_{split}/labeled.parquet"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one labeled file for {split}")
        labeled = pd.read_parquet(matches[0])
        pd.testing.assert_frame_equal(original.reset_index(drop=True),
                                      labeled[original.columns].reset_index(drop=True), check_dtype=False)
        digest = hashlib.sha256(original_path.read_bytes()).hexdigest()
        saved = next(item for item in manifest["inputs"] if item["name"] == original_path.name)
        if saved["sha256"] != digest:
            raise ValueError(f"Input hash mismatch: {original_path}")
        validation.append({"split": split, "rows": len(original), "original_columns_and_order_match": True,
                           "original_sha256_matches_manifest": True})
        labeled["split"] = split
        labeled["source_row"] = np.arange(len(labeled))
        labeled["question"] = labeled["prompt"].fillna("").map(regex["_extract_text"])
        labeled["regex_score"] = labeled["question"].map(regex["_math_score"])
        labeled["regex_domain"] = labeled["prompt"].fillna("").map(regex["_domain_id"]).map({0: "general", 1: "math"})
        labeled["regex_matches"] = labeled["question"].map(
            lambda text: ", ".join(name for name, (_, pattern) in regex["MATH_PATTERNS"].items() if pattern.search(text))
        )
        valid = labeled["qwen_label_status"].eq("ok")
        if not labeled.loc[valid, "qwen_domain"].isin(["math", "general"]).all():
            raise ValueError("An ok row lacks a valid Qwen domain")
        expected_ids = labeled.loc[valid, "qwen_domain"].map({"general": 0, "math": 1})
        if not (labeled.loc[valid, "qwen_domain_id"].astype(int).to_numpy() == expected_ids.to_numpy()).all():
            raise ValueError("Qwen domain string and ID disagree")
        labeled["comparison"] = np.where(
            valid, labeled["regex_domain"] + "_to_" + labeled["qwen_domain"].fillna("review"), "needs_review",
        )
        tables.append(labeled)
    combined = pd.concat(tables, ignore_index=True)
    report = {"model": manifest["model"], "revision": manifest["revision"], "rubric": manifest["rubric"],
              "validation": validation, "splits": {}}
    for split in ("train", "val", "test", "all"):
        data = combined if split == "all" else combined.loc[combined["split"] == split]
        accepted = data.loc[data["qwen_label_status"] == "ok"]
        counts = accepted["comparison"].value_counts().to_dict()
        agreement = int((accepted["regex_domain"] == accepted["qwen_domain"]).sum())
        n = len(accepted)
        regex_math = int(accepted["regex_domain"].eq("math").sum())
        qwen_math = int(accepted["qwen_domain"].eq("math").sum())
        expected = ((regex_math / n) * (qwen_math / n) + (1 - regex_math / n) * (1 - qwen_math / n)) if n else 0
        patterns = data.loc[data["comparison"] == "math_to_general", "regex_matches"].str.split(", ").explode().value_counts().to_dict()
        report["splits"][split] = {
            "rows": len(data), "accepted_qwen_rows": n, "statuses": data["qwen_label_status"].value_counts().to_dict(),
            "regex_math_all_rows": int(data["regex_domain"].eq("math").sum()),
            "regex_math_accepted_rows": regex_math, "qwen_math_accepted_rows": qwen_math,
            "agreement_rows": agreement, "agreement_percent": 100 * agreement / n if n else None,
            "disagreement_rows": n - agreement,
            "cohen_kappa": ((agreement / n - expected) / (1 - expected)) if n and expected < 1 else None,
            "matrix": {key: int(counts.get(key, 0)) for key in
                       ("general_to_general", "general_to_math", "math_to_general", "math_to_math")},
            "regex_patterns_in_math_to_general": patterns,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = ["split", "source_row", "question", "regex_domain", "regex_score", "regex_matches",
              "qwen_domain", "qwen_label_status", "qwen_raw_label", "comparison"]
    disagreements = combined.loc[combined["comparison"].isin(["general_to_math", "math_to_general"])]
    disagreements[fields].to_csv(args.output_dir / "disagreements.csv", index=False)
    combined.loc[combined["comparison"] == "needs_review", fields].to_csv(args.output_dir / "needs_review.csv", index=False)
    samples = [group.sample(min(50, len(group)), random_state=42) for _, group in disagreements.groupby("comparison")]
    pd.concat(samples)[fields].to_csv(args.output_dir / "disagreement_sample.csv", index=False)
    combined[fields].to_parquet(args.output_dir / "all_comparisons.parquet", index=False)
    print(json.dumps({key: value for key, value in report.items() if key != "rubric"}, indent=2))


if __name__ == "__main__":
    main()
