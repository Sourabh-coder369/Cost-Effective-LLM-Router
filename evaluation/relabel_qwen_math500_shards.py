"""Relabel saved MATH-500 shard records from their stored Qwen responses.

Early collections used ``answer_kind=scalar`` for MATH-500 and accidentally
compared its direct gold answer against the complete Qwen response. This tool
does not call Qwen again: it reparses the stored response, updates labels, and
writes a new shard directory without altering the original data.

Example:
    python evaluation/relabel_qwen_math500_shards.py \
      --input-dir data/qwen_math_other \
      --output-dir data/qwen_math_other_relabelled
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "evaluation"))
from generate_qwen_math_layer_dataset import answers_match, extract_answer_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_id = {}
    with (args.input_dir / "metadata.jsonl").open("r", encoding="utf-8") as source_metadata:
        for line in source_metadata:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source") == "math500":
                gold = extract_answer_text(str(record["gold_answer"]), answer_kind="scalar")
                predicted = extract_answer_text(str(record["qwen_response"]), answer_kind="auto")
                is_correct = answers_match(gold, predicted)
                record["predicted_answer"] = predicted
                record["is_correct"] = bool(is_correct)
                record["strong_model_wins"] = bool(not is_correct)
            metadata_by_id[str(record["record_id"])] = record

    updated = 0
    correct = 0
    total_math500 = 0
    shard_summaries = []
    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as target_metadata:
        for shard_path in sorted(args.input_dir.glob("embeddings_*.pt")):
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            record_ids = [str(item) for item in payload["record_ids"]]
            for row, record_id in enumerate(record_ids):
                record = metadata_by_id[record_id]
                if record.get("source") == "math500":
                    total_math500 += 1
                    updated += 1
                    correct += int(record["is_correct"])
                    payload["is_correct"][row] = record["is_correct"]
                    payload["strong_model_wins"][row] = record["strong_model_wins"]
                target_metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
            torch.save(payload, args.output_dir / shard_path.name)
            shard_summaries.append({"file": shard_path.name, "count": len(record_ids)})

    input_manifest = args.input_dir / "manifest.json"
    manifest = json.loads(input_manifest.read_text(encoding="utf-8")) if input_manifest.exists() else {}
    manifest.update(
        {
            "relabelled_math500_records": updated,
            "math500_qwen_correct": correct,
            "math500_qwen_failure": total_math500 - correct,
            "math500_relabel_note": "Gold scalar answer compared with a parsed Qwen response, not the complete response text.",
            "shards": shard_summaries,
        }
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "relabelled_math500_records": updated,
                "math500_qwen_correct": correct,
                "math500_qwen_failure": total_math500 - correct,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
