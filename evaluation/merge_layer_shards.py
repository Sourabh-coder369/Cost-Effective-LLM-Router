"""Stream-merge layer-embedding shard directories into one dataset.

This is useful when GSM8K has already been generated and the remaining
datasets are collected separately. Input metadata rows whose referenced shard
or row is missing are skipped, which allows recovery from a run interrupted
after metadata was written but before a partial shard was flushed.

Example:
    python evaluation/merge_layer_shards.py \
      --input-dir /kaggle/working/qwen_math_layers_full_v2 \
      --input-dir /kaggle/working/qwen_math_layers_other \
      --output-dir /kaggle/working/qwen_math_layers_combined
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import torch


def iter_valid_rows(input_dir: Path) -> Iterator[tuple[dict[str, Any], dict[str, torch.Tensor]]]:
    metadata_path = input_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl: {metadata_path}")

    cached_name: str | None = None
    cached_payload: dict[str, torch.Tensor] | None = None
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue
            metadata = json.loads(line)
            shard_name = str(metadata["embedding_shard"])
            row = int(metadata["embedding_row"])
            if shard_name != cached_name:
                shard_path = input_dir / shard_name
                if not shard_path.exists():
                    print(f"Skipping {input_dir.name}:{line_number}; missing {shard_name}", flush=True)
                    continue
                cached_payload = torch.load(shard_path, map_location="cpu", weights_only=False)
                cached_name = shard_name
            assert cached_payload is not None
            count = int(cached_payload["layer_embeddings"].shape[0])
            if row < 0 or row >= count:
                print(f"Skipping {input_dir.name}:{line_number}; row {row} outside {shard_name}", flush=True)
                continue

            row_payload = {
                "layer_embeddings": cached_payload["layer_embeddings"][row],
                "layer_last_token_embeddings": cached_payload["layer_last_token_embeddings"][row],
                "is_correct": cached_payload["is_correct"][row],
                "strong_model_wins": cached_payload["strong_model_wins"][row],
            }
            yield metadata, row_payload


def flush(
    output_dir: Path,
    shard_index: int,
    metadata_rows: list[dict[str, Any]],
    rows: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    shard_name = f"embeddings_{shard_index:05d}.pt"
    payload = {
        "record_ids": [row["record_id"] for row in metadata_rows],
        "layer_embeddings": torch.stack([row["layer_embeddings"] for row in rows]),
        "layer_last_token_embeddings": torch.stack([row["layer_last_token_embeddings"] for row in rows]),
        "is_correct": torch.stack([row["is_correct"] for row in rows]).to(torch.bool),
        "strong_model_wins": torch.stack([row["strong_model_wins"] for row in rows]).to(torch.bool),
    }
    torch.save(payload, output_dir / shard_name)
    for row_index, metadata in enumerate(metadata_rows):
        metadata["embedding_shard"] = shard_name
        metadata["embedding_row"] = row_index
    return {
        "file": shard_name,
        "count": len(metadata_rows),
        "layer_embedding_shape": list(payload["layer_embeddings"].shape[1:]),
        "dtype": str(payload["layer_embeddings"].dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=128)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    metadata_path = output_dir / "metadata.jsonl"
    pending_metadata: list[dict[str, Any]] = []
    pending_rows: list[dict[str, torch.Tensor]] = []
    shards: list[dict[str, Any]] = []
    total_seen = 0
    total_written = 0
    skipped = 0
    first_shape: tuple[int, ...] | None = None

    with metadata_path.open("w", encoding="utf-8") as output_metadata:
        for input_dir in args.input_dir:
            print(f"Reading {input_dir}", flush=True)
            before = total_written
            for metadata, row in iter_valid_rows(input_dir):
                total_seen += 1
                shape = tuple(row["layer_embeddings"].shape)
                if first_shape is None:
                    first_shape = shape
                elif shape != first_shape:
                    raise ValueError(f"Embedding shape mismatch: {shape} vs {first_shape}")
                metadata["merged_from"] = input_dir.name
                pending_metadata.append(metadata)
                pending_rows.append(row)
                total_written += 1
                if len(pending_metadata) >= args.shard_size:
                    shard = flush(output_dir, len(shards), pending_metadata, pending_rows)
                    shards.append(shard)
                    for item in pending_metadata:
                        output_metadata.write(json.dumps(item, ensure_ascii=False) + "\n")
                    pending_metadata, pending_rows = [], []
            print(f"Added {total_written - before} rows from {input_dir.name}", flush=True)

    if pending_metadata:
        shard = flush(output_dir, len(shards), pending_metadata, pending_rows)
        shards.append(shard)
        with metadata_path.open("a", encoding="utf-8") as output_metadata:
            for item in pending_metadata:
                output_metadata.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Count rows skipped due to interrupted/incomplete input shards.
    for input_dir in args.input_dir:
        input_metadata = input_dir / "metadata.jsonl"
        if input_metadata.exists():
            total_input = sum(1 for line in input_metadata.read_text(encoding="utf-8").splitlines() if line.strip())
            total_seen += 0  # keep total_seen as valid rows for readable logging
            skipped += total_input - sum(1 for _ in iter_valid_rows(input_dir))

    manifest = {
        "total_records": total_written,
        "skipped_incomplete_records": skipped,
        "input_dirs": [str(path.resolve()) for path in args.input_dir],
        "shards": shards,
        "embedding_shape_per_record": list(first_shape) if first_shape else None,
        "label_definition": {
            "is_correct": "Qwen answer matches the extracted dataset answer",
            "strong_model_wins": "not is_correct",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
