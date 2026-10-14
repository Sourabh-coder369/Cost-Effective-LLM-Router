"""Create BGE-large query embeddings on two Kaggle T4 GPUs.

Run with:
    torchrun --standalone --nproc_per_node=2 create_bge_embeddings_kaggle.py \
        --data-dir /kaggle/input/YOUR-DATASET-FOLDER \
        --output-dir /kaggle/working/bge_large_embeddings

The three outputs are float32 tensors compatible with the router training code:
    router_train_bge_large_embeddings.pt
    router_val_bge_large_embeddings.pt
    router_test_bge_large_embeddings.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from sentence_transformers import SentenceTransformer


MODEL_ID = "BAAI/bge-large-en-v1.5"


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"router_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Upload the three router_*.parquet files as a Kaggle input.")
    frame = pd.read_parquet(path)
    if "prompt" not in frame:
        raise ValueError(f"{path} does not contain a 'prompt' column.")
    return frame


def encode_split(
    model: SentenceTransformer,
    frame: pd.DataFrame,
    split: str,
    output_dir: Path,
    rank: int,
    world_size: int,
    batch_size: int,
) -> None:
    total = len(frame)
    start = rank * total // world_size
    end = (rank + 1) * total // world_size
    texts = frame["prompt"].iloc[start:end].fillna("").astype(str).tolist()
    print(f"GPU rank {rank}: {split} rows {start:,}:{end:,} ({len(texts):,})", flush=True)

    # float16 accelerates BGE inference on T4. Convert results back to float32 so
    # they can feed directly into the router's float32 Linear layers.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=(rank == 0),
        convert_to_tensor=True,
        normalize_embeddings=False,
    ).cpu().float()
    if embeddings.shape != (len(texts), 1024):
        raise RuntimeError(f"Expected [N, 1024] BGE-large embeddings; received {tuple(embeddings.shape)}")

    part_path = output_dir / f".{split}.rank{rank}.pt"
    torch.save(embeddings, part_path)
    barrier(world_size)

    if rank == 0:
        parts = [torch.load(output_dir / f".{split}.rank{i}.pt", map_location="cpu", weights_only=True)
                 for i in range(world_size)]
        full_embeddings = torch.cat(parts, dim=0)
        if len(full_embeddings) != total:
            raise RuntimeError(f"{split}: expected {total} embeddings, found {len(full_embeddings)}")
        final_path = output_dir / f"router_{split}_bge_large_embeddings.pt"
        torch.save(full_embeddings, final_path)
        print(f"Saved {final_path} with shape {tuple(full_embeddings.shape)}", flush=True)
        for i in range(world_size):
            (output_dir / f".{split}.rank{i}.pt").unlink()
    barrier(world_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing router_train/val/test.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/bge_large_embeddings"))
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size; increase to 48 or 64 if VRAM allows")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using {world_size} GPU process(es); model={MODEL_ID}; local GPU={local_rank}", flush=True)

    model = SentenceTransformer(MODEL_ID, device=f"cuda:{local_rank}")
    model.half()
    model.eval()

    for split in args.splits:
        encode_split(model, read_split(args.data_dir, split), split, args.output_dir,
                     rank, world_size, args.batch_size)

    if rank == 0:
        print("Finished. In Kaggle, use Save Version to retain /kaggle/working/bge_large_embeddings.", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
