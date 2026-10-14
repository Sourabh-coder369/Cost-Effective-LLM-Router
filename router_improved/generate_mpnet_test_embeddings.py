"""Generate the missing MPNet test embeddings for the GPT-4/Llama dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/router_test.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced/.embedding_cache/"
            "router_test.parquet_mpnet_base_v2.pt"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing embeddings: {args.output}")
    frame = pd.read_parquet(args.test_data)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    embeddings = model.encode(
        frame["prompt"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cpu",
        normalize_embeddings=False,
    ).cpu().float()
    if embeddings.shape != (len(frame), 768):
        raise RuntimeError(f"Expected {(len(frame), 768)} MPNet embeddings, got {tuple(embeddings.shape)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    print(f"Saved {tuple(embeddings.shape)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
