r"""Train and test the 256-dimensional BGE domain-specific router.

Run from the project root:
    .\.venv\Scripts\python.exe .\scripts\experiments\run_domain_router_256.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/original_gpt4_llama7b/gpt4_llama7b_data_unbalanced"
EMBEDDINGS = ROOT / "data/original_gpt4_llama7b/bge_embeddings"
OUTPUT_DIR = ROOT / "router/domain_specific/checkpoints_domain_mf_bge_modeldim_256"


def run(command: list[str]) -> None:
    print("\nRunning:\n  " + " ".join(f'"{item}"' if " " in item else item for item in command))
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    python = sys.executable

    # 1. Train the domain-specific model using train and validation splits.
    run([
        python,
        str(ROOT / "router/domain_specific/train_explicit_splits.py"),
        "--train-data", str(DATA / "router_train.parquet"),
        "--train-embeddings", str(EMBEDDINGS / "router_train_bge_large_embeddings.pt"),
        "--val-data", str(DATA / "router_val.parquet"),
        "--val-embeddings", str(EMBEDDINGS / "router_val_bge_large_embeddings.pt"),
        "--output-dir", str(OUTPUT_DIR),
        "--model-embedding-dim", "256",
        "--epochs", "10",
        "--batch-size", "512",
        "--learning-rate", "3e-4",
        "--seed", "42",
    ])

    # 2. Compare it with the equal-dimensional baseline on the held-out test set.
    run([
        python,
        str(ROOT / "router/domain_specific/evaluation/reproduce_comparison_results.py"),
        "--test-data", str(DATA / "router_test.parquet"),
        "--test-embeddings", str(EMBEDDINGS / "router_test_bge_large_embeddings.pt"),
        "--baseline-checkpoint", str(ROOT / "router/baseline/checkpoints_bge_modeldim_256/best_model.pt"),
        "--domain-checkpoint", str(OUTPUT_DIR / "best_model.pt"),
        "--output", str(OUTPUT_DIR / "comparison_results.json"),
        "--embedding-model", "BAAI/bge-large-en-v1.5",
    ])

    print("\nCompleted.")
    print(f"Checkpoint: {OUTPUT_DIR / 'best_model.pt'}")
    print(f"Comparison: {OUTPUT_DIR / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
