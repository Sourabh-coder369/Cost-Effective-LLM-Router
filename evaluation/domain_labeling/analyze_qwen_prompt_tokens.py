"""Measure prompt and labeling-input token lengths with the Qwen3-1.7B tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer

from label_math_general_qwen_kaggle import MODEL_ID, RUBRIC, extract_request

BUCKET_EDGES = [0, 128, 256, 512, 1024, 2048, 4096, 8192, float("inf")]
BUCKET_NAMES = ["0–127", "128–255", "256–511", "512–1023", "1024–2047", "2048–4095", "4096–8191", "8192+"]


def summarize(values: np.ndarray) -> dict:
    if not len(values):
        return {"rows": 0}
    percentiles = np.percentile(values, [10, 25, 50, 75, 90, 95, 99])
    counts = np.histogram(values, bins=BUCKET_EDGES)[0]
    return {
        "rows": len(values), "min": int(values.min()), "mean": round(float(values.mean()), 2),
        "std": round(float(values.std()), 2), "max": int(values.max()),
        **{f"p{pct}": round(float(value), 2) for pct, value in zip([10, 25, 50, 75, 90, 95, 99], percentiles)},
        "buckets": [{"range": name, "rows": int(count), "percent": round(float(count / len(values) * 100), 2)}
                    for name, count in zip(BUCKET_NAMES, counts)],
        "over_4096": int((values > 4096).sum()),
        "over_32760": int((values > 32760).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen_token_distribution"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Loaded {MODEL_ID} tokenizer", flush=True)
    tables = []
    for split in ("train", "val", "test"):
        path = args.input_folder / f"router_{split}.parquet"
        frame = pd.read_parquet(path, columns=["prompt"])
        requests = [extract_request(value) for value in frame["prompt"]]
        plain_counts, labeling_counts = [], []
        for start in range(0, len(requests), 512):
            batch = requests[start:start + 512]
            plain = tokenizer(batch, add_special_tokens=False, truncation=False, return_length=True)
            formatted = [tokenizer.apply_chat_template(
                [{"role": "system", "content": RUBRIC},
                 {"role": "user", "content": json.dumps({"request": request}, ensure_ascii=False)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ) for request in batch]
            labeling = tokenizer(formatted, add_special_tokens=False, truncation=False, return_length=True)
            plain_counts.extend(plain["length"])
            labeling_counts.extend(labeling["length"])
        tables.append(pd.DataFrame({"split": split, "row": np.arange(len(requests)),
                                    "prompt_tokens": plain_counts, "labeling_input_tokens": labeling_counts,
                                    "empty_prompt": [not request for request in requests]}))
        print(f"Tokenized {split}: {len(requests):,} rows", flush=True)
    counts = pd.concat(tables, ignore_index=True)
    cached_tokenizer = try_to_load_from_cache(MODEL_ID, "tokenizer.json")
    revision = tokenizer.init_kwargs.get("_commit_hash")
    if not revision and isinstance(cached_tokenizer, str):
        revision = Path(cached_tokenizer).parent.name
    report = {
        "model": MODEL_ID, "tokenizer_revision": revision,
        "definition": "Last Human: turn when present; otherwise full prompt. No truncation or padding.",
        "labeling_definition": "Exact Kaggle labeling rubric and chat template, enable_thinking=False; no generated answer.",
        "splits": {},
    }
    for split in ("train", "val", "test", "all"):
        selected = counts if split == "all" else counts.loc[counts["split"] == split]
        report["splits"][split] = {
            "empty_prompts": int(selected["empty_prompt"].sum()),
            **{column: summarize(selected[column].to_numpy()) for column in ("prompt_tokens", "labeling_input_tokens")},
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output_dir / "row_token_counts.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, column, title in zip(axes, ("prompt_tokens", "labeling_input_tokens"),
                                   ("Question text only", "Full labeling input (rubric + chat template)")):
        buckets = report["splits"]["all"][column]["buckets"]
        bars = axis.bar(BUCKET_NAMES, [bucket["percent"] for bucket in buckets], color="#3975c6")
        axis.bar_label(bars, labels=[f"{bucket['percent']:.1f}%" for bucket in buckets], fontsize=9, padding=3)
        axis.set_title(title)
        axis.set_ylabel("Percentage of prompts")
        axis.set_xlabel("Qwen3-1.7B tokens")
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylim(0, max(bucket["percent"] for bucket in buckets) * 1.15 + 1)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Qwen3-1.7B token distribution • {len(counts):,} prompts across all splits", fontsize=14)
    fig.savefig(args.output_dir / "token_distribution.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
