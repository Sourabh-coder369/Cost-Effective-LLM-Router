"""Generate Qwen3-1.7B math answers and all-layer embeddings on two T4 GPUs.

Two identical Qwen3-1.7B replicas are loaded: one on cuda:0 and one on cuda:1.
They process separate halves of each global batch concurrently. For every
question, this script stores the raw model answer and prompt hidden states from
every layer. It does not read ground-truth answers or create labels.

Default sources are GSM8K train and test plus competition_math train.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-1.7B"
SOURCES = (
    {
        "name": "gsm8k_train",
        "path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "question_field": "question",
    },
    {
        "name": "gsm8k_test",
        "path": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "question_field": "question",
    },
    {
        "name": "competition_math",
        "path": "dim/competition_math",
        "split": "train",
        "question_field": "problem",
    },
)
PROMPT = """Solve this math problem. Show brief reasoning, then finish with:
Final answer: <answer>

Problem:
{question}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/qwen3_1_7b_all_math_layers"),
    )
    parser.add_argument(
        "--source-spec",
        type=Path,
        default=None,
        help="Optional JSON source list; defaults to all configured math sources.",
    )
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=4,
        help="Questions generated together on each T4 GPU.",
    )
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit-per-source", type=int, default=None)
    return parser.parse_args()


def require_two_gpus() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle GPU T4 x2 before running this script.")


def source_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = load_dataset(
        str(spec["path"]),
        name=spec.get("config"),
        split=spec.get("split", "train"),
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
    )
    rows = dataset.to_list()
    return rows[: int(spec["limit"])] if spec.get("limit") is not None else rows


def load_runner(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}


@torch.inference_mode()
def generate_and_embed(
    questions: list[str],
    runner: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompts = [
        runner["tokenizer"].apply_chat_template(
            [{"role": "user", "content": PROMPT.format(question=question)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for question in questions
    ]
    encoded = runner["tokenizer"](
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(runner["device"])
    attention_mask = encoded["attention_mask"].to(runner["device"])

    forward = runner["model"](
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    if forward.hidden_states is None:
        raise RuntimeError("Qwen3-1.7B did not return hidden states.")
    layers = torch.stack(tuple(forward.hidden_states), dim=1)
    mask = attention_mask[:, None, :, None].to(layers.dtype)
    mean_layers = (layers * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
    last_layers = layers[:, :, -1, :]

    generated_ids = runner["model"].generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=runner["tokenizer"].pad_token_id,
    )
    responses = runner["tokenizer"].batch_decode(
        generated_ids[:, input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    return {
        "responses": [response.strip() for response in responses],
        "mean_layers": mean_layers.cpu().half(),
        "last_layers": last_layers.cpu().half(),
    }


def save_shard(
    output_dir: Path,
    index: int,
    records: list[dict[str, Any]],
    mean_layers: list[torch.Tensor],
    last_layers: list[torch.Tensor],
) -> dict[str, Any]:
    name = f"embeddings_{index:05d}.pt"
    payload = {
        "record_ids": [record["record_id"] for record in records],
        "layer_embeddings": torch.stack(mean_layers),
        "layer_last_token_embeddings": torch.stack(last_layers),
    }
    torch.save(payload, output_dir / name)
    return {
        "file": name,
        "count": len(records),
        "layer_embedding_shape": list(payload["layer_embeddings"].shape[1:]),
        "dtype": str(payload["layer_embeddings"].dtype),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size_per_gpu <= 0 or args.shard_size <= 0:
        raise ValueError("Batch and shard sizes must be positive.")
    require_two_gpus()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    specs = json.loads(args.source_spec.read_text()) if args.source_spec else [dict(item) for item in SOURCES]
    if not isinstance(specs, list) or not specs:
        raise ValueError("Source specification must be a non-empty JSON list.")

    print("Loading a Qwen3-1.7B replica on each T4 GPU.")
    runners = [load_runner("cuda:0"), load_runner("cuda:1")]
    global_batch_size = args.batch_size_per_gpu * 2
    records: list[dict[str, Any]] = []
    means: list[torch.Tensor] = []
    lasts: list[torch.Tensor] = []
    shards: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    shard_index = 0
    total = 0

    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as metadata_file:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for source_number, source in enumerate(specs):
                spec = dict(source)
                if args.limit_per_source is not None:
                    spec["limit"] = args.limit_per_source
                source_name = str(spec.get("name", f"source_{source_number}"))
                rows = source_rows(spec)
                question_field = str(spec.get("question_field", "question"))
                prepared = [
                    (index, str(row[question_field]).strip())
                    for index, row in enumerate(rows)
                    if question_field in row and str(row[question_field]).strip()
                ]
                source_counts[source_name] = 0

                for start in tqdm(range(0, len(prepared), global_batch_size), desc=source_name):
                    batch = prepared[start : start + global_batch_size]
                    split = min(args.batch_size_per_gpu, len(batch))
                    halves = (batch[:split], batch[split:])
                    futures = [
                        pool.submit(
                            generate_and_embed,
                            [question for _, question in half],
                            runner,
                            args,
                        )
                        if half
                        else None
                        for half, runner in zip(halves, runners)
                    ]
                    outputs = [future.result() if future is not None else None for future in futures]

                    for half, output in zip(halves, outputs):
                        if output is None:
                            continue
                        for (source_index, question), response, mean, last in zip(
                            half,
                            output["responses"],
                            output["mean_layers"],
                            output["last_layers"],
                        ):
                            record = {
                                "record_id": f"{source_name}:{source_index}",
                                "source": source_name,
                                "source_index": source_index,
                                "question": question,
                                "qwen3_1_7b_response": response,
                                "embedding_shard": f"embeddings_{shard_index:05d}.pt",
                                "embedding_row": len(records),
                            }
                            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            records.append(record)
                            means.append(mean)
                            lasts.append(last)
                            source_counts[source_name] += 1
                            total += 1
                            if len(records) >= args.shard_size:
                                shards.append(save_shard(args.output_dir, shard_index, records, means, lasts))
                                records, means, lasts = [], [], []
                                shard_index += 1
                    metadata_file.flush()

    if records:
        shards.append(save_shard(args.output_dir, shard_index, records, means, lasts))

    manifest = {
        "model": MODEL_ID,
        "replicas": 2,
        "devices": ["cuda:0", "cuda:1"],
        "total_records": total,
        "source_counts": source_counts,
        "shards": shards,
        "embedding_definition": "Mean-pooled and final prompt-token vectors for every hidden-state layer, including the input embedding output.",
        "ground_truth_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "total_records": total, "shards": len(shards)}, indent=2))


if __name__ == "__main__":
    main()
