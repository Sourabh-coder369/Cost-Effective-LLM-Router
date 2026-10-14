"""Add Qwen3-4B answers and validate all Qwen outputs against GSM8K labels.

This script reads the generation-only JSONL produced by generate_gsm8k_qwen.py.
It runs two replicas of Qwen3-4B: one on each Kaggle T4 GPU.
Each replica handles half of every batch, so both GPUs generate 4B answers
concurrently.

The output preserves the existing 0.5B/1.7B responses, adds 4B responses,
and adds extracted-answer and correct/incorrect fields for all three models.

Example:
    !python add_qwen3b_and_validate_gsm8k.py --batch-size-per-gpu 8
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-4B"
OUTPUT_KEY = "qwen3_4b"
EXISTING_MODEL_KEYS = ("qwen2_5_0_5b", "qwen3_1_7b")

PROMPT = """Solve this grade-school math word problem. Reason step by step,
then finish with exactly one line in this format:
Final answer: <number>

Problem:
{question}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/kaggle/working/gsm8k_qwen_answers.jsonl"),
        help="Generation-only JSONL produced by generate_gsm8k_qwen.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/gsm8k_qwen_4b_validated.jsonl"),
        help="New JSONL file with 4B outputs and all correctness labels.",
    )
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=8,
        help="Number of questions given to each T4 per generation step (default: 8).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens per 4B answer (default: 1024).",
    )
    return parser.parse_args()


def require_two_gpus() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle's GPU T4 x2 runtime before running this script.")


def normalize_number(value: str) -> str:
    value = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value.lower()
    return str(int(number)) if number.is_integer() else f"{number:.12g}"


def extract_reference_answer(solution: str) -> str:
    match = re.search(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", solution)
    if not match:
        raise ValueError(f"Could not parse GSM8K reference answer: {solution!r}")
    return normalize_number(match.group(1))


def extract_model_answer(response: str | None) -> str | None:
    if not response:
        return None
    answers = re.findall(
        r"final\s+answer\s*:\s*([-+]?\$?[\d,]+(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    )
    if answers:
        return normalize_number(answers[-1])
    boxed = re.findall(r"\\boxed\{\s*([-+]?\$?[\d,]+(?:\.\d+)?)\s*\}", response)
    return normalize_number(boxed[-1]) if boxed else None


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open(encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    if not records:
        raise ValueError("The input JSONL file has no records.")
    return records


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
    return {"device": device, "tokenizer": tokenizer, "model": model}


@torch.inference_mode()
def generate_batch(
    questions: list[str],
    runner: dict[str, Any],
    max_new_tokens: int,
) -> list[str]:
    prompts = [
        runner["tokenizer"].apply_chat_template(
            [{"role": "user", "content": PROMPT.format(question=question)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for question in questions
    ]
    inputs = runner["tokenizer"](prompts, padding=True, return_tensors="pt")
    inputs = {name: tensor.to(runner["device"]) for name, tensor in inputs.items()}
    output_ids = runner["model"].generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=runner["tokenizer"].pad_token_id,
    )
    generated_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    return runner["tokenizer"].batch_decode(generated_tokens, skip_special_tokens=True)


def ground_truth_by_id() -> dict[int, str]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    return {
        index: extract_reference_answer(example["answer"])
        for index, example in enumerate(dataset)
    }


def add_labels(record: dict[str, Any], reference_answer: str) -> None:
    record["reference_answer"] = reference_answer
    for key in (*EXISTING_MODEL_KEYS, OUTPUT_KEY):
        answer = extract_model_answer(record.get(f"{key}_response"))
        record[f"{key}_answer"] = answer
        record[f"{key}_label"] = "correct" if answer == reference_answer else "incorrect"


def main() -> None:
    args = parse_args()
    if args.batch_size_per_gpu <= 0:
        raise ValueError("--batch-size-per-gpu must be positive")
    require_two_gpus()
    records = load_records(args.input)
    gold_answers = ground_truth_by_id()

    for record in records:
        if "id" not in record or "question" not in record:
            raise ValueError("Every input row must contain integer 'id' and 'question' fields.")
        if int(record["id"]) not in gold_answers:
            raise ValueError(f"GSM8K test-set id not found: {record['id']}")

    print("Loading two Qwen3-4B replicas, one on each T4...")
    runners = [load_runner("cuda:0"), load_runner("cuda:1")]
    global_batch_size = args.batch_size_per_gpu * 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        with args.output.open("w", encoding="utf-8") as output_file:
            starts = range(0, len(records), global_batch_size)
            for start in tqdm(starts, desc="Generating Qwen3-4B and validating"):
                batch = records[start : start + global_batch_size]
                split = min(args.batch_size_per_gpu, len(batch))
                batches = (batch[:split], batch[split:])

                futures = [
                    executor.submit(
                        generate_batch,
                        [record["question"] for record in local_batch],
                        runner,
                        args.max_new_tokens,
                    )
                    if local_batch
                    else None
                    for local_batch, runner in zip(batches, runners)
                ]
                generated_batches = [
                    future.result() if future is not None else [] for future in futures
                ]

                for local_batch, responses in zip(batches, generated_batches):
                    for record, response in zip(local_batch, responses):
                        record[f"{OUTPUT_KEY}_response"] = response.strip()
                        add_labels(record, gold_answers[int(record["id"])])
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()

    total = len(records)
    model_keys = (*EXISTING_MODEL_KEYS, OUTPUT_KEY)
    summary = {
        "dataset": "openai/gsm8k",
        "split": "test",
        "num_questions": total,
        "qwen_4b_model": MODEL_ID,
        "qwen_4b_replicas": 2,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "results_file": str(args.output),
        "models": {
            key: {
                "correct": sum(record[f"{key}_label"] == "correct" for record in records),
                "incorrect": sum(record[f"{key}_label"] == "incorrect" for record in records),
            }
            for key in model_keys
        },
    }
    for stats in summary["models"].values():
        stats["accuracy"] = stats["correct"] / total

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Validated results: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
