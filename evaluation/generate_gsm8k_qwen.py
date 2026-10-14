"""Generate Qwen answers for GSM8K in Kaggle's 2xT4 GPU runtime.

Qwen2.5-0.5B-Instruct runs on cuda:0 and Qwen3-1.7B runs on cuda:1.
The script generates batches of eight questions per GPU and saves the two raw
responses for each question. It does not read or grade GSM8K reference answers.

Kaggle setup:
    !pip install -q -U "transformers>=4.51" "datasets>=3.0" accelerate tqdm

Smoke test:
    !python generate_gsm8k_qwen.py --max-samples 16

Full generation:
    !python generate_gsm8k_qwen.py
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


MODEL_SPECS = (
    {
        "key": "qwen2_5_0_5b",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": "cuda:0",
        "thinking": False,
    },
    {
        "key": "qwen3_1_7b",
        "model_id": "Qwen/Qwen3-1.7B",
        "device": "cuda:1",
        "thinking": True,
    },
)

PROMPT = """Solve this grade-school math word problem. Reason step by step,
then finish with exactly one line in this format:
Final answer: <number>

Problem:
{question}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/gsm8k_qwen_answers.jsonl"),
        help="Per-question JSONL output path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Questions generated together on each GPU (default: 8).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Generate only the first N questions; use 16 for a smoke test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens generated per answer (default: 1024).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run both models one after another rather than concurrently.",
    )
    return parser.parse_args()


def require_two_gpus() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime in Kaggle before running this script.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This script expects Kaggle GPU T4 x2. Select that accelerator, "
            "or modify MODEL_SPECS for a single-GPU run."
        )


def load_runner(spec: dict[str, Any]) -> dict[str, Any]:
    """Load one model on its assigned T4 GPU."""
    device = torch.device(spec["device"])
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
    # Decoder-only batched generation needs left padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return {**spec, "device": device, "tokenizer": tokenizer, "model": model}


@torch.inference_mode()
def generate_batch(
    questions: list[str],
    runner: dict[str, Any],
    max_new_tokens: int,
) -> list[str]:
    """Generate one answer per question as a single batched model call."""
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if runner["thinking"]:
        template_kwargs["enable_thinking"] = True

    prompts = [
        runner["tokenizer"].apply_chat_template(
            [{"role": "user", "content": PROMPT.format(question=question)}],
            **template_kwargs,
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
    prompt_length = inputs["input_ids"].shape[1]
    generated = output_ids[:, prompt_length:]
    return runner["tokenizer"].batch_decode(generated, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    require_two_gpus()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading GSM8K test split...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(
        f"Loaded {len(dataset)} questions. Loading one model on each T4 "
        f"with batch size {args.batch_size}..."
    )
    runners = [load_runner(spec) for spec in MODEL_SPECS]
    executor = None if args.sequential else ThreadPoolExecutor(max_workers=2)

    try:
        with args.output.open("w", encoding="utf-8") as output_file:
            batch_starts = range(0, len(dataset), args.batch_size)
            for start in tqdm(batch_starts, desc="GSM8K generation"):
                end = min(start + args.batch_size, len(dataset))
                examples = dataset.select(range(start, end))
                questions = list(examples["question"])

                if args.sequential:
                    responses_by_model = [
                        generate_batch(questions, runner, args.max_new_tokens)
                        for runner in runners
                    ]
                else:
                    futures = [
                        executor.submit(
                            generate_batch,
                            questions,
                            runner,
                            args.max_new_tokens,
                        )
                        for runner in runners
                    ]
                    responses_by_model = [future.result() for future in futures]

                for offset, question in enumerate(questions):
                    record: dict[str, Any] = {
                        "id": start + offset,
                        "question": question,
                    }
                    for runner, responses in zip(runners, responses_by_model):
                        record[f"{runner['key']}_response"] = responses[offset].strip()

                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    summary = {
        "dataset": "openai/gsm8k",
        "split": "test",
        "num_questions": len(dataset),
        "batch_size_per_gpu": args.batch_size,
        "results_file": str(args.output),
        "models": {
            runner["key"]: {"model_id": runner["model_id"], "device": str(runner["device"])}
            for runner in runners
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Answers: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
