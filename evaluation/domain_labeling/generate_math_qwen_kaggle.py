"""Evaluate two Qwen models on the Hendrycks MATH test set in Kaggle.

Designed for Kaggle's 2xT4 GPU runtime: Qwen2.5-0.5B runs on cuda:0 and
Qwen3-1.7B runs on cuda:1. One JSONL output row is produced for every MATH
problem, containing both generated answers and correct/incorrect labels.

Kaggle setup:
    !pip install -q -U "transformers>=4.51" "datasets>=3.0" accelerate tqdm \
        "math-verify[antlr4_13_2]"

Example:
    python evaluation/domain_labeling/generate_math_qwen_kaggle.py \
        --output /kaggle/working/math_qwen_results.jsonl

Run --max-samples 20 first as a smoke test. The complete MATH test set has
5,000 questions and can take a long time to evaluate.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from datasets import concatenate_datasets, load_dataset
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
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

MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

PROMPT = """Solve the following competition mathematics problem. Think step by
step, then give the final answer using LaTeX exactly as \\boxed{{answer}}.

Problem:
{problem}
"""

PREDICTION_EXTRACTION = (
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/math_qwen_results.jsonl"),
        help="Per-problem JSONL output path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N examples. Use 20 for a smoke test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1536,
        help="Maximum tokens generated per model answer (default: 1536).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run models one after another instead of concurrently on two GPUs.",
    )
    return parser.parse_args()


def require_two_gpus() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required. Enable GPU in Kaggle.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This script expects Kaggle's 2xT4 runtime, but fewer than two GPUs "
            "are visible. Select GPU T4 x2 in Kaggle or adapt MODEL_SPECS."
        )


def load_math_test_set(max_samples: int | None):
    """Load all seven MATH subjects and combine their test splits."""
    parts = [
        load_dataset("EleutherAI/hendrycks_math", subject, split="test")
        for subject in MATH_SUBJECTS
    ]
    dataset = concatenate_datasets(parts)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def load_runner(spec: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(spec["device"])
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return {**spec, "device": device, "tokenizer": tokenizer, "model": model}


@torch.inference_mode()
def generate(problem: str, runner: dict[str, Any], max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": PROMPT.format(problem=problem)}]
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    # Qwen3 supports a thinking-mode switch; Qwen2.5 ignores this setting.
    if runner["thinking"]:
        template_kwargs["enable_thinking"] = True

    inputs = runner["tokenizer"].apply_chat_template(messages, **template_kwargs)
    inputs = {name: tensor.to(runner["device"]) for name, tensor in inputs.items()}
    output_ids = runner["model"].generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=runner["tokenizer"].eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return runner["tokenizer"].decode(new_tokens, skip_special_tokens=True).strip()


def extract_last_boxed(text: str) -> str | None:
    """Return the last balanced ``\\boxed{...}`` expression for easy inspection."""
    positions = [match.start() for match in re.finditer(r"\\boxed\{", text)]
    for start in reversed(positions):
        index = start + len(r"\boxed{")
        depth = 1
        for end in range(index, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    return text[index:end].strip()
    return None


def is_correct(gold_solution: str, response: str) -> bool:
    """Symbolically compare the MATH solution and model response."""
    try:
        gold = parse(gold_solution)
        prediction = parse(response, extraction_config=PREDICTION_EXTRACTION)
        return bool(gold and prediction and verify(gold, prediction))
    except Exception:
        # Parsing failures count as incorrect, but raw output is retained in JSONL.
        return False


def main() -> None:
    args = parse_args()
    require_two_gpus()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading MATH test split...")
    dataset = load_math_test_set(args.max_samples)
    print(f"Loaded {len(dataset)} problems. Loading one model on each GPU...")
    runners = [load_runner(spec) for spec in MODEL_SPECS]

    correct_counts = {runner["key"]: 0 for runner in runners}
    executor = None if args.sequential else ThreadPoolExecutor(max_workers=2)
    try:
        with args.output.open("w", encoding="utf-8") as output_file:
            for index, example in enumerate(tqdm(dataset, desc="MATH evaluation")):
                problem = example["problem"]
                if args.sequential:
                    responses = [
                        generate(problem, runner, args.max_new_tokens) for runner in runners
                    ]
                else:
                    # Inference occurs concurrently because each runner uses a separate T4.
                    futures = [
                        executor.submit(generate, problem, runner, args.max_new_tokens)
                        for runner in runners
                    ]
                    responses = [future.result() for future in futures]

                record: dict[str, Any] = {
                    "id": index,
                    "subject": example["type"],
                    "level": example["level"],
                    "problem": problem,
                    "reference_solution": example["solution"],
                }
                for runner, response in zip(runners, responses):
                    correct = is_correct(example["solution"], response)
                    key = runner["key"]
                    correct_counts[key] += int(correct)
                    record[f"{key}_response"] = response
                    record[f"{key}_final_answer"] = extract_last_boxed(response)
                    record[f"{key}_label"] = "correct" if correct else "incorrect"

                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    total = len(dataset)
    summary = {
        "dataset": "EleutherAI/hendrycks_math",
        "split": "test",
        "num_problems": total,
        "results_file": str(args.output),
        "models": {
            runner["key"]: {
                "model_id": runner["model_id"],
                "correct": correct_counts[runner["key"]],
                "incorrect": total - correct_counts[runner["key"]],
                "accuracy": correct_counts[runner["key"]] / total,
            }
            for runner in runners
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
