"""Generate a Qwen trajectory/answer dataset for math routing experiments.

This script is designed for a Kaggle T4 GPU. It loads several math datasets,
asks Qwen2-0.5B-Instruct to solve each question, extracts every hidden-state
layer for the *question prompt*, and labels the result using the dataset
answer.

Routing label:
    is_correct = True  -> keep the cheap Qwen model
    is_correct = False -> route to a stronger model
    strong_model_wins = not is_correct

The output uses compact PyTorch shards rather than JSON arrays for the large
embedding tensors:

    output/
      metadata.jsonl       one row per question and Qwen response
      embeddings_00000.pt  layer embeddings for the rows in metadata.jsonl
      embeddings_00001.pt
      manifest.json        shapes, model, sources, and label definition

Each shard contains:
    layer_embeddings             [batch, layers, hidden_size] mean-pooled
    layer_last_token_embeddings  [batch, layers, hidden_size]

The hidden-state tuple includes the input embedding output followed by each
transformer block output, as returned by Hugging Face ``output_hidden_states``.
Thus a 24-block Qwen model normally produces 25 layer representations.

Example on Kaggle:

    !pip install -q -U transformers datasets accelerate sentencepiece
    !python evaluation/generate_qwen_math_layer_dataset.py \\
      --source-spec evaluation/math_sources.example.json \\
      --output-dir /kaggle/working/qwen_math_layers \\
      --model-name Qwen/Qwen2-0.5B-Instruct \\
      --batch-size 4 \\
      --max-new-tokens 512

Use ``--limit-per-source`` while validating the pipeline. Remove the limit
for the full collection run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


QUESTION_FIELDS = ("question", "problem", "prompt", "instruction", "input")
ANSWER_FIELDS = ("answer", "solution", "target", "ground_truth", "label")


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "examples", "records", "train", "test"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
    raise ValueError(f"Could not find a list of records in {path}")


def load_source_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    path_or_dataset = str(spec["path"])
    local_path = Path(path_or_dataset)
    if local_path.exists():
        suffix = local_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            rows = read_json_rows(local_path)
        elif suffix == ".parquet":
            rows = Dataset.from_parquet(str(local_path)).to_list()
        else:
            raise ValueError(f"Unsupported local source format: {local_path}")
    else:
        dataset = load_dataset(
            path_or_dataset,
            name=spec.get("config"),
            split=spec.get("split", "train"),
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
        )
        rows = dataset.to_list()

    limit = spec.get("limit")
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"Source returned no rows: {spec}")
    return rows


def choose_field(row: dict[str, Any], requested: str | None, candidates: Iterable[str], kind: str) -> str:
    if requested:
        if requested not in row:
            raise KeyError(f"Configured {kind} field {requested!r} is not in row fields {list(row)}")
        return requested
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return candidate
    raise KeyError(f"Could not infer {kind} field from row fields {list(row)}")


def scalar_to_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_braced(text: str, marker: str) -> str | None:
    start = text.find(marker)
    if start < 0:
        return None
    open_pos = text.find("{", start + len(marker))
    if open_pos < 0:
        return None
    depth = 0
    for pos in range(open_pos, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1 : pos]
    return None


def extract_answer_text(text: str, answer_kind: str = "auto") -> str | None:
    """Extract a final mathematical answer from GSM8K/MATH-style text."""
    text = scalar_to_text(text).strip()
    if not text:
        return None

    if answer_kind in {"auto", "gsm8k"}:
        matches = re.findall(r"####\s*([^\n]+)", text)
        if matches:
            return matches[-1].strip()

    if answer_kind in {"auto", "boxed", "math"}:
        boxed = extract_braced(text, r"\boxed")
        if boxed is not None:
            return boxed.strip()

    # Some datasets store a bare answer such as 42 or 1/3.
    if answer_kind == "scalar":
        return text

    # Fall back to the final numeric expression. This is intentionally a
    # fallback: dataset-specific answer markers are preferred above.
    numeric_matches = re.findall(
        r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?",
        text,
    )
    return numeric_matches[-1].strip() if numeric_matches else text


def canonical_answer(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "").replace(",", "")
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    if value.endswith("."):
        value = value[:-1]
    return value.lower()


def numeric_value(value: str | None) -> float | None:
    value = canonical_answer(value)
    if not value:
        return None
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    frac = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", value)
    try:
        if frac:
            result = float(frac.group(1)) / float(frac.group(2))
        else:
            result = float(value)
        return result / 100.0 if percent else result
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(gold: str | None, predicted: str | None, tolerance: float = 1e-4) -> bool:
    gold_num = numeric_value(gold)
    pred_num = numeric_value(predicted)
    if gold_num is not None and pred_num is not None:
        return math.isclose(gold_num, pred_num, rel_tol=tolerance, abs_tol=tolerance)
    return canonical_answer(gold) == canonical_answer(predicted)


def make_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "Solve this math problem carefully. Show the reasoning briefly and "
                "finish with the final numerical answer.\n\n" + question
            ),
        }
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\n\nAnswer:"


def extract_prompt_layer_embeddings(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean-pooled and last-token embeddings for every hidden layer."""
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    # [layers, batch, sequence, hidden] -> [batch, layers, sequence, hidden]
    stacked = torch.stack(tuple(hidden_states), dim=1)
    mask = attention_mask[:, None, :, None].to(dtype=stacked.dtype)
    mean_pooled = (stacked * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

    # With left padding, the last non-padding token is the rightmost token in
    # every row. Computing it from the mask keeps this correct if padding
    # behavior is changed later.
    position_grid = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
    last_positions = (attention_mask * position_grid).max(dim=1).values.long()
    batch_positions = torch.arange(input_ids.size(0), device=input_ids.device)
    last_token = stacked[batch_positions, :, last_positions, :]
    return mean_pooled.cpu().half(), last_token.cpu().half()


def flush_shard(
    output_dir: Path,
    shard_index: int,
    records: list[dict[str, Any]],
    mean_embeddings: list[torch.Tensor],
    last_embeddings: list[torch.Tensor],
) -> dict[str, Any]:
    shard_name = f"embeddings_{shard_index:05d}.pt"
    payload = {
        "record_ids": [record["record_id"] for record in records],
        "layer_embeddings": torch.stack(mean_embeddings, dim=0),
        "layer_last_token_embeddings": torch.stack(last_embeddings, dim=0),
        "is_correct": torch.tensor([int(record["is_correct"]) for record in records], dtype=torch.bool),
        "strong_model_wins": torch.tensor([int(record["strong_model_wins"]) for record in records], dtype=torch.bool),
    }
    torch.save(payload, output_dir / shard_name)
    return {
        "file": shard_name,
        "count": len(records),
        "layer_embedding_shape": list(payload["layer_embeddings"].shape[1:]),
        "dtype": str(payload["layer_embeddings"].dtype),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", required=True, type=Path, help="JSON list describing HF/local math datasets")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_specs = json.loads(args.source_spec.read_text(encoding="utf-8"))
    if not isinstance(source_specs, list) or not source_specs:
        raise ValueError("--source-spec must contain a non-empty JSON list")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    metadata_path = output_dir / "metadata.jsonl"
    metadata_file = metadata_path.open("w", encoding="utf-8")
    all_shards: list[dict[str, Any]] = []
    pending_records: list[dict[str, Any]] = []
    pending_mean: list[torch.Tensor] = []
    pending_last: list[torch.Tensor] = []
    shard_index = 0
    total = 0
    source_counts: dict[str, int] = {}

    try:
        for source_number, original_spec in enumerate(source_specs):
            spec = dict(original_spec)
            if args.limit_per_source is not None:
                spec["limit"] = args.limit_per_source
            source_name = str(spec.get("name", f"source_{source_number}"))
            rows = load_source_rows(spec)
            source_counts[source_name] = 0
            question_field = spec.get("question_field")
            answer_field = spec.get("answer_field")
            answer_kind = spec.get("answer_kind", "auto")
            prepared: list[tuple[int, dict[str, Any], str, str]] = []
            for source_idx, row in enumerate(rows):
                q_field = choose_field(row, question_field, QUESTION_FIELDS, "question")
                a_field = choose_field(row, answer_field, ANSWER_FIELDS, "answer")
                question = scalar_to_text(row[q_field]).strip()
                gold_raw = scalar_to_text(row[a_field])
                gold = extract_answer_text(gold_raw, answer_kind=answer_kind)
                if question:
                    prepared.append((source_idx, row, question, gold or gold_raw.strip()))

            for start in range(0, len(prepared), args.batch_size):
                batch = prepared[start : start + args.batch_size]
                prompts = [make_prompt(tokenizer, item[2]) for item in batch]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_tokens,
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                mean_embeddings, last_embeddings = extract_prompt_layer_embeddings(model, input_ids, attention_mask)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                response_ids = generated_ids[:, input_ids.shape[1] :]
                responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                for local_idx, (source_idx, row, question, gold) in enumerate(batch):
                    response = responses[local_idx].strip()
                    # ``scalar`` describes the dataset's direct gold-answer
                    # field. Qwen still emits a full response, so its answer
                    # must be parsed with the normal response extractor.
                    prediction_kind = "auto" if answer_kind == "scalar" else answer_kind
                    predicted = extract_answer_text(response, answer_kind=prediction_kind)
                    is_correct = answers_match(gold, predicted)
                    record_id = f"{source_name}:{source_idx}"
                    embedding_row = len(pending_records)
                    metadata = {
                        "record_id": record_id,
                        "source": source_name,
                        "source_index": int(source_idx),
                        "question": question,
                        "gold_answer": gold,
                        "qwen_response": response,
                        "predicted_answer": predicted,
                        "is_correct": bool(is_correct),
                        "strong_model_wins": bool(not is_correct),
                        "embedding_shard": f"embeddings_{shard_index:05d}.pt",
                        "embedding_row": int(embedding_row),
                        "prompt_tokens": int(attention_mask[local_idx].sum().item()),
                        "response_tokens": int(len(response_ids[local_idx])),
                    }
                    metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    pending_records.append(metadata)
                    pending_mean.append(mean_embeddings[local_idx])
                    pending_last.append(last_embeddings[local_idx])
                    source_counts[source_name] += 1
                    total += 1

                    if len(pending_records) >= args.shard_size:
                        all_shards.append(flush_shard(output_dir, shard_index, pending_records, pending_mean, pending_last))
                        pending_records, pending_mean, pending_last = [], [], []
                        shard_index += 1

                print(f"{source_name}: {source_counts[source_name]}/{len(prepared)} processed", flush=True)

        if pending_records:
            all_shards.append(flush_shard(output_dir, shard_index, pending_records, pending_mean, pending_last))
    finally:
        metadata_file.close()

    manifest = {
        "model_name": args.model_name,
        "device": str(device),
        "dtype": str(dtype),
        "total_records": total,
        "source_counts": source_counts,
        "shards": all_shards,
        "embedding_definition": {
            "layer_embeddings": "mean pooling over non-padding prompt tokens for every hidden-state layer",
            "layer_last_token_embeddings": "last non-padding prompt token for every hidden-state layer",
            "includes_input_embedding_output": True,
        },
        "label_definition": {
            "is_correct": "Qwen answer matches the extracted dataset answer",
            "strong_model_wins": "not is_correct; this is the routing target requested by the user",
        },
        "source_spec": source_specs,
        "answer_matching": "numeric answers compared with absolute/relative tolerance 1e-4; otherwise canonical string match",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"total_records": total, "output_dir": str(output_dir.resolve()), "shards": len(all_shards)}, indent=2))


if __name__ == "__main__":
    main()
