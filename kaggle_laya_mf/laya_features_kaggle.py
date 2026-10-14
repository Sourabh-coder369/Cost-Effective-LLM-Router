"""Generate 40-D Laya features for original_gpt4_llama7b prompts on Kaggle GPUs.

Input may be the train/val/test Parquet or CSV split files, or an existing queries.jsonl.
The ZIP includes one feature record per source row, plus reusable unique-query shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import zipfile
from pathlib import Path


BINARY_PROBES = {
    "math_reasoning": "Does correctly solving `query` require mathematical calculation, symbolic manipulation, or quantitative reasoning beyond copying an explicitly stated value?",
    "code_reasoning": "Does correctly solving `query` require writing, modifying, debugging, or reasoning about executable computer code?",
    "formal_logic": "Does solving `query` materially require formal logical, combinatorial, rule-based, or constraint-satisfaction reasoning?",
    "factual_recall": "Can `query` be answered primarily through factual knowledge or retrieval, with little derivation or multi-step reasoning?",
    "social_affective": "Does correctness materially depend on interpreting emotion, intention, social context, interpersonal meaning, or conversational affect?",
    "tool_interaction": "Does completing `query` require interaction with an external tool, software environment, API, or simulated environment rather than only producing an answer?",
    "external_knowledge": "Does solving `query` require factual information that is not explicitly supplied in the query and cannot be derived from the supplied information alone?",
    "current_information": "Would correctness materially depend on recent, changing, or time-sensitive information?",
}

TASK_FAMILY = {
    "type": "choice",
    "instructions": "Which primary capability is needed to solve `query` correctly?",
    "criteria": {
        "mathematics": "Mathematical calculation, symbolic manipulation, or quantitative reasoning is primary.",
        "code": "Writing, modifying, debugging, or reasoning about executable code is primary.",
        "logic": "Formal, combinatorial, rule-based, or constraint-satisfaction reasoning is primary.",
        "knowledge": "Factual or specialized knowledge retrieval is primary.",
        "social_affective": "Emotion, intent, social context, interpersonal meaning, or affect is primary.",
        "instruction_following": "Satisfying detailed textual output or behavioral instructions is primary.",
        "tool_use": "Interaction with an external tool, software environment, API, or simulated environment is primary.",
        "other": "None of the listed capabilities is clearly primary.",
    },
}

ORDINAL_PROBES = {
    "domain_specialization": (
        "How specialized is the knowledge required to solve `query` correctly?",
        ["Everyday or broadly accessible knowledge is sufficient.", "General educated or ordinary professional knowledge is required.", "Specialized disciplinary knowledge is required.", "Niche expert-level knowledge is required."],
    ),
    "reasoning_depth": (
        "How much sequential reasoning is required to reach a correct answer to `query`?",
        ["The answer is direct or can be identified immediately.", "One or two straightforward inference steps are needed.", "Several dependent reasoning steps are needed.", "Extended, deeply dependent reasoning is needed."],
    ),
    "constraint_density": (
        "How many independent requirements or constraints must simultaneously be satisfied by a correct answer to `query`?",
        ["Essentially one requirement determines correctness.", "A small number of compatible requirements must be satisfied.", "Several independent requirements must be tracked together.", "Many interacting constraints must simultaneously be satisfied."],
    ),
    "context_integration": (
        "How much information from different parts of `query` must be integrated to solve it correctly?",
        ["One local fact or instruction is sufficient.", "A few nearby facts or instructions must be combined.", "Multiple separated pieces of information must be integrated.", "Correctness depends on integrating many distributed details or relationships."],
    ),
    "decomposition_need": (
        "To what extent does solving `query` require decomposing it into distinct intermediate subproblems?",
        ["No meaningful decomposition is needed.", "A small decomposition may help but is not essential.", "Several distinct intermediate subproblems should be solved.", "A substantial hierarchical decomposition is required."],
    ),
    "ambiguity": (
        "How underspecified or semantically ambiguous is `query`?",
        ["The intended task and success condition are explicit.", "Minor interpretation choices exist but should not materially affect the answer.", "Important details are ambiguous or underspecified.", "Multiple materially different interpretations are plausible."],
    ),
    "exactness": (
        "How sensitive is correctness to exact details, precise constraints, exact values, or exact output behavior in `query`?",
        ["Approximate or broadly correct content is sufficient.", "Some details must be correct but small deviations are tolerable.", "Precise details or constraints materially affect correctness.", "Exact compliance or exact values are central to correctness."],
    ),
    "answer_openness": (
        "How broad is the set of answers that could reasonably count as correct for `query`?",
        ["There is essentially one correct answer or a very narrow equivalent set.", "A small family of closely related answers can be correct.", "Many substantively different answers could be acceptable.", "The task is highly open-ended with many valid solution directions."],
    ),
}

PROBES = {
    "task_family": TASK_FAMILY,
    **{name: {"type": "noul", "instructions": instruction} for name, instruction in BINARY_PROBES.items()},
    **{name: {"type": "score", "instructions": instruction, "criteria": levels} for name, (instruction, levels) in ORDINAL_PROBES.items()},
}
FEATURE_NAMES = [
    *[f"{name}__noul" for name in BINARY_PROBES],
    *[f"{name}__p__{level}" for name in ORDINAL_PROBES for level in range(4)],
]
assert len(PROBES) == 17 and len(FEATURE_NAMES) == 40


def vectorize(answers: dict) -> list[float]:
    values = []
    for name in BINARY_PROBES:
        values.append(float(answers[name]["noul"]))
    for name in ORDINAL_PROBES:
        probabilities = answers[name]["probabilities"]
        levels = [float(probabilities[str(i)]) if isinstance(probabilities, dict) else float(probabilities[i]) for i in range(4)]
        if abs(sum(levels) - 1.0) > 0.02:
            raise ValueError(f"Invalid probability mass for {name}: {levels}")
        values.extend(levels)
    if len(values) != 40 or any(not math.isfinite(x) or x < -1e-6 or x > 1 + 1e-6 for x in values):
        raise ValueError("Laya did not return 40 finite probability features in [0, 1]")
    return values


def _feature_worker(gpu_id: int, worker_index: int, worker_count: int, queries_path: str, output: str,
                    discarded_output: str, model_path: str, batch_size: int, limit: int | None,
                    max_input_tokens: int, head_max_len: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import laya
    from laya.common import build_sequence, serialize_state

    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU {gpu_id} is unavailable to the Laya worker")
    agent = laya.load(model_path, device="cuda")
    if agent.device.type != "cuda":
        raise RuntimeError(f"GPU {gpu_id}: Laya did not load on CUDA")
    done = set()
    discarded_done = set()
    output_path = Path(output)
    if output_path.exists():
        valid_lines = []
        had_invalid = False
        with output_path.open(encoding="utf-8") as existing:
            for line in existing:
                try:
                    row = json.loads(line)
                    values = row["features"]
                    if len(values) != 40 or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in values):
                        raise ValueError("Invalid cached feature vector")
                    done.add(row["query_hash"])
                    valid_lines.append(json.dumps(row, separators=(",", ":")) + "\n")
                except (ValueError, KeyError, TypeError):
                    had_invalid = True
        if had_invalid:
            cleaned = output_path.with_suffix(".cleaning")
            with cleaned.open("w", encoding="utf-8", newline="\n") as dest:
                dest.writelines(valid_lines)
            cleaned.replace(output_path)
    discarded_path = Path(discarded_output)
    if discarded_path.exists():
        valid_discarded_lines = []
        had_invalid_discarded = False
        with discarded_path.open(encoding="utf-8") as existing:
            for line in existing:
                try:
                    row = json.loads(line)
                    discarded_done.add(row["query_hash"])
                    valid_discarded_lines.append(json.dumps(row, separators=(",", ":")) + "\n")
                except (ValueError, KeyError, TypeError):
                    had_invalid_discarded = True
        if had_invalid_discarded:
            cleaned = discarded_path.with_suffix(".cleaning")
            with cleaned.open("w", encoding="utf-8", newline="\n") as dest:
                dest.writelines(valid_discarded_lines)
            cleaned.replace(discarded_path)
    pending = []
    with Path(queries_path).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key, prompt = row["query_hash"], row["prompt"]
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != key:
                raise ValueError("A query hash does not match its prompt")
            if int(key[:8], 16) % worker_count == worker_index and key not in done and key not in discarded_done:
                pending.append((key, prompt))
    if limit is not None:
        pending = pending[:limit]
    print(f"GPU {gpu_id}: checking {len(pending)} queries against the {max_input_tokens}-token limit", flush=True)
    states_to_score = []
    newly_discarded = []
    probes = {name: agent._to_internal(probe) for name, probe in PROBES.items()}
    for key, prompt in pending:
        state = {"query": prompt}
        state_text = serialize_state(state).replace(agent.tok.mask_token, " ")
        state_ids = agent.tok(state_text, add_special_tokens=False)["input_ids"]
        fits = True
        for probe in probes.values():
            _, _, stats = build_sequence(
                agent.tok, state, probe, max_len=max_input_tokens, head_max_len=head_max_len,
                state_ids=state_ids, return_truncation_stats=True,
            )
            if stats["truncated"]:
                fits = False
                break
        if fits:
            states_to_score.append((key, state, len(state_ids)))
        else:
            newly_discarded.append({"query_hash": key, "state_tokens": len(state_ids), "reason": "packed_input_exceeds_limit"})
    if newly_discarded:
        with discarded_path.open("a", encoding="utf-8", newline="\n") as dest:
            for row in newly_discarded:
                dest.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"GPU {gpu_id}: scoring {len(states_to_score)}, discarding {len(newly_discarded)}", flush=True)
    if output_path.exists() and output_path.stat().st_size:
        with output_path.open("rb+") as tail:
            tail.seek(-1, os.SEEK_END)
            if tail.read(1) != b"\n":
                tail.write(b"\n")
    with output_path.open("a", encoding="utf-8", newline="\n") as dest:
        cursor = 0
        while cursor < len(states_to_score):
            chunk = states_to_score[cursor:cursor + batch_size]
            states = [row[1] for row in chunk]
            if len(chunk) > 0:
                try:
                    responses = agent.predict_batch(states, PROBES, batch_size=len(chunk),
                                                    max_len=max_input_tokens, head_max_len=head_max_len)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(f"GPU {gpu_id}: reducing batch size to {batch_size}", flush=True)
                    continue
            if agent.device.type != "cuda":
                raise RuntimeError(f"GPU {gpu_id}: Laya fell back to CPU during inference; rerun with --laya-batch-size 8 or less")
            if len(responses) != len(chunk):
                raise RuntimeError("Laya returned an unexpected number of responses")
            for (key, _, state_tokens), response in zip(chunk, responses):
                usage = response.get("usage", {})
                row = {"query_hash": key, "features": vectorize(response["answers"]),
                       "state_tokens": usage.get("state_tokens", state_tokens)}
                dest.write(json.dumps(row, separators=(",", ":")) + "\n")
            dest.flush()
            cursor += len(chunk)
            if cursor % 100 == 0 or cursor == len(states_to_score):
                print(f"GPU {gpu_id}: completed {cursor}/{len(states_to_score)}", flush=True)


def _iter_prompt_column(path: Path):
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        if "prompt" not in parquet.schema_arrow.names:
            raise ValueError(f"The RLAIF file has no 'prompt' column: {path}")
        for batch in parquet.iter_batches(columns=["prompt"], batch_size=2048):
            yield from batch.column(0).to_pylist()
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "prompt" not in reader.fieldnames:
                raise ValueError(f"The RLAIF CSV has no 'prompt' column: {path}")
            for row in reader:
                yield row.get("prompt")


def prepare_rlaif_queries(data_dir: Path, work_dir: Path) -> tuple[Path, Path, dict]:
    """Build unique Laya inputs and an order-preserving row map from router split files."""
    split_files = {}
    for split in ("train", "val", "test"):
        parquet = data_dir / f"router_{split}.parquet"
        csv_path = data_dir / f"router_{split}.csv"
        if parquet.is_file():
            split_files[split] = parquet
        elif csv_path.is_file():
            split_files[split] = csv_path
        else:
            raise FileNotFoundError(f"Expected router_{split}.parquet or router_{split}.csv in {data_dir}")

    source_digest = hashlib.sha256()
    unique_prompts = {}
    query_counts = {"records": 0, "empty_prompts": 0, "rows_by_split": {}}
    records_tmp = work_dir / "rlaif_records.jsonl.tmp"
    with records_tmp.open("w", encoding="utf-8", newline="\n") as record_dest:
        for split, path in split_files.items():
            source_digest.update(split.encode("utf-8") + b"\0" + path.name.encode("utf-8") + b"\0")
            with path.open("rb") as raw:
                for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                    source_digest.update(chunk)
            split_rows = 0
            for row_index, raw_prompt in enumerate(_iter_prompt_column(path)):
                split_rows += 1
                query_counts["records"] += 1
                if not isinstance(raw_prompt, str) or not raw_prompt.strip():
                    query_counts["empty_prompts"] += 1
                    continue
                prompt = raw_prompt
                query_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                existing = unique_prompts.setdefault(query_hash, prompt)
                if existing != prompt:
                    raise ValueError("SHA-256 collision while preparing RLAIF prompts")
                record_dest.write(json.dumps({"split": split, "row_index": row_index,
                                             "query_hash": query_hash, "prompt": prompt},
                                            ensure_ascii=False, separators=(",", ":")) + "\n")
            query_counts["rows_by_split"][split] = split_rows
    records_path = work_dir / "rlaif_records.jsonl"
    records_tmp.replace(records_path)

    queries_tmp = work_dir / "queries.jsonl.tmp"
    queries_path = work_dir / "queries.jsonl"
    with queries_tmp.open("w", encoding="utf-8", newline="\n") as dest:
        for query_hash, prompt in sorted(unique_prompts.items()):
            dest.write(json.dumps({"query_hash": query_hash, "prompt": prompt},
                                  ensure_ascii=False, separators=(",", ":")) + "\n")
    queries_tmp.replace(queries_path)
    query_counts["unique_questions"] = len(unique_prompts)
    query_counts["duplicate_records"] = query_counts["records"] - query_counts["empty_prompts"] - len(unique_prompts)
    metadata = {"source_dataset": "original_gpt4_llama7b", "source_sha256": source_digest.hexdigest(),
                "source_files": {split: path.name for split, path in split_files.items()}, **query_counts}
    (work_dir / "rlaif_input_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"RLAIF input: {query_counts['records']} rows, {len(unique_prompts)} unique prompts, "
          f"{query_counts['empty_prompts']} empty prompts across train/val/test", flush=True)
    return queries_path, records_path, metadata


def package_features(queries_path: Path, work_dir: Path, gpu_ids: list[int], laya_model: str, laya_version: str,
                     max_input_tokens: int, head_max_len: int, record_map_path: Path | None = None,
                     source_metadata: dict | None = None) -> bool:
    expected = set()
    with queries_path.open(encoding="utf-8") as source:
        for line in source:
            expected.add(json.loads(line)["query_hash"])
    if not expected:
        raise ValueError("The uploaded queries file is empty")
    covered = set()
    discarded = set()
    feature_by_hash = {}
    shards = [work_dir / f"laya_features_gpu{gpu_id}.jsonl" for gpu_id in gpu_ids]
    discarded_shards = [work_dir / f"laya_discarded_gpu{gpu_id}.jsonl" for gpu_id in gpu_ids]
    for shard in shards:
        with shard.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if len(row["features"]) != 40 or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in row["features"]):
                    raise ValueError(f"Invalid 40-feature row in {shard}")
                if row["query_hash"] in covered:
                    raise ValueError("Duplicate Laya features found; use a fresh --work-dir")
                covered.add(row["query_hash"])
                feature_by_hash[row["query_hash"]] = row["features"]
    for shard in discarded_shards:
        if not shard.is_file():
            continue
        with shard.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row["query_hash"] in discarded:
                    raise ValueError("Duplicate discarded query found; use a fresh --work-dir")
                discarded.add(row["query_hash"])
    missing = expected - covered - discarded
    if missing:
        print(f"Laya extraction is incomplete: {len(missing)} queries missing. Rerun to resume.", flush=True)
        return False
    if (covered | discarded) - expected or covered & discarded:
        raise ValueError("Feature/discard rows do not match this input file; use a fresh --work-dir")
    metadata = {"laya_model": laya_model, "laya_version": laya_version, "gpu_ids": gpu_ids, "feature_names": FEATURE_NAMES, "query_file_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(), "queries": len(expected), "laya_discarded_queries": len(discarded), "max_input_tokens": max_input_tokens, "max_len": max_input_tokens, "head_max_len": head_max_len}
    if source_metadata:
        metadata.update(source_metadata)
    metadata_path = work_dir / "extract_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    zip_path = work_dir / "laya_features.zip"
    zip_tmp = work_dir / "laya_features.zip.tmp"
    with zipfile.ZipFile(zip_tmp, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(metadata_path, metadata_path.name)
        for shard in shards:
            bundle.write(shard, shard.name)
        with bundle.open("discarded_queries.jsonl", "w") as dest:
            for shard in discarded_shards:
                if shard.is_file():
                    dest.write(shard.read_bytes())
        if record_map_path is not None:
            with bundle.open("rlaif_features.jsonl", "w") as dest:
                with record_map_path.open(encoding="utf-8") as records:
                    for line in records:
                        record = json.loads(line)
                        query_hash = record["query_hash"]
                        output = {"split": record["split"], "row_index": record["row_index"],
                                  "query_hash": query_hash, "prompt": record["prompt"]}
                        if query_hash in feature_by_hash:
                            output["features"] = feature_by_hash[query_hash]
                            output["status"] = "featured"
                        else:
                            output["features"] = None
                            output["status"] = "discarded_over_token_limit"
                        dest.write((json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    zip_tmp.replace(zip_path)
    print(f"Laya extraction finished: {len(covered)} kept, {len(discarded)} discarded over {max_input_tokens} tokens. Download {zip_path} and place it in the local work directory.", flush=True)
    return True


def extract(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    record_map_path = None
    source_metadata = None
    if args.rlaif_data_dir is not None:
        if not args.rlaif_data_dir.is_dir():
            raise FileNotFoundError(f"RLAIF data directory not found: {args.rlaif_data_dir}")
        queries_path, record_map_path, source_metadata = prepare_rlaif_queries(args.rlaif_data_dir, args.work_dir)
    elif args.queries is not None:
        if not args.queries.is_file():
            raise FileNotFoundError(f"Queries file not found: {args.queries}")
        queries_path = args.queries
    else:
        raise ValueError("Provide either --rlaif-data-dir or --queries")
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-ids must contain distinct GPU numbers, for example 0,1")
    input_digest = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    run_config = {"query_file_sha256": input_digest, "max_input_tokens": args.max_input_tokens,
                  "head_max_len": args.head_max_len, "feature_names": FEATURE_NAMES}
    if source_metadata:
        run_config["source_sha256"] = source_metadata["source_sha256"]
        run_config["record_map_sha256"] = hashlib.sha256(record_map_path.read_bytes()).hexdigest()
    config_path = args.work_dir / "run_config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise ValueError("This output directory belongs to a different input or token limit; choose a fresh --work-dir")
    elif list(args.work_dir.glob("laya_features_gpu*.jsonl")) or list(args.work_dir.glob("laya_discarded_gpu*.jsonl")):
        raise ValueError("Existing feature shards have no run manifest; choose a fresh --work-dir")
    else:
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    model_path = args.laya_model
    if not Path(model_path).exists():
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(
            model_path,
            allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
        )
    model_config = json.loads((Path(model_path) / "rl_agent_config.json").read_text(encoding="utf-8"))
    print(f"Laya input window: max_len={args.max_input_tokens}, head_max_len={args.head_max_len} (checkpoint defaults {model_config['max_len']}/{model_config['head_max_len']})", flush=True)
    context = mp.get_context("spawn")
    processes = []
    for worker_index, gpu_id in enumerate(gpu_ids):
        output = args.work_dir / f"laya_features_gpu{gpu_id}.jsonl"
        discarded_output = args.work_dir / f"laya_discarded_gpu{gpu_id}.jsonl"
        process = context.Process(
            target=_feature_worker,
            args=(gpu_id, worker_index, len(gpu_ids), str(queries_path.resolve()), str(output.resolve()),
                  str(discarded_output.resolve()), str(model_path), args.laya_batch_size, args.limit,
                  args.max_input_tokens, args.head_max_len),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("At least one Laya worker failed; rerun extract to resume completed queries")
    if args.smoke_test:
        for gpu_id in gpu_ids:
            shard = args.work_dir / f"laya_features_gpu{gpu_id}.jsonl"
            with shard.open(encoding="utf-8") as source:
                rows = [json.loads(line) for line in source]
            dropped_path = args.work_dir / f"laya_discarded_gpu{gpu_id}.jsonl"
            dropped = sum(1 for _ in dropped_path.open(encoding="utf-8")) if dropped_path.is_file() else 0
            print(f"GPU {gpu_id} smoke test: {len(rows)} kept with 40 features; {dropped} exceeded {args.max_input_tokens} tokens", flush=True)
        return
    import laya
    package_features(queries_path, args.work_dir, gpu_ids, args.laya_model, getattr(laya, "__version__", "unknown"),
                     args.max_input_tokens, args.head_max_len, record_map_path, source_metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, help="Optional queries.jsonl input from the original workflow")
    parser.add_argument("--rlaif-data-dir", type=Path,
                        help="Directory with router_train/router_val/router_test Parquet or CSV files from original_gpt4_llama7b")
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/laya_features"))
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--laya-model", default="convaiinnovations/laya")
    parser.add_argument("--laya-batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=None,
                        help="maximum packed state+probe input (defaults to 1,024 for RLAIF, 512 for legacy queries)")
    parser.add_argument("--head-max-len", type=int, default=192)
    parser.add_argument("--limit", type=int, help="Extraction smoke test: maximum new queries per GPU")
    parser.add_argument("--smoke-test", action="store_true", help="run 16 queries per GPU in a separate output directory")
    args = parser.parse_args()
    if args.max_input_tokens is None:
        args.max_input_tokens = 1024 if args.rlaif_data_dir is not None else 512
    if args.laya_batch_size < 1:
        parser.error("--laya-batch-size must be positive")
    if args.max_input_tokens < 1 or args.head_max_len < 1 or args.head_max_len >= args.max_input_tokens:
        parser.error("--max-input-tokens and --head-max-len must be positive, with head length below total input limit")
    if args.smoke_test:
        if args.limit is not None:
            parser.error("--smoke-test already limits extraction to 16 queries per GPU")
        args.limit = 16
        if args.work_dir == Path("/kaggle/working/laya_features"):
            args.work_dir = Path("/kaggle/working/laya_smoke")
    extract(args)


if __name__ == "__main__":
    main()
