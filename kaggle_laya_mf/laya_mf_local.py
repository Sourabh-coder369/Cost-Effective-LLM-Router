"""Prepare LLMRouterBench locally, then train the 40-D MF router using Laya features from Kaggle.

Stages: prepare, train, rank. Run prepare before uploading work/queries.jsonl to Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


MODEL_POOL = (
    "DeepHermes-3-Llama-3-8B-Preview", "DeepSeek-R1-0528-Qwen3-8B",
    "DeepSeek-R1-Distill-Qwen-7B", "Fin-R1", "GLM-Z1-9B-0414",
    "Intern-S1-mini", "Llama-3.1-8B-Instruct", "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1", "MiMo-7B-RL-0530", "MiniCPM4.1-8B",
    "NVIDIA-Nemotron-Nano-9B-v2", "OpenThinker3-7B", "Qwen2.5-Coder-7B-Instruct",
    "Qwen3-8B", "claude-sonnet-4", "cogito-v1-preview-llama-8B",
    "deepseek-r1-0528", "deepseek-v3-0324", "deepseek-v3.1-terminus",
    "gemini-2.5-flash", "gemini-2.5-pro", "gemma-2-9b-it", "glm-4-9b-chat",
    "glm-4.6", "gpt-5", "gpt-5-chat", "granite-3.3-8b-instruct",
    "intern-s1", "internlm3-8b-instruct", "kimi-k2-0905",
    "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507",
)
MAX_INPUT_TOKENS = 512

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
assert len(PROBES) == 17 and len(FEATURE_NAMES) == 40 and len(MODEL_POOL) == 33


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def partition(group_hash: str, seed: int) -> str:
    bucket = int(sha(f"{seed}:{group_hash}")[:16], 16) % 10_000
    return "train" if bucket < 7000 else "validation" if bucket < 8000 else "test"


def prepare(args: argparse.Namespace) -> None:
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    target = args.work_dir / "benchmark.sqlite"
    if target.exists():
        raise FileExistsError(f"{target} already exists; choose a fresh --work-dir to prepare again")
    temporary = args.work_dir / "benchmark.sqlite.tmp"
    if temporary.exists():
        raise FileExistsError(f"Incomplete preparation exists at {temporary}; inspect it before retrying")
    db = sqlite3.connect(temporary)
    db.executescript("""
        CREATE TABLE queries (query_hash TEXT PRIMARY KEY, prompt TEXT NOT NULL, group_hash TEXT NOT NULL, partition TEXT NOT NULL);
        CREATE TABLE samples (sample_key TEXT PRIMARY KEY, query_hash TEXT NOT NULL, dataset TEXT NOT NULL, source_split TEXT NOT NULL, item_index TEXT NOT NULL);
        CREATE TABLE outcomes (sample_key TEXT NOT NULL, model TEXT NOT NULL, score REAL NOT NULL, cost REAL, PRIMARY KEY(sample_key, model));
        CREATE INDEX samples_query ON samples(query_hash);
        CREATE INDEX outcomes_sample ON outcomes(sample_key);
    """)
    invalid_scores = 0
    eligible_records = 0
    model_set = set(MODEL_POOL)
    with tarfile.open(args.archive, "r:gz") as archive:
        for file_number, member in enumerate(archive):
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            with archive.extractfile(member) as source:
                run = json.load(source)
            model = run.get("model_name")
            if model not in model_set:
                continue
            dataset, source_split = str(run["dataset_name"]), str(run["split"])
            for record in run["records"]:
                prompt = record.get("origin_query")
                if not isinstance(prompt, str) or not prompt.strip():
                    continue
                prompt = prompt.strip()
                query_hash = sha(prompt)
                group_hash = sha(re.sub(r"\s+", " ", prompt))
                sample_key = sha(f"{dataset}\0{source_split}\0{record['index']}\0{query_hash}")
                db.execute("INSERT OR IGNORE INTO queries VALUES (?, ?, ?, ?)", (query_hash, prompt, group_hash, partition(group_hash, args.seed)))
                db.execute("INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?, ?)", (sample_key, query_hash, dataset, source_split, str(record["index"])))
                score = record.get("score")
                if not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                    invalid_scores += 1
                    continue
                cost = record.get("cost")
                cost = float(cost) if isinstance(cost, (int, float)) and math.isfinite(float(cost)) else None
                db.execute("INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?)", (sample_key, model, float(score), cost))
                eligible_records += 1
            if file_number % 25 == 0:
                db.commit()
    db.commit()
    counts = {
        "queries": db.execute("SELECT COUNT(*) FROM queries").fetchone()[0],
        "samples": db.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
        "observed_outcomes": db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0],
        "invalid_scores_skipped": invalid_scores,
        "eligible_records_seen": eligible_records,
        "partition_queries": dict(db.execute("SELECT partition, COUNT(*) FROM queries GROUP BY partition")),
        "partition_outcomes": dict(db.execute("SELECT q.partition, COUNT(*) FROM outcomes o JOIN samples s USING(sample_key) JOIN queries q USING(query_hash) GROUP BY q.partition")),
    }
    queries_tmp = args.work_dir / "queries.jsonl.tmp"
    queries_path = args.work_dir / "queries.jsonl"
    with queries_tmp.open("w", encoding="utf-8", newline="\n") as dest:
        for query_hash, prompt in db.execute("SELECT query_hash, prompt FROM queries ORDER BY query_hash"):
            dest.write(json.dumps({"query_hash": query_hash, "prompt": prompt}, ensure_ascii=False) + "\n")
    queries_tmp.replace(queries_path)
    query_file_sha256 = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    db.close()
    temporary.replace(target)
    metadata = {"archive": str(args.archive), "seed": args.seed, "model_pool": list(MODEL_POOL), "feature_names": FEATURE_NAMES, "query_file_sha256": query_file_sha256, **counts}
    (args.work_dir / "prepare_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)


def load_features(work_dir: Path) -> tuple[dict[str, list[float]], set[str]]:
    features = {}
    discarded = set()
    archive_path = work_dir / "laya_features.zip"
    if not archive_path.is_file():
        sibling_archive = work_dir.parent / "laya_features" / "laya_features.zip"
        if sibling_archive.is_file():
            archive_path = sibling_archive
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            metadata = json.loads(archive.read("extract_metadata.json"))
            prepared = json.loads((work_dir / "prepare_metadata.json").read_text(encoding="utf-8"))
            if metadata["feature_names"] != FEATURE_NAMES or metadata["query_file_sha256"] != prepared["query_file_sha256"]:
                raise ValueError("The downloaded Laya features do not match this prepared benchmark")
            if metadata.get("max_input_tokens") != MAX_INPUT_TOKENS or metadata.get("max_len") != MAX_INPUT_TOKENS:
                raise ValueError(f"Expected Laya features prepared with the {MAX_INPUT_TOKENS}-token input limit")
            names = sorted(name for name in archive.namelist() if name.startswith("laya_features_gpu") and name.endswith(".jsonl"))
            if not names:
                raise ValueError("The Laya ZIP has no feature shards")
            for name in names:
                for line in archive.read(name).decode("utf-8").splitlines():
                    row = json.loads(line)
                    features[row["query_hash"]] = vectorize_from_cache(row["features"])
            if "discarded_queries.jsonl" not in archive.namelist():
                raise ValueError("The Laya ZIP is missing discarded-query metadata")
            for line in archive.read("discarded_queries.jsonl").decode("utf-8").splitlines():
                if line.strip():
                    discarded.add(json.loads(line)["query_hash"])
    else:
        for path in sorted(work_dir.glob("laya_features_gpu*.jsonl")):
            with path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        row = json.loads(line)
                        features[row["query_hash"]] = vectorize_from_cache(row["features"])
                    except (ValueError, KeyError, TypeError, IndexError):
                        continue
        for path in sorted(work_dir.glob("laya_discarded_gpu*.jsonl")):
            with path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        discarded.add(json.loads(line)["query_hash"])
                    except (ValueError, KeyError, TypeError):
                        continue
    if set(features) & discarded:
        raise ValueError("A query appears in both the feature and discarded lists")
    return features, discarded


def vectorize_from_cache(values: list[float]) -> list[float]:
    if len(values) != 40 or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in values):
        raise ValueError("Invalid cached Laya feature row")
    return [float(value) for value in values]


def train(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F

    db_path = args.work_dir / "benchmark.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError("Run prepare first")
    features_by_hash, discarded_hashes = load_features(args.work_dir)
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    all_query_rows = list(db.execute("SELECT query_hash FROM queries ORDER BY query_hash"))
    known_hashes = {row[0] for row in all_query_rows}
    missing_features = known_hashes - set(features_by_hash) - discarded_hashes
    if missing_features or (set(features_by_hash) | discarded_hashes) - known_hashes:
        raise ValueError(f"Laya feature/discard lists do not match prepared questions; {len(missing_features)} missing. Complete Kaggle extraction and download laya_features.zip")
    query_rows = [row for row in all_query_rows if row[0] in features_by_hash]
    if not query_rows:
        raise ValueError(f"The {MAX_INPUT_TOKENS}-token limit discarded every question")
    query_index = {row[0]: index for index, row in enumerate(query_rows)}
    x = np.asarray([features_by_hash[row[0]] for row in query_rows], dtype=np.float32)
    model_index = {model: index for index, model in enumerate(MODEL_POOL)}
    rows = {name: [] for name in ("train", "validation", "test")}
    dataset_counts = Counter()
    sql = """SELECT s.sample_key, s.query_hash, s.dataset, q.partition, o.model, o.score, o.cost
             FROM outcomes o JOIN samples s USING(sample_key) JOIN queries q USING(query_hash)"""
    discarded_outcomes = 0
    for sample_key, query_hash, dataset, part, model, score, cost in db.execute(sql):
        if query_hash in discarded_hashes:
            discarded_outcomes += 1
            continue
        rows[part].append((query_index[query_hash], model_index[model], float(score), dataset, sample_key, cost))
        if part == "train":
            dataset_counts[dataset] += 1
    db.close()
    if not all(rows.values()):
        raise ValueError("Train, validation, and test must all contain outcomes")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    features = torch.as_tensor(x, device=device)
    train_rows = rows["train"]
    qidx = torch.tensor([r[0] for r in train_rows], dtype=torch.long, device=device)
    midx = torch.tensor([r[1] for r in train_rows], dtype=torch.long, device=device)
    targets = torch.tensor([r[2] for r in train_rows], dtype=torch.float32, device=device)
    weights = torch.tensor([1.0 / dataset_counts[r[3]] for r in train_rows], dtype=torch.float32, device=device)
    weights *= len(weights) / weights.sum()
    vectors = torch.nn.Embedding(len(MODEL_POOL), 40, sparse=True, device=device)
    torch.nn.init.normal_(vectors.weight, mean=0.0, std=0.02)
    optimizer = torch.optim.SparseAdam(vectors.parameters(), lr=args.lr)

    def eval_bce(part: str) -> float:
        selected = rows[part]
        losses = []
        with torch.no_grad():
            for start in range(0, len(selected), args.batch_size):
                chunk = selected[start:start + args.batch_size]
                qi = torch.tensor([r[0] for r in chunk], device=device)
                mi = torch.tensor([r[1] for r in chunk], device=device)
                y = torch.tensor([r[2] for r in chunk], dtype=torch.float32, device=device)
                logits = (features[qi] * vectors(mi)).sum(dim=1)
                losses.append((F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item(), len(chunk)))
        return sum(value for value, _ in losses) / sum(count for _, count in losses)

    history = []
    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_rows), device=device)
        for batch in order.split(args.batch_size):
            logits = (features[qidx[batch]] * vectors(midx[batch])).sum(dim=1)
            loss_per_row = F.binary_cross_entropy_with_logits(logits, targets[batch], reduction="none")
            loss = (loss_per_row * weights[batch]).sum() / weights[batch].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = eval_bce("train")
        val_loss = eval_bce("validation")
        history.append({"epoch": epoch, "train_bce": train_loss, "validation_bce": val_loss})
        print(f"epoch {epoch:03d} train_bce={train_loss:.5f} validation_bce={val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best_weights = vectors.weight.detach().cpu().clone()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_weights is not None
    checkpoint = {"model_vectors": best_weights, "model_names": list(MODEL_POOL), "feature_names": FEATURE_NAMES, "best_epoch": best_epoch, "validation_bce": best_loss, "seed": args.seed}
    torch.save(checkpoint, args.work_dir / "laya_mf_40.pt")
    (args.work_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    metrics = evaluate(rows["test"], x, best_weights.numpy())
    metrics.update({"best_epoch": best_epoch, "validation_bce": best_loss, "train_outcomes": len(rows["train"]), "validation_outcomes": len(rows["validation"]), "test_outcomes": len(rows["test"]), "token_limit": MAX_INPUT_TOKENS, "discarded_questions_over_token_limit": len(discarded_hashes), "discarded_outcomes": discarded_outcomes})
    (args.work_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def evaluate(test_rows: list[tuple], features, model_vectors) -> dict:
    import numpy as np

    by_sample = defaultdict(list)
    for query_index, model_index, score, dataset, sample_key, cost in test_rows:
        by_sample[sample_key].append((model_index, score, cost, query_index, dataset))
    chosen = []
    oracle = []
    random_expected = []
    chosen_cost = []
    global_covered = 0
    by_dataset = defaultdict(list)
    for sample_key, available in by_sample.items():
        query_index, dataset = available[0][3], available[0][4]
        predictions = features[query_index] @ model_vectors.T
        global_choice = int(np.argmax(predictions))
        global_covered += any(model == global_choice for model, *_ in available)
        best = max(available, key=lambda item: float(predictions[item[0]]))
        chosen.append(float(best[1]))
        oracle.append(max(float(item[1]) for item in available))
        random_expected.append(float(np.mean([item[1] for item in available])))
        if best[2] is not None:
            chosen_cost.append(float(best[2]))
        by_dataset[dataset].append(float(best[1]))
    return {
        "evaluated_question_instances": len(by_sample),
        "observed_candidate_mean_score": float(np.mean(chosen)),
        "observed_candidate_macro_dataset_score": float(np.mean([np.mean(values) for values in by_dataset.values()])),
        "random_observed_candidate_mean_score": float(np.mean(random_expected)),
        "observed_oracle_mean_score": float(np.mean(oracle)),
        "global_argmax_outcome_coverage": global_covered / len(by_sample),
        "selected_recorded_cost_mean": float(np.mean(chosen_cost)) if chosen_cost else None,
        "per_dataset": {dataset: {"questions": len(values), "mean_score": float(np.mean(values))} for dataset, values in sorted(by_dataset.items())},
    }


def rank(args: argparse.Namespace) -> None:
    import torch

    checkpoint = torch.load(args.work_dir / "laya_mf_40.pt", map_location="cpu", weights_only=True)
    if checkpoint["feature_names"] != FEATURE_NAMES:
        raise ValueError("The checkpoint uses a different 40-feature order")
    values = vectorize_from_cache(json.loads(args.features_json.read_text(encoding="utf-8")))
    query_vector = torch.tensor(values, dtype=torch.float32)
    logits = query_vector @ checkpoint["model_vectors"].T
    ranking = sorted(zip(checkpoint["model_names"], logits.tolist()), key=lambda item: item[1], reverse=True)
    print(json.dumps({"selected_model": ranking[0][0], "ranked_models": [{"model": name, "predicted_quality": float(torch.sigmoid(torch.tensor(logit)))} for name, logit in ranking[:args.top_k]]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "rank"))
    parser.add_argument("--archive", type=Path, default=Path(__file__).with_name("bench-release.tar.gz"))
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).with_name("work"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--features-json", type=Path, help="JSON file containing 40 Laya feature values for ranking")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "train":
        train(args)
    else:
        if args.features_json is None:
            parser.error("rank requires --features-json")
        rank(args)


if __name__ == "__main__":
    main()
