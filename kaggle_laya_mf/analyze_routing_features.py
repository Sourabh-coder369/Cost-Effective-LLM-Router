"""Exploratory prompt-only feature analysis using training and validation rows.

No test labels or candidate responses enter feature construction or ranking.
Keyword flags are proxies, not semantic annotations. Permutations of the four
probabilities for one Laya property are performed jointly.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from laya_rlaif_mf_local import DEFAULT_DATASET, DEFAULT_FEATURES

OUT = Path(__file__).resolve().parent / "work" / "rlaif_feature_analysis"
SAVED = OUT.parent / "rlaif_router_comparison" / "catboost.cbm"

PATTERNS = {
    "code_task": r"\b(?:python|javascript|java|c\+\+|c#|sql|code|coding|programming|function|algorithm|debug|regex|html|css)\b",
    "math_task": r"\b(?:calculate|equation|algebra|probability|integral|derivative|arithmetic|geometry|mathematical|theorem|solve for)\b",
    "creative_writing": r"\b(?:story|poem|poetry|fiction|novel|lyrics|screenplay|creative|fairy tale)\b",
    "practical_writing": r"\b(?:email|letter|essay|blog|article|resume|cover letter|speech|marketing|advertisement)\b",
    "rewrite_or_edit": r"\b(?:rewrite|rephrase|paraphrase|proofread|correct the grammar|edit the|improve the following)\b",
    "summarization": r"\b(?:summarize|summarise|summary|summaries|tldr|tl;dr)\b",
    "translation": r"\b(?:translate|translation|translate into|translation into)\b",
    "extraction_classification": r"\b(?:extract|classify|classification|sentiment|label|categorize|part.of.speech|named entities)\b",
    "roleplay_persona": r"\b(?:act as|pretend|roleplay|role.play|you are a|you are an|in the style of)\b",
    "advice_planning": r"\b(?:advice|recommend|suggest|plan|planning|should i|how can i|how do i)\b",
    "medical_topic": r"\b(?:medical|medicine|disease|diagnosis|treatment|symptoms|diabetes|patient|doctor|cancer)\b",
    "legal_finance_topic": r"\b(?:legal|lawyer|lawsuit|contract|investment|financial|stocks|tax|mortgage)\b",
    "sexual_content_terms": r"\b(?:sex|sexual|erotic|erotica|porn|nude|naked|orgasm|penis|vagina)\b",
    "harm_illegality_terms": r"\b(?:kill|murder|suicide|bomb|weapon|illegal|steal|hack|malware|explosive)\b",
    "jailbreak_terms": r"\b(?:ignore (?:all |previous |your )?instructions|no restrictions|without restrictions|bypass|jailbreak|unfiltered|uncensored)\b",
    "format_requirement": r"\b(?:json|csv|xml|markdown|bullet points|table format|output format|numbered list)\b",
    "brevity_requirement": r"\b(?:concise|brief|short answer|one sentence|single sentence|no explanation|only (?:output|answer|return))\b",
    "detail_requirement": r"\b(?:detailed|in detail|comprehensive|step.by.step|explain (?:your|the) reasoning)\b",
    "explicit_length_requirement": r"\b\d+\s*(?:words|sentences|paragraphs|characters|lines)\b",
    "provided_context_marker": r"\b(?:following (?:text|passage|document|article|code|sentence)|given (?:text|passage|context)|based on|context:|input:)\b",
    "few_shot_marker": r"\b(?:example\s*\d*\s*:|input\s*\d*\s*:|output\s*\d*\s*:|follow the example|new instance)\b",
    "multiple_choice_marker": r"\([a-dA-D]\)|\b(?:which of the following|multiple choice|choose (?:the|one))\b",
    "current_information": r"\b(?:today|latest|current|recent|yesterday|news|202[4-9])\b",
}
REGEX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}


def extract(prompt):
    text = re.sub(r"\A\s*Human:\s*", "", prompt)
    text = re.sub(r"\s*Assistant:\s*\Z", "", text)
    words = re.findall(r"\b\w+\b", text)
    n = max(1, len(text))
    row = {
        "log_char_count": np.log1p(len(text)),
        "log_word_count": np.log1p(len(words)),
        "log_line_count": np.log1p(text.count("\n") + 1),
        "question_count": text.count("?"),
        "digit_fraction": sum(c.isdigit() for c in text) / n,
        "non_ascii_fraction": sum(ord(c) > 127 for c in text) / n,
        "code_punctuation_fraction": sum(c in "{}[]();=<>`" for c in text) / n,
        "code_fence_count": text.count("```"),
        "conversation_turn_count": len(re.findall(r"(?:\A|\n)\s*(?:Human|User|Assistant):", prompt, re.I)),
        "constraint_word_count": len(re.findall(r"\b(?:must|only|exactly|without|avoid|include|exclude|do not|at least|at most)\b", text, re.I)),
        "list_item_count": len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s", text)),
        "placeholder_count": len(re.findall(r"\bNAME_\d+\b", text)),
    }
    row.update({k: float(bool(pattern.search(text))) for k, pattern in REGEX.items()})
    return row


def load():
    source = {s: pq.read_table(DEFAULT_DATASET / f"router_{s}.parquet", columns=["prompt", "binary_label"]).to_pydict() for s in ("train", "val")}
    features, indices = {s: [] for s in source}, {s: [] for s in source}
    with zipfile.ZipFile(DEFAULT_FEATURES) as archive:
        names = json.loads(archive.read("extract_metadata.json"))["feature_names"]
        with archive.open("rlaif_features.jsonl") as rows:
            for line in rows:
                row = json.loads(line)
                s = row["split"]
                if s not in source or row["status"] != "featured":
                    continue
                i = int(row["row_index"])
                assert row["prompt"] == source[s]["prompt"][i]
                indices[s].append(i)
                features[s].append(row["features"])
    prompts = {s: [source[s]["prompt"][i] for i in indices[s]] for s in source}
    labels = {s: np.asarray([source[s]["binary_label"][i] for i in indices[s]]) for s in source}
    engineered = {s: [extract(p) for p in prompts[s]] for s in source}
    cheap_names = list(engineered["train"][0])
    cheap = {s: np.asarray([[r[n] for n in cheap_names] for r in engineered[s]], dtype=np.float32) for s in source}
    return prompts, labels, {s: np.asarray(features[s], dtype=np.float32) for s in source}, names, cheap, cheap_names


def scores(y, p):
    return {"n": len(y), "auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)), "accuracy_at_0_5": float(accuracy_score(y, p >= .5))}


def permutations(model, x, y, groups, repeats=5):
    assert sorted(i for columns in groups.values() for i in columns) == list(range(x.shape[1]))
    base = model.predict_proba(x)[:, 1]
    base_auc, base_loss = roc_auc_score(y, base), log_loss(y, base)
    results = []
    for name, columns in groups.items():
        rng = np.random.default_rng(42)
        auc_drops, loss_increases = [], []
        for _ in range(repeats):
            copied = x.copy()
            copied[:, columns] = x[rng.permutation(len(x))][:, columns]
            p = model.predict_proba(copied)[:, 1]
            auc_drops.append(base_auc - roc_auc_score(y, p))
            loss_increases.append(log_loss(y, p) - base_loss)
        results.append({"feature_group": name, "auc_drop": float(np.mean(auc_drops)), "auc_drop_shuffle_sd": float(np.std(auc_drops)), "log_loss_increase": float(np.mean(loss_increases))})
    return sorted(results, key=lambda r: r["auc_drop"], reverse=True)


def fit(train, val, labels):
    model = CatBoostClassifier(iterations=500, depth=5, learning_rate=.05, l2_leaf_reg=5,
        loss_function="Logloss", eval_metric="Logloss", od_type="Iter", od_wait=50,
        thread_count=4, random_seed=42, verbose=False, allow_writing_files=False)
    model.fit(train, labels["train"], eval_set=(val, labels["val"]), use_best_model=True)
    return model


def matched_comparisons(report, labels, laya, cheap):
    model = fit(laya["train"], laya["val"], labels)
    report["models"]["laya_matched_catboost"] = {
        "validation": scores(labels["val"], model.predict_proba(laya["val"])[:, 1]),
        "best_iteration": model.get_best_iteration(),
        "note": "Same depth=5 and training configuration as the new prompt feature experiments."}
    model.save_model(str(OUT / "laya_matched_catboost.cbm"))
    for name, matrices in (("linear_laya", laya), ("linear_laya_plus_prompt", {
            s: np.concatenate([laya[s], cheap[s]], axis=1) for s in labels})):
        linear = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42))
        linear.fit(matrices["train"], labels["train"])
        report["models"][name] = {"validation": scores(labels["val"], linear.predict_proba(matrices["val"])[:, 1]),
            "note": "Linear logistic scorer has the same decision-function form as fixed-query two-model MF with biases; regularization differs."}
    for name in ("laya_matched_catboost", "linear_laya", "linear_laya_plus_prompt"):
        print(name, report["models"][name], flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prompts, labels, laya, laya_names, cheap, cheap_names = load()
    groups = {}
    for i, name in enumerate(laya_names):
        groups.setdefault(name.split("__")[0], []).append(i)
    normalize = lambda p: re.sub(r"\s+", " ", p).strip().casefold()
    train_set = {normalize(p) for p in prompts["train"]}
    report = {"dataset": "original_gpt4_llama7b", "scope": "Laya-retained training and validation rows only; no test evaluation",
              "caveats": ["Exploratory validation evidence; no claim of globally optimal features.",
                          "Keyword flags are noisy proxies for semantic properties.",
                          "Early stopping uses validation labels; confirm selected features with a fresh grouped holdout.",
                          "Permutation standard deviation measures shuffle variability, not sampling confidence.",
                          "Targets are RLAIF preferences, not objective answer correctness."],
              "counts": {s: len(y) for s,y in labels.items()},
              "gpt4_win_rates": {s: float(y.mean()) for s,y in labels.items()},
              "validation_normalized_prompt_overlap_with_train": sum(normalize(p) in train_set for p in prompts["val"]),
              "laya_names": laya_names, "prompt_feature_names": cheap_names, "models": {}, "slices": {}}
    model = CatBoostClassifier()
    model.load_model(str(SAVED))
    report["models"]["existing_laya40"] = {"validation": scores(labels["val"], model.predict_proba(laya["val"])[:,1]),
                                            "permutation_groups": permutations(model, laya["val"], labels["val"], groups)}
    print("Existing Laya grouped importances complete", flush=True)
    for name, matrices, feature_groups in (
        ("prompt_features", cheap, {n: [i] for i,n in enumerate(cheap_names)}),
        ("laya_plus_prompt", {s: np.concatenate([laya[s],cheap[s]], axis=1) for s in labels},
            {**groups, **{f"prompt__{n}": [i+40] for i,n in enumerate(cheap_names)}}),
    ):
        model = fit(matrices["train"], matrices["val"], labels)
        result = {"validation": scores(labels["val"], model.predict_proba(matrices["val"])[:,1]), "best_iteration": model.get_best_iteration()}
        print(name, result, flush=True)
        result["permutation_groups"] = permutations(model, matrices["val"], labels["val"], feature_groups)
        report["models"][name] = result
        model.save_model(str(OUT / f"{name}.cbm"))
    for j,name in enumerate(cheap_names):
        if name not in PATTERNS:
            continue
        report["slices"][name] = {}
        for s in labels:
            mask = cheap[s][:,j] > 0
            report["slices"][name][s] = {"n": int(mask.sum()), "fraction": float(mask.mean()), "gpt4_win_rate": float(labels[s][mask].mean()) if mask.any() else None,
                                        "gpt4_win_rate_absent": float(labels[s][~mask].mean()) if (~mask).any() else None}
    report["length_slices"] = {}
    for low, high in ((0,50),(50,100),(100,250),(250,500),(500,1000),(1000,10000000)):
        row = {}
        for s in labels:
            count = np.expm1(cheap[s][:,1])
            mask = (count >= low-.001) & (count < high-.001)
            row[s] = {"n": int(mask.sum()), "gpt4_win_rate": float(labels[s][mask].mean()) if mask.any() else None}
        report["length_slices"][f"{low}-{high}_words"] = row
    matched_comparisons(report, labels, laya, cheap)
    (OUT / "feature_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved", OUT / "feature_analysis.json", flush=True)


if __name__ == "__main__":
    main()
