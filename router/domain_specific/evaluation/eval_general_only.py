"""
General-Only Evaluation: Baseline MF vs Domain MF Router
=========================================================
Filters test set to ONLY General (non-math) queries and compares both routers.

Usage:
    cd router/domain_specific
    python eval_general_only.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from domain_specific.domain_model   import DomainMFRouter
from domain_specific.domain_dataset import DomainRouterDataset
from baseline.model                 import MatrixFactorizationRouter
from baseline.dataset               import RouterDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent.parent     # capstone/

TEST_PARQUET  = BASE / "gpt4_llama7b_data_unbalanced/router_test.parquet"
TEST_EMB      = BASE / "router/data/router_test_bge_large_embeddings.pt"
DOMAIN_CKPT   = BASE / "router/domain_specific/checkpoints_domain_mf/best_model.pt"
BASELINE_CKPT = BASE / "router/baseline/checkpoints_bge_large_unbalanced/best_model.pt"

# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(probs, labels):
    results = []
    for t in np.linspace(0, 1, 1000):
        route_strong = probs >= t
        cost    = route_strong.mean()
        quality = np.where(route_strong, labels, 1 - labels).mean()
        results.append({"t": t, "cost": cost, "quality": quality})
    return results


def cost_at_drop(sweep, labels, target_drop_pct):
    baseline_q = labels.mean()
    target_q   = baseline_q - target_drop_pct / 100.0
    valid = [r for r in sweep if r["quality"] >= target_q]
    if not valid:
        return None, None, None
    best = min(valid, key=lambda x: x["cost"])
    return best["cost"], best["quality"], best["t"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  GENERAL-ONLY Evaluation: Baseline MF vs Domain MF Router")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    # ── 1. Load Domain Dataset (gives us domain IDs) ──────────────────────────
    print("Loading test dataset...")
    test_ds = DomainRouterDataset(str(TEST_PARQUET), str(TEST_EMB))

    general_mask  = test_ds.domain_ids.numpy() == 0   # 0 = general
    general_count = general_mask.sum()
    print(f"  Total test samples    : {len(test_ds):,}")
    print(f"  General test queries  : {general_count:,}  ({general_count/len(test_ds)*100:.1f}%)")

    # ── 2. Domain MF inference ────────────────────────────────────────────────
    print("\nRunning Domain MF Router inference...")
    ckpt = torch.load(DOMAIN_CKPT, map_location=device, weights_only=False)
    domain_model = DomainMFRouter(
        query_embedding_dim=test_ds.embeddings.shape[1],
        model_embedding_dim=64,
        num_models=2,
        num_domains=2,
    ).to(device)
    domain_model.load_state_dict(ckpt["model_state_dict"])
    domain_model.eval()

    all_emb, all_lbl, all_dom = [], [], []
    for emb, lbl, dom in DataLoader(test_ds, batch_size=256, shuffle=False):
        all_emb.append(emb); all_lbl.append(lbl); all_dom.append(dom)
    emb_t  = torch.cat(all_emb).to(device)
    lbl_np = torch.cat(all_lbl).numpy()
    dom_t  = torch.cat(all_dom).to(device)

    with torch.no_grad():
        domain_probs = domain_model(emb_t, dom_t).cpu().numpy().squeeze()

    # ── 3. Baseline inference ─────────────────────────────────────────────────
    print("Running Baseline MF Router inference...")
    bl_ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline_query_dim = bl_ckpt["model_state_dict"]["text_proj.0.weight"].shape[1]
    baseline = MatrixFactorizationRouter(
        query_embedding_dim=baseline_query_dim,
        model_embedding_dim=bl_ckpt["config"]["model_embedding_dim"],
        num_models=2,
    ).to(device)
    baseline.load_state_dict(bl_ckpt["model_state_dict"])
    baseline.eval()

    test_standard = RouterDataset(str(TEST_PARQUET),
                                  precomputed_embeddings_path=str(TEST_EMB))
    all_emb_b = []
    for emb, lbl in DataLoader(test_standard, batch_size=256, shuffle=False):
        all_emb_b.append(emb)
    emb_b = torch.cat(all_emb_b).to(device)

    with torch.no_grad():
        base_probs = baseline(emb_b).cpu().numpy().squeeze()

    # ── 4. Filter to ONLY General queries ────────────────────────────────────
    gen_domain_probs = domain_probs[general_mask]
    gen_base_probs   = base_probs[general_mask]
    gen_labels       = lbl_np[general_mask]

    always_strong_quality = gen_labels.mean()
    print(f"\n  Always-GPT4 baseline quality (General only): {always_strong_quality*100:.2f}%")

    # ── 5. Confidence score stats ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  CONFIDENCE SCORES ON GENERAL QUERIES")
    print(f"{'='*65}")
    print(f"  {'Metric':<35} {'Baseline':>12} {'Domain MF':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Mean Probability':<35} {gen_base_probs.mean():>11.4f}  {gen_domain_probs.mean():>11.4f}")
    print(f"  {'Median Probability':<35} {np.median(gen_base_probs):>11.4f}  {np.median(gen_domain_probs):>11.4f}")
    print(f"  {'Std Dev of Probability':<35} {gen_base_probs.std():>11.4f}  {gen_domain_probs.std():>11.4f}")
    print(f"  {'% Routed to GPT-4 (t=0.5)':<35} {(gen_base_probs>=0.5).mean()*100:>10.2f}%  {(gen_domain_probs>=0.5).mean()*100:>10.2f}%")
    print(f"  {'% Routed to Llama  (t=0.5)':<35} {(gen_base_probs<0.5).mean()*100:>10.2f}%  {(gen_domain_probs<0.5).mean()*100:>10.2f}%")

    # ── 6. Accuracy & F1 (at threshold = 0.5) ────────────────────────────────
    base_preds   = (gen_base_probs   >= 0.5).astype(int)
    domain_preds = (gen_domain_probs >= 0.5).astype(int)

    base_acc   = (base_preds   == gen_labels).mean() * 100
    domain_acc = (domain_preds == gen_labels).mean() * 100
    base_f1    = f1_score(gen_labels, base_preds,   zero_division=0) * 100
    domain_f1  = f1_score(gen_labels, domain_preds, zero_division=0) * 100

    print(f"\n{'='*65}")
    print("  ACCURACY & F1 ON GENERAL QUERIES (threshold=0.5)")
    print(f"{'='*65}")
    print(f"  {'Metric':<35} {'Baseline':>12} {'Domain MF':>12} {'Delta':>8}")
    print(f"  {'-'*68}")
    print(f"  {'Accuracy':<35} {base_acc:>10.2f}%  {domain_acc:>10.2f}%  {domain_acc-base_acc:>+7.2f}%")
    print(f"  {'F1 Score':<35} {base_f1:>10.2f}%  {domain_f1:>10.2f}%  {domain_f1-base_f1:>+7.2f}%")

    # ── 7. Cost vs Quality sweep — General Only ───────────────────────────────
    base_sweep   = threshold_sweep(gen_base_probs,   gen_labels)
    domain_sweep = threshold_sweep(gen_domain_probs, gen_labels)

    print(f"\n{'='*65}")
    print("  COST vs QUALITY ON GENERAL QUERIES ONLY")
    print(f"  Always-GPT4 Quality Ceiling: {always_strong_quality*100:.2f}%")
    print(f"{'='*65}")
    print(f"  {'Drop':>6} | {'Baseline':^26} | {'Domain MF':^26} | {'Delta':>8}")
    print(f"  {'':6} | {'Quality':>10}  {'Cost Adv':>10} | {'Quality':>10}  {'Cost Adv':>10} |")
    print(f"  {'-'*75}")

    for drop in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        b_cost, b_qual, b_t = cost_at_drop(base_sweep,   gen_labels, drop)
        d_cost, d_qual, d_t = cost_at_drop(domain_sweep, gen_labels, drop)
        b_cadv = (1 - b_cost) * 100 if b_cost is not None else None
        d_cadv = (1 - d_cost) * 100 if d_cost is not None else None
        delta  = (d_cadv - b_cadv) if (d_cadv and b_cadv) else None

        b_q_s = f"{b_qual*100:.2f}%" if b_qual else "N/A"
        b_c_s = f"{b_cadv:.2f}%"     if b_cadv else "N/A"
        d_q_s = f"{d_qual*100:.2f}%" if d_qual else "N/A"
        d_c_s = f"{d_cadv:.2f}%"     if d_cadv else "N/A"
        d_str = (f"+{delta:.2f}%" if delta and delta >= 0 else f"{delta:.2f}%") if delta else "N/A"

        print(f"  {drop:>5.1f}% | {b_q_s:>10}  {b_c_s:>10} | {d_q_s:>10}  {d_c_s:>10} | {d_str:>8}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
