"""
Train Domain-Specific MF Router and Compare vs Baseline
=========================================================
1. Trains the DomainMFRouter (math + general vectors)
2. Loads the best existing baseline MF checkpoint
3. Evaluates BOTH on the test set
4. Prints a side-by-side comparison table

Usage:
    cd router
    python train_domain_router.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domain_specific.domain_model   import DomainMFRouter
from domain_specific.domain_dataset import DomainRouterDataset, create_domain_dataloaders
from baseline.model          import MatrixFactorizationRouter
from baseline.dataset        import RouterDataset

# ─────────────────────────────────────────────────────────────────────────────
# Paths — adjust if yours differ
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent    # capstone/

TRAIN_PARQUET = BASE / "gpt4_llama7b_data_unbalanced/router_train.parquet"
VAL_PARQUET   = BASE / "gpt4_llama7b_data_unbalanced/router_val.parquet"
TEST_PARQUET  = BASE / "gpt4_llama7b_data_unbalanced/router_test.parquet"

TRAIN_EMB = BASE / "router/data/router_train_bge_large_embeddings.pt"
VAL_EMB   = BASE / "router/data/router_val_bge_large_embeddings.pt"
TEST_EMB  = BASE / "router/data/router_test_bge_large_embeddings.pt"

# Baseline checkpoint (best existing model)
BASELINE_CKPT = BASE / "router/baseline/checkpoints_bge_large_unbalanced/best_model.pt"

# Where to save the domain model
DOMAIN_CKPT_DIR = BASE / "router/domain_specific/checkpoints_domain_mf"

# Training hyper-parameters
CONFIG = {
    "query_embedding_dim" : 1024,
    "model_embedding_dim" : 64,
    "batch_size"          : 64,
    "num_epochs"          : 0,
    "learning_rate"       : 3e-4,
    "weight_decay"        : 1e-5,
    "early_stopping"      : 8,
    "seed"                : 42,
}

QUALITY_DROPS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for emb, labels, domains in tqdm(loader, desc="  train", leave=False):
        emb, labels, domains = emb.to(device), labels.to(device), domains.to(device)
        optimizer.zero_grad()
        logits = model.forward_logits(emb, domains)
        loss   = criterion(logits, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * emb.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.no_grad():
        for emb, labels, domains in loader:
            emb, labels, domains = emb.to(device), labels.to(device), domains.to(device)
            logits = model.forward_logits(emb, domains)
            probs  = torch.sigmoid(logits)
            loss   = criterion(logits, labels.unsqueeze(1))
            total_loss += loss.item() * emb.size(0)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)
    acc = np.mean(all_preds == all_labels)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    return total_loss / len(loader.dataset), acc, f1, auc


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (threshold sweep)
# ─────────────────────────────────────────────────────────────────────────────

def get_domain_predictions(model, dataset, device):
    """Run inference over a DomainRouterDataset, return probs + labels."""
    all_emb, all_lbl, all_dom = [], [], []
    for emb, lbl, dom in DataLoader(dataset, batch_size=256, shuffle=False,
                                     collate_fn=lambda b: (
                                         torch.stack([x[0] for x in b]),
                                         torch.stack([x[1] for x in b]),
                                         torch.stack([x[2] for x in b]),
                                     )):
        all_emb.append(emb); all_lbl.append(lbl); all_dom.append(dom)

    emb    = torch.cat(all_emb).to(device)
    labels = torch.cat(all_lbl).numpy()
    doms   = torch.cat(all_dom).to(device)

    with torch.no_grad():
        probs = model(emb, doms).cpu().numpy().squeeze()
    return probs, labels


def get_baseline_predictions(model, dataset, device):
    """Run inference over standard RouterDataset."""
    all_emb, all_lbl = [], []
    for emb, lbl in DataLoader(dataset, batch_size=256, shuffle=False):
        all_emb.append(emb); all_lbl.append(lbl)
    emb    = torch.cat(all_emb).to(device)
    labels = torch.cat(all_lbl).numpy()
    with torch.no_grad():
        probs = model(emb).cpu().numpy().squeeze()
    return probs, labels


def threshold_sweep(probs, labels):
    thresholds = np.linspace(0, 1, 1000)
    results = []
    for t in thresholds:
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


def tune_threshold(val_probs, val_labels):
    """
    Find the threshold that maximises accuracy on the validation set.
    Returns the best threshold.
    """
    thresholds = np.linspace(0.01, 0.99, 500)
    best_t, best_acc = 0.5, 0.0
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        acc   = (preds == val_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t   = t
    return best_t, best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  Domain-Specific MF Router — Train + Compare")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    # ── 1. Load datasets ──────────────────────────────────────────────────────
    print("► Loading datasets...")
    train_ds = DomainRouterDataset(str(TRAIN_PARQUET), str(TRAIN_EMB))
    val_ds   = DomainRouterDataset(str(VAL_PARQUET),   str(VAL_EMB))
    test_ds  = DomainRouterDataset(str(TEST_PARQUET),  str(TEST_EMB))

    def collate(batch):
        e, l, d = zip(*batch)
        return torch.stack(e), torch.stack(l), torch.stack(d)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0, collate_fn=collate)

    emb_dim = train_ds.embeddings.shape[1]
    print(f"  Embedding dim: {emb_dim}")

    # ── 2. Build Domain MF model ──────────────────────────────────────────────
    print("\n► Building DomainMFRouter...")
    domain_model = DomainMFRouter(
        query_embedding_dim=emb_dim,
        model_embedding_dim=CONFIG["model_embedding_dim"],
        num_models=2,
        num_domains=2,
    ).to(device)

    total_params = sum(p.numel() for p in domain_model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Class-weighted loss
    n_pos = (train_ds.df["binary_label"] == 1).sum()
    n_neg = (train_ds.df["binary_label"] == 0).sum()
    pos_w = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    print(f"  Class weight (pos): {pos_w.item():.4f}")

    optimizer = optim.Adam(domain_model.parameters(),
                           lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # ── 3. Training loop ──────────────────────────────────────────────────────
    print(f"\n► Training for up to {CONFIG['num_epochs']} epochs...")
    DOMAIN_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss   = float("inf")
    patience_count  = 0
    history         = []
    start_epoch     = 0
    
    # --- RESUME CAPABILITY ---
    ckpt_path = DOMAIN_CKPT_DIR / "best_model.pt"
    if ckpt_path.exists():
        print(f"► Found existing checkpoint at {ckpt_path}. Resuming training...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        domain_model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1  # Start from next epoch
        best_val_loss = ckpt.get("val_loss", float("inf"))
        
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            
        print(f"  Resumed from epoch {start_epoch-1} with best val loss: {best_val_loss:.4f}")
        
        if start_epoch >= CONFIG["num_epochs"]:
            print(f"  Target epochs ({CONFIG['num_epochs']}) already reached. Skipping training.")

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        tr_loss = train_epoch(domain_model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_f1, vl_auc = validate(domain_model, val_loader, criterion, device)
        scheduler.step(vl_loss)
        lr = optimizer.param_groups[0]["lr"]

        history.append({"epoch": epoch+1, "train_loss": tr_loss,
                         "val_loss": vl_loss, "val_acc": vl_acc, "val_auc": vl_auc})

        marker = ""
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": domain_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": vl_loss,
                "val_acc":  vl_acc,
                "config":   CONFIG,
            }, ckpt_path)
            marker = "  ✓ saved"

        else:
            patience_count += 1

        print(f"  Epoch {epoch+1:>2} | "
              f"train={tr_loss:.4f}  val={vl_loss:.4f}  "
              f"acc={vl_acc:.4f}  auc={vl_auc:.4f}  lr={lr:.2e}{marker}")

        if patience_count >= CONFIG["early_stopping"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ── 4. Load best domain checkpoint ───────────────────────────────────────
    print("\n► Loading best Domain MF checkpoint...")
    ckpt = torch.load(DOMAIN_CKPT_DIR / "best_model.pt", map_location=device, weights_only=False)
    domain_model.load_state_dict(ckpt["model_state_dict"])
    domain_model.eval()
    print(f"  Best val loss: {ckpt['val_loss']:.4f}  acc: {ckpt['val_acc']:.4f}")

    # ── 5. Load baseline model ────────────────────────────────────────────────
    print(f"\n► Loading Baseline MF checkpoint from {BASELINE_CKPT}...")
    bl_ckpt  = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    bl_cfg   = bl_ckpt["config"]
    baseline_query_dim = bl_ckpt["model_state_dict"]["text_proj.0.weight"].shape[1]
    baseline = MatrixFactorizationRouter(
        query_embedding_dim=baseline_query_dim,
        model_embedding_dim=bl_cfg["model_embedding_dim"],
        num_models=2,
    ).to(device)
    baseline.load_state_dict(bl_ckpt["model_state_dict"])
    baseline.eval()

    # We also need a standard RouterDataset for the baseline
    test_standard = RouterDataset(
        data_path=str(TEST_PARQUET),
        precomputed_embeddings_path=str(TEST_EMB),
    )

    # ── 6. Get predictions on val + test sets ────────────────────────────────
    print("\n► Running inference on validation set (for threshold tuning)...")
    domain_val_probs, domain_val_labels = get_domain_predictions(domain_model, val_ds, device)
    base_val_probs,   base_val_labels   = get_baseline_predictions(
        baseline,
        RouterDataset(str(VAL_PARQUET), precomputed_embeddings_path=str(VAL_EMB)),
        device,
    )

    print("► Tuning thresholds on validation set...")
    domain_best_t, domain_best_acc = tune_threshold(domain_val_probs, domain_val_labels)
    base_best_t,   base_best_acc   = tune_threshold(base_val_probs,   base_val_labels)
    print(f"  Baseline  — best val threshold: {base_best_t:.3f}  (val acc: {base_best_acc*100:.2f}%)")
    print(f"  Domain MF — best val threshold: {domain_best_t:.3f}  (val acc: {domain_best_acc*100:.2f}%)")

    print("\n► Running inference on test set...")
    domain_probs, domain_labels = get_domain_predictions(domain_model, test_ds, device)
    base_probs,   base_labels   = get_baseline_predictions(baseline, test_standard, device)

    # Both datasets share the same labels
    labels = domain_labels   # same test set

    baseline_quality = labels.mean()   # always-strong upper bound

    # ── 7. Threshold sweep ────────────────────────────────────────────────────
    domain_sweep = threshold_sweep(domain_probs, labels)
    base_sweep   = threshold_sweep(base_probs,   labels)

    # ── 8. Print comparison table ─────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  COMPARISON: Baseline MF  vs  Domain-Specific MF")
    print(f"  Test set baseline (always GPT-4): {baseline_quality*100:.2f}%  quality")
    print(f"{'='*75}")
    print(f"\n  {'Drop':>6} | {'Baseline':^26} | {'Domain MF':^26} | {'Δ Cost Adv':>10}")
    print(f"  {'':6} | {'Quality':>10}  {'Cost Adv':>10} | {'Quality':>10}  {'Cost Adv':>10} |")
    print(f"  {'-'*6}-+-{'-'*26}-+-{'-'*26}-+-{'-'*10}")

    comparison_rows = []
    for drop in QUALITY_DROPS:
        b_cost, b_qual, b_t   = cost_at_drop(base_sweep,   labels, drop)
        d_cost, d_qual, d_t   = cost_at_drop(domain_sweep, labels, drop)

        b_cadv = (1 - b_cost) * 100 if b_cost is not None else None
        d_cadv = (1 - d_cost) * 100 if d_cost is not None else None
        delta  = (d_cadv - b_cadv) if (d_cadv is not None and b_cadv is not None) else None

        b_q_str    = f"{b_qual*100:.2f}%" if b_qual  else "N/A"
        b_cadv_str = f"{b_cadv:.2f}%"     if b_cadv  else "N/A"
        d_q_str    = f"{d_qual*100:.2f}%" if d_qual  else "N/A"
        d_cadv_str = f"{d_cadv:.2f}%"     if d_cadv  else "N/A"
        delta_str  = (f"+{delta:.2f}%" if delta and delta >= 0 else f"{delta:.2f}%") if delta else "N/A"

        print(f"  {drop:>5.1f}% | {b_q_str:>10}  {b_cadv_str:>10} | "
              f"{d_q_str:>10}  {d_cadv_str:>10} | {delta_str:>10}")

        comparison_rows.append({
            "drop": drop,
            "baseline_quality": b_qual, "baseline_cost_adv": b_cadv,
            "domain_quality":   d_qual, "domain_cost_adv":   d_cadv,
            "delta_cost_adv":   delta,
        })

    print(f"  {'-'*6}-+-{'-'*26}-+-{'-'*26}-+-{'-'*10}")

    # ── 9. Per-domain breakdown (using tuned thresholds) ─────────────────────
    print(f"\n{'='*75}")
    print(f"  PER-DOMAIN BREAKDOWN on Test Set")
    print(f"  Baseline threshold (tuned): {base_best_t:.3f}")
    print(f"  Domain MF threshold (tuned): {domain_best_t:.3f}")
    print(f"{'='*75}")

    test_domain_ids = test_ds.domain_ids.numpy()
    math_mask    = test_domain_ids == 1
    general_mask = test_domain_ids == 0

    # Overall Metrics
    overall_b_preds = (base_probs >= base_best_t).astype(int)
    overall_d_preds = (domain_probs >= domain_best_t).astype(int)
    
    b_overall_acc = (overall_b_preds == labels).mean() * 100
    d_overall_acc = (overall_d_preds == labels).mean() * 100
    b_overall_f1  = f1_score(labels, overall_b_preds) * 100
    d_overall_f1  = f1_score(labels, overall_d_preds) * 100
    
    print(f"  OVERALL:")
    print(f"    Baseline Accuracy: {b_overall_acc:.2f}% | F1 Score: {b_overall_f1:.2f}%")
    print(f"    Domain MF Accuracy: {d_overall_acc:.2f}% | F1 Score: {d_overall_f1:.2f}%\n")

    for name, mask in [("MATH", math_mask), ("GENERAL", general_mask)]:
        n = mask.sum()
        if n == 0:
            continue

        b_preds = (base_probs[mask]   >= base_best_t).astype(int)
        d_preds = (domain_probs[mask] >= domain_best_t).astype(int)
        lbl_sub = labels[mask]
        always_strong_q = lbl_sub.mean()

        b_acc  = (b_preds == lbl_sub).mean() * 100
        d_acc  = (d_preds == lbl_sub).mean() * 100

        b_strong_usage = b_preds.mean() * 100
        d_strong_usage = d_preds.mean() * 100

        # Quality = fraction of queries answered by the better model
        b_quality = np.where(b_preds, lbl_sub, 1 - lbl_sub).mean() * 100
        d_quality = np.where(d_preds, lbl_sub, 1 - lbl_sub).mean() * 100

        print(f"\n  {name} queries ({n:,} samples, always-GPT4-quality={always_strong_q*100:.1f}%)")
        print(f"  {'Metric':<25} {'Baseline':>12} {'Domain MF':>12} {'Δ':>8}")
        print(f"  {'-'*60}")
        print(f"  {'Accuracy':<25} {b_acc:>11.2f}% {d_acc:>11.2f}% {d_acc-b_acc:>+7.2f}%")
        print(f"  {'Quality':<25} {b_quality:>11.2f}% {d_quality:>11.2f}% {d_quality-b_quality:>+7.2f}%")
        print(f"  {'GPT-4 usage':<25} {b_strong_usage:>11.2f}% {d_strong_usage:>11.2f}% {d_strong_usage-b_strong_usage:>+7.2f}%")

    # ── 10. Save results ──────────────────────────────────────────────────────
    results_path = DOMAIN_CKPT_DIR / "comparison_results.json"

    def to_native(obj):
        if isinstance(obj, (np.floating, float)):  return float(obj)
        if isinstance(obj, (np.integer, int)):      return int(obj)
        if isinstance(obj, dict):  return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [to_native(x) for x in obj]
        return obj

    results = {
        "baseline_always_strong_quality": float(baseline_quality),
        "comparison": to_native(comparison_rows),
        "training_history": history,
        "config": CONFIG,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {results_path}")
    print(f"\n{'='*75}")
    print("  DONE")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
