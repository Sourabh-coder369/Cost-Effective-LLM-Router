"""
RouteLLM Paper-Style Evaluation
================================
Implements all key evaluation metrics from the RouteLLM paper (ICLR 2025), Section 3.2:

    1. Cost Efficiency  c(MRα)  — Eq. 4: % of calls routed to strong model
    2. Response Quality  r(MRα) — Eq. 5: average response quality under routing
    3. PGR (Performance Gap Recovered)     — Eq. 6
    4. APGR (Average PGR)                  — Eq. 8
    5. CPT (Call-Performance Threshold)     — Section 3.2
    6. Random Baseline with 95% CI         — Tables 1-3
    7. Cost Savings Ratio                  — Section 5.4
    8. Performance-Cost Plot               — Figure 1

Usage:
    python evaluation/routellm_evaluation.py \
        --checkpoint router/checkpoints/best.pt \
        --test_data router/data/test.parquet \
        --output evaluation/results.json
"""

import torch
import numpy as np
import json
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path to import router module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from router.dataset import RouterDataset
from router.model import MatrixFactorizationRouter


# =============================================================================
# Core Metric Functions (RouteLLM Paper Section 3.2)
# =============================================================================

def compute_cost_efficiency(probabilities: np.ndarray, threshold: float) -> float:
    """
    Cost Efficiency — Eq. 4
    
    c(MRα) = (1/|Q|) * Σ I{Rα(q) = M_strong}
    
    Fraction of queries routed to the strong model.
    
    Args:
        probabilities: P(wins | q) for each query, shape [N]
        threshold: Routing threshold α
        
    Returns:
        Fraction of queries sent to strong model (0.0 to 1.0)
    """
    route_to_strong = probabilities >= threshold
    return float(route_to_strong.mean())


def compute_response_quality(probabilities: np.ndarray, labels: np.ndarray,
                              threshold: float) -> float:
    """
    Response Quality — Eq. 5

    r(MRα) = (1/|Q|) * Σ s(MRα(q))

    Quality is measured as:
      - If routed to strong AND strong wins → correct (1)
      - If routed to weak AND weak wins    → correct (1)
      - Otherwise                          → incorrect (0)

    Args:
        probabilities: P(wins | q) for each query, shape [N]
        labels: Ground truth, 1 = strong wins, 0 = weak wins/tie, shape [N]
        threshold: Routing threshold α

    Returns:
        Average response quality (0.0 to 1.0)
    """
    route_to_strong = probabilities >= threshold
    # Quality = 1 when routing matches ground truth
    #   route_to_strong=True  and label=1 → strong wins, routed correctly
    #   route_to_strong=False and label=0 → weak wins, routed correctly
    quality = np.where(route_to_strong, labels, 1 - labels)
    return float(quality.mean())


def compute_pgr(router_quality: float, weak_quality: float,
                strong_quality: float) -> float:
    """
    Performance Gap Recovered (PGR) — Eq. 6

    PGR(MRα) = (r(MRα) - r(Mw)) / (r(Ms) - r(Mw))

    How much of the gap between the weak and strong model the router recovers.
    PGR = 0 means router performs like the weak model.
    PGR = 1 means router performs like the strong model.

    Args:
        router_quality: Quality of the router r(MRα)
        weak_quality:   Quality when always using weak model r(Mw)
        strong_quality: Quality when always using strong model r(Ms)

    Returns:
        PGR value (can be < 0 or > 1 in edge cases)
    """
    gap = strong_quality - weak_quality
    if abs(gap) < 1e-10:
        return 1.0  # models are equivalent
    return (router_quality - weak_quality) / gap


def compute_apgr(probabilities: np.ndarray, labels: np.ndarray,
                 weak_quality: float, strong_quality: float,
                 num_bins: int = 10) -> float:
    """
    Average Performance Gap Recovered (APGR) — Eq. 8

    APGR ≈ (1/N) * Σ PGR(MRαi)    for i = 1..N cost bins

    Discretizes the cost axis into `num_bins` equal bins and computes
    the average PGR across all cost levels.

    Args:
        probabilities: P(wins | q) for each query, shape [N]
        labels: Ground truth labels, shape [N]
        weak_quality:   Quality when always using weak model
        strong_quality: Quality when always using strong model
        num_bins: Number of cost bins (default 10, as in paper)

    Returns:
        APGR value (0 to 1 for a good router; 0.5 = random)
    """
    # Target cost levels: 10%, 20%, ..., 100% strong calls
    cost_levels = np.linspace(1 / num_bins, 1.0, num_bins)

    # Sort probabilities descending — highest P(wins) first
    sorted_probs = np.sort(probabilities)[::-1]
    n = len(sorted_probs)

    pgr_values = []
    for target_cost in cost_levels:
        # Find threshold such that ~target_cost fraction goes to strong
        # k = number of queries to route to strong
        k = int(np.round(target_cost * n))
        k = max(1, min(k, n))

        if k < n:
            threshold = sorted_probs[k - 1]
        else:
            threshold = sorted_probs[-1] - 1e-10  # route everything to strong

        quality = compute_response_quality(probabilities, labels, threshold)
        pgr = compute_pgr(quality, weak_quality, strong_quality)
        pgr_values.append(pgr)

    return float(np.mean(pgr_values))


def compute_cpt(probabilities: np.ndarray, labels: np.ndarray,
                weak_quality: float, strong_quality: float,
                target_pgr: float) -> float:
    """
    Call-Performance Threshold (CPT) — Section 3.2

    CPT(x%) = minimum % of strong model calls needed to achieve PGR ≥ x%.

    Uses fine-grained threshold sweep to find the minimum cost (strong call %)
    that achieves the desired PGR.

    Args:
        probabilities: P(wins | q) for each query, shape [N]
        labels: Ground truth labels, shape [N]
        weak_quality:   Quality when always using weak model
        strong_quality: Quality when always using strong model
        target_pgr: Desired PGR as a fraction (e.g. 0.5 for 50%)

    Returns:
        Minimum % of strong calls needed (0 to 100), or None if unreachable
    """
    thresholds = np.linspace(0, 1, 1000)

    best_cost = None
    for t in thresholds:
        cost = compute_cost_efficiency(probabilities, t)
        quality = compute_response_quality(probabilities, labels, t)
        pgr = compute_pgr(quality, weak_quality, strong_quality)

        if pgr >= target_pgr:
            cost_pct = cost * 100
            if best_cost is None or cost_pct < best_cost:
                best_cost = cost_pct

    return best_cost


# =============================================================================
# Evaluation Helper Functions
# =============================================================================

def evaluate_at_cost_levels(probabilities: np.ndarray, labels: np.ndarray,
                            weak_quality: float, strong_quality: float,
                            num_levels: int = 10) -> list:
    """
    Evaluate router performance at evenly-spaced cost levels.

    For each cost level (10%, 20%, ..., 100% strong calls), finds the
    threshold that approximately satisfies the cost constraint and
    computes quality, PGR, and actual cost.

    Args:
        probabilities: P(wins | q) for each query
        labels: Ground truth labels
        weak_quality: Weak model quality
        strong_quality: Strong model quality
        num_levels: Number of cost levels

    Returns:
        List of dicts with keys: target_cost, actual_cost, quality, pgr, threshold
    """
    sorted_probs = np.sort(probabilities)[::-1]
    n = len(sorted_probs)
    results = []

    for i in range(1, num_levels + 1):
        target_cost = i / num_levels
        k = int(np.round(target_cost * n))
        k = max(1, min(k, n))

        if k < n:
            threshold = sorted_probs[k - 1]
        else:
            threshold = sorted_probs[-1] - 1e-10

        actual_cost = compute_cost_efficiency(probabilities, threshold)
        quality = compute_response_quality(probabilities, labels, threshold)
        pgr = compute_pgr(quality, weak_quality, strong_quality)

        results.append({
            'target_cost': target_cost * 100,
            'actual_cost': actual_cost * 100,
            'quality': quality,
            'pgr': pgr,
            'threshold': float(threshold),
        })

    return results


def compute_cost_savings_ratio(cpt_value: float,
                                strong_cost_per_mtok: float = 24.7,
                                weak_cost_per_mtok: float = 0.24) -> float:
    """
    Cost Savings Ratio — Section 5.4

    Ratio of always-strong cost to router cost, approximated as the
    inverse of the fraction of calls to the strong model, since the
    strong model dominates cost.

    Example: If CPT(50%) = 25%, savings ratio = 100/25 = 4x.

    Args:
        cpt_value: CPT percentage (e.g., 25.0 for 25%)
        strong_cost_per_mtok: Cost of strong model per million tokens
        weak_cost_per_mtok: Cost of weak model per million tokens

    Returns:
        Cost savings multiplier (e.g. 2.0 means 2x cheaper)
    """
    if cpt_value is None or cpt_value <= 0:
        return float('inf')

    strong_frac = cpt_value / 100.0
    weak_frac = 1.0 - strong_frac

    always_strong_cost = strong_cost_per_mtok
    router_cost = strong_frac * strong_cost_per_mtok + weak_frac * weak_cost_per_mtok

    return always_strong_cost / router_cost


def compute_random_baseline(labels: np.ndarray,
                            weak_quality: float, strong_quality: float,
                            num_iters: int = 100,
                            num_levels: int = 10) -> dict:
    """
    Random Router Baseline with 95% CI — Tables 1-3

    Simulates a random router that routes queries randomly under a cost
    constraint (i.e., at each cost level, exactly that fraction of queries
    go to strong). Repeats over multiple iterations to compute confidence
    intervals.

    Args:
        labels: Ground truth labels
        weak_quality: Weak model quality
        strong_quality: Strong model quality
        num_iters: Number of random iterations
        num_levels: Number of cost levels

    Returns:
        Dict with cpt_50, cpt_80 (mean ± CI), apgr (mean ± CI)
    """
    n = len(labels)
    apgr_values = []
    cpt50_values = []
    cpt80_values = []

    for _ in range(num_iters):
        random_probs = np.random.rand(n)

        apgr = compute_apgr(random_probs, labels, weak_quality, strong_quality, num_levels)
        apgr_values.append(apgr)

        cpt50 = compute_cpt(random_probs, labels, weak_quality, strong_quality, 0.5)
        cpt80 = compute_cpt(random_probs, labels, weak_quality, strong_quality, 0.8)
        if cpt50 is not None:
            cpt50_values.append(cpt50)
        if cpt80 is not None:
            cpt80_values.append(cpt80)

    def stats(values):
        arr = np.array(values)
        mean = arr.mean()
        ci = 1.96 * arr.std() / np.sqrt(len(arr))
        return mean, ci

    result = {}
    result['apgr_mean'], result['apgr_ci'] = stats(apgr_values)

    if cpt50_values:
        result['cpt50_mean'], result['cpt50_ci'] = stats(cpt50_values)
    else:
        result['cpt50_mean'], result['cpt50_ci'] = None, None

    if cpt80_values:
        result['cpt80_mean'], result['cpt80_ci'] = stats(cpt80_values)
    else:
        result['cpt80_mean'], result['cpt80_ci'] = None, None

    return result


# =============================================================================
# Visualization
# =============================================================================

def generate_performance_cost_plot(probabilities: np.ndarray, labels: np.ndarray,
                                    weak_quality: float, strong_quality: float,
                                    output_path: str = None,
                                    num_points: int = 50) -> None:
    """
    Performance-Cost Plot — Figure 1 (left) in the RouteLLM paper.

    X-axis: Strong Model Calls (%)
    Y-axis: Response Quality (or PGR)

    Includes horizontal lines for weak and strong model baselines,
    and the random router diagonal.

    Args:
        probabilities: P(wins | q) for each query
        labels: Ground truth labels
        weak_quality: Weak model quality
        strong_quality: Strong model quality
        output_path: File path to save the plot (if None, shows interactively)
        num_points: Number of threshold points to sample
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        return

    thresholds = np.linspace(0, 1, num_points)
    costs = []
    qualities = []

    for t in thresholds:
        c = compute_cost_efficiency(probabilities, t) * 100
        q = compute_response_quality(probabilities, labels, t)
        costs.append(c)
        qualities.append(q)

    # Random baseline: quality interpolates linearly from weak to strong
    random_costs = np.linspace(0, 100, 50)
    random_qualities = [weak_quality + (c / 100) * (strong_quality - weak_quality)
                        for c in random_costs]

    plt.figure(figsize=(8, 6))
    plt.plot(costs, qualities, 'b-o', markersize=3, label='Router', linewidth=2)
    plt.plot(random_costs, random_qualities, 'k--', alpha=0.5, label='Random', linewidth=1.5)
    plt.axhline(y=weak_quality, color='grey', linestyle=':', label=f'Weak Model ({weak_quality:.3f})')
    plt.axhline(y=strong_quality, color='red', linestyle=':', label=f'Strong Model ({strong_quality:.3f})')

    plt.xlabel('Strong Model Calls (%)', fontsize=12)
    plt.ylabel('Response Quality', fontsize=12)
    plt.title('Router Performance / Cost Trade-off', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# Printing / Reporting
# =============================================================================

def print_results_table(results: dict) -> None:
    """
    Print results in the RouteLLM paper's table format (Tables 1-3).

    Columns: Method | CPT(50%) | CPT(80%) | APGR | Improvement
    """
    print("\n" + "=" * 75)
    print("  RouteLLM Paper-Style Results")
    print("=" * 75)
    print(f"\n  Model Pair: {results.get('model_pair', 'S: GPT-4, L: Llama-2-7b')}")
    print(f"  Strong Model Quality: {results['strong_quality']*100:.2f}%")
    print(f"  Weak Model Quality:   {results['weak_quality']*100:.2f}%")
    print(f"  Performance Gap:      {(results['strong_quality'] - results['weak_quality'])*100:.2f}%")

    # Random baseline
    rb = results.get('random_baseline', {})
    if rb:
        print(f"\n  {'Method':<25} {'CPT(50%)':<15} {'CPT(80%)':<15} {'APGR':<18} {'Improvement':<12}")
        print("  " + "-" * 73)

        cpt50_str = f"{rb['cpt50_mean']:.2f}(±{rb['cpt50_ci']:.0f})%" if rb.get('cpt50_mean') else "N/A"
        cpt80_str = f"{rb['cpt80_mean']:.2f}(±{rb['cpt80_ci']:.0f})%" if rb.get('cpt80_mean') else "N/A"
        apgr_str = f"{rb['apgr_mean']:.3f}(±{rb['apgr_ci']:.2f})"
        print(f"  {'Random (95% CI)':<25} {cpt50_str:<15} {cpt80_str:<15} {apgr_str:<18} {'(+0%)':<12}")

    # Router results
    r = results['router']
    cpt50_str = f"{r['cpt_50']:.2f}%" if r.get('cpt_50') is not None else "N/A"
    cpt80_str = f"{r['cpt_80']:.2f}%" if r.get('cpt_80') is not None else "N/A"
    apgr_str = f"{r['apgr']:.3f}"

    improvement = ""
    if rb and rb.get('apgr_mean'):
        imp_pct = ((r['apgr'] - rb['apgr_mean']) / rb['apgr_mean']) * 100
        improvement = f"({'+' if imp_pct >= 0 else ''}{imp_pct:.1f}%)"

    print(f"  {'Matrix Factorization':<25} {cpt50_str:<15} {cpt80_str:<15} {apgr_str:<18} {improvement:<12}")
    print("  " + "-" * 73)

    # Cost Savings
    print(f"\n  Cost Savings (Section 5.4):")
    if r.get('cpt_50') is not None:
        savings_50 = compute_cost_savings_ratio(r['cpt_50'])
        print(f"    At CPT(50%): {savings_50:.2f}x cheaper than always-strong")
    if r.get('cpt_80') is not None:
        savings_80 = compute_cost_savings_ratio(r['cpt_80'])
        print(f"    At CPT(80%): {savings_80:.2f}x cheaper than always-strong")

    print()


def print_cost_level_table(cost_level_results: list) -> None:
    """
    Print quality and PGR at each cost level (performance-cost breakdown).
    """
    print("\n" + "=" * 70)
    print("  Performance at Each Cost Level")
    print("=" * 70)
    print(f"  {'Strong Calls %':<16} {'Actual Cost %':<16} {'Quality':<12} {'PGR':<12} {'Threshold':<12}")
    print("  " + "-" * 66)

    for r in cost_level_results:
        print(f"  {r['target_cost']:<16.0f} {r['actual_cost']:<16.2f} "
              f"{r['quality']*100:<12.2f} {r['pgr']:<12.4f} {r['threshold']:<12.4f}")

    print("  " + "-" * 66)


def print_cpt_summary(results: dict) -> None:
    """
    Print CPT values at multiple PGR targets.
    """
    print("\n" + "=" * 55)
    print("  CPT at Various PGR Targets")
    print("=" * 55)
    print(f"  {'PGR Target':<15} {'CPT (Strong %)':<20} {'Cost Savings':<15}")
    print("  " + "-" * 50)

    for target, value in results['router']['cpt_all'].items():
        if value is not None:
            savings = compute_cost_savings_ratio(value)
            print(f"  {target:<15} {value:<20.2f}% {savings:<15.2f}x")
        else:
            print(f"  {target:<15} {'N/A':<20} {'N/A':<15}")

    print("  " + "-" * 50)


# =============================================================================
# Model Loading & Inference
# =============================================================================

def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained router model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']

    # Infer actual dimensions from saved weights to handle config mismatches
    # text_proj.0.weight shape is [query_dim, query_dim] (first Linear layer)
    if 'text_proj.0.weight' in state_dict:
        actual_query_dim = state_dict['text_proj.0.weight'].shape[1]
    else:
        actual_query_dim = config['query_embedding_dim']

    if 'model_embeddings.weight' in state_dict:
        actual_model_dim = state_dict['model_embeddings.weight'].shape[1]
    else:
        actual_model_dim = config['model_embedding_dim']

    if actual_query_dim != config['query_embedding_dim']:
        print(f"  Note: Config says query_dim={config['query_embedding_dim']}, "
              f"but checkpoint weights are {actual_query_dim}-dim. Using {actual_query_dim}.")
        config['query_embedding_dim'] = actual_query_dim

    if actual_model_dim != config['model_embedding_dim']:
        print(f"  Note: Config says model_dim={config['model_embedding_dim']}, "
              f"but checkpoint weights are {actual_model_dim}-dim. Using {actual_model_dim}.")
        config['model_embedding_dim'] = actual_model_dim

    model = MatrixFactorizationRouter(
        query_embedding_dim=config['query_embedding_dim'],
        model_embedding_dim=config['model_embedding_dim'],
        num_models=2,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def get_predictions(model, dataset, device: str) -> tuple:
    """Run inference on the dataset, return (probabilities, labels)."""
    embeddings = []
    labels = []

    for i in range(len(dataset)):
        emb, label = dataset[i]
        embeddings.append(emb)
        labels.append(label)

    embeddings = torch.stack(embeddings).to(device)
    labels = np.array(labels)

    with torch.no_grad():
        probabilities = model(embeddings).cpu().numpy().squeeze()

    return probabilities, labels


# =============================================================================
# Full Evaluation Orchestrator
# =============================================================================

def run_full_evaluation(model, dataset, device: str,
                        model_pair: str = 'S: GPT-4, L: Llama-2-7b',
                        random_iters: int = 50) -> dict:
    """
    Run the complete RouteLLM paper-style evaluation.

    Computes all metrics: PGR, APGR, CPT(20/50/80%), cost levels,
    random baseline, and cost savings.

    Args:
        model: Trained MatrixFactorizationRouter
        dataset: RouterDataset instance
        device: 'cpu' or 'cuda'
        model_pair: Description string for the model pair
        random_iters: Iterations for random baseline

    Returns:
        Complete results dictionary
    """
    print("Running inference...")
    probabilities, labels = get_predictions(model, dataset, device)

    # Baseline qualities
    # strong_quality = r(Ms) = expected quality when always routing to strong
    # Since label=1 means strong wins, r(Ms) = mean(labels)
    strong_quality = float(labels.mean())
    # weak_quality = r(Mw) = expected quality when always routing to weak
    # Since label=0 means weak wins, r(Mw) = mean(1 - labels)
    weak_quality = float((1 - labels).mean())

    print(f"  Strong quality (always strong): {strong_quality*100:.2f}%")
    print(f"  Weak quality (always weak):     {weak_quality*100:.2f}%")
    print(f"  Performance gap:                {(strong_quality - weak_quality)*100:.2f}%")

    # --- APGR ---
    print("Computing APGR...")
    apgr = compute_apgr(probabilities, labels, weak_quality, strong_quality)

    # --- CPT at multiple targets ---
    print("Computing CPT values...")
    cpt_targets = {'PGR=20%': 0.2, 'PGR=50%': 0.5, 'PGR=80%': 0.8, 'PGR=90%': 0.9}
    cpt_all = {}
    for name, target in cpt_targets.items():
        cpt_val = compute_cpt(probabilities, labels, weak_quality, strong_quality, target)
        cpt_all[name] = cpt_val

    # --- Cost-level breakdown ---
    print("Evaluating at cost levels...")
    cost_level_results = evaluate_at_cost_levels(
        probabilities, labels, weak_quality, strong_quality
    )

    # --- Random baseline ---
    print(f"Computing random baseline ({random_iters} iterations)...")
    random_baseline = compute_random_baseline(
        labels, weak_quality, strong_quality, num_iters=random_iters
    )

    # --- Accuracy at default threshold (0.5) ---
    predictions_05 = (probabilities >= 0.5).astype(int)
    accuracy_05 = float((predictions_05 == labels).mean())

    # --- Assemble results ---
    results = {
        'model_pair': model_pair,
        'strong_quality': strong_quality,
        'weak_quality': weak_quality,
        'performance_gap': strong_quality - weak_quality,
        'num_samples': len(labels),
        'label_distribution': {
            'strong_wins': int(labels.sum()),
            'weak_wins': int((1 - labels).sum()),
            'strong_win_rate': float(labels.mean()) * 100,
        },
        'router': {
            'apgr': apgr,
            'cpt_50': cpt_all.get('PGR=50%'),
            'cpt_80': cpt_all.get('PGR=80%'),
            'cpt_all': cpt_all,
            'accuracy_at_0_5': accuracy_05,
            'cost_level_results': cost_level_results,
        },
        'random_baseline': random_baseline,
    }

    return results


# =============================================================================
# JSON Serialization Helper
# =============================================================================

def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RouteLLM Paper-Style Evaluation — PGR, APGR, CPT metrics"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data (.parquet)')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Optional path to validation data (.parquet)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path for saving results')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output path for performance-cost plot (.png)')
    parser.add_argument('--random_iters', type=int, default=50,
                        help='Number of random baseline iterations (default: 50)')
    parser.add_argument('--model_pair', type=str, default='S: GPT-4, L: Llama-2-7b',
                        help='Model pair description')
    parser.add_argument('--test_embeddings', type=str, default=None,
                        help='Path to precomputed test embeddings (.pt)')
    parser.add_argument('--val_embeddings', type=str, default=None,
                        help='Path to precomputed validation embeddings (.pt)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device=str(device))

    # ---- Evaluate Test Set ----
    print(f"\nLoading test data from: {args.test_data}")
    test_dataset = RouterDataset(
        data_path=args.test_data,
        embedding_model_name=config['embedding_model_name'],
        cache_embeddings=True,
        precomputed_embeddings_path=args.test_embeddings,
    )

    print("\n" + "=" * 75)
    print("  TEST SET EVALUATION")
    print("=" * 75)
    test_results = run_full_evaluation(
        model, test_dataset, device,
        model_pair=args.model_pair,
        random_iters=args.random_iters,
    )

    # Print formatted results
    print_results_table(test_results)
    print_cpt_summary(test_results)
    print_cost_level_table(test_results['router']['cost_level_results'])

    # ---- Evaluate Validation Set (optional) ----
    val_results = None
    if args.val_data:
        print(f"\nLoading validation data from: {args.val_data}")
        val_dataset = RouterDataset(
            data_path=args.val_data,
            embedding_model_name=config['embedding_model_name'],
            cache_embeddings=True,
            precomputed_embeddings_path=args.val_embeddings,
        )

        print("\n" + "=" * 75)
        print("  VALIDATION SET EVALUATION")
        print("=" * 75)
        val_results = run_full_evaluation(
            model, val_dataset, device,
            model_pair=args.model_pair,
            random_iters=args.random_iters,
        )
        print_results_table(val_results)
        print_cpt_summary(val_results)
        print_cost_level_table(val_results['router']['cost_level_results'])

    # ---- Generate Plot ----
    if args.plot:
        print(f"\nGenerating performance-cost plot...")
        test_probs, test_labels = get_predictions(model, test_dataset, device)
        generate_performance_cost_plot(
            test_probs, test_labels,
            test_results['weak_quality'], test_results['strong_quality'],
            output_path=args.plot,
        )

    # ---- Save results ----
    if args.output:
        output_data = {
            'test': convert_to_native(test_results),
        }
        if val_results:
            output_data['validation'] = convert_to_native(val_results)

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # ---- Final Summary ----
    print("\n" + "=" * 75)
    print("  EVALUATION COMPLETE")
    print("=" * 75)
    r = test_results['router']
    print(f"  APGR:          {r['apgr']:.4f}")
    print(f"  CPT(50%):      {r['cpt_50']:.2f}%" if r['cpt_50'] else "  CPT(50%):      N/A")
    print(f"  CPT(80%):      {r['cpt_80']:.2f}%" if r['cpt_80'] else "  CPT(80%):      N/A")
    print(f"  Accuracy@0.5:  {r['accuracy_at_0_5']*100:.2f}%")

    if r.get('cpt_50') is not None:
        savings = compute_cost_savings_ratio(r['cpt_50'])
        print(f"  Cost Savings:  {savings:.2f}x at CPT(50%)")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
