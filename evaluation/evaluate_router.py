"""
Comprehensive Router Evaluation Script

Evaluates the router model at:
1. Quality drop targets (matching research paper format)
2. Accuracy drop targets (cost advantage analysis)

Also includes functionality to view saved results.
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


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = MatrixFactorizationRouter(
        query_embedding_dim=config['query_embedding_dim'],
        model_embedding_dim=config['model_embedding_dim'],
        num_models=2,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def get_predictions(model, dataset, device):
    """Run inference and return probabilities and labels."""
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


def evaluate_at_thresholds(probabilities, labels):
    """Evaluate at many thresholds, return results list."""
    thresholds = np.linspace(0, 1, 1000)
    threshold_results = []
    
    for threshold in thresholds:
        route_to_strong = probabilities >= threshold
        cost = route_to_strong.mean()
        quality = np.where(route_to_strong, labels, 1 - labels).mean()
        predictions = route_to_strong.astype(int)
        accuracy = (predictions == labels).mean()
        
        threshold_results.append({
            'threshold': threshold,
            'cost': cost,
            'quality': quality,
            'accuracy': accuracy,
        })
    
    return threshold_results


def evaluate_quality_drops(threshold_results, labels, target_drops):
    """Evaluate at quality drop targets (research paper format)."""
    always_strong_quality = labels.mean()
    
    results = []
    for drop_pct in target_drops:
        target_quality = always_strong_quality - (drop_pct / 100.0)
        valid = [r for r in threshold_results if r['quality'] >= target_quality]
        
        if valid:
            if drop_pct == 0.0:
                best = max(valid, key=lambda x: x['cost'])
            else:
                best = min(valid, key=lambda x: x['cost'])
            
            actual_drop = (always_strong_quality - best['quality']) * 100
            cost_adv = (1.0 - best['cost']) * 100
            
            results.append({
                'target_drop': drop_pct,
                'actual_drop': actual_drop,
                'target_quality': target_quality,
                'actual_quality': best['quality'],
                'cost': best['cost'] * 100,
                'cost_advantage': cost_adv,
                'threshold': best['threshold'],
            })
        else:
            results.append({
                'target_drop': drop_pct,
                'actual_drop': None,
                'target_quality': target_quality,
                'actual_quality': None,
                'cost': None,
                'cost_advantage': None,
                'threshold': None,
            })
    
    return results, always_strong_quality


def evaluate_accuracy_drops(threshold_results, target_drops):
    """Evaluate at accuracy drop targets."""
    max_acc_result = max(threshold_results, key=lambda x: x['accuracy'])
    max_accuracy = max_acc_result['accuracy']
    
    results = []
    for drop_pct in target_drops:
        target_accuracy = max_accuracy - (drop_pct / 100.0)
        valid = [r for r in threshold_results if r['accuracy'] >= target_accuracy]
        
        if valid:
            if drop_pct == 0.0:
                best = max_acc_result
            else:
                best = min(valid, key=lambda x: x['cost'])
            
            actual_drop = (max_accuracy - best['accuracy']) * 100
            cost_advantage = (1.0 - best['cost']) * 100
            
            results.append({
                'target_drop': drop_pct,
                'target_accuracy': target_accuracy * 100,
                'actual_accuracy': best['accuracy'] * 100,
                'actual_drop': actual_drop,
                'strong_usage': best['cost'] * 100,
                'weak_usage': 100.0 - (best['cost'] * 100),
                'cost_advantage': cost_advantage,
                'quality': best['quality'] * 100,
                'threshold': best['threshold'],
            })
        else:
            results.append({
                'target_drop': drop_pct,
                'target_accuracy': target_accuracy * 100,
                'actual_accuracy': None,
                'actual_drop': None,
                'strong_usage': None,
                'weak_usage': None,
                'cost_advantage': None,
                'quality': None,
                'threshold': None,
            })
    
    return results, max_accuracy, max_acc_result


def print_quality_drop_results(val_metrics, test_metrics, target_drops):
    """Print results in research paper format."""
    print("\n" + "="*80)
    print("RESULTS: Quality Drop Analysis (Research Paper Format)")
    print("="*80)
    print(f"\nModel Pair: S: GPT-4, L: Llama-2-7b")
    print(f"\nBaseline (Always Strong):")
    print(f"  Validation Quality: {val_metrics['always_strong_quality']*100:.2f}%")
    print(f"  Test Quality: {test_metrics['always_strong_quality']*100:.2f}%")
    
    print(f"\n{'':12} {'Validation':28} {'Test':28}")
    print(f"{'Perf. Drop':12} {'Perf. Drop':<14} {'Cost Adv.':<14} {'Perf. Drop':<14} {'Cost Adv.':<14}")
    print("-" * 68)
    
    for i, drop in enumerate(target_drops):
        val_result = val_metrics['results'][i]
        test_result = test_metrics['results'][i]
        
        val_drop_str = f"{val_result['actual_drop']:.2f}%" if val_result['actual_drop'] is not None else "N/A"
        val_cost_str = f"{val_result['cost_advantage']:.2f}%" if val_result['cost_advantage'] is not None else "N/A"
        test_drop_str = f"{test_result['actual_drop']:.2f}%" if test_result['actual_drop'] is not None else "N/A"
        test_cost_str = f"{test_result['cost_advantage']:.2f}%" if test_result['cost_advantage'] is not None else "N/A"
        
        print(f"{drop:.1f}%{'':<8} {val_drop_str:<14} {val_cost_str:<14} {test_drop_str:<14} {test_cost_str:<14}")
    
    print("-" * 68)


def print_accuracy_drop_results(results, max_accuracy, max_acc_result):
    """Print accuracy drop analysis results."""
    print("\n" + "="*80)
    print("RESULTS: Accuracy Drop Analysis")
    print("="*80)
    print(f"\nMax Accuracy: {max_accuracy*100:.2f}% at threshold {max_acc_result['threshold']:.3f}")
    print(f"Strong usage at max accuracy: {max_acc_result['cost']*100:.2f}%")
    print(f"Quality at max accuracy: {max_acc_result['quality']*100:.2f}%\n")
    
    print(f"{'Target Drop':>12} {'Target Acc':>11} {'Actual Acc':>11} {'Strong Usage':>13} "
          f"{'Weak Usage':>11} {'Cost Adv':>10} {'Quality':>8} {'Threshold':>10}")
    print("-"*80)
    
    for r in results:
        if r['actual_accuracy'] is not None:
            print(f"{r['target_drop']:>12.1f} {r['target_accuracy']:>11.2f} {r['actual_accuracy']:>11.2f} "
                  f"{r['strong_usage']:>13.2f} {r['weak_usage']:>11.2f} "
                  f"{r['cost_advantage']:>10.2f} {r['quality']:>8.2f} {r['threshold']:>10.3f}")
        else:
            print(f"{r['target_drop']:>12.1f} {r['target_accuracy']:>11.2f} {'N/A':>11} "
                  f"{'N/A':>13} {'N/A':>11} {'N/A':>10} {'N/A':>8} {'N/A':>10}")
    
    print("="*80)


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj


def view_saved_results(results_path):
    """View previously saved evaluation results."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*70)
    print("Saved Evaluation Results")
    print("="*70)
    
    # Handle different result formats
    if 'test' in results and 'validation' in results:
        # Full evaluation format
        test = results['test']
        val = results['validation']
        
        print(f"\nModel: {results.get('model_pair', 'Unknown')}")
        print(f"\nAccuracy:")
        print(f"  Test @ 0.5: {test.get('accuracy_at_0_5', 0)*100:.2f}%")
        print(f"  Test Max: {test.get('max_accuracy', 0)*100:.2f}% (threshold={test.get('best_accuracy_threshold', 0):.3f})")
        print(f"  Val @ 0.5: {val.get('accuracy_at_0_5', 0)*100:.2f}%")
        print(f"  Val Max: {val.get('max_accuracy', 0)*100:.2f}%")
        
        print(f"\nBaseline Quality (Always Strong):")
        print(f"  Test: {test.get('always_strong_quality', 0)*100:.2f}%")
        print(f"  Val: {val.get('always_strong_quality', 0)*100:.2f}%")
        
        if 'results' in test:
            print(f"\nTest Set Results at Drop Targets:")
            print(f"{'Drop':>8} {'Quality':>10} {'Cost Adv':>12} {'Threshold':>12}")
            print("-"*45)
            for r in test['results']:
                if r.get('actual_quality') is not None:
                    print(f"{r['target_drop']:>8.1f} {r['actual_quality']*100:>10.2f} "
                          f"{r['cost_advantage']:>12.2f} {r['threshold']:>12.3f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive router evaluation - quality drops, accuracy drops, and result viewing"
    )
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--val_data', type=str, help='Path to validation data')
    parser.add_argument('--test_data', type=str, help='Path to test data')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--view', type=str, help='View saved results from JSON file')
    parser.add_argument('--mode', type=str, choices=['quality', 'accuracy', 'both'], 
                        default='both', help='Evaluation mode')
    
    args = parser.parse_args()
    
    # View mode - just show saved results
    if args.view:
        view_saved_results(args.view)
        return
    
    # Evaluation mode - need checkpoint and data
    if not args.checkpoint or not args.test_data:
        parser.error("--checkpoint and --test_data are required for evaluation")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device=str(device))
    
    # Target drops
    quality_drops = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    accuracy_drops = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    output_data = {
        'model_pair': 'S: GPT-4, L: Llama-2-7b',
        'config': config,
    }
    
    # Evaluate validation set if provided
    if args.val_data:
        print(f"\nLoading validation data from: {args.val_data}")
        val_dataset = RouterDataset(
            data_path=args.val_data,
            embedding_model_name=config['embedding_model_name'],
            cache_embeddings=True,
        )
        
        print("Evaluating validation set...")
        val_probs, val_labels = get_predictions(model, val_dataset, device)
        val_threshold_results = evaluate_at_thresholds(val_probs, val_labels)
        
        val_quality_results, val_baseline = evaluate_quality_drops(val_threshold_results, val_labels, quality_drops)
        val_accuracy_results, val_max_acc, val_max_acc_result = evaluate_accuracy_drops(val_threshold_results, accuracy_drops)
        
        predictions_05 = (val_probs >= 0.5).astype(int)
        accuracy_05 = (predictions_05 == val_labels).mean()
        
        output_data['validation'] = {
            'always_strong_quality': float(val_baseline),
            'accuracy_at_0_5': float(accuracy_05),
            'max_accuracy': float(val_max_acc),
            'best_accuracy_threshold': float(val_max_acc_result['threshold']),
            'quality_drop_results': val_quality_results,
            'accuracy_drop_results': val_accuracy_results,
        }
    
    # Evaluate test set
    print(f"\nLoading test data from: {args.test_data}")
    test_dataset = RouterDataset(
        data_path=args.test_data,
        embedding_model_name=config['embedding_model_name'],
        cache_embeddings=True,
    )
    
    print("Evaluating test set...")
    test_probs, test_labels = get_predictions(model, test_dataset, device)
    test_threshold_results = evaluate_at_thresholds(test_probs, test_labels)
    
    test_quality_results, test_baseline = evaluate_quality_drops(test_threshold_results, test_labels, quality_drops)
    test_accuracy_results, test_max_acc, test_max_acc_result = evaluate_accuracy_drops(test_threshold_results, accuracy_drops)
    
    predictions_05 = (test_probs >= 0.5).astype(int)
    accuracy_05 = (predictions_05 == test_labels).mean()
    
    output_data['test'] = {
        'always_strong_quality': float(test_baseline),
        'accuracy_at_0_5': float(accuracy_05),
        'max_accuracy': float(test_max_acc),
        'best_accuracy_threshold': float(test_max_acc_result['threshold']),
        'quality_drop_results': test_quality_results,
        'accuracy_drop_results': test_accuracy_results,
    }
    
    # Print results
    if args.mode in ['quality', 'both']:
        if args.val_data:
            val_metrics = {'always_strong_quality': val_baseline, 'results': val_quality_results}
            test_metrics = {'always_strong_quality': test_baseline, 'results': test_quality_results}
            print_quality_drop_results(val_metrics, test_metrics, quality_drops)
        else:
            print("\n" + "="*60)
            print("Quality Drop Results (Test Set)")
            print("="*60)
            print(f"\nBaseline: {test_baseline*100:.2f}%")
            for r in test_quality_results:
                if r['actual_quality'] is not None:
                    print(f"  {r['target_drop']:.1f}% drop: quality={r['actual_quality']*100:.2f}%, "
                          f"cost_adv={r['cost_advantage']:.2f}%, thresh={r['threshold']:.3f}")
    
    if args.mode in ['accuracy', 'both']:
        print_accuracy_drop_results(test_accuracy_results, test_max_acc, test_max_acc_result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test Accuracy @ 0.5: {accuracy_05*100:.2f}%")
    print(f"Test Max Accuracy: {test_max_acc*100:.2f}% (threshold={test_max_acc_result['threshold']:.3f})")
    print(f"Baseline Quality: {test_baseline*100:.2f}%")
    
    # Save results
    if args.output:
        output_data = convert_to_native(output_data)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to: {args.output}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
