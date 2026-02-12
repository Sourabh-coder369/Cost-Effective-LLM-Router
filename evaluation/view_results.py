"""
View Router Evaluation Results
"""
import json
import os

def view_results():
    # Handle both running from root or evaluation folder
    if os.path.exists('gpt4_llama7b_checkpoints/complete_evaluation.json'):
        path = 'gpt4_llama7b_checkpoints/complete_evaluation.json'
    else:
        path = '../gpt4_llama7b_checkpoints/complete_evaluation.json'
    
    with open(path, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*70)
    print("Evaluation Results on Unbalanced Test Set")
    print(f"({results['data']['test_distribution']['gpt4_win_rate']}% GPT-4 wins, "
          f"{results['data']['test_distribution']['llama_win_rate']}% Llama wins)")
    print("="*70)
    
    router_accuracy = results['summary']['model_accuracy'] * 100
    baseline = results['evaluation']['baseline_quality'] * 100
    print(f"\nRouter Accuracy: {router_accuracy:.2f}%")
    print(f"Baseline: Always using GPT-4 = {baseline:.2f}% quality\n")
    
    print("Performance at Different Drop Targets:\n")
    print(f"{'Target Drop':>13} {'Actual Drop':>13} {'Quality':>9} {'Strong Usage':>13} "
          f"{'Weak Usage':>11} {'Cost Advantage':>15} {'Threshold':>10}")
    print("-" * 105)
    
    for r in results['evaluation']['results_at_drop_targets']:
        print(f"{r['target_drop']:>13.1f} "
              f"{r['actual_drop']:>13.2f} "
              f"{r['actual_quality']*100:>9.2f} "
              f"{r['cost']:>13.2f} "
              f"{100-r['cost']:>11.2f} "
              f"{r['cost_advantage']:>15.2f} "
              f"{r['threshold']:>10.3f}")
    
    print("\n" + "="*70)
    print("Key Metrics at 1% Performance Drop:")
    print("="*70)
    
    summary = results['summary']['at_1pct_drop']
    print(f"\nRouter Quality: {summary['quality']*100:.2f}% (vs {baseline:.2f}% baseline)")
    print(f"Routes {summary['weak_usage_pct']:.1f}% of queries to Llama-2-7b "
          f"(saving {summary['cost_advantage_pct']:.1f}% of GPT-4 calls)")
    print(f"Routes {summary['strong_usage_pct']:.1f}% of queries to GPT-4")
    print(f"Threshold: {summary['threshold']}")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    view_results()
