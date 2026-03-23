"""
Threshold Tuning Script for Router
Finds the optimal threshold for routing decisions based on F1 score.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import json

from model import MatrixFactorizationRouter
from dataset import RouterDataset


def find_best_threshold(
    checkpoint_path: str,
    val_data_path: str,
    thresholds: list = None,
    device: str = "auto"
):
    """
    Find the best threshold for the router based on validation data.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        val_data_path: Path to validation data parquet file
        thresholds: List of thresholds to try (default: 0.1 to 0.9)
        device: Device to use
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05).tolist()
    
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Load model
    model = MatrixFactorizationRouter(
        query_embedding_dim=config['query_embedding_dim'],
        model_embedding_dim=config['model_embedding_dim'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load validation dataset
    print(f"\nLoading validation data from {val_data_path}...")
    val_dataset = RouterDataset(
        data_path=val_data_path,
        embedding_model_name=config['embedding_model_name'],
        cache_embeddings=True,
    )
    
    # Get all predictions
    print("\nGenerating predictions...")
    all_embeddings = val_dataset.embeddings.to(device)
    all_labels = torch.tensor(val_dataset.df['binary_label'].values, dtype=torch.float32)
    
    with torch.no_grad():
        win_probs = model(all_embeddings).squeeze().cpu().numpy()
    
    all_labels = all_labels.numpy()
    
    # Evaluate each threshold
    print("\n" + "="*70)
    print(f"{'Threshold':^10} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1':^10}")
    print("="*70)
    
    results = []
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (win_probs >= threshold).astype(float)
        
        acc = accuracy_score(all_labels, predictions)
        prec = precision_score(all_labels, predictions, zero_division=0)
        rec = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        })
        
        marker = " <-- BEST" if f1 > best_f1 else ""
        print(f"{threshold:^10.2f} | {acc:^10.4f} | {prec:^10.4f} | {rec:^10.4f} | {f1:^10.4f}{marker}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("="*70)
    
    # Calculate AUC (threshold-independent)
    try:
        auc = roc_auc_score(all_labels, win_probs)
    except:
        auc = 0.5
    
    print(f"\nAUC-ROC (threshold-independent): {auc:.4f}")
    print(f"\n{'='*70}")
    print(f"BEST THRESHOLD: {best_threshold:.2f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"{'='*70}")
    
    # Save results
    checkpoint_dir = Path(checkpoint_path).parent
    results_path = checkpoint_dir / 'threshold_tuning_results.json'
    
    output = {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'auc': auc,
        'all_results': results,
    }
    
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return best_threshold, best_f1, results


def main():
    parser = argparse.ArgumentParser(description="Find optimal routing threshold")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data parquet file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    
    args = parser.parse_args()
    
    find_best_threshold(
        checkpoint_path=args.checkpoint,
        val_data_path=args.val_data,
        device=args.device,
    )


if __name__ == "__main__":
    main()
