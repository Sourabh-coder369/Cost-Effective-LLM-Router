"""
Learning Rate Comparison Study

Compares lr = 3e-4 (current), 1e-4, 5e-5, 3e-5
on the unbalanced dataset with MPNet embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
import json
import sys
import os

# Add parent directory so we can import dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import create_dataloaders
from sklearn.metrics import f1_score, roc_auc_score


class MLPRouter(nn.Module):
    def __init__(self, query_embedding_dim=768, model_embedding_dim=64, num_models=2):
        super().__init__()
        self.model_embeddings = nn.Embedding(num_models, model_embedding_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(query_embedding_dim, query_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(query_embedding_dim, model_embedding_dim, bias=False)
        )
        self.classifier = nn.Linear(model_embedding_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.model_embeddings.weight)
        for layer in self.text_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward_logits(self, query_embedding):
        batch_size = query_embedding.size(0)
        vm_weak = F.normalize(self.model_embeddings(torch.tensor([0], device=query_embedding.device)), p=2, dim=1).expand(batch_size, -1)
        score_weak = self.classifier(vm_weak * self.text_proj(query_embedding))
        vm_strong = F.normalize(self.model_embeddings(torch.tensor([1], device=query_embedding.device)), p=2, dim=1).expand(batch_size, -1)
        score_strong = self.classifier(vm_strong * self.text_proj(query_embedding))
        return score_strong - score_weak


def train_and_evaluate(model, train_loader, val_loader, device, num_epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_metrics = {}
    epoch_history = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for embeddings, labels in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{num_epochs}", leave=False):
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model.forward_logits(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * embeddings.size(0)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device).unsqueeze(1)
                logits = model.forward_logits(embeddings)
                probs = torch.sigmoid(logits)
                val_loss += criterion(logits, labels).item() * embeddings.size(0)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend((probs > 0.5).cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_preds == all_labels)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
        })

        print(f"    Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'best_epoch': epoch + 1,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
            }

    best_metrics['epoch_history'] = epoch_history
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Learning Rate Comparison")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--query_dim', type=int, default=768)
    parser.add_argument('--num_epochs', type=int, default=15)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data once
    print("\nLoading data...")
    train_loader, val_loader, _ = create_dataloaders(
        train_path=args.data_path,
        embedding_model_name=args.embedding_model,
        batch_size=64, val_split=0.1, seed=42,
    )

    learning_rates = [3e-4, 1e-4, 5e-5, 3e-5]
    results = {}

    for lr in learning_rates:
        lr_str = f"{lr:.0e}"
        print("\n" + "=" * 60)
        print(f"Training with lr = {lr_str}")
        print("=" * 60)

        # Reset model with same seed for fair comparison
        torch.manual_seed(42)
        model = MLPRouter(query_embedding_dim=args.query_dim, model_embedding_dim=64).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

        metrics = train_and_evaluate(model, train_loader, val_loader, device, args.num_epochs, lr)
        results[lr_str] = metrics

    # Summary
    print("\n" + "=" * 80)
    print("LEARNING RATE COMPARISON RESULTS")
    print("=" * 80)
    print(f"\n{'LR':<10} | {'Best Epoch':<12} | {'Val Loss':<10} | {'Accuracy':<10} | {'F1':<10} | {'AUC':<10}")
    print("-" * 80)

    for lr in learning_rates:
        lr_str = f"{lr:.0e}"
        r = results[lr_str]
        print(f"{lr_str:<10} | {r['best_epoch']:<12} | {r['val_loss']:<10.4f} | {r['accuracy']:<10.4f} | {r['f1']:<10.4f} | {r['auc']:<10.4f}")

    print("-" * 80)

    best_lr = max(results.keys(), key=lambda k: results[k]['f1'])
    print(f"\nBest LR by F1: {best_lr} (F1={results[best_lr]['f1']:.4f})")

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'lr_comparison_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
