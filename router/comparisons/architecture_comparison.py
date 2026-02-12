"""
Architecture Comparison: Matrix Factorization vs Direct Classifier

Tests whether removing model embeddings and using a simple MLP classifier
on query embeddings performs better than the current MF approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import create_dataloaders
from sklearn.metrics import f1_score, roc_auc_score


# ──────────────────────────────────────────────────────────────
# Model 1: Current Matrix Factorization Router
# ──────────────────────────────────────────────────────────────
class MFRouter(nn.Module):
    """Current architecture: query * model_embedding → classifier"""

    def __init__(self, query_dim=768, model_dim=64):
        super().__init__()
        self.model_embeddings = nn.Embedding(2, model_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(query_dim, model_dim, bias=False),
        )
        self.classifier = nn.Linear(model_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.model_embeddings.weight)
        for layer in self.text_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward_logits(self, x):
        b = x.size(0)
        proj = self.text_proj(x)
        vm_w = F.normalize(self.model_embeddings(torch.tensor([0], device=x.device)), p=2, dim=1).expand(b, -1)
        vm_s = F.normalize(self.model_embeddings(torch.tensor([1], device=x.device)), p=2, dim=1).expand(b, -1)
        return self.classifier(vm_s * proj) - self.classifier(vm_w * proj)


# ──────────────────────────────────────────────────────────────
# Model 2: Direct MLP Classifier (no model embeddings)
# ──────────────────────────────────────────────────────────────
class DirectClassifier(nn.Module):
    """Simple approach: query → MLP → binary output"""

    def __init__(self, query_dim=768, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward_logits(self, x):
        return self.mlp(x)


# ──────────────────────────────────────────────────────────────
# Model 3: Deeper Direct Classifier
# ──────────────────────────────────────────────────────────────
class DeepClassifier(nn.Module):
    """Deeper MLP: query → 3 hidden layers → binary output"""

    def __init__(self, query_dim=768, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward_logits(self, x):
        return self.mlp(x)


# ──────────────────────────────────────────────────────────────
# Training and evaluation
# ──────────────────────────────────────────────────────────────
def train_and_evaluate(model, train_loader, val_loader, device, num_epochs=15, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_metrics = {}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for embeddings, labels in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{num_epochs}", leave=False):
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model.forward_logits(embeddings), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * embeddings.size(0)
        train_loss /= len(train_loader.dataset)

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
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'best_epoch': epoch + 1,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'accuracy': acc,
                'f1': f1,
                'auc': auc,
            }
            marker = " *"

        print(f"    Ep {epoch+1}: Train={train_loss:.4f} Val={val_loss:.4f} Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}{marker}")

    return best_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading data...")
    train_loader, val_loader, _ = create_dataloaders(
        train_path='../gpt4_llama7b_data_unbalanced/router_train.parquet',
        embedding_model_name='all-mpnet-base-v2',
        batch_size=64, val_split=0.1, seed=42,
    )

    models = {
        'MF Router (current)': lambda: MFRouter(query_dim=768, model_dim=64),
        'Direct Classifier (256)': lambda: DirectClassifier(query_dim=768, hidden_dim=256),
        'Deep Classifier (256→128→64)': lambda: DeepClassifier(query_dim=768, hidden_dim=256),
    }

    results = {}

    for name, model_fn in models.items():
        print("\n" + "=" * 65)
        print(f"  {name}")
        print("=" * 65)

        torch.manual_seed(42)
        model = model_fn().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

        metrics = train_and_evaluate(model, train_loader, val_loader, device, num_epochs=15, lr=1e-4)
        metrics['params'] = params
        results[name] = metrics

    # Summary
    print("\n" + "=" * 90)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("=" * 90)
    print(f"\n{'Architecture':<30} | {'Params':<10} | {'Epoch':<6} | {'Val Loss':<9} | {'Acc':<7} | {'F1':<7} | {'AUC':<7}")
    print("-" * 90)

    for name, r in results.items():
        print(f"{name:<30} | {r['params']:<10,} | {r['best_epoch']:<6} | {r['val_loss']:<9.4f} | {r['accuracy']:<7.4f} | {r['f1']:<7.4f} | {r['auc']:<7.4f}")

    print("-" * 90)

    best = max(results.keys(), key=lambda k: results[k]['f1'])
    print(f"\nBest architecture by F1: {best}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'architecture_comparison_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
