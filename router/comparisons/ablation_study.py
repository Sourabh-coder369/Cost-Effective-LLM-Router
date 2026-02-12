"""
Ablation Study: Compare Linear vs MLP (Non-linear) Projection

This script trains two router models on the same data:
1. Linear projection (original design)
2. MLP projection (with ReLU non-linearity)

And compares their performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json

from dataset import create_dataloaders
from sklearn.metrics import f1_score, roc_auc_score


class LinearRouter(nn.Module):
    """Router with LINEAR projection (no non-linearity)."""
    
    def __init__(self, query_embedding_dim=384, model_embedding_dim=64, num_models=2):
        super().__init__()
        self.model_embeddings = nn.Embedding(num_models, model_embedding_dim)
        
        # LINEAR projection - NO activation function
        self.text_proj = nn.Linear(query_embedding_dim, model_embedding_dim, bias=False)
        
        self.classifier = nn.Linear(model_embedding_dim, 1, bias=False)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.model_embeddings.weight)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def forward_logits(self, query_embedding):
        batch_size = query_embedding.size(0)
        
        # Weak model score
        vm_weak = F.normalize(self.model_embeddings(torch.tensor([0], device=query_embedding.device)), p=2, dim=1)
        vm_weak = vm_weak.expand(batch_size, -1)
        score_weak = self.classifier(vm_weak * self.text_proj(query_embedding))
        
        # Strong model score
        vm_strong = F.normalize(self.model_embeddings(torch.tensor([1], device=query_embedding.device)), p=2, dim=1)
        vm_strong = vm_strong.expand(batch_size, -1)
        score_strong = self.classifier(vm_strong * self.text_proj(query_embedding))
        
        return score_strong - score_weak
    
    def forward(self, query_embedding):
        return torch.sigmoid(self.forward_logits(query_embedding))


class MLPRouter(nn.Module):
    """Router with MLP projection (WITH non-linearity)."""
    
    def __init__(self, query_embedding_dim=384, model_embedding_dim=64, num_models=2):
        super().__init__()
        self.model_embeddings = nn.Embedding(num_models, model_embedding_dim)
        
        # MLP projection - WITH activation function
        self.text_proj = nn.Sequential(
            nn.Linear(query_embedding_dim, query_embedding_dim),
            nn.ReLU(),  # <-- Non-linearity here!
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
        
        vm_weak = F.normalize(self.model_embeddings(torch.tensor([0], device=query_embedding.device)), p=2, dim=1)
        vm_weak = vm_weak.expand(batch_size, -1)
        score_weak = self.classifier(vm_weak * self.text_proj(query_embedding))
        
        vm_strong = F.normalize(self.model_embeddings(torch.tensor([1], device=query_embedding.device)), p=2, dim=1)
        vm_strong = vm_strong.expand(batch_size, -1)
        score_strong = self.classifier(vm_strong * self.text_proj(query_embedding))
        
        return score_strong - score_weak
    
    def forward(self, query_embedding):
        return torch.sigmoid(self.forward_logits(query_embedding))


def train_and_evaluate(model, train_loader, val_loader, device, num_epochs=10, lr=3e-4):
    """Train model and return best validation metrics."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_metrics = {}
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
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
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * embeddings.size(0)
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
            }
        
        print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}")
    
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Linear vs MLP")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--query_dim', type=int, default=384)
    parser.add_argument('--num_epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, _ = create_dataloaders(
        train_path=args.data_path,
        embedding_model_name=args.embedding_model,
        batch_size=64,
        val_split=0.1,
        seed=42,
    )
    
    # Train LINEAR model
    print("\n" + "="*60)
    print("Training LINEAR Router (no non-linearity)")
    print("="*60)
    linear_model = LinearRouter(query_embedding_dim=args.query_dim).to(device)
    linear_params = sum(p.numel() for p in linear_model.parameters())
    print(f"Parameters: {linear_params:,}")
    linear_results = train_and_evaluate(linear_model, train_loader, val_loader, device, args.num_epochs)
    
    # Train MLP model
    print("\n" + "="*60)
    print("Training MLP Router (WITH non-linearity)")
    print("="*60)
    mlp_model = MLPRouter(query_embedding_dim=args.query_dim).to(device)
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"Parameters: {mlp_params:,}")
    mlp_results = train_and_evaluate(mlp_model, train_loader, val_loader, device, args.num_epochs)
    
    # Compare
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"\n{'Metric':<15} | {'Linear':<12} | {'MLP':<12} | {'Winner':<10}")
    print("-"*55)
    
    for metric in ['val_loss', 'accuracy', 'f1', 'auc']:
        linear_val = linear_results[metric]
        mlp_val = mlp_results[metric]
        
        if metric == 'val_loss':
            winner = "MLP" if mlp_val < linear_val else "Linear"
        else:
            winner = "MLP" if mlp_val > linear_val else "Linear"
        
        print(f"{metric:<15} | {linear_val:<12.4f} | {mlp_val:<12.4f} | {winner:<10}")
    
    print("-"*55)
    
    # Save results
    results = {
        'linear': linear_results,
        'mlp': mlp_results,
        'linear_params': linear_params,
        'mlp_params': mlp_params,
    }
    
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: ablation_results.json")


if __name__ == "__main__":
    main()
