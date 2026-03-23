"""
Unified training script for Matrix Factorization Router.
Works with any dataset in the standard router format.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import argparse

# Add router module to path
from model import MatrixFactorizationRouter
from dataset import create_dataloaders
from sklearn.metrics import f1_score, roc_auc_score


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for embeddings, labels in tqdm(train_loader, desc="Training", leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model.forward_logits(embeddings)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * embeddings.size(0)
    
    return total_loss / len(train_loader.dataset)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Forward pass
            logits = model.forward_logits(embeddings)
            win_probs = torch.sigmoid(logits)
            
            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item() * embeddings.size(0)

            # Store predictions
            all_probs.extend(win_probs.cpu().numpy().flatten())
            all_preds.extend((win_probs > 0.5).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(val_loader.dataset)
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    accuracy = np.mean(all_preds == all_labels)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return avg_loss, accuracy, f1, auc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_dir: Path,
) -> dict:
    """Main training loop."""
    
    # Optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Calculate class weights
    print("Calculating class weights...")
    try:
        if hasattr(train_loader.dataset, 'dataset'):
            # Subset
            indices = train_loader.dataset.indices
            all_labels = train_loader.dataset.dataset.df.iloc[indices]['binary_label']
        else:
            # Full dataset
            all_labels = train_loader.dataset.df['binary_label']
        
        n_pos = (all_labels == 1).sum()
        n_neg = (all_labels == 0).sum()
    except:
        print("Warning: Could not extract labels efficiently. Analyzing loader...")
        n_pos = 0
        n_neg = 0
        for _, labels in train_loader:
            n_pos += (labels == 1).sum().item()
            n_neg += (labels == 0).sum().item()
            
    print(f"Class distribution: {n_pos} positive, {n_neg} negative")
    if n_pos > 0:
        pos_weight_val = n_neg / n_pos
        pos_weight = torch.tensor([pos_weight_val], device=device)
        print(f"Using positive class weight: {pos_weight_val:.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print("Warning: No positive samples found!")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print("\nStarting training...")
    print(f"Device: {device}")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Initial learning rate: {config['learning_rate']}")
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_accuracy, val_f1, val_auc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'] = history.get('val_f1', []) + [val_f1]
        history['val_auc'] = history.get('val_auc', []) + [val_auc]
        history['learning_rate'].append(current_lr)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_accuracy:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'config': config,
            }
            
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{config['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'config': config,
    }
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save(final_checkpoint, final_path)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*60}\n")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Matrix Factorization Router")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to router training data (parquet file)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Model arguments
    parser.add_argument('--query_embedding_dim', type=int, default=384,
                        help='Dimension of query embeddings (default: 384 for sentence-transformers)')
    parser.add_argument('--model_embedding_dim', type=int, default=64,
                        help='Dimension of model embeddings')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    parser.add_argument('--train_embeddings', type=str, default=None,
                        help='Path to precomputed train embeddings (.pt file)')
    parser.add_argument('--val_embeddings', type=str, default=None,
                        help='Path to precomputed val embeddings (.pt file)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = {
        'data_path': args.data_path,
        'query_embedding_dim': args.query_embedding_dim,
        'model_embedding_dim': args.model_embedding_dim,
        'embedding_model_name': args.embedding_model,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'val_split': args.val_split,
        'seed': args.seed,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save config
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")
    
    # Create dataloaders
    print("\nPreparing data...")
    train_loader, val_loader, query_dim = create_dataloaders(
        train_path=args.data_path,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        train_embeddings_path=args.train_embeddings,
        val_embeddings_path=args.val_embeddings,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Initialize model (use auto-detected query_dim from the embedding model)
    print(f"\nInitializing model (detected embedding dim: {query_dim})...")
    model = MatrixFactorizationRouter(
        query_embedding_dim=query_dim,
        model_embedding_dim=args.model_embedding_dim,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    history = train(model, train_loader, val_loader, config, device, checkpoint_dir)
    
    # Save history
    history_path = checkpoint_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()
