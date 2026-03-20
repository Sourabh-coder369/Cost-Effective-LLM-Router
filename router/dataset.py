"""
Dataset and DataLoader for Router Training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple
import numpy as np
import os
import hashlib
from huggingface_hub import hf_hub_download

def get_hf_path(local_path: str, repo_id: str = "MSourav/capstone-datasets") -> str:
    """Check if file exists locally, otherwise try to download from Hugging Face."""
    if not local_path:
        return local_path
        
    if os.path.exists(local_path):
        return local_path
        
    print(f"File {local_path} not found locally. Attempting to download from Hugging Face ({repo_id})...")
    
    # Standardize path
    hf_path = local_path.replace("\\", "/")
    
    # Extract relative paths usually mapped in the HF repo
    if "gpt4_llama7b_data_unbalanced" in hf_path:
        hf_path = "gpt4_llama7b_data_unbalanced" + hf_path.split("gpt4_llama7b_data_unbalanced")[1]
    elif "router/data" in hf_path:
        hf_path = "router/data" + hf_path.split("router/data")[1]
    else:
        # Fallback to cleaning leading prefixes
        if hf_path.startswith("./"):
            hf_path = hf_path[2:]
        elif hf_path.startswith("../"):
            hf_path = hf_path.replace("../", "")
            
    try:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=hf_path, repo_type="dataset")
        print(f"Successfully downloaded to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Warning: Failed to download {hf_path} from HF: {e}")
        return local_path


class RouterDataset(Dataset):
    """
    Dataset for training the router.
    
    Each sample contains:
    - query embedding
    - label: 0 = weak wins/tie (route to weak), 1 = strong wins (route to strong)
    """
    
    def __init__(
        self,
        data_path: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        include_ties_as: str = "weak",  # "weak", "strong", or "exclude"
        precomputed_embeddings_path: str = None,
    ):
        """
        Args:
            data_path: Path to parquet file with 'prompt' and 'label' columns
            embedding_model_name: Sentence transformer model to use
            cache_embeddings: Whether to precompute and cache all embeddings
            include_ties_as: How to treat ties - "weak" (default), "strong", or "exclude"
            precomputed_embeddings_path: Path to a .pt file with precomputed embeddings
        """
        self.data_path = get_hf_path(data_path)
        self.embedding_model_name = embedding_model_name
        self.cache_embeddings = cache_embeddings
        self.include_ties_as = include_ties_as
        self.precomputed_embeddings_path = get_hf_path(precomputed_embeddings_path) if precomputed_embeddings_path else None
        
        # Load data
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # Handle ties and build binary label
        # If a precomputed binary_label exists (e.g., from a balanced split), respect it
        if 'binary_label' in self.df.columns:
            print("Using existing binary_label column from dataset")
        else:
            if include_ties_as == "exclude":
                self.df = self.df[self.df['label'] != 'tie'].reset_index(drop=True)
                print(f"Excluded ties. Remaining samples: {len(self.df)}")
            
            # Convert labels to binary
            # 1 = strong should win (route to strong)
            # 0 = weak should win or tie (route to weak)
            self.df['binary_label'] = self.df['label'].apply(
                lambda x: 1 if x == 'wins' else (1 if x == 'tie' and include_ties_as == 'strong' else 0)
            )
        
        print(f"Dataset size: {len(self.df)}")
        print(f"Label distribution:")
        print(f"  Route to strong (1): {(self.df['binary_label'] == 1).sum()}")
        print(f"  Route to weak (0): {(self.df['binary_label'] == 0).sum()}")
        
        # Load embeddings
        self.embeddings = None
        self.embedding_model = None
        
        if self.precomputed_embeddings_path and os.path.exists(self.precomputed_embeddings_path):
            # Use precomputed embeddings (e.g., generated on Kaggle GPU)
            print(f"Loading precomputed embeddings from {self.precomputed_embeddings_path}...")
            self.embeddings = torch.load(self.precomputed_embeddings_path, map_location='cpu')
            assert len(self.embeddings) == len(self.df), (
                f"Embedding count ({len(self.embeddings)}) != dataset rows ({len(self.df)}). "
                f"Make sure the .pt file matches the parquet file."
            )
            print(f"Loaded {len(self.embeddings)} precomputed embeddings of dimension {self.embeddings.shape[1]}")
        else:
            # Load embedding model and compute embeddings
            print(f"Loading embedding model: {embedding_model_name}...")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            
            if cache_embeddings:
                # Generate cache file path based on data file and model
                cache_key = f"{self.data_path}_{embedding_model_name}".encode('utf-8')
                cache_hash = hashlib.md5(cache_key).hexdigest()[:16]
                
                # Use a safe cache directory even if downloaded from HF 
                # (which might reside in a read-only cache dir)
                base_dir = os.path.dirname(self.data_path)
                if not base_dir or not os.access(base_dir, os.W_OK):
                    base_dir = "."
                    
                cache_dir = os.path.join(base_dir, '.embedding_cache')
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f"{os.path.basename(self.data_path)}_{cache_hash}.pt")
                
                # Try to load from cache
                if os.path.exists(cache_file):
                    print(f"Loading cached embeddings from {cache_file}...")
                    self.embeddings = torch.load(cache_file)
                    print(f"Loaded {len(self.embeddings)} cached embeddings of dimension {self.embeddings.shape[1]}")
                else:
                    print("Caching embeddings...")
                    prompts = self.df['prompt'].tolist()
                    self.embeddings = self.embedding_model.encode(
                        prompts,
                        convert_to_tensor=True,
                        show_progress_bar=True,
                        batch_size=64
                    )
                    print(f"Cached {len(self.embeddings)} embeddings of dimension {self.embeddings.shape[1]}")
                    
                    # Save to disk
                    print(f"Saving embeddings to {cache_file}...")
                    torch.save(self.embeddings, cache_file)
                    print("Embeddings saved to disk")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (query_embedding, label)
        """
        if self.embeddings is not None:
            embedding = self.embeddings[idx]
        else:
            prompt = self.df.iloc[idx]['prompt']
            embedding = self.embedding_model.encode(
                prompt,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        
        label = torch.tensor(self.df.iloc[idx]['binary_label'], dtype=torch.float32)
        
        return embedding, label


def create_dataloaders(
    train_path: str,
    val_split: float = 0.1,
    val_path: str = None,
    batch_size: int = 64,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    include_ties_as: str = "weak",
    seed: int = 42,
    train_embeddings_path: str = None,
    val_embeddings_path: str = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and validation dataloaders.
    
    If val_path is provided, use that file as the validation set (no random split).
    Otherwise, split the training data according to val_split.
    
    If train_embeddings_path / val_embeddings_path are provided, load precomputed
    embeddings instead of running the embedding model.
    """

    if val_path:
        print(f"Using explicit train/val files:\n  train={train_path}\n  val={val_path}")
        train_dataset = RouterDataset(
            data_path=train_path,
            embedding_model_name=embedding_model_name,
            cache_embeddings=True,
            include_ties_as=include_ties_as,
            precomputed_embeddings_path=train_embeddings_path,
        )
        val_dataset = RouterDataset(
            data_path=val_path,
            embedding_model_name=embedding_model_name,
            cache_embeddings=True,
            include_ties_as=include_ties_as,
            precomputed_embeddings_path=val_embeddings_path,
        )
    else:
        # Load full dataset and split
        full_dataset = RouterDataset(
            data_path=train_path,
            embedding_model_name=embedding_model_name,
            cache_embeddings=True,
            include_ties_as=include_ties_as,
            precomputed_embeddings_path=train_embeddings_path,
        )
        n_total = len(full_dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        
        torch.manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )
        print(f"\nTrain size: {n_train}, Val size: {n_val}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Get embedding dimension
    # Handle Subset objects from random_split
    dataset_obj = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
    
    if hasattr(dataset_obj, 'embeddings') and dataset_obj.embeddings is not None:
        embedding_dim = dataset_obj.embeddings.shape[1]
    else:
        embedding_dim = dataset_obj.embedding_model.get_sentence_embedding_dimension()
    
    return train_loader, val_loader, embedding_dim


if __name__ == "__main__":
    # Test dataset
    import os
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "strong_weak_dataset_t025.parquet")
    
    if os.path.exists(data_path):
        dataset = RouterDataset(data_path, cache_embeddings=True, include_ties_as="weak")
        
        # Get a sample
        embedding, label = dataset[0]
        print(f"\nSample embedding shape: {embedding.shape}")
        print(f"Sample label: {label}")
    else:
        print(f"Data file not found: {data_path}")
