"""
Matrix Factorization Router Model
Based on RouteLLM paper (ICLR 2025)

The goal is to learn a scoring function δ(M, q) that predicts how well model M 
will perform on query q. The win probability is modeled as:
    P(wins | q) = σ(δ(M_strong, q) - δ(M_weak, q))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorizationRouter(nn.Module):
    """
    Matrix Factorization Router that learns to predict which model 
    (strong or weak) will perform better on a given query.
    
    Following RouteLLM's implementation:
    δ(M, q) = classifier(normalize(vm) * text_proj(vq))
    
    where:
    - vm: model embedding (learned, L2 normalized)
    - vq: query embedding (from sentence transformer)
    - text_proj: projection matrix without bias
    - classifier: linear layer to produce final scalar
    - *: element-wise product
    """
    
    def __init__(
        self,
        query_embedding_dim: int = 384,  # dimension of sentence embeddings
        model_embedding_dim: int = 64,   # dimension of learned model embeddings
        num_models: int = 2,             # strong and weak
    ):
        super().__init__()
        
        self.query_embedding_dim = query_embedding_dim
        self.model_embedding_dim = model_embedding_dim
        
        # Learned model embeddings: index 0 = weak, index 1 = strong
        # These will be L2 normalized before use
        self.model_embeddings = nn.Embedding(num_models, model_embedding_dim)
        
        # Projection layer with MLP for non-linearity
        self.text_proj = nn.Sequential(
            nn.Linear(query_embedding_dim, query_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(query_embedding_dim, model_embedding_dim, bias=False)
        )
        
        # Classifier layer to produce final scalar
        self.classifier = nn.Linear(model_embedding_dim, 1, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.xavier_uniform_(self.model_embeddings.weight)
        # Initialize Linear layers in the MLP
        for layer in self.text_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def compute_score(self, query_embedding: torch.Tensor, model_idx: int) -> torch.Tensor:
        """
        Compute the score δ(M, q) for a given model and query.
        
        Following RouteLLM: classifier(normalize(vm) * text_proj(vq))
        
        Args:
            query_embedding: Query embedding from sentence transformer [batch_size, query_dim]
            model_idx: 0 for weak model, 1 for strong model
            
        Returns:
            Score tensor [batch_size, 1]
        """
        batch_size = query_embedding.size(0)
        
        # Get model embedding and L2 normalize [model_dim]
        model_idx_tensor = torch.tensor([model_idx], device=query_embedding.device)
        vm = self.model_embeddings(model_idx_tensor)  # [1, model_dim]
        vm = F.normalize(vm, p=2, dim=1)  # L2 normalize
        vm = vm.expand(batch_size, -1)  # [batch_size, model_dim]
        
        # Project query embedding (no bias)
        vq = self.text_proj(query_embedding)  # [batch_size, model_dim]
        
        # Element-wise product
        combined = vm * vq  # [batch_size, model_dim]
        
        # Classifier to produce final score
        score = self.classifier(combined)  # [batch_size, 1]
        
        return score
    
    def forward_logits(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits for the strong model winning.
        score_strong - score_weak
        """
        score_weak = self.compute_score(query_embedding, model_idx=0)
        score_strong = self.compute_score(query_embedding, model_idx=1)
        return score_strong - score_weak

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability that the strong model wins.
        
        P(wins | q) = σ(δ(M_strong, q) - δ(M_weak, q))
        
        Args:
            query_embedding: Query embedding from sentence transformer [batch_size, query_dim]
            
        Returns:
            Win probability [batch_size, 1]
        """
        logits = self.forward_logits(query_embedding)
        return torch.sigmoid(logits)
    
    def predict(self, query_embedding: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict routing decision.
        
        Args:
            query_embedding: Query embedding [batch_size, query_dim]
            threshold: If P(wins) >= threshold, route to strong model
            
        Returns:
            Routing decision: 1 = strong, 0 = weak [batch_size]
        """
        win_prob = self.forward(query_embedding)
        return (win_prob >= threshold).squeeze(-1).long()

if __name__ == "__main__":
    # Quick test
    print("Testing MatrixFactorizationRouter...")
    
    router = MatrixFactorizationRouter(query_embedding_dim=384, model_embedding_dim=64)
    
    # Dummy query embeddings
    batch_size = 4
    dummy_embeddings = torch.randn(batch_size, 384)
    
    # Forward pass
    win_probs = router(dummy_embeddings)
    print(f"Win probabilities: {win_probs.squeeze().tolist()}")
    
    # Predictions
    predictions = router.predict(dummy_embeddings, threshold=0.5)
    print(f"Predictions (1=strong, 0=weak): {predictions.tolist()}")
    
    print("\nModel parameters:")
    for name, param in router.named_parameters():
        print(f"  {name}: {param.shape}")
