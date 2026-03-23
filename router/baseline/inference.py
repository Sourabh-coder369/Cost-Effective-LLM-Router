"""
Inference script for Matrix Factorization Router
"""

import torch
import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from model import MatrixFactorizationRouter


class Router:
    """
    Router for inference - routes queries to either strong or weak model.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
    ):
        """
        Load a trained router.
        
        Args:
            checkpoint_path: Path to saved checkpoint (.pt file)
            device: Device to use
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Load sentence transformer
        print(f"Loading embedding model: {config['embedding_model_name']}...")
        self.embedding_model = SentenceTransformer(config['embedding_model_name'])
        
        # Create and load router model
        self.model = MatrixFactorizationRouter(
            query_embedding_dim=config['query_embedding_dim'],
            model_embedding_dim=config['model_embedding_dim'],
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.config = config
        print(f"Router loaded! Val accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
    
    def get_embeddings(self, queries: List[str]) -> torch.Tensor:
        """Get embeddings for queries."""
        embeddings = self.embedding_model.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings.to(self.device)
    
    def get_win_probability(self, queries: List[str]) -> List[float]:
        """
        Get the probability that the strong model will win for each query.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of probabilities (0-1)
        """
        embeddings = self.get_embeddings(queries)
        
        with torch.no_grad():
            win_probs = self.model(embeddings)
        
        return win_probs.squeeze().cpu().tolist()
    
    def route(
        self,
        queries: List[str],
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Route queries to either strong or weak model.
        
        Args:
            queries: List of query strings
            threshold: If P(strong wins) >= threshold, route to strong
            
        Returns:
            List of (model_choice, probability) tuples
            model_choice is either "strong" or "weak"
        """
        probabilities = self.get_win_probability(queries)
        
        if isinstance(probabilities, float):
            probabilities = [probabilities]
        
        results = []
        for prob in probabilities:
            if prob >= threshold:
                results.append(("strong", prob))
            else:
                results.append(("weak", prob))
        
        return results
    
    def route_single(self, query: str, threshold: float = 0.5) -> Tuple[str, float]:
        """Route a single query."""
        results = self.route([query], threshold)
        return results[0]


def demo():
    """Demo the router with some example queries."""
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using: python train.py")
        return
    
    # Load router
    router = Router(checkpoint_path)
    
    # Example queries
    queries = [
        "What is 2 + 2?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What is the capital of France?",
        "Develop a comprehensive business plan for a tech startup including market analysis, financial projections, and competitive strategy.",
        "Hi, how are you?",
        "Explain the implications of Gödel's incompleteness theorems on the foundations of mathematics.",
    ]
    
    print("\n" + "="*60)
    print("Routing Demo")
    print("="*60)
    
    for query in queries:
        choice, prob = router.route_single(query, threshold=0.5)
        print(f"\nQuery: {query[:60]}...")
        print(f"  → Route to: {choice.upper()} (P(strong wins) = {prob:.3f})")


if __name__ == "__main__":
    demo()
