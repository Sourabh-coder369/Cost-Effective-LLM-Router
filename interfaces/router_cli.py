"""
Terminal-based Router Interface
Input a question and see where the router decides to route it.
"""

import torch
import sys
import os

# Add router directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'router'))

from sentence_transformers import SentenceTransformer
from router.model import MatrixFactorizationRouter


class RouterCLI:
    def __init__(self, checkpoint_path: str = "router/checkpoints/best_model.pt"):
        """Load the trained router model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading router model...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        self.embedding_model = SentenceTransformer(config['embedding_model_name'])
        
        self.model = MatrixFactorizationRouter(
            query_embedding_dim=config['query_embedding_dim'],
            model_embedding_dim=config['model_embedding_dim'],
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.val_accuracy = checkpoint.get('val_accuracy', 'N/A')
        print(f"Router loaded! (Val accuracy: {self.val_accuracy:.1%})\n")
    
    def route(self, query: str, threshold: float = 0.5):
        """Route a query and return decision."""
        # Get embedding
        embedding = self.embedding_model.encode(
            [query],
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(self.device)
        
        # Get probability
        with torch.no_grad():
            prob = self.model(embedding).item()
        
        return prob >= threshold, prob

    def run(self):
        """Run the interactive CLI."""
        print("=" * 60)
        print("       LLM Query Router - Terminal Interface")
        print("=" * 60)
        print("\nEnter a question and I'll tell you whether to route it")
        print("to a STRONG model (GPT-4) or WEAK model (GPT-3.5).")
        print("\nCommands: 'quit' to exit, 'threshold X' to set threshold")
        print("-" * 60)
        
        threshold = 0.5
        
        while True:
            print()
            query = input("Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower().startswith('threshold'):
                try:
                    threshold = float(query.split()[1])
                    print(f"  Threshold set to {threshold:.2f}")
                except:
                    print(f"  Current threshold: {threshold:.2f}")
                continue
            
            # Route the query
            use_strong, prob = self.route(query, threshold)
            
            print()
            if use_strong:
                print(f"  🚀 Route to: STRONG MODEL (GPT-4, Claude)")
            else:
                print(f"  ⚡ Route to: WEAK MODEL (GPT-3.5, smaller LLMs)")
            
            print(f"  📊 P(strong wins): {prob:.1%}")
            print(f"  📏 Threshold: {threshold:.1%}")


if __name__ == "__main__":
    router = RouterCLI()
    router.run()
