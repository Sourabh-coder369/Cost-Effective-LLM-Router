"""
Interactive Router Interface
Input a question and see where the router decides to route it.
"""

import gradio as gr
import torch
import sys
import os

# Add router directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'router'))

from sentence_transformers import SentenceTransformer
from router.model import MatrixFactorizationRouter


class RouterInterface:
    def __init__(self, checkpoint_path: str = "gpt4_llama7b_checkpoints/best_model.pt"):
        """Load the trained router model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        print(f"Loading embedding model: {config['embedding_model_name']}...")
        self.embedding_model = SentenceTransformer(config['embedding_model_name'])
        
        self.model = MatrixFactorizationRouter(
            query_embedding_dim=config['query_embedding_dim'],
            model_embedding_dim=config['model_embedding_dim'],
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.config = config
        self.val_accuracy = checkpoint.get('val_accuracy', 'N/A')
        print(f"Router loaded! Val accuracy: {self.val_accuracy}")
    
    def route(self, query: str, threshold: float = 0.5):
        """Route a query and return decision with explanation."""
        if not query.strip():
            return "❌ Please enter a question.", "", 0.5
        
        # Get embedding
        embedding = self.embedding_model.encode(
            [query],
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(self.device)
        
        # Get probability
        with torch.no_grad():
            prob = self.model(embedding).item()
        
        # Make routing decision
        if prob >= threshold:
            decision = "🚀 **STRONG MODEL** (e.g., GPT-4, Claude)"
            explanation = f"""
### Routing Decision: Strong Model

**Probability Score:** {prob:.1%}

**Reason:** This query appears to require advanced reasoning, complex analysis, 
or nuanced understanding that would benefit from a more capable model.

**Threshold:** {threshold:.1%} (queries above this go to strong model)
"""
        else:
            decision = "⚡ **WEAK MODEL** (e.g., GPT-3.5, Smaller LLMs)"
            explanation = f"""
### Routing Decision: Weak Model

**Probability Score:** {prob:.1%}

**Reason:** This query appears straightforward enough that a smaller, 
faster model can handle it effectively, saving cost and latency.

**Threshold:** {threshold:.1%} (queries below this go to weak model)
"""
        
        return decision, explanation, prob


def create_interface():
    """Create the Gradio interface."""
    
    # Load router
    router = RouterInterface()
    
    def route_query(query: str, threshold: float):
        decision, explanation, prob = router.route(query, threshold)
        return decision, explanation, prob
    
    # Example queries
    examples = [
        ["What is 2 + 2?", 0.5],
        ["Explain quantum computing in simple terms.", 0.5],
        ["Write a Python function to sort a list.", 0.5],
        ["What is the capital of France?", 0.5],
        ["Develop a comprehensive business plan for a tech startup.", 0.5],
        ["Explain the implications of Gödel's incompleteness theorems.", 0.5],
        ["Hi, how are you?", 0.5],
        ["What's the weather like today?", 0.5],
        ["Analyze the socioeconomic factors contributing to income inequality.", 0.5],
    ]
    
    with gr.Blocks(title="LLM Router", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🔀 LLM Query Router
        
This router uses a trained Matrix Factorization model to decide whether your query 
should go to a **Strong Model** (like GPT-4) or a **Weak Model** (like GPT-3.5).

**How it works:** The router analyzes your query and predicts the probability that 
a strong model would provide a significantly better response. Based on the threshold 
you set, it routes accordingly.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="Type your question here...",
                    lines=3,
                )
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Routing Threshold",
                    info="Higher = more queries go to weak model (cost savings)"
                )
                route_btn = gr.Button("🔀 Route Query", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                decision_output = gr.Markdown(label="Routing Decision")
                prob_output = gr.Number(label="Strong Model Win Probability", precision=3)
        
        explanation_output = gr.Markdown(label="Explanation")
        
        gr.Examples(
            examples=examples,
            inputs=[query_input, threshold_slider],
            label="Example Queries"
        )
        
        # Connect the button
        route_btn.click(
            fn=route_query,
            inputs=[query_input, threshold_slider],
            outputs=[decision_output, explanation_output, prob_output]
        )
        
        # Also route on Enter key
        query_input.submit(
            fn=route_query,
            inputs=[query_input, threshold_slider],
            outputs=[decision_output, explanation_output, prob_output]
        )
        
        gr.Markdown(f"""
---
**Model Info:** Validation Accuracy: {router.val_accuracy if isinstance(router.val_accuracy, str) else f'{router.val_accuracy:.1%}'} | 
Embedding Model: `{router.config['embedding_model_name']}`
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
