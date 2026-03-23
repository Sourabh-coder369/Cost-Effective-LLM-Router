"""
Domain-Specific Matrix Factorization Router
=============================================
Extends the baseline MF router by learning SEPARATE capability vectors
for each (model, domain) pair — addressing the "Averaging Trap".

Domains:
  0 = general
  1 = math

Model indices:
  0 = weak  (Llama-2-7b)
  1 = strong (GPT-4)

Embedding lookup:
  index = model_idx * num_domains + domain_id
  → 4 vectors total: weak_general, weak_math, strong_general, strong_math
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainMFRouter(nn.Module):
    """
    Domain-Specific Matrix Factorization Router.

    Instead of a single vector per model, learns one vector per
    (model × domain) combination:

        δ(M, d, q) = classifier(normalize(v_{m,d}) * text_proj(vq))

    where v_{m,d} is the capability vector for model M in domain d.
    """

    # Domain constants (used externally)
    DOMAIN_GENERAL = 0
    DOMAIN_MATH    = 1
    NUM_DOMAINS    = 2

    def __init__(
        self,
        query_embedding_dim: int = 1024,
        model_embedding_dim: int = 64,
        num_models: int = 2,
        num_domains: int = 2,
    ):
        super().__init__()

        self.query_embedding_dim = query_embedding_dim
        self.model_embedding_dim = model_embedding_dim
        self.num_models  = num_models
        self.num_domains = num_domains

        # num_models * num_domains = 4 vectors:
        #   index 0 → weak_general
        #   index 1 → weak_math
        #   index 2 → strong_general
        #   index 3 → strong_math
        self.model_domain_embeddings = nn.Embedding(
            num_models * num_domains,
            model_embedding_dim
        )

        # Query projection (shared across domains)
        self.text_proj = nn.Sequential(
            nn.Linear(query_embedding_dim, query_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(query_embedding_dim, model_embedding_dim, bias=False),
        )

        # Final scalar scorer
        self.classifier = nn.Linear(model_embedding_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.model_domain_embeddings.weight)
        for layer in self.text_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.classifier.weight)

    def _embedding_idx(self, model_idx: int, domain_ids: torch.Tensor) -> torch.Tensor:
        """Return embedding table indices for each sample in the batch."""
        # domain_ids: [batch_size]  (0=general, 1=math)
        return model_idx * self.num_domains + domain_ids   # [batch_size]

    def compute_score(
        self,
        query_embedding: torch.Tensor,   # [B, query_dim]
        domain_ids: torch.Tensor,         # [B]  long tensor
        model_idx: int,                   # 0=weak, 1=strong
    ) -> torch.Tensor:
        """Compute δ(M, d, q) for each sample in the batch."""
        batch_size = query_embedding.size(0)

        # Get per-sample domain-specific model vector
        emb_indices = self._embedding_idx(model_idx, domain_ids)   # [B]
        vm = self.model_domain_embeddings(emb_indices)              # [B, model_dim]
        vm = F.normalize(vm, p=2, dim=1)                           # L2 normalise

        # Project query
        vq = self.text_proj(query_embedding)   # [B, model_dim]

        # Element-wise product → scalar score
        combined = vm * vq                     # [B, model_dim]
        score    = self.classifier(combined)   # [B, 1]
        return score

    def forward_logits(
        self,
        query_embedding: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """score_strong - score_weak  → [B, 1]"""
        score_weak   = self.compute_score(query_embedding, domain_ids, model_idx=0)
        score_strong = self.compute_score(query_embedding, domain_ids, model_idx=1)
        return score_strong - score_weak

    def forward(
        self,
        query_embedding: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """P(strong wins | q, d) = σ(logits)  → [B, 1]"""
        return torch.sigmoid(self.forward_logits(query_embedding, domain_ids))

    def predict(
        self,
        query_embedding: torch.Tensor,
        domain_ids: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Routing decision: 1=strong, 0=weak  → [B]"""
        win_prob = self.forward(query_embedding, domain_ids)
        return (win_prob >= threshold).squeeze(-1).long()


if __name__ == "__main__":
    print("Testing DomainMFRouter...")
    model = DomainMFRouter(query_embedding_dim=1024, model_embedding_dim=64)

    B = 8
    emb     = torch.randn(B, 1024)
    domains = torch.randint(0, 2, (B,))   # random math/general

    probs = model(emb, domains)
    print(f"Win probs: {probs.squeeze().tolist()}")
    print(f"Decisions: {model.predict(emb, domains).tolist()}")

    print("\nEmbedding table (4 vectors):")
    for name, param in model.named_parameters():
        if "model_domain" in name:
            print(f"  {name}: {param.shape}")
            # Show which index maps to what
    print("  idx 0 = weak_general | idx 1 = weak_math")
    print("  idx 2 = strong_general | idx 3 = strong_math")
