"""A soft-gated, domain-aware router for a fixed strong/weak model pair."""

from __future__ import annotations

import torch
from torch import nn


class SoftDomainRouter(nn.Module):
    """Mixture of routing experts whose domain assignment is learned.

    The gate is conditioned on the query embedding rather than a hand-written
    label.  Expert scores are mixed in logit space, which lets the output remain
    a calibrated binary strong-model-win probability after temperature scaling.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 192,
        num_experts: int = 6,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.encoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.expert_heads = nn.Linear(hidden_dim, num_experts)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))

    def expert_logits_and_weights(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(embeddings)
        weights = torch.softmax(self.gate(features), dim=-1)
        logits = self.expert_heads(features) + self.expert_bias
        return logits, weights

    def forward_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        expert_logits, weights = self.expert_logits_and_weights(embeddings)
        return (expert_logits * weights).sum(dim=-1, keepdim=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(embeddings))
