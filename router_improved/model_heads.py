"""Pairwise router with one shared query projection and two small model heads."""

import torch
from torch import nn


class ModelHeadRouter(nn.Module):
    """Return GPT and Llama scores whose difference predicts GPT preference.

    With winner-only supervision, the two scores are relative latent scores;
    neither is independently calibrated as a model-quality estimate.
    """

    def __init__(self, embedding_dim: int, projection_dim: int = 256, head_dim: int = 32) -> None:
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.weak_head = nn.Sequential(nn.Linear(projection_dim, head_dim), nn.GELU(), nn.Linear(head_dim, 1))
        self.strong_head = nn.Sequential(nn.Linear(projection_dim, head_dim), nn.GELU(), nn.Linear(head_dim, 1))

    def model_scores(self, query_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.query_projection(query_embedding)
        return self.weak_head(projected), self.strong_head(projected)

    def forward_logits(self, query_embedding: torch.Tensor) -> torch.Tensor:
        weak, strong = self.model_scores(query_embedding)
        return strong - weak

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(query_embedding))
