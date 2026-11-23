"""A small PatchTST-style return forecaster implemented for this project."""

from __future__ import annotations

import torch
from torch import nn


class PatchTransformer(nn.Module):
    """Encode non-overlapping multivariate time-series patches and predict one return."""

    def __init__(
        self,
        feature_count: int,
        context_length: int = 60,
        patch_length: int = 5,
        embedding_dim: int = 64,
        attention_heads: int = 4,
        encoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if context_length % patch_length:
            raise ValueError("context_length must be divisible by patch_length")
        patch_count = context_length // patch_length
        self.patch_length = patch_length
        self.patch_embedding = nn.Linear(feature_count * patch_length, embedding_dim)
        self.position = nn.Parameter(torch.zeros(1, patch_count, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """sequence has shape [batch, context_length, feature_count]."""
        batch, length, features = sequence.shape
        if length != self.position.shape[1] * self.patch_length:
            raise ValueError("Unexpected context length")
        patches = sequence.reshape(batch, -1, self.patch_length * features)
        encoded = self.encoder(self.patch_embedding(patches) + self.position)
        return self.head(encoded[:, -1]).squeeze(-1)
