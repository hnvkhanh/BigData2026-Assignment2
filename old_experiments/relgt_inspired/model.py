"""RelGT-inspired encoder over sampled temporal ego-graph tokens.

Follows the structure of the Relational Graph Transformer (Dwivedi et al. 2025,
https://github.com/snap-stanford/relgt) in plain PyTorch:

  * the sample has no seed token, so a learned user token is prepended and acts as the seed;
  * each token is encoded by five elements (entity, node type, hop / relation, row features and
    time relative to the cutoff), concatenated and mixed by an MLP;
  * a local transformer attends over the tokens of one sampled subgraph;
  * the seed token also attends to a bank of learned global centroids;
  * a feed-forward block and a head map the seed representation to the six activity logits.

Differences from the official code: the centroids are learned by gradient descent instead of an
EMA vector quantiser, there is no GIN positional encoding (the sample has no explicit edges), and
the row features are a small hand-made vector instead of torch_frame embeddings.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .data import LABELS, REPO_MEMBERSHIP_RELATION

NUM_ACTIONS = len(LABELS)
NODE_FEATURE_DIM = 4
NUM_NODE_TYPES = 4  # padding, pull request, issue, repository
NUM_HOPS = 3
NUM_RELATIONS = REPO_MEMBERSHIP_RELATION + 1  # padding / user, six actions, repository membership


class TimeEncoder(nn.Module):
    """Sinusoidal features of log-scaled age in days, followed by a linear map."""

    def __init__(self, hidden_dim: int, num_frequencies: int = 8):
        super().__init__()
        self.register_buffer("frequencies", torch.logspace(-1, 1, num_frequencies))
        self.proj = nn.Linear(2 * num_frequencies + 1, hidden_dim)

    def forward(self, relative_days: torch.Tensor) -> torch.Tensor:
        age = torch.log1p(relative_days.clamp(min=0)).unsqueeze(-1)
        angles = age * self.frequencies
        return self.proj(torch.cat([age, angles.sin(), angles.cos()], dim=-1))


class TokenEncoder(nn.Module):
    """Concatenate the five token elements and mix them with an MLP (shared with the GNN)."""

    def __init__(self, num_entities: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.entity = nn.Embedding(num_entities, hidden_dim, padding_idx=0)
        self.node_type = nn.Embedding(NUM_NODE_TYPES, hidden_dim)
        self.relation = nn.Embedding(NUM_RELATIONS, hidden_dim)
        self.hop = nn.Embedding(NUM_HOPS, hidden_dim)
        self.features = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time = TimeEncoder(hidden_dim)
        self.mix = nn.Sequential(
            nn.Linear(6 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        entity_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        hop_ids: torch.Tensor,
        node_features: torch.Tensor,
        relative_days: torch.Tensor,
    ) -> torch.Tensor:
        parts = [
            self.entity(entity_ids),
            self.node_type(node_type_ids),
            self.relation(relation_ids),
            self.hop(hop_ids),
            self.features(node_features),
            self.time(relative_days),
        ]
        return self.mix(torch.cat(parts, dim=-1))


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class RelGTInspired(nn.Module):
    def __init__(
        self,
        num_entities: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_local_layers: int = 2,
        dropout: float = 0.1,
        num_centroids: int = 32,
    ):
        super().__init__()
        self.tokens = TokenEncoder(num_entities, hidden_dim, dropout)
        self.user_token = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local = nn.TransformerEncoder(
            layer, num_layers=num_local_layers, enable_nested_tensor=False
        )
        self.readout_query = nn.Linear(hidden_dim, hidden_dim)
        self.centroids = nn.Parameter(torch.randn(num_centroids, hidden_dim) / math.sqrt(hidden_dim))
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.combine = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU())
        self.ffn = FeedForward(hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_ACTIONS),
        )

    def forward(
        self,
        entity_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        hop_ids: torch.Tensor,
        node_features: torch.Tensor,
        relative_days: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tokens(
            entity_ids, node_type_ids, relation_ids, hop_ids, node_features, relative_days
        )
        padding = node_type_ids == 0
        user = self.user_token.expand(x.shape[0], 1, -1)
        x = torch.cat([user, x], dim=1)
        padding = torch.cat([padding.new_zeros(x.shape[0], 1), padding], dim=1)
        x = self.local(x, src_key_padding_mask=padding)
        seed = x[:, 0]  # position 0 is the learned user token

        # Attention readout: the seed scores every real token of its subgraph.
        scores = (self.readout_query(seed).unsqueeze(1) * x).sum(-1) / math.sqrt(x.shape[-1])
        weights = scores.masked_fill(padding, float("-inf")).softmax(dim=-1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)

        centroids = self.centroids.unsqueeze(0).expand(x.shape[0], -1, -1)
        global_context, _ = self.global_attention(seed.unsqueeze(1), centroids, centroids)
        combined = self.combine(torch.cat([seed, pooled, global_context.squeeze(1)], dim=-1))
        return self.head(self.ffn(combined))
