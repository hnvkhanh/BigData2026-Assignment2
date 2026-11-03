"""A compact RelGT-inspired encoder for sampled temporal neighborhoods."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .data import LABELS
from .gnn_model import user_embedding_index


class RelGTInspired(nn.Module):
    """Encode typed user/entity/repository neighborhoods with attention.

    The encoder follows RelGT's feature/type/hop/time/structure tokenization and
    local-plus-global attention pattern, but uses this assignment's sampled
    sampled two-hop graph instead of the upstream RelBench data pipeline.
    """

    def __init__(
        self,
        num_entities: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_local_layers: int = 2,
        num_centroids: int = 8,
        dropout: float = 0.1,
        known_user_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(4, hidden_dim)
        self.relation_embedding = nn.Embedding(len(LABELS) + 2, hidden_dim)
        self.hop_embedding = nn.Embedding(3, hidden_dim)
        self.feature_projection = nn.Linear(4, hidden_dim)
        self.time_projection = nn.Linear(7, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        # Users seen in training get a learned vector added to the user token;
        # index 0 (zero vector) is shared by every other user.
        known = (
            np.unique(np.asarray(known_user_ids, dtype=np.int64))
            if known_user_ids is not None
            else np.empty(0, dtype=np.int64)
        )
        self.register_buffer("known_user_ids", torch.as_tensor(known))
        self.user_embedding = (
            nn.Embedding(len(known) + 1, hidden_dim, padding_idx=0)
            if len(known)
            else None
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.local_encoder = nn.TransformerEncoder(
            layer, num_layers=num_local_layers, enable_nested_tensor=False
        )
        self.centroids = nn.Parameter(torch.empty(num_centroids, hidden_dim))
        nn.init.normal_(self.centroids, mean=0.0, std=hidden_dim**-0.5)
        self.centroid_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.global_readout = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(LABELS)),
        )

    @staticmethod
    def _time_features(relative_days: torch.Tensor) -> torch.Tensor:
        age = relative_days.clamp(min=0)
        log_age = torch.log1p(age).unsqueeze(-1)
        periods = age.unsqueeze(-1) * (2 * math.pi) / age.new_tensor(
            [7.0, 30.0, 90.0]
        )
        return torch.cat(
            [log_age, torch.sin(periods), torch.cos(periods)], dim=-1
        )

    def forward(
        self,
        entity_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        hop_ids: torch.Tensor,
        node_features: torch.Tensor,
        relative_days: torch.Tensor,
        user_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return six logits for each user in a batch."""
        return self.output(
            self.encode(
                entity_ids,
                node_type_ids,
                relation_ids,
                hop_ids,
                node_features,
                relative_days,
                user_ids,
            )
        )

    def encode(
        self,
        entity_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        hop_ids: torch.Tensor,
        node_features: torch.Tensor,
        relative_days: torch.Tensor,
        user_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return each user's local-plus-global relational graph representation."""
        if (self.user_embedding is None) != (user_ids is None):
            raise ValueError(
                "user_ids must be given exactly when the model has known_user_ids."
            )
        batch_size = entity_ids.shape[0]
        device = entity_ids.device
        is_neighbor = entity_ids.ne(0)

        node_types = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                node_type_ids.to(torch.long),
            ],
            dim=1,
        )
        relation_ids = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                relation_ids.to(torch.long),
            ],
            dim=1,
        )
        hop_ids = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                hop_ids.to(torch.long),
            ],
            dim=1,
        )

        entity_tokens = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                entity_ids.to(torch.long),
            ],
            dim=1,
        )
        structural_tokens = torch.cat(
            [
                node_features.new_zeros((batch_size, 1, node_features.shape[-1])),
                node_features,
            ],
            dim=1,
        )
        age_tokens = torch.cat(
            [
                relative_days.new_zeros((batch_size, 1)),
                relative_days,
            ],
            dim=1,
        )
        local_tokens = self.token_norm(
            self.entity_embedding(entity_tokens)
            + self.node_type_embedding(node_types)
            + self.relation_embedding(relation_ids)
            + self.hop_embedding(hop_ids)
            + self.feature_projection(structural_tokens)
            + self.time_projection(self._time_features(age_tokens))
        )
        if self.user_embedding is not None:
            user_vectors = self.user_embedding(
                user_embedding_index(self.known_user_ids, user_ids)
            )
            local_tokens = torch.cat(
                [local_tokens[:, :1] + user_vectors.unsqueeze(1), local_tokens[:, 1:]],
                dim=1,
            )

        padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
                ~is_neighbor,
            ],
            dim=1,
        )
        local_tokens = self.local_encoder(
            local_tokens, src_key_padding_mask=padding_mask
        )

        centroid_queries = self.centroids.unsqueeze(0).expand(batch_size, -1, -1)
        global_tokens, _ = self.centroid_attention(
            centroid_queries,
            local_tokens,
            local_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        global_user, _ = self.global_readout(
            local_tokens[:, :1],
            global_tokens,
            global_tokens,
            need_weights=False,
        )
        user_representation = torch.cat(
            [local_tokens[:, 0], global_user[:, 0]], dim=-1
        )
        return user_representation
