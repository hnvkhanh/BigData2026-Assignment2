"""Relation-aware message passing for the sampled temporal user graphs."""

from __future__ import annotations

import math

import torch
from torch import nn

from .data import LABELS


class RelationalMessageLayer(nn.Module):
    """Apply relation-specific messages on the sampled user-centered graph."""

    def __init__(self, hidden_dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.relation_weights = nn.Parameter(
            torch.empty(num_relations, hidden_dim, hidden_dim)
        )
        for relation_weights in self.relation_weights:
            nn.init.xavier_uniform_(relation_weights)
        self.root_self = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_self = nn.Linear(hidden_dim, hidden_dim)
        self.root_to_neighbor = nn.Linear(hidden_dim, hidden_dim)
        self.root_norm = nn.LayerNorm(hidden_dim)
        self.neighbor_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        root: torch.Tensor,
        neighbors: torch.Tensor,
        relation_ids: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.relation_weights[relation_ids]
        messages = torch.einsum("bnh,bnhd->bnd", neighbors, weights)
        mask = neighbor_mask.unsqueeze(-1)
        messages = messages * mask
        degrees = mask.sum(dim=1).clamp(min=1)
        pooled_messages = messages.sum(dim=1) / degrees

        updated_root = torch.nn.functional.gelu(
            self.root_norm(self.root_self(root) + pooled_messages)
        )
        root_messages = self.root_to_neighbor(root).unsqueeze(1)
        updated_neighbors = torch.nn.functional.gelu(
            self.neighbor_norm(
                self.neighbor_self(neighbors) + root_messages
            )
        )
        updated_root = self.dropout(updated_root)
        updated_neighbors = self.dropout(updated_neighbors) * mask
        return updated_root, updated_neighbors


class RelationalGNN(nn.Module):
    """Classify activity from relation-aware messages over sampled ego graphs.

    Each snapshot is represented as a user root connected to its typed,
    relation-labeled entity and repository neighbors. Hop IDs distinguish
    direct activity neighbors from sampled repository context.
    """

    def __init__(
        self,
        num_entities: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(4, hidden_dim)
        self.relation_embedding = nn.Embedding(len(LABELS) + 2, hidden_dim)
        self.hop_embedding = nn.Embedding(3, hidden_dim)
        self.feature_projection = nn.Linear(4, hidden_dim)
        self.time_projection = nn.Linear(7, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.root = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.root, mean=0.0, std=hidden_dim**-0.5)
        self.layers = nn.ModuleList(
            RelationalMessageLayer(hidden_dim, len(LABELS) + 2, dropout)
            for _ in range(num_layers)
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
        return torch.cat([log_age, torch.sin(periods), torch.cos(periods)], dim=-1)

    def forward(
        self,
        entity_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        hop_ids: torch.Tensor,
        node_features: torch.Tensor,
        relative_days: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_mask = entity_ids.ne(0)
        neighbors = self.token_norm(
            self.entity_embedding(entity_ids.to(torch.long))
            + self.node_type_embedding(node_type_ids.to(torch.long))
            + self.relation_embedding(relation_ids.to(torch.long))
            + self.hop_embedding(hop_ids.to(torch.long))
            + self.feature_projection(node_features)
            + self.time_projection(self._time_features(relative_days))
        )
        mask = neighbor_mask.unsqueeze(-1)
        neighbors = neighbors * mask
        root = self.root.unsqueeze(0).expand(entity_ids.shape[0], -1)
        for layer in self.layers:
            root, neighbors = layer(root, neighbors, relation_ids, neighbor_mask)

        degrees = mask.sum(dim=1).clamp(min=1)
        pooled_neighbors = neighbors.sum(dim=1) / degrees
        return self.output(torch.cat([root, pooled_neighbors], dim=-1))
