"""Relation-aware message passing over the sampled user neighborhood.

The graph is rebuilt from the token arrays (see data.py). A virtual user node is linked to every
token in both directions; messages from a token to the user use one weight matrix per relation
(the six activities and repository membership). A push object is a repository, so it is linked
both ways to the repository membership token with the same entity id. Messages are mean-aggregated
per edge class and added to a self transform. The user node and the mean of all tokens feed the head.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import NUM_ACTIONS, NUM_RELATIONS, TokenEncoder

NUM_TOKEN_RELATIONS = NUM_RELATIONS - 1  # relation ids 1..7


class RelationalLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_loop = nn.Linear(hidden_dim, hidden_dim)
        self.token_to_user = nn.Linear(
            hidden_dim, NUM_TOKEN_RELATIONS * hidden_dim, bias=False
        )
        self.user_to_token = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.same_entity = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    @staticmethod
    def _mean(adjacency: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adjacency @ messages / degree

    def forward(
        self,
        x: torch.Tensor,
        relations: torch.Tensor,
        to_user: torch.Tensor,
        to_token: torch.Tensor,
        same_entity: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, _ = x.shape
        per_relation = self.token_to_user(x).view(
            batch, nodes, NUM_TOKEN_RELATIONS, self.hidden_dim
        )
        picked = per_relation.gather(
            2, relations.view(batch, nodes, 1, 1).expand(-1, -1, 1, self.hidden_dim)
        ).squeeze(2)
        update = (
            self.self_loop(x)
            + self._mean(to_user, picked)
            + self._mean(to_token, self.user_to_token(x))
            + self._mean(same_entity, self.same_entity(x))
        )
        return self.norm(x + self.dropout(torch.relu(update)))


class RelationalGNN(nn.Module):
    def __init__(
        self,
        num_entities: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokens = TokenEncoder(num_entities, hidden_dim, dropout)
        self.user_token = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.layers = nn.ModuleList(
            RelationalLayer(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
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
        tokens = self.tokens(
            entity_ids, node_type_ids, relation_ids, hop_ids, node_features, relative_days
        )
        batch, count, _ = tokens.shape
        x = torch.cat([self.user_token.expand(batch, 1, -1), tokens], dim=1)
        dtype = x.dtype

        valid = node_type_ids != 0
        # Node 0 is the user; nodes 1.. are tokens. Padding tokens (relation 0) use relation 1's
        # weights but have no edges.
        relations = torch.cat(
            [relation_ids.new_ones(batch, 1), relation_ids.clamp(min=1)], dim=1
        ).long() - 1

        to_user = torch.zeros(batch, count + 1, count + 1, dtype=dtype, device=x.device)
        to_user[:, 0, 1:] = valid.to(dtype)
        to_token = torch.zeros_like(to_user)
        to_token[:, 1:, 0] = valid.to(dtype)

        # Hop-1 repositories (push objects) <-> hop-2 repository tokens with the same id.
        is_repo = node_type_ids == 3
        link = (
            (entity_ids.unsqueeze(2) == entity_ids.unsqueeze(1))
            & is_repo.unsqueeze(2)
            & is_repo.unsqueeze(1)
            & (hop_ids.unsqueeze(2) != hop_ids.unsqueeze(1))
        )
        same_entity = torch.zeros_like(to_user)
        same_entity[:, 1:, 1:] = link.to(dtype)

        for layer in self.layers:
            x = layer(x, relations, to_user, to_token, same_entity)
        mask = torch.cat([valid.new_ones(batch, 1), valid], dim=1).unsqueeze(-1).to(dtype)
        mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(torch.cat([x[:, 0], mean], dim=-1))
