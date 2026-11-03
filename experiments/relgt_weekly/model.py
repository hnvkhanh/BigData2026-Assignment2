"""RelGT-style encoder whose tokens are weekly activity vectors."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from experiments.relgt.data import LABELS
from experiments.relgt.gnn_model import RelationalMessageLayer, user_embedding_index
from experiments.relgt.model import RelGTInspired


class WeeklyRelGT(RelGTInspired):
    """Local transformer + centroid attention + global readout over week tokens.

    The sequence is a root token for the user followed by one token per week.
    Each week token is a learned projection of that week's activity vector plus
    an embedding of how many weeks before the cutoff it is, so the model sees
    weekly totals and never an event's day or time of day. The transformer,
    centroids, readout, and output head are the ones from ``RelGTInspired``.
    """

    def __init__(
        self,
        num_weeks: int,
        num_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_local_layers: int = 2,
        num_centroids: int = 8,
        dropout: float = 0.1,
        known_user_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            num_entities=1,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_local_layers=num_local_layers,
            num_centroids=num_centroids,
            dropout=dropout,
            known_user_ids=known_user_ids,
        )
        # The graph-token embeddings do not apply to week tokens.
        for name in (
            "entity_embedding",
            "node_type_embedding",
            "relation_embedding",
            "hop_embedding",
            "feature_projection",
            "time_projection",
        ):
            delattr(self, name)
        self.num_weeks = num_weeks
        # Index 0 is the user root; index w + 1 is the week ending w weeks ago.
        self.week_embedding = nn.Embedding(num_weeks + 1, hidden_dim)
        self.week_projection = nn.Linear(num_features, hidden_dim)

    def forward(
        self, week_features: torch.Tensor, user_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return six logits for each user in a batch."""
        return self.output(self.encode(week_features, user_ids))

    def encode(
        self, week_features: torch.Tensor, user_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (self.user_embedding is None) != (user_ids is None):
            raise ValueError(
                "user_ids must be given exactly when the model has known_user_ids."
            )
        batch_size, num_weeks, _ = week_features.shape
        if num_weeks != self.num_weeks:
            raise ValueError(f"Expected {self.num_weeks} weeks, got {num_weeks}.")
        device = week_features.device

        positions = torch.arange(num_weeks + 1, device=device)
        week_tokens = self.week_projection(week_features) + self.week_embedding(
            positions[1:]
        )
        root = self.week_embedding(positions[:1]).expand(batch_size, -1)
        if self.user_embedding is not None:
            root = root + self.user_embedding(
                user_embedding_index(self.known_user_ids, user_ids)
            )
        tokens = self.token_norm(torch.cat([root.unsqueeze(1), week_tokens], dim=1))

        tokens = self.local_encoder(tokens)
        centroid_queries = self.centroids.unsqueeze(0).expand(batch_size, -1, -1)
        global_tokens, _ = self.centroid_attention(
            centroid_queries, tokens, tokens, need_weights=False
        )
        global_user, _ = self.global_readout(
            tokens[:, :1], global_tokens, global_tokens, need_weights=False
        )
        return torch.cat([tokens[:, 0], global_user[:, 0]], dim=-1)


def _known_users_and_embedding(
    known_user_ids: np.ndarray | None, hidden_dim: int
) -> tuple[torch.Tensor, nn.Embedding | None]:
    """Sorted known ids and an embedding where index 0 is shared by unseen users."""
    known = (
        np.unique(np.asarray(known_user_ids, dtype=np.int64))
        if known_user_ids is not None
        else np.empty(0, dtype=np.int64)
    )
    embedding = (
        nn.Embedding(len(known) + 1, hidden_dim, padding_idx=0) if len(known) else None
    )
    return torch.as_tensor(known), embedding


def _head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, len(LABELS)),
    )


class WeeklyNN(nn.Module):
    """MLP over the weekly vectors concatenated into one flat input."""

    def __init__(
        self,
        num_weeks: int,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        known_user_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.num_weeks = num_weeks
        known, self.user_embedding = _known_users_and_embedding(known_user_ids, hidden_dim)
        self.register_buffer("known_user_ids", known)
        flat = num_weeks * num_features
        self.input = nn.Sequential(nn.LayerNorm(flat), nn.Linear(flat, hidden_dim))
        self.hidden = nn.Sequential(
            *[
                layer
                for _ in range(num_layers - 1)
                for layer in (
                    nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
                )
            ]
        )
        self.output = _head(hidden_dim, hidden_dim, dropout)

    def forward(
        self, week_features: torch.Tensor, user_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (self.user_embedding is None) != (user_ids is None):
            raise ValueError(
                "user_ids must be given exactly when the model has known_user_ids."
            )
        if week_features.shape[1] != self.num_weeks:
            raise ValueError(
                f"Expected {self.num_weeks} weeks, got {week_features.shape[1]}."
            )
        hidden = self.input(week_features.flatten(1))
        if self.user_embedding is not None:
            hidden = hidden + self.user_embedding(
                user_embedding_index(self.known_user_ids, user_ids)
            )
        return self.output(self.hidden(hidden))


class WeeklyGNN(nn.Module):
    """Message passing on a star graph: the user root and one node per week.

    Each week node sends its message through a weight matrix specific to its age
    (the "relation" is how many weeks before the cutoff it is), the root pools
    the messages, and the root sends its state back to every week node.
    """

    def __init__(
        self,
        num_weeks: int,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        known_user_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.num_weeks = num_weeks
        known, self.user_embedding = _known_users_and_embedding(known_user_ids, hidden_dim)
        self.register_buffer("known_user_ids", known)
        self.register_buffer("week_ids", torch.arange(num_weeks))
        self.week_projection = nn.Linear(num_features, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.root = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.root, mean=0.0, std=hidden_dim**-0.5)
        self.layers = nn.ModuleList(
            RelationalMessageLayer(hidden_dim, num_weeks, dropout)
            for _ in range(num_layers)
        )
        self.output = _head(hidden_dim * 2, hidden_dim, dropout)

    def forward(
        self, week_features: torch.Tensor, user_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (self.user_embedding is None) != (user_ids is None):
            raise ValueError(
                "user_ids must be given exactly when the model has known_user_ids."
            )
        batch_size, num_weeks, _ = week_features.shape
        if num_weeks != self.num_weeks:
            raise ValueError(f"Expected {self.num_weeks} weeks, got {num_weeks}.")
        weeks = self.token_norm(self.week_projection(week_features))
        root = self.root.unsqueeze(0).expand(batch_size, -1)
        if self.user_embedding is not None:
            root = root + self.user_embedding(
                user_embedding_index(self.known_user_ids, user_ids)
            )
        week_ids = self.week_ids.unsqueeze(0).expand(batch_size, -1)
        mask = torch.ones((batch_size, num_weeks), dtype=torch.bool, device=weeks.device)
        for layer in self.layers:
            root, weeks = layer(root, weeks, week_ids, mask)
        return self.output(torch.cat([root, weeks.mean(dim=1)], dim=-1))
