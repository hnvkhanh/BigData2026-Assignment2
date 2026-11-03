"""Relational Graph Transformer over database rows (PyTorch-only port).

Ports snap-stanford/relgt ``model.py``, ``local_module.py``, ``codebook.py``
and ``encoders.py``. Each sampled token is a database row described by five
elements, as in the paper:

- features: the row's columns, encoded per table (torch_frame-style per-stype
  column encoders followed by a ResNet);
- node type: the row's table;
- hop distance from the seed row (0, 1, 2, or 3 for fallback tokens);
- relative time: seed time minus row time;
- local structure: a GIN over the sampled subgraph on random node features.

A local Transformer attends over each token set, and a global module attends
to an EMA vector-quantized codebook of seed representations.

torch_frame and PyTorch Geometric are replaced by small equivalents so the
model runs on CPU, CUDA, or MPS. Column features are masked per token at its
seed time, so late columns such as ``merged_at`` are visible only once known.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import LABELS
from .database import NEVER, SECONDS_PER_DAY, TableFeatures


TIMESTAMP_FEATURES = 6


def _known(known_at: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    return known_at < cutoff.unsqueeze(-1)


class TableEncoder(nn.Module):
    """Encode the rows of one table at their seed times.

    Every column becomes one ``channels``-dimensional vector (as torch_frame's
    per-stype encoders do), the vectors are concatenated, and a residual MLP
    maps them to a single row representation (torch_frame's ResNet).
    """

    def __init__(
        self,
        features: TableFeatures,
        channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "numerical", torch.as_tensor(features.numerical), persistent=False
        )
        self.register_buffer(
            "categorical",
            torch.as_tensor(features.categorical, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "timestamps", torch.as_tensor(features.timestamps), persistent=False
        )
        # Column -1 of known_at is a "known from the start" column.
        known_at = np.concatenate(
            [features.known_at, np.full((features.num_rows, 1), -1, np.int64)], axis=1
        )
        self.register_buffer("known_at", torch.as_tensor(known_at), persistent=False)
        for index, tokens in enumerate(features.text):
            self.register_buffer(
                f"text_{index}", torch.as_tensor(tokens), persistent=False
            )
        for stype in ("numerical", "categorical", "text", "timestamp"):
            self.register_buffer(
                f"{stype}_known_at",
                torch.as_tensor(getattr(features, f"{stype}_known_at")),
                persistent=False,
            )
        self.num_text = len(features.text)

        num_numerical = features.numerical.shape[1]
        self.numerical_weight = nn.Parameter(torch.empty(num_numerical, channels))
        self.numerical_bias = nn.Parameter(torch.zeros(num_numerical, channels))
        self.numerical_missing = nn.Parameter(torch.zeros(num_numerical, channels))
        cardinalities = features.categorical_cardinalities
        self.register_buffer(
            "categorical_offsets",
            torch.as_tensor(np.cumsum([0] + cardinalities[:-1]), dtype=torch.long),
            persistent=False,
        )
        self.categorical_embedding = nn.Embedding(max(sum(cardinalities), 1), channels)
        self.text_projection = nn.Parameter(
            torch.empty(self.num_text, channels, channels)
        )
        num_timestamps = features.timestamps.shape[1]
        self.timestamp_weight = nn.Parameter(
            torch.empty(num_timestamps, TIMESTAMP_FEATURES, channels)
        )
        self.timestamp_bias = nn.Parameter(torch.zeros(num_timestamps, channels))

        num_columns = num_numerical + len(cardinalities) + self.num_text + num_timestamps
        self.num_columns = num_columns
        self.input = nn.Linear(max(num_columns, 1) * channels, channels)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(channels, channels),
            )
            for _ in range(num_layers)
        )
        self.output = nn.Sequential(nn.LayerNorm(channels), nn.ReLU(), nn.Linear(channels, channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.numerical_weight, std=0.1)
        nn.init.zeros_(self.numerical_bias)
        nn.init.zeros_(self.numerical_missing)
        nn.init.normal_(self.categorical_embedding.weight, std=0.1)
        if self.num_text:
            nn.init.xavier_uniform_(self.text_projection)
        nn.init.normal_(self.timestamp_weight, std=0.1)
        nn.init.zeros_(self.timestamp_bias)

    def _timestamp_features(
        self, rows: torch.Tensor, cutoff: torch.Tensor
    ) -> torch.Tensor:
        values = self.timestamps[rows]
        known = (values < cutoff.unsqueeze(-1)) & (values != NEVER)
        known &= _known(self.known_at[rows][:, self.timestamp_known_at], cutoff)
        safe = torch.where(known, values, cutoff.unsqueeze(-1))
        age_days = (cutoff.unsqueeze(-1) - safe).float() / SECONDS_PER_DAY
        day = (safe % SECONDS_PER_DAY).float() / SECONDS_PER_DAY * 2 * math.pi
        # 1970-01-01 was a Thursday; the offset only shifts the phase.
        week = (safe % (7 * SECONDS_PER_DAY)).float() / (7 * SECONDS_PER_DAY) * 2 * math.pi
        features = torch.stack(
            [
                known.float(),
                torch.log1p(age_days),
                torch.sin(day),
                torch.cos(day),
                torch.sin(week),
                torch.cos(week),
            ],
            dim=-1,
        )
        return features * known.unsqueeze(-1)

    def forward(
        self, rows: torch.Tensor, cutoff: torch.Tensor, text_embedding: nn.Embedding
    ) -> torch.Tensor:
        """Encode ``rows`` (long [n]) as known at ``cutoff`` (int64 seconds [n])."""
        known_at = self.known_at[rows]
        columns: list[torch.Tensor] = []
        if self.numerical.shape[1]:
            values = self.numerical[rows]
            present = ~torch.isnan(values) & _known(
                known_at[:, self.numerical_known_at], cutoff
            )
            values = torch.where(present, values, torch.zeros_like(values))
            encoded = values.unsqueeze(-1) * self.numerical_weight + self.numerical_bias
            columns.append(
                torch.where(present.unsqueeze(-1), encoded, self.numerical_missing)
            )
        if self.categorical.shape[1]:
            codes = self.categorical[rows].long()
            codes = codes * _known(known_at[:, self.categorical_known_at], cutoff)
            columns.append(self.categorical_embedding(codes + self.categorical_offsets))
        if self.num_text:
            known_text = _known(known_at[:, self.text_known_at], cutoff)
            for index in range(self.num_text):
                tokens = getattr(self, f"text_{index}")[rows].long()
                mask = tokens.ne(0) & known_text[:, index : index + 1]
                vectors = text_embedding(tokens) * mask.unsqueeze(-1)
                mean = vectors.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
                columns.append((mean @ self.text_projection[index]).unsqueeze(1))
        if self.timestamps.shape[1]:
            features = self._timestamp_features(rows, cutoff)
            columns.append(
                torch.einsum("nct,ctd->ncd", features, self.timestamp_weight)
                + self.timestamp_bias
            )
        if not columns:
            hidden = self.input.weight.new_zeros((len(rows), self.input.in_features))
        else:
            hidden = torch.cat(columns, dim=1).flatten(1)
        hidden = self.input(hidden)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class TfsEncoder(nn.Module):
    """Upstream ``NeighborTfsEncoder``: one table encoder per node type."""

    def __init__(
        self,
        tables: list[TableFeatures],
        channels: int,
        text_buckets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channels = channels
        # Text tokens share one hashed vocabulary across columns, as upstream
        # shares one GloVe embedder across text columns.
        self.text_embedding = nn.Embedding(text_buckets, channels, padding_idx=0)
        nn.init.normal_(self.text_embedding.weight, std=0.1)
        with torch.no_grad():
            self.text_embedding.weight[0].zero_()
        self.encoders = nn.ModuleList(
            TableEncoder(table, channels, dropout=dropout) for table in tables
        )

    def forward(
        self, node_types: torch.Tensor, node_rows: torch.Tensor, cutoff: torch.Tensor
    ) -> torch.Tensor:
        batch, tokens = node_types.shape
        flat_types = node_types.reshape(-1)
        flat_rows = node_rows.reshape(-1).long()
        flat_cutoff = cutoff.unsqueeze(1).expand(batch, tokens).reshape(-1)
        output = self.text_embedding.weight.new_zeros((batch * tokens, self.channels))
        for node_type, encoder in enumerate(self.encoders):
            positions = torch.nonzero(flat_types == node_type, as_tuple=True)[0]
            if len(positions):
                output[positions] = encoder(
                    flat_rows[positions], flat_cutoff[positions], self.text_embedding
                ).to(output.dtype)
        return output.view(batch, tokens, self.channels)


class TimeEncoder(nn.Module):
    """Upstream ``NeighborTimeEncoder``: PyG positional encoding + linear."""

    def __init__(self, channels: int, base_frequency: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer(
            "frequency", torch.logspace(0, 1, channels // 2, base_frequency)
        )
        self.linear = nn.Linear(2 * (channels // 2), channels)
        self.mask_vector = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.mask_vector, mean=0.0, std=0.02)

    def forward(self, relative_days: torch.Tensor) -> torch.Tensor:
        angles = relative_days.unsqueeze(-1) * self.frequency
        encoded = self.linear(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))
        masked = (relative_days < 0).unsqueeze(-1)
        return torch.where(masked, self.mask_vector, encoded)


class GINConv(nn.Module):
    """PyG ``GINConv`` with a trainable epsilon and sum aggregation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregated = torch.zeros_like(x).index_add_(0, edge_index[1], x[edge_index[0]])
        return self.mlp((1 + self.eps) * x + aggregated)


class GNNPEEncoder(nn.Module):
    """Upstream ``GNNPEEncoder`` with random scalar inputs (``pe_dim=0``)."""

    def __init__(self, channels: int, num_layers: int = 4) -> None:
        super().__init__()
        hidden = max(channels // 4, 1)
        self.input_proj = nn.Linear(1, hidden)
        self.convs = nn.ModuleList(GINConv(hidden) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(num_layers))
        self.final_transform = nn.Linear(hidden, channels)

    def forward(
        self, edge_index: torch.Tensor, batch_size: int, num_tokens: int
    ) -> torch.Tensor:
        total = batch_size * num_tokens
        x = self.input_proj(
            torch.randn(total, 1, device=edge_index.device)
        )
        for conv, norm in zip(self.convs, self.norms):
            x = x + F.relu(norm(conv(x, edge_index)))
        return self.final_transform(x).view(batch_size, num_tokens, -1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.ffn_net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_in(x.transpose(1, 2)).transpose(1, 2)
        x = self.ffn_net(x)
        return self.bn_out(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Pre-norm self-attention block from upstream ``local_module.py``."""

    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        dropout: float,
        attention_dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        head_dim = dim // self.num_heads
        normed = self.self_attention_norm(x)
        q, k, v = (
            projection(normed).view(batch, length, self.num_heads, head_dim).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        # Upstream applies attention dropout in eval too; only training does here.
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attention_dropout if self.training else 0.0
        )
        attention = attention.transpose(1, 2).reshape(batch, length, dim)
        x = x + self.self_attention_dropout(self.out_proj(attention))
        return x + self.ffn(self.ffn_norm(x))


class LocalModule(nn.Module):
    """Transformer over a token set with attention readout to the seed."""

    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.att_embeddings_nope = nn.Linear(channels, channels)
        self.layers = nn.ModuleList(
            EncoderLayer(channels, 2 * channels, dropout, attention_dropout, num_heads)
            for _ in range(num_layers)
        )
        self.final_ln = nn.LayerNorm(channels)
        self.attn_layer = nn.Linear(2 * channels, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.att_embeddings_nope(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        node, neighbors = x[:, :1], x[:, 1:]
        weights = torch.softmax(
            self.attn_layer(torch.cat([node.expand_as(neighbors), neighbors], dim=2)),
            dim=1,
        )
        return node[:, 0] + (neighbors * weights).sum(dim=1)


class VectorQuantizerEMA(nn.Module):
    """Upstream ``codebook.py`` (adapted from GOAT)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.register_buffer("embedding", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer(
            "embedding_output", torch.randn(num_embeddings, embedding_dim)
        )
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.bn(x)
        distances = (
            inputs.pow(2).sum(1, keepdim=True)
            + self.embedding.pow(2).sum(1)
            - 2 * inputs @ self.embedding.t()
        )
        indices = distances.argmin(dim=1)
        if self.training:
            encodings = F.one_hot(indices, self.num_embeddings).to(inputs.dtype)
            self.ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            total = self.ema_cluster_size.sum()
            self.ema_cluster_size.copy_(
                (self.ema_cluster_size + 1e-5)
                / (total + self.num_embeddings * 1e-5)
                * total
            )
            self.ema_w.mul_(self.decay).add_(encodings.t() @ inputs, alpha=1 - self.decay)
            self.embedding.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
            std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(0)
            self.embedding_output.copy_(self.embedding * std + self.bn.running_mean)
        return indices


class RelGTLayer(nn.Module):
    """Local token attention plus global attention to codebook centroids."""

    def __init__(
        self,
        channels: int,
        global_dim: int,
        num_layers: int,
        num_heads: int,
        num_seed_nodes: int,
        num_centroids: int,
        dropout: float,
        conv_type: str,
    ) -> None:
        super().__init__()
        if conv_type not in {"local", "global", "full"}:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        self.conv_type = conv_type
        self.channels = channels
        self.num_heads = num_heads
        self.attention_dropout = dropout
        self.num_centroids = num_centroids
        self.local_module = LocalModule(channels, num_layers, num_heads, dropout, dropout)
        self.layer_norm_local = nn.LayerNorm(channels)
        if conv_type != "local":
            self.vq = VectorQuantizerEMA(num_centroids, global_dim)
            self.register_buffer(
                "c_idx", torch.randint(0, num_centroids, (num_seed_nodes,))
            )
            self.lin_proj_g = nn.Linear(channels, global_dim)
            self.lin_key_g = nn.Linear(global_dim, channels)
            self.lin_query_g = nn.Linear(global_dim, channels)
            self.lin_value_g = nn.Linear(global_dim, channels)
            self.layer_norm_global = nn.LayerNorm(channels)

    def global_forward(self, x: torch.Tensor, seed_indices: torch.Tensor) -> torch.Tensor:
        heads = self.num_heads
        projected = self.lin_proj_g(x)
        # Cloned because the EMA update below rewrites the codebook in place.
        centroids = self.vq.embedding_output.detach().clone()
        q = self.lin_query_g(projected).view(len(x), heads, -1).transpose(0, 1)
        k = self.lin_key_g(centroids).view(self.num_centroids, heads, -1).transpose(0, 1)
        v = self.lin_value_g(centroids).view(self.num_centroids, heads, -1).transpose(0, 1)
        dots = q @ k.transpose(1, 2) / math.sqrt(self.channels)
        counts = torch.bincount(self.c_idx, minlength=self.num_centroids)
        attention = torch.softmax(dots + torch.log(counts.to(dots.dtype)), dim=-1)
        attention = F.dropout(attention, p=self.attention_dropout, training=self.training)
        out = (attention @ v).transpose(0, 1).reshape(len(x), -1)
        if self.training:
            with torch.no_grad():
                self.c_idx[seed_indices] = self.vq.update(projected.detach())
        return out

    def forward(
        self, tokens: torch.Tensor, seed: torch.Tensor, seed_indices: torch.Tensor
    ) -> torch.Tensor:
        outputs = []
        if self.conv_type != "global":
            outputs.append(self.layer_norm_local(self.local_module(tokens)))
        if self.conv_type != "local":
            outputs.append(self.layer_norm_global(self.global_forward(seed, seed_indices)))
        return torch.cat(outputs, dim=1)


class RelGT(nn.Module):
    """Upstream ``RelGT`` with a six-label head for the weekly activity task."""

    ABLATIONS = ("none", "type", "hop", "time", "tfs", "gnn")

    def __init__(
        self,
        tables: list[TableFeatures],
        num_seed_nodes: int,
        channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_centroids: int = 512,
        dropout: float = 0.1,
        conv_type: str = "full",
        ablate: str = "none",
        text_buckets: int = 2**16,
        max_hop: int = 3,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads.")
        if ablate not in self.ABLATIONS:
            raise ValueError(f"ablate must be one of {self.ABLATIONS}.")
        self.ablate = ablate
        self.type_encoder = nn.Embedding(len(tables) + 1, channels)
        self.hop_encoder = nn.Embedding(max_hop + 2, channels)
        self.time_encoder = TimeEncoder(channels)
        self.tfs_encoder = TfsEncoder(tables, channels, text_buckets, dropout)
        self.pe_encoder = GNNPEEncoder(channels)
        self.norms = nn.ModuleDict(
            {f"{name}_norm": nn.LayerNorm(channels) for name in self.ABLATIONS[1:]}
        )
        components = 5 if ablate == "none" else 4
        self.in_mixture = nn.Sequential(
            nn.Linear(components * channels, 2 * channels),
            nn.ReLU(),
            nn.Linear(2 * channels, channels),
        )
        self.conv = RelGTLayer(
            channels,
            channels // 2,
            num_layers,
            num_heads,
            num_seed_nodes,
            num_centroids,
            dropout,
            conv_type,
        )
        width = 2 if conv_type == "full" else 1
        self.ff = nn.Sequential(
            nn.BatchNorm1d(width * channels),
            nn.Linear(width * channels, 2 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
            nn.Dropout(dropout),
            nn.BatchNorm1d(channels),
        )
        # PyG MLP(channels, channels, out, num_layers=2) with batch norm.
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, len(LABELS)),
        )

    def forward(
        self,
        node_types: torch.Tensor,
        node_rows: torch.Tensor,
        hops: torch.Tensor,
        relative_days: torch.Tensor,
        cutoff_seconds: torch.Tensor,
        seed_indices: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return six logits per seed; inputs are ``[batch, K]`` token tensors."""
        batch, tokens = node_types.shape
        components = {
            "type": self.type_encoder(node_types.long()),
            "hop": self.hop_encoder(hops.long() + 1),
            "time": self.time_encoder(relative_days.float()),
            "tfs": self.tfs_encoder(node_types, node_rows, cutoff_seconds),
            "gnn": self.pe_encoder(edge_index, batch, tokens),
        }
        x_set = torch.cat(
            [
                self.norms[f"{name}_norm"](value)
                for name, value in components.items()
                if name != self.ablate
            ],
            dim=-1,
        )
        x_set = self.in_mixture(x_set)
        hidden = self.ff(self.conv(x_set, x_set[:, 0], seed_indices))
        return self.head(hidden)
