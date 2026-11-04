"""LSTM, Informer and FEDformer forecasters ported from DM2026-Final.

The layers follow ``lstm_v2.py``, ``informer_updated.py`` and
``fedformer_updated.py`` of https://github.com/hnvkhanh/DM2026-Final. The
changes for this task:

- the region embedding is a per-user embedding (users unseen in training share
  a zero vector, as in ``experiments.relgt.gnn_model``);
- the five forecast horizons become the six action labels: the LSTM is
  conditioned on a label embedding instead of a horizon embedding, the Informer
  decodes one query token per label, and the FEDformer head emits one value per
  label;
- every model returns logits of shape ``(batch, len(LABELS))`` for BCE.

All models take ``(sequence, aux, user_ids)``; only the LSTM uses ``aux``, as in
the original code.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from experiments.relgt.data import LABELS
from experiments.relgt.gnn_model import user_embedding_index


class UserEmbedding(nn.Module):
    """Learned vector per training user; unknown users (and ``dim=0``) get zeros."""

    def __init__(self, known_user_ids: np.ndarray | None, dim: int) -> None:
        super().__init__()
        known = (
            np.unique(np.asarray(known_user_ids, dtype=np.int64))
            if known_user_ids is not None and dim > 0
            else np.empty(0, dtype=np.int64)
        )
        self.register_buffer("known_user_ids", torch.as_tensor(known))
        self.dim = dim if len(known) else 0
        self.embedding = nn.Embedding(len(known) + 1, dim, padding_idx=0) if len(known) else None

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            return torch.zeros(len(user_ids), 0, device=user_ids.device)
        return self.embedding(user_embedding_index(self.known_user_ids, user_ids))


# --- LSTM (lstm_v2.py) --------------------------------------------------------


class LSTMForecaster(nn.Module):
    """LSTM over the window; the head is run once per label (``LSTMRegressor``)."""

    def __init__(
        self,
        input_size: int,
        aux_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        user_emb_dim: int = 32,
        label_emb_dim: int = 4,
        known_user_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.user_emb = UserEmbedding(known_user_ids, user_emb_dim)
        self.label_emb = nn.Embedding(len(LABELS), label_emb_dim)
        head_in = hidden_size + self.user_emb.dim + label_emb_dim + aux_size
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, seq: torch.Tensor, aux: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(seq)
        # One head input per (example, label): [hidden, user, label, aux].
        context = torch.cat([hidden[-1], self.user_emb(user_ids)], dim=1)
        num_labels = len(LABELS)
        combined = torch.cat(
            [
                context.unsqueeze(1).expand(-1, num_labels, -1),
                self.label_emb.weight.unsqueeze(0).expand(seq.size(0), -1, -1),
                aux.unsqueeze(1).expand(-1, num_labels, -1),
            ],
            dim=2,
        )
        return self.head(combined).squeeze(-1)


# --- Informer (informer_updated.py) ----------------------------------------------


def causal_mask(batch_size, length, source_length=None, device=None):
    source_length = length if source_length is None else source_length
    return torch.triu(
        torch.ones(batch_size, 1, length, source_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.token_conv(x.transpose(1, 2)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, input_size, d_model, dropout, max_len=1024):
        super().__init__()
        self.value_embedding = TokenEmbedding(input_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class FullAttention(nn.Module):
    def __init__(self, mask_flag, scale=None, attention_dropout=0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size, query_len, _, depth = queries.shape
        scale = self.scale or 1.0 / math.sqrt(depth)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = causal_mask(batch_size, query_len, keys.size(1), device=queries.device)
            scores = scores.masked_fill(attn_mask, -torch.inf)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attention, values)
        return values.contiguous(), None


class ProbAttention(nn.Module):
    """ProbSparse self-attention: only the top-u most "active" queries attend."""

    def __init__(self, mask_flag, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_qk(self, queries, keys, sample_k, n_top):
        batch_size, heads, key_len, depth = keys.shape
        _, _, query_len, _ = queries.shape
        index_sample = torch.randint(key_len, (query_len, sample_k), device=queries.device)
        keys_expand = keys.unsqueeze(-3).expand(batch_size, heads, query_len, key_len, depth)
        keys_sample = keys_expand[
            :, :, torch.arange(query_len, device=queries.device).unsqueeze(1), index_sample, :
        ]
        sampled_scores = torch.matmul(queries.unsqueeze(-2), keys_sample.transpose(-2, -1)).squeeze(
            -2
        )
        sparsity = sampled_scores.max(dim=-1).values - sampled_scores.mean(dim=-1)
        top_index = sparsity.topk(n_top, sorted=False).indices
        queries_top = queries[
            torch.arange(batch_size, device=queries.device)[:, None, None],
            torch.arange(heads, device=queries.device)[None, :, None],
            top_index,
            :,
        ]
        return torch.matmul(queries_top, keys.transpose(-2, -1)), top_index

    def _initial_context(self, values, query_len):
        batch_size, heads, value_len, depth = values.shape
        if not self.mask_flag:
            context = values.mean(dim=-2).unsqueeze(-2)
            return context.expand(batch_size, heads, query_len, depth).clone()
        if query_len != value_len:
            raise ValueError("Causal ProbAttention requires query_len == value_len.")
        return values.cumsum(dim=-2)

    def _update_context(self, context, values, scores, index, query_len):
        batch_size, heads, value_len, _ = values.shape
        if self.mask_flag:
            mask = torch.ones(query_len, value_len, dtype=torch.bool, device=values.device).triu(1)
            prob_mask = mask[None, None, :, :].expand(batch_size, heads, query_len, value_len)[
                torch.arange(batch_size, device=values.device)[:, None, None],
                torch.arange(heads, device=values.device)[None, :, None],
                index,
                :,
            ]
            scores = scores.masked_fill(prob_mask, -torch.inf)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        context[
            torch.arange(batch_size, device=values.device)[:, None, None],
            torch.arange(heads, device=values.device)[None, :, None],
            index,
            :,
        ] = torch.matmul(attention, values).type_as(context)
        return context

    def forward(self, queries, keys, values, attn_mask=None):
        _, query_len, _, depth = queries.shape
        key_len = keys.size(1)
        queries, keys, values = (t.transpose(1, 2) for t in (queries, keys, values))
        sample_k = min(key_len, max(1, self.factor * int(math.ceil(math.log(max(key_len, 2))))))
        n_top = min(query_len, max(1, self.factor * int(math.ceil(math.log(max(query_len, 2))))))
        scores_top, top_index = self._prob_qk(queries, keys, sample_k, n_top)
        scores_top = scores_top * (self.scale or 1.0 / math.sqrt(depth))
        context = self._initial_context(values, query_len)
        context = self._update_context(context, values, scores_top, top_index, query_len)
        return context.transpose(1, 2).contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("--d-model must be divisible by --n-heads.")
        self.inner_attention = attention
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size, query_len, _ = queries.shape
        key_len = keys.size(1)
        queries = self.query_projection(queries).view(
            batch_size, query_len, self.n_heads, self.d_head
        )
        keys = self.key_projection(keys).view(batch_size, key_len, self.n_heads, self.d_head)
        values = self.value_projection(values).view(batch_size, key_len, self.n_heads, self.d_head)
        out, attention = self.inner_attention(queries, keys, values, attn_mask)
        return self.out_projection(out.view(batch_size, query_len, -1)), attention


class ConvLayer(nn.Module):
    """Distilling layer: halves the encoder sequence length between attention layers."""

    def __init__(self, d_model):
        super().__init__()
        self.down_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.activation(self.norm(self.down_conv(x.transpose(1, 2))))
        return self.max_pool(x).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attention = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y), attention


class Encoder(nn.Module):
    def __init__(self, attention_layers, conv_layers=None):
        super().__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None
        self.norm = nn.LayerNorm(attention_layers[0].norm1.normalized_shape)

    def forward(self, x, attn_mask=None):
        if self.conv_layers is not None:
            for attention_layer, conv_layer in zip(self.attention_layers[:-1], self.conv_layers):
                x, _ = attention_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x, _ = self.attention_layers[-1](x, attn_mask=attn_mask)
        else:
            for attention_layer in self.attention_layers:
                x, _ = attention_layer(x, attn_mask=attn_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff, dropout, activation):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0]))
        x = self.norm2(
            x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        )
        y = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(layers[0].norm1.normalized_shape)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        return self.norm(x)


class InformerForecaster(nn.Module):
    """Informer encoder-decoder; the decoder's "future" tokens are one per label.

    As in the original, the decoder input is the last ``label_days`` of the
    window followed by zero tokens plus a learned per-step (here per-label)
    embedding, under a causal mask, so label ``j`` attends to labels before it
    in ``LABELS`` order.
    """

    def __init__(
        self,
        input_size: int,
        window_days: int,
        label_days: int = 45,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,
        factor: int = 5,
        attn: str = "prob",
        distil: bool = True,
        dropout: float = 0.2,
        activation: str = "gelu",
        user_emb_dim: int = 32,
        known_user_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if label_days <= 0 or label_days > window_days:
            raise ValueError("--label-days must be in the range [1, --window-days].")
        self.label_days = label_days
        self.pred_len = len(LABELS)
        self.enc_embedding = DataEmbedding(
            input_size, d_model, dropout, max_len=window_days + self.pred_len + 8
        )
        self.dec_embedding = DataEmbedding(
            input_size, d_model, dropout, max_len=label_days + self.pred_len + 8
        )
        self.future_step_embedding = nn.Embedding(self.pred_len, d_model)

        def self_attention(mask_flag):
            if attn == "prob":
                inner = ProbAttention(mask_flag=mask_flag, factor=factor, attention_dropout=dropout)
            else:
                inner = FullAttention(mask_flag=mask_flag, attention_dropout=dropout)
            return AttentionLayer(inner, d_model, n_heads)

        self.encoder = Encoder(
            [
                EncoderLayer(self_attention(False), d_model, d_ff, dropout, activation)
                for _ in range(e_layers)
            ],
            conv_layers=[ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self_attention(True),
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=dropout), d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                )
                for _ in range(d_layers)
            ]
        )
        self.user_emb = UserEmbedding(known_user_ids, user_emb_dim)
        hidden = max(d_model // 2, 16)
        self.head = nn.Sequential(
            nn.Linear(d_model + self.user_emb.dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: torch.Tensor, aux: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        batch_size = seq.size(0)
        enc_out = self.encoder(self.enc_embedding(seq))
        future_tokens = seq.new_zeros(batch_size, self.pred_len, seq.size(2))
        dec_tokens = torch.cat([seq[:, -self.label_days :, :], future_tokens], dim=1)
        dec_out = self.dec_embedding(dec_tokens)
        future_index = torch.arange(self.pred_len, device=seq.device)
        dec_out[:, -self.pred_len :, :] = dec_out[
            :, -self.pred_len :, :
        ] + self.future_step_embedding(future_index).unsqueeze(0)
        dec_mask = causal_mask(batch_size, dec_out.size(1), device=seq.device)
        future_states = self.decoder(dec_out, enc_out, x_mask=dec_mask)[:, -self.pred_len :, :]
        user_state = self.user_emb(user_ids).unsqueeze(1).expand(-1, self.pred_len, -1)
        return self.head(torch.cat([future_states, user_state], dim=2)).squeeze(-1)


# --- FEDformer (fedformer_updated.py) -------------------------------------------


def select_frequency_modes(seq_len, modes, mode_select, seed=0):
    freq_count = seq_len // 2 + 1
    modes = int(min(max(modes, 1), freq_count))
    if mode_select == "random":
        index = np.sort(np.random.default_rng(seed).permutation(freq_count)[:modes])
    else:
        index = np.arange(modes)
    return torch.tensor(index, dtype=torch.long)


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.kernel_size <= 1:
            return x
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        padded = torch.cat(
            [x[:, :1, :].repeat(1, left, 1), x, x[:, -1:, :].repeat(1, right, 1)], dim=1
        )
        return self.avg(padded.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """Split into seasonal + trend with a softmax-weighted mix of moving averages."""

    def __init__(self, kernels):
        super().__init__()
        if isinstance(kernels, int):
            kernels = [kernels]
        if not kernels:
            raise ValueError("At least one moving-average kernel is required.")
        self.avgs = nn.ModuleList([MovingAverage(kernel) for kernel in kernels])
        self.kernel_logits = nn.Parameter(torch.zeros(len(kernels)))

    def forward(self, x):
        if len(self.avgs) == 1:
            moving_mean = self.avgs[0](x)
        else:
            weights = torch.softmax(self.kernel_logits, dim=0)
            moving_mean = torch.stack(
                [avg(x) * weights[i] for i, avg in enumerate(self.avgs)], dim=0
            ).sum(dim=0)
        return x - moving_mean, moving_mean


class SeasonalLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x = self.layernorm(x)
        return x - x.mean(dim=1, keepdim=True)


class FourierMixing(nn.Module):
    """Frequency enhanced block: rFFT, complex mixing on selected modes, iRFFT."""

    def __init__(self, d_model, seq_len, modes, mode_select, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.register_buffer(
            "mode_index", select_frequency_modes(seq_len, modes, mode_select), persistent=False
        )
        weights = torch.randn(len(self.mode_index), d_model, d_model) / math.sqrt(d_model)
        self.weight = nn.Parameter(torch.complex(weights, torch.zeros_like(weights)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, seq_len, channels = x.shape
        if channels != self.d_model or seq_len != self.seq_len:
            raise ValueError(f"Expected (*, {self.seq_len}, {self.d_model}), got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = torch.zeros_like(x_ft)
        selected = x_ft.index_select(dim=1, index=self.mode_index)
        out_ft.index_copy_(1, self.mode_index, torch.einsum("bmd,mdo->bmo", selected, self.weight))
        return self.dropout(torch.fft.irfft(out_ft, n=seq_len, dim=1))


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, seq_len, modes, mode_select, moving_avg, dropout):
        super().__init__()
        self.norm1 = SeasonalLayerNorm(d_model)
        self.fourier = FourierMixing(d_model, seq_len, modes, mode_select, dropout)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.norm2 = SeasonalLayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.decomp2 = SeriesDecomposition(moving_avg)

    def forward(self, seasonal):
        seasonal = seasonal + self.fourier(self.norm1(seasonal))
        seasonal, trend1 = self.decomp1(seasonal)
        seasonal = seasonal + self.dropout(self.feed_forward(self.norm2(seasonal)))
        seasonal, trend2 = self.decomp2(seasonal)
        return seasonal, trend1 + trend2


class FEDformerForecaster(nn.Module):
    """FEDformer encoder; seasonal/trend states are pooled into a six-label head."""

    def __init__(
        self,
        input_size: int,
        window_days: int,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 2,
        modes: int = 16,
        mode_select: str = "low",
        moving_avg: tuple[int, ...] = (7, 14, 28),
        dropout: float = 0.2,
        user_emb_dim: int = 32,
        known_user_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_decomp = SeriesDecomposition(list(moving_avg))
        self.encoder = nn.ModuleList(
            FEDformerEncoderLayer(
                d_model, d_ff, window_days, modes, mode_select, list(moving_avg), dropout
            )
            for _ in range(e_layers)
        )
        self.norm = SeasonalLayerNorm(d_model)
        self.user_emb = UserEmbedding(known_user_ids, user_emb_dim)
        hidden = max(d_model // 2, 16)
        self.head = nn.Sequential(
            nn.Linear(4 * d_model + self.user_emb.dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(LABELS)),
        )

    def forward(self, seq: torch.Tensor, aux: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        seasonal, trend_total = self.input_decomp(self.input_projection(seq))
        for layer in self.encoder:
            seasonal, layer_trend = layer(seasonal)
            trend_total = trend_total + layer_trend
        seasonal = self.norm(seasonal)
        pooled = torch.cat(
            [
                seasonal[:, -1, :],
                seasonal.mean(dim=1),
                trend_total[:, -1, :],
                trend_total.mean(dim=1),
                self.user_emb(user_ids),
            ],
            dim=1,
        )
        return self.head(pooled)
