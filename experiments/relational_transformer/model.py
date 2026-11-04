"""Relational Transformer over cell tokens (port of stanford-star ``rt.model.net``).

Kept from upstream: per-type value encoders plus a column-name encoder, all
RMS-normalized and summed into one vector per cell; learned mask embeddings
for target cells; blocks of column / feature / neighbor masked attention (and
optionally the v1 ``full`` attention) followed by a SwiGLU FFN, all
pre-norm residual; QK-RMSNorm, a learned per-head ``log(kv_size)`` query
scale, ``1 / head_dim`` attention scale, a sigmoid output gate, and
zero-initialized residual outputs and decoders.

Changed for this repository: there are no pretrained checkpoints or frozen
sentence embeddings, so column names and categories are learned embeddings,
and text cells are mean-pooled hashed-token embeddings shared across
columns. Attention uses dense boolean masks with PyTorch SDPA instead of
``flex_attention`` (no ``torch.compile`` requirement). Targets are binary, so
the shared number decoder is trained with BCE on the six target cells.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from experiments.relgt.data import LABELS

from .cells import (
    DATETIME_FEATURES,
    PAD_ROW,
    SEM_CATEGORY,
    SEM_DATETIME,
    SEM_NUMBER,
    SEM_TEXT,
    SEM_TYPES,
)


ATTENTION_TYPES = ("col", "feat", "nbr", "full")


class MaskedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.wo.weight)
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.wg = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.wg.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``mask`` bool [batch, queries, keys]; a query with no keys outputs 0."""
        batch, length, _ = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(heads(self.wq(x)))
        k = self.k_norm(heads(self.wk(x)))
        v = heads(self.wv(x))
        kv_sizes = mask.sum(-1, keepdim=True).clamp_min(1).to(q.dtype)
        q = q * self.scale.to(q.dtype) * torch.log(kv_sizes).unsqueeze(1)
        has_keys = mask.any(-1, keepdim=True)
        # SDPA returns NaN for a query that sees no key; let it see itself.
        eye = torch.eye(length, dtype=torch.bool, device=x.device)
        safe_mask = mask | (eye & ~has_keys)
        out = F.scaled_dot_product_attention(
            q, k, v.to(q.dtype), attn_mask=safe_mask.unsqueeze(1), scale=1.0 / self.head_dim
        )
        out = out.transpose(1, 2).reshape(batch, length, -1) * has_keys.to(out.dtype)
        return self.wo(2 * torch.sigmoid(self.wg(x)) * out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, attention_types: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.attention_types = attention_types
        self.norms = nn.ModuleDict(
            {name: nn.RMSNorm(d_model, eps=1e-6) for name in (*attention_types, "ffn")}
        )
        self.attns = nn.ModuleDict(
            {name: MaskedAttention(d_model, num_heads) for name in attention_types}
        )
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, masks: dict[str, torch.Tensor]) -> torch.Tensor:
        for name in self.attention_types:
            x = x + self.attns[name](self.norms[name](x), masks[name])
        return x + self.ffn(self.norms["ffn"](x))


def attention_masks(
    batch: dict[str, torch.Tensor], attention_types: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Upstream RT masks, from local row ids and per-row foreign-key parents.

    - ``col``: same column (column ids already include the table);
    - ``feat``: the query's own row and the rows it references;
    - ``nbr``: rows that reference the query's row;
    - ``full``: every cell.
    """
    row, row_parents, padding = batch["row"], batch["row_parents"], batch["padding"]
    batch_size, num_rows, _ = row_parents.shape
    valid = ~padding[:, :, None] & ~padding[:, None, :]
    # is_parent[b, r, p]: row r references row p.
    is_parent = torch.zeros(
        batch_size, num_rows + 1, num_rows + 1, dtype=torch.bool, device=row.device
    )
    b, r, slot = torch.nonzero(row_parents != PAD_ROW, as_tuple=True)
    is_parent[b, r, row_parents[b, r, slot]] = True
    safe_row = torch.where(row == PAD_ROW, num_rows, row)
    index = torch.arange(batch_size, device=row.device)[:, None, None]
    parent_of = is_parent[index, safe_row[:, :, None], safe_row[:, None, :]]  # q row -> kv row
    masks = {}
    if "col" in attention_types:
        masks["col"] = (batch["column"][:, :, None] == batch["column"][:, None, :]) & valid
    if "feat" in attention_types:
        masks["feat"] = ((safe_row[:, :, None] == safe_row[:, None, :]) | parent_of) & valid
    if "nbr" in attention_types:
        masks["nbr"] = parent_of.transpose(1, 2) & valid
    if "full" in attention_types:
        masks["full"] = valid
    return masks


class RelationalTransformer(nn.Module):
    def __init__(
        self,
        num_columns: int,
        num_categories: int,
        text_buckets: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        d_ff: int,
        attention_types: tuple[str, ...] = ("col", "feat", "nbr"),
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        unknown = set(attention_types) - set(ATTENTION_TYPES)
        if unknown or not attention_types:
            raise ValueError(f"attention_types must be a non-empty subset of {ATTENTION_TYPES}.")
        self.attention_types = tuple(attention_types)
        self.column_embedding = nn.Embedding(num_columns, d_model)
        self.category_embedding = nn.Embedding(num_categories, d_model)
        self.text_embedding = nn.Embedding(text_buckets, d_model, padding_idx=0)
        self.encoders = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model),
                "text": nn.Linear(d_model, d_model),
                "datetime": nn.Linear(DATETIME_FEATURES, d_model),
            }
        )
        self.norms = nn.ModuleDict(
            {name: nn.RMSNorm(d_model, eps=1e-6) for name in (*SEM_TYPES, "column")}
        )
        self.mask_embeddings = nn.Parameter(torch.randn(len(SEM_TYPES), d_model))
        self.blocks = nn.ModuleList(
            RelationalBlock(d_model, num_heads, d_ff, self.attention_types)
            for _ in range(num_blocks)
        )
        self.norm_out = nn.RMSNorm(d_model, eps=1e-6)
        # Target logits and masked-number reconstruction get separate decoders:
        # history label cells are reconstructed as values, targets as logits.
        self.number_decoder = nn.Linear(d_model, 1)
        self.mtp_decoder = nn.Linear(d_model, 1)
        for decoder in (self.number_decoder, self.mtp_decoder):
            nn.init.zeros_(decoder.weight)
            nn.init.zeros_(decoder.bias)

    def encode(self, batch: dict[str, torch.Tensor], masked: torch.Tensor) -> torch.Tensor:
        sem, padding = batch["sem"], batch["padding"]
        x = self.norms["column"](self.column_embedding(batch["column"]))
        tokens = batch["text"]
        present = tokens.ne(0).unsqueeze(-1)
        text = (self.text_embedding(tokens) * present).sum(-2) / present.sum(-2).clamp(min=1)
        values = {
            SEM_NUMBER: self.norms["number"](
                self.encoders["number"](batch["number"].unsqueeze(-1))
            ),
            SEM_CATEGORY: self.norms["category"](self.category_embedding(batch["category"])),
            SEM_TEXT: self.norms["text"](self.encoders["text"](text)),
            SEM_DATETIME: self.norms["datetime"](self.encoders["datetime"](batch["datetime"])),
        }
        for sem_type, value in values.items():
            is_type = (sem == sem_type) & ~padding
            x = x + value * (is_type & ~masked).unsqueeze(-1)
            x = x + self.mask_embeddings[sem_type] * (is_type & masked).unsqueeze(-1)
        return x * (~padding).unsqueeze(-1)

    def forward(
        self, batch: dict[str, torch.Tensor], mtp_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return target logits [batch, labels] and the masked-number loss.

        ``mtp_mask`` (bool [batch, cells]) hides extra number cells, which
        ``mtp_decoder`` must reconstruct (upstream's masked token prediction).
        """
        masked = batch["is_target"] if mtp_mask is None else batch["is_target"] | mtp_mask
        x = self.encode(batch, masked)
        masks = attention_masks(batch, self.attention_types)
        for block in self.blocks:
            x = block(x, masks)
        x = self.norm_out(x)
        index = batch["target_pos"].unsqueeze(-1).expand(-1, -1, x.shape[-1])
        logits = self.number_decoder(x.gather(1, index)).squeeze(-1)
        if logits.shape[1] != len(LABELS):
            raise AssertionError("Expected one target cell per label.")
        if mtp_mask is None or not mtp_mask.any():
            return logits, None
        reconstruction = self.mtp_decoder(x[mtp_mask]).squeeze(-1).float()
        return logits, F.huber_loss(reconstruction, batch["number"][mtp_mask].float())
