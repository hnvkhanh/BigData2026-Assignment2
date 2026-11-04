"""Relational Graph Transformer (RelGT, Dwivedi et al. 2025) for the 6 weekly activity labels.

Port of the official implementation (https://github.com/snap-stanford/relgt: model.py,
local_module.py, codebook.py, encoders.py, utils.py) to this data, in plain PyTorch (no PyTorch
Geometric, torch_frame or relbench).

Relational entity graph: one node per row of each of the 8 tables and one undirected edge per
foreign key (Pushes.user_id -> Users, IssueComments.issue_id -> Issues, ...).
PullRequests.merged_by_id is left out because it is only known after the merge.

Examples are the baseline's: (cohort user, week start s) for users passing the selection rule,
6 labels, the same weeks and time split (experiments/baseline/run.py). A node of table T is
visible to an example only if its row time is before s - lag_T, the staleness each table has at
the test week, so nothing after the cutoff leaks in.

Tokens (utils.py): the seed user plus K-1 nodes sampled uniformly from its visible 1- and 2-hop
neighbours (without replacement if there are enough, with replacement otherwise, random visible
nodes as hop 3 if there are none). Each token is encoded by five elements, concatenated and
mixed by an MLP: node type, hop distance, time relative to s, the row's own columns, and a GIN
positional encoding of the sampled subgraph (random inputs). Then a local transformer over the
K tokens with an attention readout to the seed, plus global attention from the seed to an EMA
vector-quantised codebook of centroids, a feed-forward block and a 2-layer head with 6 outputs.

Differences from the official code:
  * Row features are hand-picked columns per table (flags, counts, one-hot categories, text as
    its length) encoded by a per-table residual MLP, instead of torch_frame column-type
    inference, GloVe text embeddings and torch_frame's ResNet. Columns known only later
    (merged_at, closed_at, dismissed_at) enter as "happened before the cutoff" flags.
  * The neighbour caps (5,000 1-hop nodes, 1,000 per 1-hop node) are applied after the time
    filter rather than before it.
  * The codebook assignment buffer covers the cohort users (the only seeds), not every node.
  * Attention dropout is applied only in training.
  * Defaults are smaller than the paper's (channels 128 vs 512, K 100 vs 300) to fit one 12 GB
    GPU, and train longer at a higher learning rate (20 epochs at 3e-4 vs 10 at 1e-4, which was
    still improving when it stopped); see --help.
  * As in the other experiments, the epoch is chosen on validation (mean average precision over
    the 6 labels, the official multilabel metric), thresholds maximise F1 there, and the final
    model is retrained on all labelled weeks.

Using that pr_review_comment ~ subset of pr_review (see experiments/gnn_review), both off by default:
  * --habit: the user's review-commenting habit over the 90 days before the cutoff (the 11
    features of experiments/gnn_review) is encoded by an MLP and added to the seed token.
  * --review-aux: a 7th training output, "pr_review or pr_review_comment" this week.
Every run also tunes the thresholds jointly on the score over the validation weeks and keeps
them if they beat the per-label F1 thresholds.

Run from the repo root (needs PyTorch, a GPU is used if available):
    python experiments/relgt/run.py
    python experiments/relgt/run.py --habit --review-aux --name habit_aux
"""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch  # after numpy: importing torch first trips a duplicate-OpenMP error on this machine
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("baseline_run", HERE.parent / "baseline" / "run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
_spec = importlib.util.spec_from_file_location("gnn_review_run", HERE.parent / "gnn_review" / "run.py")
review = importlib.util.module_from_spec(_spec)  # habit features, joint thresholds
_spec.loader.exec_module(review)

EVENT_COLS, WEEK, DAY = base.EVENT_COLS, base.WEEK, base.DAY
IR, IC = review.IR, review.IC
OUT_DIR = HERE / "outputs"
CACHE_DIR = HERE / "cache"

TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments", "PRReviews",
          "PRReviewComments", "Pushes"]
TIME_COL = {"Users": "first_seen_at", "Repositories": "first_seen_at", "PullRequests": "first_seen_at",
            "Issues": "first_seen_at", "IssueComments": "created_at", "PRReviews": "first_seen_at",
            "PRReviewComments": "created_at", "Pushes": "created_at"}
# Which source's data end sets the staleness of each table (Users / Repositories: the latest).
LAG_KEY = {"PullRequests": "open_pr", "Issues": "open_issue", "IssueComments": "issue_comment",
           "PRReviews": "pr_review", "PRReviewComments": "pr_review_comment", "Pushes": "push"}
FKEYS = {
    "PullRequests": [("repo_id", "Repositories"), ("author_id", "Users")],
    "Issues": [("repo_id", "Repositories"), ("author_id", "Users"), ("assignee_id", "Users"),
               ("pull_request_id", "PullRequests")],
    "IssueComments": [("issue_id", "Issues"), ("repo_id", "Repositories"), ("author_id", "Users")],
    "PRReviews": [("pr_id", "PullRequests"), ("repo_id", "Repositories"), ("reviewer_id", "Users")],
    "PRReviewComments": [("pr_id", "PullRequests"), ("repo_id", "Repositories"), ("author_id", "Users"),
                         ("pull_request_review_id", "PRReviews"), ("in_reply_to_id", "PRReviewComments")],
    "Pushes": [("user_id", "Users"), ("repo_id", "Repositories")],
}
COLUMNS = {
    "Users": ["is_bot", "is_cohort"],
    "Repositories": ["org_id"],
    "PullRequests": ["number", "is_cross_repo", "base_ref", "created_at", "merged_at", "closed_at"],
    "Issues": ["is_pull_request", "title", "body", "num_labels", "milestone_title", "issue_type",
               "locked", "created_via_app", "created_at", "closed_at"],
    "IssueComments": ["is_pull_request", "body", "via_app"],
    "PRReviews": ["state", "body", "submitted_at", "dismissed_at"],
    "PRReviewComments": ["file_extension", "subject_type", "position", "body"],
    "Pushes": ["branch", "is_new_ref"],
}
FILE_EXTENSIONS = ["py", "ts", "md", "rs", "go", "java", "tsx", "cpp", "yml", "cs", "js", "c", "h",
                   "yaml", "json", "kt"]
USER_SETS = ["all", "test", "exclude-test"]
MAX_HOP = 3  # 1- and 2-hop neighbours, 3 = random fallback node


# --- Relational entity graph -----------------------------------------------------------------

def days(t):
    """Days since the data start as float32; missing times become +inf (never visible)."""
    d = ((pd.Series(t) - base.DATA_START) / DAY).to_numpy(dtype="float64", na_value=np.inf)
    return d.astype(np.float32)


def row_features(name, d):
    """Static columns of a table as floats, and the columns that are known only later."""
    def flag(s):
        return s.fillna(False).astype(bool)

    def text_len(s):
        return np.log1p(s.fillna("").str.len())

    def lag(late, early):
        return np.log1p(((late - early) / DAY).clip(lower=0).fillna(0))

    f, later = {}, {}
    if name == "Users":
        f = {"bot": d.is_bot, "cohort": d.is_cohort}
    elif name == "Repositories":
        f = {"org": d.org_id.notna()}
    elif name == "PullRequests":
        f = {"cross_repo": flag(d.is_cross_repo), "has_author": d.author_id.notna(),
             "opened_seen": d.created_at.notna(), "seen_lag": lag(d.first_seen_at, d.created_at),
             "number": np.log1p(d.number.astype(float).fillna(0)),
             "default_base": d.base_ref.isin(["main", "master"])}
        later = {"merged": d.merged_at, "closed": d.closed_at}
    elif name == "Issues":
        f = {"is_pr": d.is_pull_request, "pr_known": d.pull_request_id.notna(),
             "labels": np.log1p(d.num_labels.astype(float).fillna(0)), "assignee": d.assignee_id.notna(),
             "milestone": d.milestone_title.notna(), "typed": d.issue_type.notna(), "locked": flag(d.locked),
             "via_app": d.created_via_app.notna(), "title_len": text_len(d.title), "body_len": text_len(d.body),
             "seen_lag": lag(d.first_seen_at, d.created_at)}
        later = {"closed": d.closed_at}
    elif name == "IssueComments":
        f = {"is_pr": d.is_pull_request, "via_app": d.via_app.notna(), "body_len": text_len(d.body)}
    elif name == "PRReviews":
        # "dismissed" is a later state, so it is not one-hot encoded; dismissed_at covers it.
        f = {s: d.state == s for s in ["approved", "commented", "changes_requested"]}
        f.update({"body_len": text_len(d.body), "seen_lag": lag(d.first_seen_at, d.submitted_at)})
        later = {"dismissed": d.dismissed_at}
    elif name == "PRReviewComments":
        f = {"line": d.subject_type == "line", "reply": d.in_reply_to_id.notna(),
             "in_review": d.pull_request_review_id.notna(), "outdated": d.position.isna(),
             "body_len": text_len(d.body)}
        f.update({f"ext_{e}": d.file_extension == e for e in FILE_EXTENSIONS})
    elif name == "Pushes":
        f = {"new_ref": flag(d.is_new_ref), "default_branch": d.branch.isin(["main", "master"]),
             "branch_len": text_len(d.branch)}
    x = np.nan_to_num(pd.DataFrame(f).astype("float64").values).astype(np.float32)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
    lat = np.stack([days(v) for v in later.values()], 1) if later else np.zeros((len(d), 0), np.float32)
    return x.astype(np.float32), lat


def csr(src, dst, n):
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=n))
    return ptr, dst[order]


class Graph:
    """All rows of all tables as one node set: type, row time, features and FK adjacency."""

    def __init__(self):
        t0 = time.time()
        sizes, times, self.feats, self.later, fsrc, fdst = [], [], [], [], [], []
        frames = {}
        for name in TABLES:
            cols = [TIME_COL[name]] + [c for c, _ in FKEYS.get(name, [])] + COLUMNS[name]
            frames[name] = pd.read_parquet(base.DB / f"{name}.parquet", columns=list(dict.fromkeys(cols)))
            sizes.append(len(frames[name]))
        self.offset = np.concatenate([[0], np.cumsum(sizes)])
        self.n = int(self.offset[-1])
        self.ntype = np.repeat(np.arange(len(TABLES), dtype=np.int64), sizes)
        for t, name in enumerate(TABLES):
            d = frames.pop(name)
            times.append(days(d[TIME_COL[name]]))
            for col, ref in FKEYS.get(name, []):
                v = d[col].to_numpy(dtype="float64", na_value=np.nan)
                m = ~np.isnan(v)
                fsrc.append(self.offset[t] + np.flatnonzero(m))
                fdst.append(self.offset[TABLES.index(ref)] + v[m].astype(np.int64))
            x, lat = row_features(name, d)
            self.feats.append(x)
            self.later.append(lat)
        self.time = np.concatenate(times)
        fsrc, fdst = np.concatenate(fsrc), np.concatenate(fdst)
        self.fptr, self.fadj = csr(fsrc, fdst, self.n)  # forward FK edges, for subgraph adjacency
        self.ptr, self.adj = csr(np.concatenate([fsrc, fdst]), np.concatenate([fdst, fsrc]), self.n)
        print(f"graph: {self.n:,} nodes, {len(fsrc):,} FK edges ({time.time() - t0:.0f}s)")


# --- Tokenisation (utils.py: gather_1_and_2_hop_with_seed_time, _process_one_seed) -----------

def expand(ptr, adj, ids, nodes):
    """(id, neighbour) for every neighbour of every node."""
    start, cnt = ptr[nodes], ptr[nodes + 1] - ptr[nodes]
    within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return np.repeat(ids, cnt), adj[np.repeat(start, cnt) + within]


def cap(group, value, k, rng):
    """Keep at most k random values per group."""
    if not len(group) or np.bincount(group).max() <= k:
        return group, value
    order = np.lexsort((rng.random(len(group)), group))
    group, value = group[order], value[order]
    first = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    rank = np.arange(len(group)) - np.repeat(first, np.diff(np.r_[first, len(group)]))
    return group[rank < k], value[rank < k]


def sample_tokens(G, seeds, cutoff, K, rng, cap1=5000, cap2=1000):
    """K tokens (node, hop) per seed from its 1- and 2-hop neighbours visible before the cutoff."""
    visible = G.time < cutoff[G.ntype]
    b1, v1 = expand(G.ptr, G.adj, np.arange(len(seeds)), seeds)
    m = visible[v1]
    b1, v1 = cap(b1[m], v1[m], cap1, rng)
    i2, v2 = expand(G.ptr, G.adj, np.arange(len(v1)), v1)
    m = visible[v2] & (v2 != seeds[b1[i2]])
    i2, v2 = cap(i2[m], v2[m], cap2, rng)

    # Union of both hops, each node once at its smallest hop, in random order per seed.
    b = np.concatenate([b1, b1[i2]])
    v = np.concatenate([v1, v2])
    hop = np.concatenate([np.ones(len(v1), np.int8), np.full(len(v2), 2, np.int8)])
    key = b * G.n + v
    order = np.lexsort((hop, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    b, v, hop = b[order][first], v[order][first], hop[order][first]
    order = np.lexsort((rng.random(len(b)), b))
    v, hop = v[order], hop[order]

    cnt = np.bincount(b, minlength=len(seeds))
    start = np.cumsum(cnt) - cnt
    pick = np.where(cnt[:, None] >= K - 1, np.arange(K - 1)[None, :],
                    (rng.random((len(seeds), K - 1)) * cnt[:, None]).astype(np.int64))
    nodes = np.empty((len(seeds), K), np.int64)
    hops = np.zeros((len(seeds), K), np.int8)
    nodes[:, 0] = seeds
    has = cnt > 0
    pos = (start[:, None] + pick)[has]
    nodes[has, 1:], hops[has, 1:] = v[pos], hop[pos]
    if (~has).any():
        nodes[~has, 1:] = rng.choice(np.flatnonzero(visible), size=((~has).sum(), K - 1))
        hops[~has, 1:] = MAX_HOP
    return nodes, hops


def subgraph_edges(G, nodes):
    """Edges among each seed's K tokens: (seed row, token i, token j), both directions."""
    B, K = nodes.shape
    flat = nodes.ravel()
    p, f = expand(G.fptr, G.fadj, np.arange(B * K), flat)
    row = p // K
    keys = np.repeat(np.arange(B), K) * G.n + flat
    order = np.argsort(keys)
    q = np.minimum(np.searchsorted(keys[order], row * G.n + f), len(keys) - 1)
    hit = keys[order][q] == row * G.n + f
    i, j, r = p[hit] % K, order[q[hit]] % K, row[hit]
    return np.concatenate([r, r]), np.concatenate([i, j]), np.concatenate([j, i])


# --- Model (model.py, local_module.py, codebook.py, encoders.py) -----------------------------

class PositionalEncoding(nn.Module):
    """torch_geometric.nn.PositionalEncoding: sinusoids of the relative time."""

    def __init__(self, d, base_freq=1e-4):
        super().__init__()
        self.register_buffer("frequency", torch.logspace(0, 1, d // 2, base_freq))

    def forward(self, x):
        out = x.view(-1, 1) * self.frequency.view(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


class TimeEncoder(nn.Module):
    """NeighborTimeEncoder: positional encoding + linear, a learned vector for masked times."""

    def __init__(self, d):
        super().__init__()
        self.pos = PositionalEncoding(d)
        self.linear = nn.Linear(d, d)
        self.mask_vector = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_vector, std=0.02)

    def forward(self, rel_time):
        B, K = rel_time.shape
        out = self.linear(self.pos(rel_time.reshape(-1))).view(B, K, -1)
        mask = (rel_time < 0).unsqueeze(-1).float()
        return (1 - mask) * out + mask * self.mask_vector


class TableEncoder(nn.Module):
    """Stand-in for torch_frame's ResNet: input layer, residual blocks, LayerNorm-ReLU-Linear."""

    def __init__(self, d_in, out, hidden=128, layers=4, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)


class RowEncoder(nn.Module):
    """NeighborTfsEncoder: each token's row through the encoder of its table."""

    def __init__(self, G, cutoffs, week_day, channels, dropout):
        super().__init__()
        self.encoders = nn.ModuleList([
            TableEncoder(G.feats[t].shape[1] + G.later[t].shape[1], channels, dropout=dropout)
            for t in range(len(TABLES))])
        for t in range(len(TABLES)):
            self.register_buffer(f"feat{t}", torch.tensor(G.feats[t]), persistent=False)
            self.register_buffer(f"later{t}", torch.tensor(G.later[t]), persistent=False)
        self.register_buffer("ntype", torch.tensor(G.ntype), persistent=False)
        self.register_buffer("offset", torch.tensor(G.offset[:-1]), persistent=False)
        self.register_buffer("node_time", torch.tensor(G.time), persistent=False)
        self.register_buffer("cutoffs", torch.tensor(cutoffs), persistent=False)  # [weeks, tables]
        self.register_buffer("week_day", torch.tensor(week_day), persistent=False)
        self.channels = channels

    def forward(self, nodes, week):
        flat = nodes.reshape(-1)
        types = self.ntype[flat]
        local = flat - self.offset[types]
        wk = week.repeat_interleave(nodes.shape[1])
        out = torch.zeros(len(flat), self.channels, device=flat.device)
        for t, enc in enumerate(self.encoders):
            m = torch.nonzero(types == t).squeeze(1)
            if not len(m):
                continue
            x = getattr(self, f"feat{t}")[local[m]]
            later = getattr(self, f"later{t}")
            if later.shape[1]:
                happened = later[local[m]] < self.cutoffs[wk[m], t][:, None]
                x = torch.cat([x, happened.float()], 1)
            out[m] = enc(x)
        return out.view(*nodes.shape, -1), types.view(nodes.shape)

    def rel_time(self, nodes, week):
        t = self.week_day[week][:, None] - self.node_time[nodes]
        t[:, 0] = 0.0  # the seed token has time 0, as in utils.py
        return t


class GNNPEEncoder(nn.Module):
    """GIN over each sampled subgraph with random scalar inputs (encoders.py, pe_dim=0)."""

    def __init__(self, d, layers=4):
        super().__init__()
        h = d // 4
        self.input_proj = nn.Linear(1, h)
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.BatchNorm1d(2 * h), nn.ReLU(),
                                                 nn.Linear(2 * h, h)) for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))  # GINConv(train_eps=True)
        self.bns = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(layers)])
        self.final_transform = nn.Linear(h, d)
        for lin in [self.input_proj, self.final_transform]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, edge_index, B, K):
        x = self.input_proj(torch.randn(B * K, 1, device=edge_index.device))
        src, dst = edge_index
        for i, mlp in enumerate(self.mlps):
            agg = torch.zeros_like(x).index_add_(0, dst, x[src])
            x = x + F.relu(self.bns[i](mlp((1 + self.eps[i]) * x + agg)))
        return self.final_transform(x).view(B, K, -1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d, ffn, dropout):
        super().__init__()
        self.bn_in, self.bn_out = nn.BatchNorm1d(d), nn.BatchNorm1d(d)
        self.ffn_net = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(ffn, d), nn.Dropout(dropout))

    def forward(self, x):
        x = self.bn_in(x.transpose(1, 2)).transpose(1, 2)
        x = self.ffn_net(x)
        return self.bn_out(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Pre-LayerNorm transformer layer (local_module.py)."""

    def __init__(self, d, ffn, dropout, attn_dropout, heads):
        super().__init__()
        self.heads, self.attn_dropout = heads, attn_dropout
        self.self_attention_norm = nn.LayerNorm(d)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        self.self_attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = FeedForwardNetwork(d, ffn, dropout)

    def forward(self, x):
        B, L, D = x.shape
        h = self.self_attention_norm(x)
        q, k, v = (p(h).view(B, L, self.heads, D // self.heads).transpose(1, 2)
                   for p in [self.q_proj, self.k_proj, self.v_proj])
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.self_attention_dropout(self.out_proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.ffn(self.ffn_norm(x))


class LocalModule(nn.Module):
    """Transformer over the K tokens; readout = seed token + attention-weighted neighbours."""

    def __init__(self, d, layers, heads, dropout, attn_dropout):
        super().__init__()
        self.att_embeddings_nope = nn.Linear(d, d)
        self.layers = nn.ModuleList([EncoderLayer(d, 2 * d, dropout, attn_dropout, heads)
                                     for _ in range(layers)])
        self.final_ln = nn.LayerNorm(d)
        self.attn_layer = nn.Linear(2 * d, 1)

    def forward(self, x_set):
        x = self.att_embeddings_nope(x_set)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        node, nbr = x[:, :1], x[:, 1:]
        w = F.softmax(self.attn_layer(torch.cat([node.expand_as(nbr), nbr], 2)), dim=1)
        return (node + (nbr * w).sum(1, keepdim=True)).squeeze(1)


class VectorQuantizerEMA(nn.Module):
    """Codebook of centroids updated by exponential moving averages (codebook.py)."""

    def __init__(self, num_embeddings, dim, decay=0.99):
        super().__init__()
        self.num_embeddings, self.decay = num_embeddings, decay
        self.register_buffer("_embedding", torch.randn(num_embeddings, dim))
        self.register_buffer("_embedding_output", torch.randn(num_embeddings, dim))
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", torch.randn(num_embeddings, dim))
        self.bn = nn.BatchNorm1d(dim, affine=False)

    def get_k(self):
        """Centroids in the input scale; keys and values of the global attention."""
        return self._embedding_output

    @torch.no_grad()
    def update(self, x):
        x = self.bn(x)
        dist = (x ** 2).sum(1, keepdim=True) + (self._embedding ** 2).sum(1) - 2 * x @ self._embedding.t()
        idx = dist.argmin(1)
        enc = F.one_hot(idx, self.num_embeddings).float()
        self._ema_cluster_size.mul_(self.decay).add_((1 - self.decay) * enc.sum(0))
        n = self._ema_cluster_size.sum()
        self._ema_cluster_size.copy_((self._ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n)
        self._ema_w.mul_(self.decay).add_((1 - self.decay) * enc.t() @ x)
        self._embedding.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(0)
        self._embedding_output.copy_(self._embedding * std + self.bn.running_mean.unsqueeze(0))
        return idx


class RelGTLayer(nn.Module):
    def __init__(self, d, global_dim, local_layers, heads, dropout, attn_dropout, conv_type,
                 num_centroids, num_seeds):
        super().__init__()
        self.d, self.heads, self.conv_type, self.attn_dropout = d, heads, conv_type, attn_dropout
        self.num_centroids = num_centroids
        if conv_type != "global":
            self.local_module = LocalModule(d, local_layers, heads, dropout, attn_dropout)
            self.layer_norm_local = nn.LayerNorm(d)
        if conv_type != "local":
            self.vq = VectorQuantizerEMA(num_centroids, global_dim)
            self.register_buffer("c_idx", torch.randint(0, num_centroids, (num_seeds,)))
            self.lin_proj_g = nn.Linear(d, global_dim)
            self.lin_key_g, self.lin_query_g, self.lin_value_g = (nn.Linear(global_dim, d) for _ in range(3))
            self.layer_norm_global = nn.LayerNorm(d)

    def global_forward(self, x, seed_idx):
        q_x = self.lin_proj_g(x)
        codebook = self.vq.get_k().detach().clone()  # updated in place below
        k, v = self.lin_key_g(codebook), self.lin_value_g(codebook)
        q = self.lin_query_g(q_x)
        q, k, v = (t.view(t.shape[0], self.heads, -1).transpose(0, 1) for t in (q, k, v))
        dots = q @ k.transpose(1, 2) / math.sqrt(self.d)
        count = torch.bincount(self.c_idx, minlength=self.num_centroids).float()
        attn = F.softmax(dots + torch.log(count).view(1, 1, -1), dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        out = (attn @ v).transpose(0, 1).reshape(x.shape[0], -1)
        if self.training:
            self.c_idx[seed_idx] = self.vq.update(q_x.detach())
        return out

    def forward(self, x_set, x, seed_idx):
        out = []
        if self.conv_type != "global":
            out.append(self.layer_norm_local(self.local_module(x_set)))
        if self.conv_type != "local":
            out.append(self.layer_norm_global(self.global_forward(x, seed_idx)))
        return torch.cat(out, 1)


class RelGT(nn.Module):
    ELEMENTS = ["type", "hop", "time", "tfs", "gnn"]

    def __init__(self, G, cutoffs, week_day, num_seeds, n_out, channels=128, local_layers=1, heads=4,
                 dropout=0.3, num_centroids=1024, conv_type="full", ablate="none", n_seed_feats=0):
        super().__init__()
        c = channels
        self.rows = RowEncoder(G, cutoffs, week_day, c, dropout)
        self.seed_encoder = (nn.Sequential(nn.Linear(n_seed_feats, c), nn.ReLU(), nn.Linear(c, c))
                             if n_seed_feats else None)
        self.type_encoder = nn.Embedding(len(TABLES) + 1, c)
        self.hop_encoder = nn.Embedding(MAX_HOP + 2, c)
        self.time_encoder = TimeEncoder(c)
        self.pe_encoder = GNNPEEncoder(c)
        self.norms = nn.ModuleDict({f"{k}_norm": nn.LayerNorm(c) for k in self.ELEMENTS})
        self.elements = [k for k in self.ELEMENTS if k != ablate]
        self.in_mixture = nn.Sequential(nn.Linear(len(self.elements) * c, 2 * c), nn.ReLU(), nn.Linear(2 * c, c))
        self.conv = RelGTLayer(c, c // 2, local_layers, heads, dropout, dropout, conv_type, num_centroids,
                               num_seeds)
        h = 2 if conv_type == "full" else 1
        self.ff = nn.Sequential(nn.BatchNorm1d(h * c), nn.Linear(h * c, 2 * c), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(2 * c, c), nn.Dropout(dropout), nn.BatchNorm1d(c))
        # torch_geometric MLP(num_layers=2): Linear, BatchNorm, ReLU, Linear
        self.head = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c), nn.ReLU(), nn.Linear(c, n_out))

    def forward(self, nodes, hops, week, seed_idx, edge_index, seed_x=None):
        B, K = nodes.shape
        tfs, types = self.rows(nodes, week)
        parts = {
            "type": self.type_encoder(types),
            "hop": self.hop_encoder(hops + 1),
            "time": self.time_encoder(self.rows.rel_time(nodes, week)),
            "tfs": tfs,
            "gnn": self.pe_encoder(edge_index, B, K),
        }
        x_set = self.in_mixture(torch.cat([self.norms[f"{k}_norm"](parts[k]) for k in self.elements], -1))
        if self.seed_encoder is not None:  # per-user summary features join the seed token
            x_set = torch.cat([x_set[:, :1] + self.seed_encoder(seed_x).unsqueeze(1), x_set[:, 1:]], 1)
        out = self.ff(self.conv(x_set, x_set[:, 0], seed_idx))
        return self.head(out)


# --- Training ------------------------------------------------------------------------------

class Samples:
    """Tokens, labels and subgraph edges of every (user, week) example, on the device."""

    def __init__(self, nodes, hops, week, cidx, y, edges, device, seed_x=None):
        self.n, self.K = nodes.shape
        self.seed_x = None if seed_x is None else torch.tensor(seed_x, dtype=torch.float32, device=device)
        self.nodes = torch.tensor(nodes, dtype=torch.int32, device=device)
        self.hops = torch.tensor(hops, device=device)
        self.week = torch.tensor(week, device=device)
        self.cidx = torch.tensor(cidx, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        r, i, j = edges
        order = np.argsort(r, kind="stable")
        self.esrc, self.edst = i[order], j[order]  # int16 token positions
        self.eptr = np.zeros(self.n + 1, np.int64)
        self.eptr[1:] = np.cumsum(np.bincount(r, minlength=self.n))
        self.device = device

    def batch(self, ids):
        start, cnt = self.eptr[ids], self.eptr[ids + 1] - self.eptr[ids]
        pos = np.repeat(start - np.cumsum(cnt) + cnt, cnt) + np.arange(cnt.sum())
        shift = np.repeat(np.arange(len(ids)) * self.K, cnt)
        edge_index = torch.tensor(np.stack([self.esrc[pos] + shift, self.edst[pos] + shift]),  # int64
                                  device=self.device)
        t = torch.tensor(ids, device=self.device)
        seed_x = None if self.seed_x is None else self.seed_x[t]
        return (self.nodes[t].long(), self.hops[t].long(), self.week[t], self.cidx[t], edge_index, seed_x), self.y[t]


def masked_bce(logits, y):
    m = ~torch.isnan(y)
    loss = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
    return (loss * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def predict(model, S, ids, batch_size=2048):
    model.eval()
    torch.manual_seed(0)  # the subgraph encoding draws random inputs
    out = [torch.sigmoid(model(*S.batch(ids[i:i + batch_size])[0])).cpu().numpy()
           for i in range(0, len(ids), batch_size)]
    return np.concatenate(out)


def mean_ap(S, ids, p):
    y = S.y[torch.tensor(ids, device=S.device)].cpu().numpy()
    aps = [average_precision_score(y[~np.isnan(y[:, j]), j], p[~np.isnan(y[:, j]), j])
           for j in range(len(EVENT_COLS)) if (y[~np.isnan(y[:, j]), j] == 1).any()]  # not the auxiliary output
    return float(np.mean(aps))


def fit(make_model, S, train_ids, args, epochs, val_ids=None):
    """Adam + BCE as in main_node_ddp.py; with val_ids, keep the epoch with the best mean AP."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, best_epoch, history = -np.inf, None, epochs, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = rng.permutation(train_ids)
        for step, i in enumerate(range(0, len(perm), args.batch_size)):
            ids = perm[i:i + args.batch_size]
            if (args.max_steps_per_epoch and step >= args.max_steps_per_epoch) or len(ids) < 2:
                break
            inputs, y = S.batch(ids)
            opt.zero_grad()
            loss = masked_bce(model(*inputs), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        msg = f"  epoch {epoch:2d}  train loss {np.mean(losses):.4f}  {time.time() - t0:.0f}s"
        if val_ids is not None:
            ap = mean_ap(S, val_ids, predict(model, S, val_ids))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_map": ap})
            msg += f"  val mean AP {ap:.4f}"
            if ap > best:
                best, best_epoch = ap, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def tokenize(G, groups, cutoffs, K, seed):
    """Tokens of every example, cached on disk (sampling is the slow part)."""
    key = hashlib.md5(repr((K, seed, cutoffs.tolist(), [g.tolist() for g in groups], G.n)).encode()).hexdigest()
    path = CACHE_DIR / f"tokens_K{K}_{key[:12]}.npz"
    if path.exists():
        z = np.load(path)
        print(f"tokens from {path.name}")
        return z["nodes"], z["hops"], (z["er"], z["ei"], z["ej"])
    t0 = time.time()
    rng = np.random.default_rng(seed)
    nodes, hops, er, ei, ej, n = [], [], [], [], [], 0
    for w, seeds in enumerate(groups):
        nd, hp = sample_tokens(G, seeds, cutoffs[w], K, rng)
        r, i, j = subgraph_edges(G, nd)
        nodes.append(nd.astype(np.int32))
        hops.append(hp)
        er.append((r + n).astype(np.int32))
        ei.append(i.astype(np.int16))
        ej.append(j.astype(np.int16))
        n += len(seeds)
    out = np.concatenate(nodes), np.concatenate(hops), (np.concatenate(er), np.concatenate(ei), np.concatenate(ej))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, nodes=out[0], hops=out[1], er=out[2][0], ei=out[2][1], ej=out[2][2])
    print(f"sampled {n:,} examples x {K} tokens, {len(out[2][0]):,} subgraph edges ({time.time() - t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description="RelGT for the weekly activity labels")
    base.add_period_args(ap)
    ap.add_argument("--num-neighbors", type=int, default=100, help="K tokens per example (paper: 300)")
    ap.add_argument("--channels", type=int, default=128, help="paper: 512")
    ap.add_argument("--num-layers", type=int, default=1, help="local transformer layers (paper: 1-8)")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-centroids", type=int, default=1024)
    ap.add_argument("--conv-type", choices=["full", "local", "global"], default="full")
    ap.add_argument("--ablate", choices=["none"] + RelGT.ELEMENTS, default="none",
                    help="leave one token element out")
    ap.add_argument("--dropout", type=float, default=0.3, help="feed-forward and attention dropout")
    ap.add_argument("--lr", type=float, default=3e-4, help="paper: 1e-4")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20, help="paper: 10")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--habit", action="store_true",
                    help="add the review-commenting habit features to the seed token")
    ap.add_argument("--review-aux", action="store_true",
                    help="extra training output: pr_review or pr_review_comment")
    g = ap.add_argument_group("users (test users = those in test_input.csv)")
    for stage, what in [("train", "training"), ("val", "validation / thresholds"), ("final", "final fit")]:
        g.add_argument(f"--{stage}-users", choices=USER_SETS, default="all",
                       help=f"users whose examples are used for {what}")
    ap.add_argument("--name", default="default", help="output sub-directory")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = OUT_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, ev, own, data_end = base.load()
    n = len(cohort)
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    weeks, train_weeks, val_weeks, final_weeks = base.split_weeks(args, data_end)
    print(f"device {device}, {len(train_weeks)} train weeks, validation {[str(w)[:10] for w in val_weeks]}")

    tables = {w: base.build_week(w, ev, own, lags, data_end, n) for w in weeks}
    test = pd.read_csv(base.EVAL_DIR / "test_input.csv")
    test_cidx = pd.Series(np.arange(n), index=cohort.github_user_id.values).loc[test.UserID].values
    all_weeks = weeks + [base.TEST_START]

    # Per week and table: the cutoff in days; rows with an earlier row time are visible.
    min_lag = min(lags.values())
    cutoffs = np.array([[((s - lags[LAG_KEY[t]] if t in LAG_KEY else s - min_lag) - base.DATA_START) / DAY
                         for t in TABLES] for s in all_weeks], np.float32)
    week_day = np.array([(s - base.DATA_START) / DAY for s in all_weeks], np.float32)

    # Examples: eligible users of every labelled week, then the test users.
    cidx, week, y = [], [], []
    for i, w in enumerate(weeks):
        m = np.flatnonzero(tables[w].eligible.values)
        cidx.append(m)
        week.append(np.full(len(m), i))
        y.append(tables[w][[f"y_{k}" for k in EVENT_COLS]].values[m])
    cidx.append(test_cidx)
    week.append(np.full(len(test_cidx), len(weeks)))
    y.append(np.full((len(test_cidx), len(EVENT_COLS)), np.nan))
    sizes = np.cumsum([0] + [len(c) for c in cidx])
    y = np.concatenate(y)
    if args.review_aux:  # 1 if either review label is 1, 0 if both are 0, unknown otherwise
        r, c = y[:, IR], y[:, IC]
        union = np.where((r == 1) | (c == 1), 1.0, np.where((r == 0) & (c == 0), 0.0, np.nan))
        y = np.concatenate([y, union[:, None]], 1)
    G = Graph()
    seeds = [G.offset[TABLES.index("Users")] + cohort.user_id.values[c] for c in cidx]
    nodes, hops, edges = tokenize(G, seeds, cutoffs, args.num_neighbors, args.seed)
    ids = {w: np.arange(sizes[i], sizes[i + 1]) for i, w in enumerate(all_weeks)}
    is_test = np.zeros(n, bool)
    is_test[test_cidx] = True
    user_sets = {"all": np.ones(n, bool), "test": is_test, "exclude-test": ~is_test}
    cid = dict(zip(all_weeks, cidx))  # cohort index of each week's examples

    def pick(wks, mode):
        """Boolean masks (per week) of the examples of the chosen user set."""
        return {w: user_sets[mode][cid[w]] for w in wks}

    def select(wks, mode):
        sel = pick(wks, mode)
        return np.concatenate([ids[w][sel[w]] for w in wks])

    train_ids = select(train_weeks, args.train_users)
    val_ids = select(val_weeks, args.val_users)
    all_ids = select(final_weeks, args.final_users)
    if not len(train_ids) or not len(val_ids) or not len(all_ids):
        raise SystemExit("no examples for the chosen weeks and user sets")
    print(f"{len(train_ids):,} training examples, {len(val_ids):,} validation, {len(ids[base.TEST_START]):,} test")

    seed_x = None
    if args.habit:  # log1p, standardised on the training examples
        R, C = review.load_review_detail(cohort)
        seed_x = np.log1p(np.concatenate([review.habit_features(w, R, C, lags, n).values[c]
                                          for w, c in zip(all_weeks, cidx)]))
        seed_x = (seed_x - seed_x[train_ids].mean(0)) / (seed_x[train_ids].std(0) + 1e-6)
        print(f"{seed_x.shape[1]} habit features on the seed token")
    S = Samples(nodes, hops, np.concatenate(week), np.concatenate(cidx), y, edges, device, seed_x)

    def make_model():
        return RelGT(G, cutoffs, week_day, n, len(EVENT_COLS) + args.review_aux, channels=args.channels,
                     local_layers=args.num_layers, heads=args.num_heads, dropout=args.dropout,
                     num_centroids=args.num_centroids, conv_type=args.conv_type, ablate=args.ablate,
                     n_seed_feats=0 if seed_x is None else seed_x.shape[1]).to(device)

    print(f"model parameters: {sum(p.numel() for p in make_model().parameters()):,}")
    model, best_epoch, history = fit(make_model, S, train_ids, args, args.epochs, val_ids=val_ids)
    print(f"best epoch {best_epoch}")

    # Thresholds on the pooled validation weeks (same rule as the baseline).
    metrics = {"args": vars(args), "best_epoch": best_epoch, "history": history, "per_label": {},
               "validation": []}
    usel = pick(val_weeks, args.val_users)
    val_p = {w: predict(model, S, ids[w][usel[w]])[:, :len(EVENT_COLS)] for w in val_weeks}
    rates, pooled, thr = {}, {}, np.zeros(len(EVENT_COLS))
    for j, k in enumerate(EVENT_COLS):
        yv = np.concatenate([tables[w][f"y_{k}"].values[tables[w].eligible.values][usel[w]]
                             for w in val_weeks])
        pv = np.concatenate([val_p[w][:, j] for w in val_weeks])
        m = ~np.isnan(yv)
        pooled[k] = pv[m]
        thr[j], f1 = base.best_threshold(yv[m], pv[m])
        rates[k] = float((pv[m] >= thr[j]).mean())
        metrics["per_label"][k] = {"threshold": thr[j], "val_f1": f1, "val_pred_rate": rates[k],
                                   "val_ap": float(average_precision_score(yv[m], pv[m]))}
        print(f"{k:18s} val F1 {f1:.4f}  AP {metrics['per_label'][k]['val_ap']:.4f}  threshold {thr[j]:.3f}  "
              f"predicted {rates[k]:.3f}  positives {yv[m].mean():.3f}")

    # Per-week scores on validation weeks where all 6 labels are known.
    for w in val_weeks:
        t = tables[w]
        m = t.eligible.values
        if t.loc[m, [f"y_{k}" for k in EVENT_COLS]].isna().any().any():
            continue
        gids = cohort.github_user_id.values[m][usel[w]]
        truth = base.to_output(gids, {k: t[f"y_{k}"].values[m][usel[w]] for k in EVENT_COLS})
        pred = base.to_output(gids, {k: val_p[w][:, j] >= metrics["per_label"][k]["threshold"]
                                     for j, k in enumerate(EVENT_COLS)})
        r = {"week": str(w), "users": int(usel[w].sum()), **base.scores(pred, truth)}
        metrics["validation"].append(r)
        print(f"validation {str(w)[:10]}: score {100 * r['score']:.2f}")
    metrics["val_score_mean"] = float(np.mean([r["score"] for r in metrics["validation"]]))
    print(f"mean validation score {100 * metrics['val_score_mean']:.2f}")

    # Thresholds tuned jointly on the score over the same weeks (experiments/gnn_review); kept if better.
    def labels(w):
        return tables[w][[f"y_{k}" for k in EVENT_COLS]].values[tables[w].eligible.values][usel[w]]

    full = [w for w in val_weeks if not np.isnan(labels(w)).any()]
    joint, _ = review.joint_thresholds(np.concatenate([labels(w) for w in full]).astype(bool),
                                       np.concatenate([val_p[w] for w in full]), thr, False)
    metrics["variants"] = {}
    for v, t in [("per-label F1 thresholds", thr), ("joint thresholds", joint)]:
        ws = [review.score(labels(w).astype(bool), val_p[w] >= t) for w in full]
        metrics["variants"][v] = {"weekly": ws, "mean": float(np.mean(ws)), "thresholds": t.tolist()}
        print(f"  {v:25s} " + "  ".join(f"{100 * s:.2f}" for s in ws) + f"   mean {100 * np.mean(ws):.2f}")
    metrics["chosen"] = max(metrics["variants"], key=lambda v: metrics["variants"][v]["mean"])
    chosen = np.array(metrics["variants"][metrics["chosen"]]["thresholds"])
    rates = {k: float((pooled[k] >= chosen[j]).mean()) for j, k in enumerate(EVENT_COLS)}
    print(f"chosen: {metrics['chosen']}")

    # Final model: all labelled weeks for the selected number of epochs; the validation share of
    # predicted positives is the decision rule on the test week (as in the baseline).
    final, _, _ = fit(make_model, S, all_ids, args, best_epoch)
    p = predict(final, S, ids[base.TEST_START])
    out = pd.DataFrame({"UserID": test.UserID.values})
    for j, k in enumerate(EVENT_COLS):
        out[k] = (p[:, j] >= np.quantile(p[:, j], 1 - rates[k])).astype(int)
    out.to_csv(out_dir / "output.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {out_dir / 'output.csv'}")


if __name__ == "__main__":
    main()
