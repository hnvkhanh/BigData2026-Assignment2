"""RelGNN (Chen, Kanatsoulis & Leskovec, ICML 2025): composite message passing over atomic routes.

Port of the official implementation (https://github.com/snap-stanford/RelGNN: examples/atomic_routes.py,
relgnn_conv.py, relgnn_hetero_conv.py, relgnn_nn.py, relgnn_model.py) in plain PyTorch (no PyTorch
Geometric, torch_frame or relbench), for the token subgraphs of experiments/relgt: one node per row of the 8
tables, K tokens per (user, week) example and the edges among them. It is trained and scored by
experiments/gnn/run.py (--token-model relgnn), with the weeks, examples, protocol and outputs of RowGNN.
Design and differences from the official code: experiments/relgnn/README.md.

Atomic routes (atomic_routes): a table with several foreign keys is a "fact" between the rows it references.
For every ordered pair (q, v) of its keys, the route dim_v -> fact -> dim_q lets dim_q read the fact row
already fused with its dim_v row, in one layer (e.g. Repositories -> Pushes -> Users: a user sees each push
together with its repository). A table with one key, and every self-referencing key (a reply to a review
comment; on the chain graph, a row's next row of the same table and actor) gives two plain routes, the key and
its reverse.

One layer (RelGNNConv, RelGNN_HeteroConv): per route,
  dim-fact-dim  fused = SAGEConv(aggr="sum")((x_v, x_fact))      the fact row + the sum of its dim_v rows
                out_q = final_proj(TransformerConv((fused, x_q)))  dim_q attends over its fact rows
                (both outputs are kept: out_q for the dim_q nodes, fused for the fact nodes)
  dim-dim       out   = final_proj(TransformerConv((x_src, x_dst)))
TransformerConv: multi-head dot-product attention softmaxed over each destination's edges, heads
concatenated, plus a skip (root weight) projection of the destination. Every node type sums the outputs
of the routes that reach it, then LayerNorm per node type and ReLU.
"""
import itertools
import math
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from experiments.relgt import run as relgt

TABLES, FKEYS, RELATIONS, N_REL = relgt.TABLES, relgt.FKEYS, relgt.RELATIONS, relgt.N_REL
CHAIN_REL, N_ALL_REL = relgt.CHAIN_REL, relgt.N_ALL_REL
N_TYPES = len(TABLES)
READOUTS = ["user", "seeds"]


class Route(NamedTuple):
    """Edge types are (relation id, source node type), relation ids as in experiments/relgt: FK r for the
    edge row -> referenced row (the official 'f2p_<key>'), r + N_REL for its reverse ('rev_f2p_<key>'), and
    on the chain graph CHAIN_REL (row -> the actor's next row of its table) and CHAIN_REL + 1."""
    kind: str                     # "dim-dim" or "dim-fact-dim"
    attn: tuple                   # edge type of the attention edges (into dst)
    dst: int                      # node type that receives the attention output
    aggr: Optional[tuple] = None  # dim-fact-dim: edge type of the aggregation edges dim_v -> fact
    fact: Optional[int] = None    # dim-fact-dim: the fact node type, which also receives the fused rows

    def name(self):
        if self.kind == "dim-dim":
            return f"{TABLES[self.attn[1]]} -> {TABLES[self.dst]} ({rel_name(self.attn[0])})"
        return (f"{TABLES[self.aggr[1]]} -> {TABLES[self.fact]} -> {TABLES[self.dst]} "
                f"({rel_name(self.aggr[0])}, {rel_name(self.attn[0])})")


def rel_name(r):
    if r >= CHAIN_REL:
        return "chain next" if r == CHAIN_REL else "chain previous"
    return ("rev " if r >= N_REL else "") + RELATIONS[r % N_REL][1]


def atomic_routes(chain):
    """get_atomic_routes (atomic_routes.py) on this schema. Each key is (forward id, reverse id, referenced
    type), grouped by the table that holds it; a self-referencing key is a group of its own (the official
    'src--rel' key). With chain, each activity table also has its chain as a self-referencing key."""
    groups = {}
    for name, keys in FKEYS.items():
        t = TABLES.index(name)
        for col, ref in keys:
            r = RELATIONS.index((name, col))
            groups.setdefault((t, col) if ref == name else (t,), []).append((r, r + N_REL, TABLES.index(ref)))
    if chain:
        for name in relgt.ACTOR_COL:
            t = TABLES.index(name)
            groups[(t, "chain")] = [(CHAIN_REL, CHAIN_REL + 1, t)]
    routes = []
    for (t, *_), keys in groups.items():
        if len(keys) == 1:  # dim-dim: the key and its reverse
            fwd, rev, ref = keys[0]
            routes += [Route("dim-dim", (fwd, t), ref), Route("dim-dim", (rev, ref), t)]
            continue
        for (fq, _, bq), (_, rv, av) in itertools.permutations(keys, 2):
            routes.append(Route("dim-fact-dim", (fq, t), bq, (rv, av), t))
    return routes


def segment_softmax(score, index, n):
    """torch_geometric.utils.softmax: softmax of score [E, heads] over the edges that share index."""
    idx = index.unsqueeze(1).expand_as(score)
    mx = score.new_full((n, score.shape[1]), -math.inf).scatter_reduce(0, idx, score.detach(), "amax")
    e = (score - mx.gather(0, idx)).exp()
    return e / (score.new_zeros(n, score.shape[1]).scatter_add(0, idx, e).gather(0, idx) + 1e-16)


class RouteConv(nn.Module):
    """RelGNNConv of one route: torch_geometric TransformerConv(heads, concat=True, root_weight=True,
    beta=False, dropout=0) followed by final_proj; a dim-fact-dim route first fuses dim_v into the fact
    with SAGEConv(aggr="sum") (lin_l on the summed neighbours with bias, lin_r on the fact without)."""

    def __init__(self, kind, d, heads):
        super().__init__()
        self.heads, self.d = heads, d
        if kind == "dim-fact-dim":
            self.sage_l = nn.Linear(d, d)
            self.sage_r = nn.Linear(d, d, bias=False)
        self.lin_query, self.lin_key, self.lin_value, self.lin_skip = (nn.Linear(d, heads * d) for _ in range(4))
        self.final_proj = nn.Linear(heads * d, d)

    def fuse(self, x_v, x_fact, src, dst):
        """SAGEConv((x_v, x_fact)) over the edges dim_v -> fact: the fact rows fused with their dim_v rows."""
        agg = x_fact.new_zeros(x_fact.shape).index_add(0, dst, x_v[src])
        return self.sage_l(agg) + self.sage_r(x_fact)

    def attend(self, x_src, x_dst, src, dst):
        """final_proj(TransformerConv((x_src, x_dst))) for every dst node. final_proj is linear, so it is
        applied per edge before the sum, and to the skip term as one composed map: the same function and
        parameters without a [nodes, heads * d] tensor per route."""
        H, d = self.heads, self.d
        q = self.lin_query(x_dst[dst]).view(-1, H, d)
        k = self.lin_key(x_src[src]).view(-1, H, d)
        v = self.lin_value(x_src[src]).view(-1, H, d)
        alpha = segment_softmax((q * k).sum(-1) / math.sqrt(d), dst, len(x_dst))
        W = self.final_proj.weight
        msg = F.linear((v * alpha.unsqueeze(-1)).reshape(-1, H * d), W)
        skip = F.linear(x_dst, W @ self.lin_skip.weight, W @ self.lin_skip.bias + self.final_proj.bias)
        return skip.index_add(0, dst, msg)


class RelGNN(nn.Module):
    """RelGNN_Model on one example's K tokens, the forward signature of RowGNN (experiments/gnn).

    rows is relgt's RowEncoder (the stand-in for relbench's HeteroEncoder, a ResNet per table); the
    HeteroTemporalEncoder is relgt's PositionalEncoding of the token's age in days before the week, with a
    linear layer per node type, added to the row encoding. readout="user": the head (one linear layer, as
    the official MLP(num_layers=1)) reads the example's user token, the entity node; "seeds": the mean of
    the hop-0 tokens (RowGNN's readout; differs only with --chain-seed latest). The per-user history
    features (seed_x) are encoded by an MLP and added to the readout token(s) before message passing."""

    def __init__(self, rows, user_node, n_seed_feats, n_out, chain, d=128, layers=2, heads=4, readout="user"):
        super().__init__()
        self.rows, self.readout = rows, readout
        self.routes = atomic_routes(chain)
        self.register_buffer("user_node", torch.as_tensor(user_node, dtype=torch.long), persistent=False)
        self.time_pos = relgt.PositionalEncoding(d)
        self.time_lin = nn.ModuleList([nn.Linear(d, d) for _ in range(N_TYPES)])
        self.seed_encoder = (nn.Sequential(nn.Linear(n_seed_feats, d), nn.ReLU(), nn.Linear(d, d))
                             if n_seed_feats else None)
        self.convs = nn.ModuleList([nn.ModuleList([RouteConv(r.kind, d, heads) for r in self.routes])
                                    for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(d) for _ in range(N_TYPES)])
                                    for _ in range(layers)])
        self.head = nn.Linear(d, n_out)

    def forward(self, nodes, hops, paths, week, cidx, edge_index, edge_type, seed_x=None):
        B, K = nodes.shape
        N = B * K
        dev = nodes.device
        tfs, types = self.rows(nodes, week)
        x, t = tfs.reshape(N, -1), types.reshape(N)
        rows_ = torch.arange(B, device=dev)
        if self.readout == "user":  # the user's first token (repeats of a row carry no edges)
            first = (nodes == self.user_node[cidx].unsqueeze(1)).float().argmax(1)
            read = torch.zeros(B, K, dtype=torch.bool, device=dev)
            read[rows_, first] = True
        else:
            read = hops == 0
        if self.seed_encoder is not None:
            x = x + (read.unsqueeze(-1) * self.seed_encoder(seed_x).unsqueeze(1)).reshape(N, -1)
        age = self.rows.week_day[week].repeat_interleave(K) - self.rows.node_time[nodes.reshape(N)]

        # Tokens grouped by node type (the official x_dict); every edge type's edges in local indices.
        order = torch.argsort(t, stable=True)
        key = edge_type * N_TYPES + t[edge_index[0]]
        counts = torch.cat([torch.bincount(t, minlength=N_TYPES),
                            torch.bincount(key, minlength=N_ALL_REL * N_TYPES)]).tolist()  # the one sync
        tp = [0, *itertools.accumulate(counts[:N_TYPES])]
        ep = [0, *itertools.accumulate(counts[N_TYPES:])]
        pos = torch.empty_like(order)
        pos[order] = torch.arange(N, device=dev)
        loc = pos - torch.tensor(tp[:-1], device=dev)[t]
        src_loc, dst_loc = loc[edge_index[0]], loc[edge_index[1]]
        eorder = torch.argsort(key)

        def edges(etype):
            k = etype[0] * N_TYPES + etype[1]
            e = eorder[ep[k]:ep[k + 1]]
            return src_loc[e], dst_loc[e]

        route_edges = [(edges(r.attn), edges(r.aggr) if r.aggr else None) for r in self.routes]
        x, pe = x[order], self.time_pos(age)[order]
        xs = [x[tp[i]:tp[i + 1]] + lin(pe[tp[i]:tp[i + 1]]) for i, lin in enumerate(self.time_lin)]

        for convs, norms in zip(self.convs, self.norms):
            outs = [[] for _ in range(N_TYPES)]
            for r, conv, (attn_e, aggr_e) in zip(self.routes, convs, route_edges):
                if r.kind == "dim-dim":
                    outs[r.dst].append(conv.attend(xs[r.attn[1]], xs[r.dst], *attn_e))
                else:
                    fused = conv.fuse(xs[r.aggr[1]], xs[r.fact], *aggr_e)
                    outs[r.dst].append(conv.attend(fused, xs[r.dst], *attn_e))
                    outs[r.fact].append(fused)
            xs = [F.relu(norm(sum(o))) if o else h for norm, o, h in zip(norms, outs, xs)]

        h = torch.cat(xs)[pos].view(B, K, -1)
        if self.readout == "user":
            return self.head(h[rows_, first])
        m = read.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))
