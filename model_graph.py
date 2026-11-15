"""Draw what one user's example looks like to each graph model, built with the models' own code.

--model picks the graph (all built for one week start --week, rows visible before each table's cutoff):
  ego        experiments/gnn/run.py --graph-structure ego (default model): ego_tokens(); the star graph of
             the user and its 48 token slots. Edges as EgoGNN.forward: user <-> every token (colour =
             relation), push repo <-> member repo of the same repository (dashed). Inner ring hop 1,
             outer ring hop 2 (a label only: both rings link to the user directly).
  user-repo  --graph-structure user-repo: week_graph(); one graph per week of all cohort users and the
             repos they touched in 90 days, one edge per (user, repo, activity) with windowed counts.
             Shown: the user, its repos (hop 1) and the other users of those repos (hop 2), i.e. the
             receptive field of the model's 2 rounds user -> repo -> user.
  hetero     --graph-structure hetero: hetero_graph(); users, PRs, issues and repos, one edge per event
             (latest 32 per user and activity), repo -> user "received", PR / issue -> its repo. Every
             relation also runs reversed, so the 2-layer model sees the 2-hop neighbourhood drawn here.
             Parallel events between the same two nodes are drawn as one edge (hover: count, ages).
  chain      --graph-structure chain (RowGNN): the K = 100 tokens of experiments/relgt's chain row graph
             (sample_walk) and the edges among them (subgraph_edges), as the GNN uses them.
  relgt      experiments/relgt/run.py: the same tokens on --relgt-graph fk (default, sample_tokens: 1- and
             2-hop FK neighbours) or chain (sample_walk). RelGT attends over all K tokens; the edges
             drawn are the subgraph its GIN positional encoding sees.
The week graphs can be large: --max-per-hop keeps the nodes with the most edges to the previous hop
(the title says how many were dropped). Hover any node or edge for its features; a table below the
graph lists every node.

Run from the repo root:
    python model_graph.py                  # interactive app at http://127.0.0.1:8052: switch model, user,
                                           # week and the token options in the page
    python model_graph.py --html --model hetero --user <login or GitHub id> --week "2026-09-26 16:00"
    python model_graph.py --html --model chain --chain-seed latest --num-seeds 16  # as script_chain_ho0927.sh
    python model_graph.py --html --model relgt --relgt-graph chain --chain-seed latest --num-seeds 16
--html writes one static file, figures/graph_<model>_<login>_<date>.html, instead of serving the app.
In the app the command-line options are the starting values; graphs already built are cached.
"""
import argparse
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gnn_run", ROOT / "experiments" / "gnn" / "run.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
base = G.base

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           "#0b8a8f", "#8c564b", "#52514e", "#b7a400"]
SYMBOL = {"user": "star", "PR": "square", "issue": "diamond", "repo": "circle", "Users": "star",
          "Repositories": "circle", "PullRequests": "square", "Issues": "diamond",
          "IssueComments": "triangle-up", "PRReviews": "triangle-down", "PRReviewComments": "cross",
          "Pushes": "x"}
INK, MUTED, RING = "#1f1f1e", "#6b6a64", "#e4e3df"


# --- Names of users, PRs, issues and repos ---------------------------------------------------------

class Names:
    """Readable names. PR / issue / repo ids are row positions, as every model uses them."""

    def __init__(self):
        db = base.DB
        self.login = pd.read_parquet(db / "Users.parquet", columns=["login"]).login.values
        self.repo = pd.read_parquet(db / "Repositories.parquet", columns=["name"]).name.values
        self.pr = pd.read_parquet(db / "PullRequests.parquet", columns=["repo_id", "number", "title"])
        self.issue = pd.read_parquet(db / "Issues.parquet", columns=["repo_id", "number", "title"])

    def user(self, user_id):
        return str(self.login[user_id])

    def repo_name(self, repo_id):
        return str(self.repo[repo_id]) if 0 <= repo_id < len(self.repo) else str(repo_id)

    def _numbered(self, d, i):
        r = d.iloc[i]
        title = r.title if isinstance(r.title, str) else ""
        return f"{self.repo_name(int(r.repo_id))}#{r.number}", title

    def pr_name(self, pr_id):
        return self._numbered(self.pr, pr_id)

    def issue_name(self, issue_id):
        return self._numbered(self.issue, issue_id)


def entity_name(names, e, node_type, n_prs, repo_offset):
    """load_entity_events' global entity id (1 + PRs | issues | repos) -> (name, title)."""
    if node_type == 1:
        return names.pr_name(e - 1)
    if node_type == 2:
        return names.issue_name(e - 1 - n_prs)
    return names.repo_name(e - repo_offset), ""


# --- Graph spec: nodes and edges, then one drawing function ---------------------------------------
# nodes: key, kind (legend group / symbol), hop, name, hover, size; edges: a, b, rel, hover, dash.

def node(key, kind, hop, name, hover, size=10):
    return dict(key=key, kind=kind, hop=hop, name=name, hover=hover, size=size)


def edge(a, b, rel, hover="", dash=False):
    return dict(a=a, b=b, rel=rel, hover=hover, dash=dash)


def keep_neighbourhood(user, edges, hops, max_per_hop):
    """BFS from user over undirected edges, up to `hops`. Each hop keeps the max_per_hop nodes with the
    most edges to the previous hop. Returns {node: hop} and the number of nodes dropped per hop."""
    adj = {}
    for e in edges:
        adj.setdefault(e["a"], []).append(e["b"])
        adj.setdefault(e["b"], []).append(e["a"])
    hop_of, frontier, dropped = {user: 0}, [user], []
    for h in range(1, hops + 1):
        links = {}
        for v in frontier:
            for w in adj.get(v, []):
                if w not in hop_of:
                    links[w] = links.get(w, 0) + 1
        ranked = sorted(links, key=lambda w: (-links[w], str(w)))
        frontier = ranked[:max_per_hop]
        dropped.append(len(ranked) - len(frontier))
        hop_of.update({w: h for w in frontier})
    return hop_of, dropped


def ring_layout(nodes, edges):
    """Hop h on ring h; a node is placed near the mean angle of its neighbours on the inner ring."""
    pos = {}
    by_hop = {}
    for n in nodes:
        by_hop.setdefault(n["hop"], []).append(n["key"])
    nbrs = {}
    for e in edges:
        nbrs.setdefault(e["a"], set()).add(e["b"])
        nbrs.setdefault(e["b"], set()).add(e["a"])
    angle = {}
    for h in sorted(by_hop):
        keys = by_hop[h]
        if h == 0 and len(keys) == 1:
            pos[keys[0]], angle[keys[0]] = (0.0, 0.0), 0.0
            continue
        pref = []
        for k in keys:
            inner = [angle[w] for w in nbrs.get(k, ()) if w in angle]
            pref.append(np.angle(np.exp(1j * np.array(inner)).mean()) if inner else 0.0)
        order = np.argsort(pref, kind="stable")
        radius = 0.35 if h == 0 else h
        for j, i in enumerate(order):
            a = np.pi / 2 - 2 * np.pi * j / len(keys)
            pos[keys[i]], angle[keys[i]] = (radius * np.cos(a), radius * np.sin(a)), a
    return pos


def spring_layout(nodes, edges, pos, iters=300):
    """Fruchterman-Reingold from the ring layout, the hop-0 nodes pinned; for the token graphs."""
    keys = [n["key"] for n in nodes]
    idx = {k: i for i, k in enumerate(keys)}
    p = np.array([pos[k] for k in keys], float)
    pinned = np.array([n["hop"] == 0 for n in nodes])
    ab = np.array([(idx[e["a"]], idx[e["b"]]) for e in edges if e["a"] in idx and e["b"] in idx]).reshape(-1, 2)
    span = max(np.abs(p).max(), 1.0)
    p /= span
    k = 1.0 / np.sqrt(len(keys))
    for it in range(iters):
        d = p[:, None] - p[None]
        dist = np.linalg.norm(d, axis=-1) + 1e-6
        disp = (d / dist[..., None] * (k * k / dist)[..., None]).sum(1)
        if len(ab):
            dd = p[ab[:, 0]] - p[ab[:, 1]]
            dl = np.linalg.norm(dd, axis=-1, keepdims=True) + 1e-6
            f = dd / dl * dl ** 2 / k
            np.add.at(disp, ab[:, 0], -f)
            np.add.at(disp, ab[:, 1], f)
        t = 0.1 * (1 - it / iters) + 1e-3
        ln = np.linalg.norm(disp, axis=-1, keepdims=True) + 1e-9
        p[~pinned] += (disp / ln * np.minimum(ln, t))[~pinned]
    return {kk: tuple(v * span) for kk, v in zip(keys, p)}


def draw(nodes, edges, title, rings, layout, table_cols):
    nodes = sorted(nodes, key=lambda n: n["hop"])
    keys = {n["key"] for n in nodes}
    edges = [e for e in edges if e["a"] in keys and e["b"] in keys and e["a"] != e["b"]]
    pos = ring_layout(nodes, edges)
    if layout == "spring":
        pos = spring_layout(nodes, edges, pos)
    fig = go.Figure()
    rels = sorted({e["rel"] for e in edges})
    color = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(rels)}
    if layout == "ring":
        for h, txt in rings.items():
            fig.add_shape(type="circle", x0=-h, y0=-h, x1=h, y1=h, line=dict(color=RING, dash="dot"), layer="below")
            fig.add_annotation(x=0, y=h + 0.08, text=txt, showarrow=False, font=dict(size=10, color=MUTED))
    for r in rels:
        es = [e for e in edges if e["rel"] == r]
        xs = np.array([[pos[e["a"]][0], pos[e["b"]][0], np.nan] for e in es]).ravel()
        ys = np.array([[pos[e["a"]][1], pos[e["b"]][1], np.nan] for e in es]).ravel()
        dash = "dash" if es[0]["dash"] else None
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{r} ({len(es)})", legendgroup="edges",
                                 legendgrouptitle_text="edges (relation)", line=dict(color=color[r], width=1.4, dash=dash),
                                 opacity=0.65, hoverinfo="skip"))
        # Invisible midpoints carry the edge hover.
        mx = [(pos[e["a"]][0] + pos[e["b"]][0]) / 2 for e in es]
        my = [(pos[e["a"]][1] + pos[e["b"]][1]) / 2 for e in es]
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=6, color=color[r], opacity=0),
                                 hovertext=[f"<b>{r}</b><br>{e['hover']}" for e in es], hoverinfo="text",
                                 showlegend=False, legendgroup="edges"))
    kinds = list(dict.fromkeys(n["kind"] for n in nodes))
    for i, kd in enumerate(kinds):
        ns = [n for n in nodes if n["kind"] == kd]
        fig.add_trace(go.Scatter(
            x=[pos[n["key"]][0] for n in ns], y=[pos[n["key"]][1] for n in ns], mode="markers+text",
            name=f"{kd} ({len(ns)})", legendgroup="nodes", legendgrouptitle_text="nodes",
            text=[n["name"].split("/")[-1][:16] if n["hop"] <= 1 or len(nodes) < 60 else "" for n in ns],
            textposition="top center", textfont=dict(size=9, color=MUTED),
            marker=dict(size=[34 if n["hop"] == 0 else n["size"] for n in ns],
                        symbol=SYMBOL.get(kd.split(" ")[0], "circle"),
                        color=INK if ns[0]["hop"] == 0 and kd.startswith(("user", "Users")) else
                        PALETTE[(i + 3) % len(PALETTE)], line=dict(color="white", width=1)),
            hovertext=[n["hover"] for n in ns], hoverinfo="text"))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), width=1100, height=950, plot_bgcolor="white",
                      paper_bgcolor="white", xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                      hoverlabel=dict(align="left"))
    t = pd.DataFrame([{c: n.get(c, "") for c in table_cols} for n in nodes])
    return fig, t


def static_html(fig, t):
    html = [fig.to_html(include_plotlyjs="cdn", full_html=False)]
    if len(t):
        tab = go.Figure(go.Table(header=dict(values=list(t.columns), fill_color="#f2f1ed", align="left"),
                                 cells=dict(values=[t[c] for c in t.columns], align="left")))
        tab.update_layout(width=1100, height=min(60 + 26 * len(t), 4000), margin=dict(t=10, b=10))
        html.append(tab.to_html(include_plotlyjs=False, full_html=False))
    return "".join(html)


# --- Per model -----------------------------------------------------------------------------------

def ego_graph(ctx, u):
    """--graph-structure ego: the 48 slots of ego_tokens() and the edges of EgoGNN.forward."""
    ent_ev, _, repo_offset = ctx.entity_events()
    tok = G.ego_tokens(ent_ev, ctx.s, ctx.lags, np.ones(ctx.n, bool), ctx.args.ego_lookback, repo_offset)
    rel_names = ["padding"] + list(G.EVENT_COLS) + ["member"]
    types = ["padding", "PR", "issue", "repo"]
    feat_names = ["log1p user events", "log1p user active days", "log1p cohort events", "users / relations"]
    login = ctx.login(u)
    nodes = [node("user", "user", 0, login, f"user {login}<br>learned node (+ 60 user features with "
                  "--user-features)")]
    edges = []
    slot_of = {}
    for j in np.flatnonzero(tok["type"][u] != 0):
        t, e, rel, hop = (int(tok[k][u, j]) for k in ("type", "entity", "rel", "hop"))
        name, title = entity_name(ctx.names, e, t, ctx.n_prs, repo_offset)
        f = tok["feat"][u, j]
        age = float(tok["days"][u, j])
        hover = (f"slot {j} · hop {hop} · {types[t]} · relation {rel_names[rel]}<br><b>{name}</b>"
                 + (f"<br>{title[:80]}" if title else "") + f"<br>age {age:.1f} days before cutoff<br>"
                 + "<br>".join(f"{a}: {b:.2f}" for a, b in zip(feat_names, f)))
        nodes.append(dict(node(("slot", j), f"{types[t]} (hop {hop})", hop, name, hover, 8 + 4 * float(f[0])),
                          slot=j, relation=rel_names[rel], age=round(age, 1),
                          **{a: round(float(b), 2) for a, b in zip(feat_names, f)}))
        edges.append(edge("user", ("slot", j), rel_names[rel], f"user <-> {name}<br>age {age:.1f} d"))
        if t == 3:
            slot_of.setdefault(e, []).append((j, hop))
    for e, sl in slot_of.items():
        for ja, ha in sl:
            for jb, hb in sl:
                if ha == 1 and hb == 2:
                    edges.append(edge(("slot", ja), ("slot", jb), "same repo (push <-> member)", "", dash=True))
    n_tok = len(nodes) - 1
    title = (f"<b>ego</b> GNN · {login} · week {ctx.s:%Y-%m-%d %H:%M} UTC · {n_tok} of {G.EGO_K} slots, last "
             f"{ctx.args.ego_lookback:g} days before each table's cutoff"
             + (" · quiet example: the graph is the user node only" if n_tok == 0 else ""))
    rings = {1: "hop 1: 4 latest objects per activity", 2: "hop 2: 24 latest repos (linked to the user directly)"}
    return nodes, edges, title, rings, "ring", ["slot", "hop", "kind", "relation", "name", "age"] + feat_names


def user_repo_graph(ctx, u):
    """--graph-structure user-repo: week_graph() of the whole week, cut to the user's 2-hop field."""
    window = ctx.args.history_days or 90
    xr, edges_w, ages = G.week_graph(ctx.ev, ctx.s, ctx.lags, ctx.repo_is_org, np.ones(ctx.n, bool), times=True,
                                     window=window)
    # week_graph numbers its repos 0..R-1 in sorted order of the repo ids it saw; rebuild that order.
    seen = []
    for k in G.EDGE_TYPES:
        c = np.datetime64(ctx.s - ctx.lags[k])
        e = ctx.ev[k]
        m = ((e.avail.values < c) & (e.t.values >= c - np.timedelta64(int(window * 86400), "s"))
             & ~np.isnan(e.repo.values) & (e.uid.values >= 0))
        seen.append(e.repo.values[m].astype(np.int64))
    repo_ids = np.unique(np.concatenate(seen))
    assert len(repo_ids) == len(xr), "repo order differs from week_graph"
    windows = [w for w in G.EDGE_WINDOWS if w < window] + [window]
    all_edges = []
    for k, (src, dst, ef), (ag) in zip(G.EDGE_TYPES, edges_w, ages):
        for a, b, f, (last, first) in zip(src, dst, ef, ag):
            cnt = ", ".join(f"{w}d {np.expm1(x):.0f}" for w, x in zip(windows, f[:-1]))
            all_edges.append(edge(("u", int(a)), ("r", int(b)), k,
                                  f"events: {cnt}<br>days since last {last:.1f}, first {first:.1f}"))
    hop_of, dropped = keep_neighbourhood(("u", u), all_edges, 2, ctx.args.max_per_hop)
    nodes = []
    act = G.EDGE_TYPES
    for key, h in hop_of.items():
        if key[0] == "u":
            login = ctx.login(key[1])
            nodes.append(node(key, "user" if h == 0 else "other user", h, login, f"user {login} · hop {h}<br>"
                              "node features: the baseline's 60 user features"))
        else:
            r = int(key[1])
            name = ctx.names.repo_name(int(repo_ids[r]))
            x = xr[r]
            per = "<br>".join(f"{k}: 30d {np.expm1(x[1 + 3 * i]):.0f}, 90d {np.expm1(x[2 + 3 * i]):.0f}, "
                              f"users {np.expm1(x[3 + 3 * i]):.0f}" for i, k in enumerate(act))
            nodes.append(node(key, "repo", h, name, f"repo <b>{name}</b> · hop {h} · org {bool(x[0])}<br>{per}"))
    keep = set(hop_of)
    # Only edges the 2 rounds use: user - hop-1 repo, hop-1 repo - hop-2 user.
    edges = [e for e in all_edges if e["a"] in keep and e["b"] in keep and abs(hop_of[e["a"]] - hop_of[e["b"]]) == 1]
    login = ctx.login(u)
    title = (f"<b>user-repo</b> GNN · {login} · week {ctx.s:%Y-%m-%d %H:%M} UTC · {window}-day window · "
             f"{sum(h == 1 for h in hop_of.values())} repos, {sum(h == 2 for h in hop_of.values())} other users"
             + (f" (dropped {dropped[0]} / {dropped[1]} by --max-per-hop)" if any(dropped) else ""))
    return nodes, edges, title, {1: "hop 1: repos", 2: "hop 2: other cohort users"}, "ring", ["hop", "kind", "name"]


def hetero_graph(ctx, u):
    """--graph-structure hetero: hetero_graph() of the whole week, cut to the user's 2-hop field."""
    ent_ev, _, repo_offset = ctx.entity_events()
    window = ctx.args.history_days or 90
    g = G.hetero_graph(ent_ev, ctx.ev, ctx.s, ctx.lags, np.ones(ctx.n, bool), window, ctx.args.het_max_events,
                       ctx.repo_is_org, max_seen=ctx.args.history_days)
    tname = G.HET_TYPES
    agg = {}
    for (rel, st, dt, _), (src, dst, age) in zip(G.HET_RELS, g["edges"]):
        for a, b, ag in zip(src, dst, age):
            key = ((tname[st], int(a)), (tname[dt], int(b)), rel)
            agg.setdefault(key, []).append(float(ag))
    all_edges = []
    for (a, b, rel), ag in agg.items():
        what = "age" if rel.endswith("_in") else "event ages"
        all_edges.append(edge(a, b, rel, f"{len(ag)} edge(s), {what} (days): "
                              + ", ".join(f"{x:.1f}" for x in sorted(ag)[:8]) + (" ..." if len(ag) > 8 else "")))
    hop_of, dropped = keep_neighbourhood(("user", u), all_edges, 2, ctx.args.max_per_hop)
    feat = {1: [f"{k} events" for k in G.EVENT_COLS] + ["users", "days since first seen"],
            2: [f"{k} events" for k in G.EVENT_COLS] + ["users", "days since first seen"],
            3: [f"{k} events" for k in G.EVENT_COLS] + ["received", "users", "org"]}
    nodes = []
    for key, h in hop_of.items():
        t = tname.index(key[0])
        if t == 0:
            login = ctx.login(key[1])
            nodes.append(node(key, "user" if h == 0 else "other user", h, login,
                              f"user {login} · hop {h}<br>node features: the baseline's 60 user features"))
            continue
        e = int(g["ent"][t][key[1]])
        name, title = (entity_name(ctx.names, e, t, ctx.n_prs, repo_offset) if t < 3
                       else (ctx.names.repo_name(e), ""))
        x = g["x"][t][key[1]]
        vals = [f"{f}: {np.expm1(v) if f != 'org' else v:.0f}" for f, v in zip(feat[t], x) if v > 0 or f == "org"]
        kind = {1: "PR", 2: "issue", 3: "repo"}[t]
        nodes.append(node(key, kind, h, name, f"{kind} <b>{name}</b> · hop {h}" + (f"<br>{title[:80]}" if title else "")
                          + "<br>" + "<br>".join(vals)))
    login = ctx.login(u)
    title = (f"<b>hetero</b> GNN · {login} · week {ctx.s:%Y-%m-%d %H:%M} UTC · {window}-day window, latest "
             f"{ctx.args.het_max_events} events per user and activity · {len(hop_of) - 1} nodes within 2 hops"
             + (f" (dropped {dropped[0]} / {dropped[1]} by --max-per-hop)" if any(dropped) else ""))
    rings = {1: "hop 1: the user's PRs, issues, repos", 2: "hop 2: their repos, other users"}
    return nodes, all_edges, title, rings, "ring", ["hop", "kind", "name"]


def token_graph(ctx, u, structure):
    """--graph-structure chain (gnn) and experiments/relgt: the K tokens of one example and their edges."""
    R = G.load_relgt()
    a = ctx.args
    Gr = ctx.relgt_graph(structure)
    min_lag = min(ctx.lags.values())
    cutoff = np.array([((ctx.s - ctx.lags[R.LAG_KEY[t]] if t in R.LAG_KEY else ctx.s - min_lag) - base.DATA_START)
                       / base.DAY for t in R.TABLES], np.float32)
    users_t = R.TABLES.index("Users")
    seed = np.array([Gr.offset[users_t] + int(ctx.cohort.user_id.values[u])])
    rng = np.random.default_rng(a.seed)
    if structure == "chain":
        nodes_, hops, _ = R.sample_walk(Gr, seed, cutoff, a.num_neighbors, rng, a.max_hops, lookback=a.lookback_days,
                                        start=a.chain_seed, exclude_push=a.exclude_push, num_seeds=a.num_seeds)
        fallback = a.max_hops + 1
    else:
        nodes_, hops, _ = R.sample_tokens(Gr, seed, cutoff, a.num_neighbors, rng, lookback=a.lookback_days,
                                          exclude_push=a.exclude_push)
        fallback = R.MAX_HOP
    visible = R.visible_nodes(Gr, cutoff, a.lookback_days, a.exclude_push)
    tok, hp = nodes_[0], hops[0]
    _, ei, ej, rel = R.subgraph_edges(Gr, nodes_, "none", seed, visible)
    actor_rels = set(Gr.actor_rel.tolist())
    pair_rel = {}
    for i, j, r in zip(ei, ej, rel):
        va, vb = int(tok[i]), int(tok[j])
        if va == vb:
            continue
        if r >= R.CHAIN_REL:
            name = "chain: next row of the same user and table"
        else:
            t, col = R.RELATIONS[r % R.N_REL]
            name = f"{t}.{col}" + (" (owner)" if structure == "chain" and r % R.N_REL in actor_rels else "")
        pair_rel.setdefault(tuple(sorted((va, vb))), name)
    edges = [edge(a_, b_, r, f"{r}") for (a_, b_), r in pair_rel.items()]

    hop_of = {}
    for v, h in zip(tok[::-1], hp[::-1]):  # the first (smallest index) occurrence wins
        hop_of[int(v)] = int(h)
    nodes = []
    for v, h in hop_of.items():
        t = R.TABLES[Gr.ntype[v]]
        row = v - int(Gr.offset[Gr.ntype[v]])
        title = ""
        if t == "Users":
            name = ctx.names.user(row)
        elif t == "Repositories":
            name = ctx.names.repo_name(row)
        elif t == "PullRequests":
            name, title = ctx.names.pr_name(row)
        elif t == "Issues":
            name, title = ctx.names.issue_name(row)
        else:
            rp = int(Gr.repo[v])
            name = f"{t} row {row}" + (f" in {ctx.names.repo_name(rp - int(Gr.offset[1]))}" if rp >= 0 else "")
        age = cutoff[Gr.ntype[v]] - Gr.time[v]
        hop_txt = "seed" if h == 0 else ("random fallback" if h == fallback else f"hop {h}")
        hover = (f"{t} · {hop_txt}<br><b>{name}</b>" + (f"<br>{title[:80]}" if title else "")
                 + (f"<br>{age:.1f} days before its table's cutoff" if np.isfinite(age) else ""))
        nodes.append(dict(node(v, t, min(h, fallback), name, hover, 9), table=t,
                          age=round(float(age), 1) if np.isfinite(age) else ""))
    login = ctx.login(u)
    model = "chain GNN (RowGNN)" if a.model == "chain" else f"RelGT, {structure} graph"
    seed_txt = "user" if a.chain_seed == "user" or structure == "fk" else f"{a.num_seeds} latest actions"
    title = (f"<b>{model}</b> · {login} · week {ctx.s:%Y-%m-%d %H:%M} UTC · K = {a.num_neighbors} tokens, "
             f"{len(hop_of)} distinct · seed: {seed_txt}"
             + (f" · max hops {a.max_hops}" if structure == "chain" else " · 1- and 2-hop FK neighbours")
             + (f" · lookback {a.lookback_days:g} d" if a.lookback_days else ""))
    return nodes, edges, title, {}, "spring", ["hop", "table", "name", "age"]


# --- Main ----------------------------------------------------------------------------------------

class Context:
    def __init__(self, args):
        self.args = args
        self.s = pd.Timestamp(args.week)
        self.cohort, self.ev, _, data_end = base.load()
        self.lags = {k: base.TEST_START - v for k, v in data_end.items()}
        self.n = len(self.cohort)
        repos = pd.read_parquet(base.DB / "Repositories.parquet", columns=["repo_id", "org_id"])
        self.repo_is_org = np.zeros(repos.repo_id.max() + 1, bool)
        self.repo_is_org[repos.repo_id.values] = repos.org_id.notna().values
        self.names = Names()
        self.n_prs = len(self.names.pr)
        self._ent = None
        self._graphs = {}
        self.week_cache = {}

    def relgt_graph(self, structure):
        if structure not in self._graphs:
            self._graphs[structure] = G.load_relgt().Graph(structure)
        return self._graphs[structure]

    def entity_events(self):
        if self._ent is None:
            self._ent = G.load_entity_events(self.cohort)
        return self._ent

    def login(self, cidx):
        return self.names.user(int(self.cohort.user_id.values[cidx]))

    def find_user(self, text):
        """Cohort index of a login or GitHub user id, or None."""
        if not hasattr(self, "_logins"):
            self._logins = np.char.lower(np.array([self.login(i) for i in range(self.n)]).astype(str))
        hit = np.flatnonzero((self._logins == str(text).strip().lower())
                             | (self.cohort.github_user_id.astype(str).values == str(text).strip()))
        return int(hit[0]) if len(hit) else None

    def pick_user(self):
        """--user, else the test user at the 75th percentile of visible events in the last 30 days."""
        if self.args.user:
            u = self.find_user(self.args.user)
            if u is None:
                raise SystemExit(f"{self.args.user!r} is not a cohort user")
            return u
        counts = np.zeros(self.n)
        for k in G.EVENT_COLS:
            c = np.datetime64(self.s - self.lags[k])
            e = self.ev[k]
            m = (e.avail.values < c) & (e.t.values >= c - np.timedelta64(30, "D"))
            counts += np.bincount(e.uid.values[m], minlength=self.n)
        test = np.flatnonzero(np.isin(self.cohort.github_user_id.values, pd.read_csv(base.TEST_INPUT).UserID.values))
        target = np.quantile(counts[test], 0.75)
        return int(test[np.argmin(np.abs(counts[test] - target))])


BUILDERS = {"ego": ego_graph, "user-repo": user_repo_graph, "hetero": hetero_graph,
            "chain": lambda ctx, u: token_graph(ctx, u, "chain"),
            "relgt": lambda ctx, u: token_graph(ctx, u, ctx.args.relgt_graph)}
TOKEN_OPTS = ["num_neighbors", "max_hops", "chain_seed", "num_seeds", "lookback_days", "exclude_push", "seed"]
MODEL_OPTS = {"ego": ["ego_lookback"], "user-repo": ["history_days", "max_per_hop"],
              "hetero": ["history_days", "het_max_events", "max_per_hop"], "chain": TOKEN_OPTS,
              "relgt": ["relgt_graph"] + TOKEN_OPTS}


def build(ctx, args, u):
    """Figure and node table of one example, cached by everything that changes it."""
    key = (args.model, args.week, u) + tuple(getattr(args, o) for o in MODEL_OPTS[args.model])
    if key not in ctx.week_cache:
        ctx.args, ctx.s = args, pd.Timestamp(args.week)
        nodes, edges, title, rings, layout, cols = BUILDERS[args.model](ctx, u)
        keys = {n["key"] for n in nodes}
        edges = [e for e in edges if e["a"] in keys and e["b"] in keys and e["a"] != e["b"]]
        fig, table = draw(nodes, edges, title, rings, layout, cols)
        ctx.week_cache[key] = (fig, table, len(nodes), len(edges))
    return ctx.week_cache[key]


def serve(ctx, args, u0, host, port):
    from dash import Dash, Input, Output, State, dash_table, dcc, html

    weeks = [base.TEST_START - i * base.WEEK for i in range(0, 46)]
    weeks = [w for w in weeks if w >= base.DATA_START + 30 * base.DAY]
    week_opts = [{"label": f"{w:%Y-%m-%d %H:%M} UTC" + (" (test week)" if w == base.TEST_START else ""),
                  "value": str(w)} for w in weeks]
    if str(pd.Timestamp(args.week)) not in [o["value"] for o in week_opts]:
        week_opts.insert(0, {"label": str(pd.Timestamp(args.week)), "value": str(pd.Timestamp(args.week))})
    test = set(pd.read_csv(base.TEST_INPUT).UserID.values)
    user_opts = [{"label": ctx.login(i) + (" · test" if g in test else ""), "value": i}
                 for i, g in enumerate(ctx.cohort.github_user_id.values)]
    models = [("ego", "Ego GNN"), ("user-repo", "User-repo GNN"), ("hetero", "Hetero GNN"),
              ("chain", "Chain GNN"), ("relgt-fk", "RelGT, FK graph"), ("relgt-chain", "RelGT, chain graph")]
    start_model = args.model if args.model != "relgt" else f"relgt-{args.relgt_graph}"

    def field(label, comp):
        return html.Div([html.Label(label, style={"fontSize": 12, "color": MUTED}), comp],
                        style={"display": "flex", "flexDirection": "column", "gap": 2})

    num = dict(type="number", debounce=True, style={"width": 90})
    app = Dash(__name__, title="Model graphs")
    app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif", "padding": "12px 16px", "color": INK,
                                 "background": "white"}, children=[
        dcc.RadioItems(id="model", options=[{"label": l, "value": v} for v, l in models], value=start_model,
                       inline=True, inputStyle={"marginRight": 4}, labelStyle={"marginRight": 18, "fontSize": 15}),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": 16, "alignItems": "end", "margin": "10px 0"},
                 children=[
            field("user (type to search)", dcc.Dropdown(id="user", options=user_opts, value=u0, clearable=False,
                                                        style={"width": 300})),
            field("week start", dcc.Dropdown(id="week", options=week_opts, value=str(pd.Timestamp(args.week)),
                                             clearable=False, style={"width": 260})),
            field("ego lookback (d)", dcc.Input(id="ego_lookback", value=args.ego_lookback, min=1, **num)),
            field("window (d, user-repo / hetero)", dcc.Input(id="history_days", value=args.history_days or 90,
                                                              min=1, **num)),
            field("max per hop", dcc.Input(id="max_per_hop", value=args.max_per_hop, min=1, **num)),
            field("K tokens", dcc.Input(id="num_neighbors", value=args.num_neighbors, min=2, **num)),
            field("chain seed", dcc.RadioItems(id="chain_seed", options=["user", "latest"], value=args.chain_seed,
                                               inline=True, labelStyle={"marginRight": 8})),
            field("num seeds", dcc.Input(id="num_seeds", value=args.num_seeds, min=1, **num)),
            field("token lookback (d)", dcc.Input(id="lookback_days", value=args.lookback_days, min=1, **num)),
            field("", dcc.Checklist(id="exclude_push", options=[{"label": " exclude push", "value": "x"}],
                                    value=["x"] if args.exclude_push else [])),
        ]),
        html.Div(id="status", style={"fontSize": 13, "color": MUTED, "minHeight": 18}),
        dcc.Loading(dcc.Graph(id="graph", style={"height": "82vh"}, config={"scrollZoom": True})),
        dash_table.DataTable(id="table", page_size=50, sort_action="native", filter_action="native",
                             style_cell={"fontSize": 12, "textAlign": "left", "maxWidth": 420,
                                         "overflow": "hidden", "textOverflow": "ellipsis"},
                             style_header={"background": "#f2f1ed", "fontWeight": 600}),
    ])

    @app.callback(Output("graph", "figure"), Output("table", "data"), Output("table", "columns"),
                  Output("status", "children"),
                  Input("model", "value"), Input("user", "value"), Input("week", "value"),
                  Input("ego_lookback", "value"), Input("history_days", "value"), Input("max_per_hop", "value"),
                  Input("num_neighbors", "value"), Input("chain_seed", "value"), Input("num_seeds", "value"),
                  Input("lookback_days", "value"), Input("exclude_push", "value"))
    def update(model, u, week, ego_lb, hist, mph, k, chain_seed, num_seeds, tok_lb, nopush):
        a = copy.copy(args)
        a.model, a.relgt_graph = (model.split("-")[0], model.split("-")[1]) if model.startswith("relgt") \
            else (model, args.relgt_graph)
        a.week, a.ego_lookback = week, float(ego_lb or 14)
        a.history_days = None if not hist or int(hist) == 90 else int(hist)
        a.max_per_hop, a.num_neighbors = int(mph or 60), int(k or 100)
        a.chain_seed, a.num_seeds = chain_seed, int(num_seeds or 1)
        a.lookback_days, a.exclude_push = (float(tok_lb) if tok_lb else None), bool(nopush)
        if a.num_seeds > 1 and a.chain_seed != "latest":
            a.num_seeds = 1
        fig, t, n_nodes, n_edges = build(ctx, a, int(u))
        fig = go.Figure(fig).update_layout(width=None, height=None, autosize=True, margin=dict(l=10, r=10, t=60, b=10))
        used = ", ".join(o.replace("_", " ") for o in MODEL_OPTS[a.model])
        status = (f"{n_nodes} nodes, {n_edges} edges · options used by this model: {used}"
                  + (" · num seeds needs chain seed = latest" if int(num_seeds or 1) > 1 and chain_seed != "latest"
                     and a.model in ("chain", "relgt") else ""))
        cols = [{"name": c, "id": c} for c in t.columns]
        return fig, t.astype(str).to_dict("records"), cols, status

    print(f"serving on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(BUILDERS), default="ego")
    ap.add_argument("--user", help="login or GitHub user id (default: a test user at the 75th percentile of "
                                   "visible events in the last 30 days)")
    ap.add_argument("--week", default=str(base.TEST_START), help="week start (UTC), as in the models")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-per-hop", type=int, default=60, help="user-repo / hetero: nodes drawn per hop")
    g = ap.add_argument_group("model options (same names and defaults as the models)")
    g.add_argument("--ego-lookback", type=float, default=14, help="ego")
    g.add_argument("--history-days", type=int, default=None, help="user-repo / hetero window (default 90)")
    g.add_argument("--het-max-events", type=int, default=32, help="hetero")
    g.add_argument("--relgt-graph", choices=["fk", "chain"], default="fk", help="relgt: --graph-structure of relgt")
    g.add_argument("--num-neighbors", type=int, default=100, help="chain / relgt: K tokens")
    g.add_argument("--max-hops", type=int, default=50, help="chain graph walk")
    g.add_argument("--chain-seed", choices=["user", "latest"], default="user")
    g.add_argument("--num-seeds", type=int, default=1)
    g.add_argument("--lookback-days", type=float, default=None, help="chain / relgt")
    g.add_argument("--exclude-push", action="store_true", help="chain / relgt")
    g.add_argument("--seed", type=int, default=0, help="chain / relgt token sampling")
    ap.add_argument("--html", action="store_true", help="write one static HTML file instead of serving the app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8052)
    args = ap.parse_args()

    ctx = Context(args)
    u = ctx.pick_user()
    login = ctx.login(u)
    print(f"model {args.model}, user {login} (github id {ctx.cohort.github_user_id.values[u]}), week {ctx.s}")
    for k in G.EVENT_COLS:
        print(f"  cutoff {k:18s} {ctx.s - ctx.lags[k]}")
    if not args.html:
        serve(ctx, args, u, args.host, args.port)
        return
    fig, table, n_nodes, n_edges = build(ctx, args, u)
    html = static_html(fig, table)
    print(f"{n_nodes} nodes, {n_edges} edges")
    with pd.option_context("display.max_rows", 40, "display.width", 200, "display.max_colwidth", 50):
        print(table)
    tag = args.model + (f"_{args.relgt_graph}" if args.model == "relgt" else "")
    out = args.out or ROOT / "figures" / f"graph_{tag}_{login}_{ctx.s:%Y%m%d}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{tag} graph {login}</title></head>"
                   f"<body style='background:white'>{html}</body></html>", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
