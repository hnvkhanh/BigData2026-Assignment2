"""RelGT's input for one test user: the sampled token subgraph and what the sample covers.

Builds the relational entity graph of experiments/relgt/run.py (every row a node, every foreign
key an edge), samples the K = 100 tokens of each test user at the test week exactly as training
does, and draws, for one example user:
  * left: the 100 tokens and the edges among them, coloured by table;
  * right: when the user's own activity rows happened vs which of them were sampled.
Also prints, over all test users, how much of the recent activity the sample keeps.

--graph-structure, --max-hops, --action-edges, --lookback-days, --exclude-push and --fk-order are the same
options as in experiments/relgt/run.py; chain and action edges are drawn dark, FK edges light.

--interactive serves the graph as a Dash app instead of writing the figure: pick any test user from
the search box (or click a test user's node) to see every visible node around them, with the K
sampled tokens highlighted. "Every node" is the whole 1+2-hop neighbourhood for the FK graph; on the
chain graph the walk stops at --max-hops or --max-nodes, and the page says when the limit cut it.
The page also picks the seed: the user, or the user's k latest visible actions. The seeds count
towards the K tokens. experiments/relgt/run.py trains on the user or, on the chain graph, on the k
latest actions (--chain-seed latest --num-seeds k); latest-action seeds on the FK graph are a preview.
On the chain graph a user links to their latest row of each table, and another user's sampled row
brings that user in, with the link drawn in its own colour.
It also picks the cut-off date, which acts like a training week's start: at TEST_START's time of day,
each table visible up to its usual data lag before it. Actions can be coloured by their age, the time
from the cut-off date to the row, the same clock as the model's relative time.

Run from the repo root:
    python relgt_graph.py                                   # figures/relgt_token_graph.png
    python relgt_graph.py --action-edges both --lookback-days 90
                                                            # figures/relgt_token_graph_both_lb90.png
    python relgt_graph.py --graph-structure chain           # figures/relgt_token_graph_chain.png
    python relgt_graph.py --interactive                     # http://127.0.0.1:8051
"""
import argparse
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("relgt_run", ROOT / "experiments" / "relgt" / "run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
base, TABLES = R.base, R.TABLES
OUT = ROOT / "figures" / "relgt_token_graph.png"
K = 100

# Categorical slots in fixed order, one per table.
COLOR = dict(zip(TABLES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                          "#6250d6", "#e34948"]))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
K_BOX = {"user": {"display": "none"}, "latest": {}}  # the k control shows only for latest-action seeds
# Age colouring: one hue, dark = recent, on a fixed log scale so colours compare across users and dates.
# It starts at 10 days: the data lag (at least 10 days) keeps every visible action at least that old.
AGE_SCALE = [[0, "#104281"], [0.5, "#2a78d6"], [1, "#b7d3f6"]]
AGE_TICKS = [10, 14, 21, 30, 60, 90, 180, 365]
NO_AGE = "#a8a7a1"  # Users and Repositories: not actions
OWNER_EDGE = "#c2410c"  # chain graph: an action and the user it belongs to
MAX_USER_LABELS = 40  # chain graph: write the logins on the graph up to this many users
SYMBOL = dict(zip(TABLES, ["pentagon", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "circle"]))


def log_age(days):
    return np.log10(np.clip(days, AGE_TICKS[0], AGE_TICKS[-1]))


def table_cutoffs(s, lags):
    """Per table, the cutoff in days for a week starting at s: rows with an earlier row time are visible."""
    min_lag = min(lags.values())
    return np.array([((s - lags[R.LAG_KEY[t]] if t in R.LAG_KEY else s - min_lag) - base.DATA_START) / base.DAY
                     for t in TABLES], np.float32)


def neighbourhood(G, seed, visible):
    """Visible 1-hop and 2-hop neighbours of one or more seeds (no caps), each node at its smallest hop."""
    seed = np.atleast_1d(seed)
    _, n1 = R.expand(G.ptr, G.adj, np.zeros(len(seed), np.int64), seed)
    n1 = np.setdiff1d(n1[visible[n1]], seed)
    _, n2 = R.expand(G.ptr, G.adj, np.zeros(len(n1), np.int64), n1)
    n2 = np.setdiff1d(np.unique(n2[visible[n2]]), np.r_[n1, seed])
    return n1, n2


def own_rows(G, seed, visible):
    """The seed user's visible activity rows."""
    r = G.aadj[G.aptr[seed]:G.aptr[seed + 1]]
    return r[visible[r]]


def edge_pairs(G, nodes, action_edges, user, visible):
    """Undirected edges among the given nodes, by kind ("owner": chain graph, a row and its user)."""
    _, ei, ej, rel = R.subgraph_edges(G, nodes[None, :], action_edges, np.array([user]), visible)
    pairs = {"fk": set(), "owner": set(), "chain": set(), "action": set()}
    actor_rels = set(G.actor_rel.tolist())  # not any row-user FK: an issue's assignee can be its author
    for a, b, t in zip(ei, ej, rel):
        if nodes[a] != nodes[b]:
            kind = "fk" if t < 2 * R.N_REL else "chain" if t >= R.CHAIN_REL else "action"
            if kind == "fk" and G.structure == "chain" and t % R.N_REL in actor_rels:
                kind = "owner"
            pairs[kind].add(tuple(sorted((int(nodes[a]), int(nodes[b])))))
    pairs["fk"] -= pairs["owner"]
    return pairs


def seed_set(G, user, visible, mode, k):
    """Seeds as R.sample_walk picks them. mode "user": the user. "latest": the k latest actions
    (newest first; the user if it has none)."""
    o = own_rows(G, user, visible)
    if mode == "user" or not len(o):
        return np.array([user])
    return o[np.lexsort((o, G.time[o]))][::-1][:k]


def all_nodes(G, seeds, visible):
    """FK graph: every visible node around the seeds and its hop, 1+2 hops."""
    n1, n2 = neighbourhood(G, seeds, visible)
    return np.r_[seeds, n1, n2], np.r_[np.zeros(len(seeds), np.int8), np.ones(len(n1), np.int8),
                                       np.full(len(n2), 2, np.int8)]


def radial_layout(G, nodes, hop):
    """x, y of sorted unique nodes: hop h on ring h. Each ring runs by parent, then repository, then
    table and time, and an outer node sits near the middle of its parents, so FK edges stay short.
    One seed sits at the centre; several sit on a small inner ring, newest first."""
    angle = np.zeros(len(nodes))
    radius = hop.astype(np.float64)
    is_repo = G.ntype[nodes] == TABLES.index("Repositories")
    repo = np.where(is_repo, nodes, G.repo[nodes])
    rng = np.random.default_rng(0)
    ring0 = np.flatnonzero(hop == 0)
    if len(ring0) > 1:
        angle[ring0[np.argsort(-G.time[nodes[ring0]], kind="stable")]] = 2 * np.pi * np.arange(len(ring0)) / len(ring0)
        radius[ring0] = 0.5
    for h in np.unique(hop[hop > 0]):
        ring = np.flatnonzero(hop == h)
        i, v = R.expand(G.ptr, G.adj, np.arange(len(ring)), nodes[ring])
        if G.structure == "chain":  # users have no static edge to their rows
            i2, v2 = R.expand(G.aptr, G.aadj, np.arange(len(ring)), nodes[ring])
            i, v = np.r_[i, i2], np.r_[v, v2]
        j = np.minimum(np.searchsorted(nodes, v), len(nodes) - 1)
        hit = (nodes[j] == v) & (hop[j] == h - 1)
        parent = np.arctan2(np.bincount(i[hit], np.sin(angle[j[hit]]), len(ring)),
                            np.bincount(i[hit], np.cos(angle[j[hit]]), len(ring))) % (2 * np.pi)
        o = np.lexsort((G.time[nodes[ring]], G.ntype[nodes[ring]], repo[ring], parent))
        even = 2 * np.pi * np.arange(len(ring)) / len(ring)
        # Ring 1 of a single seed hangs from it alone: spread evenly. Else: stay by the parents, spread a little.
        off = (even - parent[o] + np.pi) % (2 * np.pi) - np.pi
        angle[ring[o]] = even if h == 1 and len(ring0) == 1 else parent[o] + 0.25 * off
        radius[ring] += rng.uniform(-0.3, 0.3, len(ring)) * min(1, len(ring) / 200)  # spread crowded rings
    return radius * np.cos(angle), radius * np.sin(angle)


def serve(args, G, lags, test, seeds, nodes, hops, example):
    """Dash app: one test user's graph at a chosen cut-off date, chosen from a search box or by clicking
    their node, grown from the user or from their k latest actions."""
    logins = pd.read_parquet(base.DB / "Users.parquet", columns=["login"]).login.values
    user0 = G.offset[TABLES.index("Users")]
    test_of = {int(sd): i for i, sd in enumerate(seeds)}
    chain = args.graph_structure == "chain"
    cli_seed = args.chain_seed if chain else "user"  # what the precomputed nodes and hops start from
    options = sorted(({"label": f"{logins[sd - user0]} ({uid})", "value": i}
                      for i, (sd, uid) in enumerate(zip(seeds, test.UserID.values))),
                     key=lambda o: o["label"].lower())

    def node_label(v):
        t = TABLES[G.ntype[v]]
        if t != "Users":
            by = f", by {logins[G.actor[v] - user0]}" if G.actor[v] >= 0 else ""
            return f"{t} row {v - G.offset[G.ntype[v]]}{by}"
        return f"{logins[v - user0]}" + (" (test user, click to select)" if int(v) in test_of else "")

    @lru_cache(maxsize=8)
    def state_at(date):
        """Cut-off moment, per-table cutoffs, its day number and the visible rows for a picked date."""
        s = pd.Timestamp(date).normalize() + (base.TEST_START - base.TEST_START.normalize())
        cutoff = table_cutoffs(s, lags)
        return s, cutoff, (s - base.DATA_START) / base.DAY, R.visible_nodes(G, cutoff, args.lookback_days, args.exclude_push)

    def visible_until(cutoff):
        """Where each table's visible rows end, naming only the tables that differ from the most common day."""
        days = pd.Series(base.DATA_START + cutoff * base.DAY).dt.strftime("%Y-%m-%d").values
        groups = pd.Series(TABLES).groupby(days).agg(list)
        main_day = groups.map(len).idxmax()
        rest = [f"{', '.join(groups[d])} until {d}" for d in groups.index if d != main_day]
        return f"Rows visible until {main_day}" + (f" ({'; '.join(rest)})." if rest else ".")

    def walk_of(i, mode, k, cutoff, size, cap1, rng, fk_order):
        """R.sample_walk for one test user: size tokens, seeds first."""
        n, h, _ = R.sample_walk(G, seeds[i:i + 1], cutoff, size, rng, args.max_hops, cap1=cap1,
                                lookback=args.lookback_days, start=mode, exclude_push=args.exclude_push, num_seeds=k,
                                fk_order=fk_order)
        return n[0], h[0]

    def tokens_of(i, mode, k, roots, cutoff, s, visible, fk_order):
        """The K tokens, seeds first; the same ones as the static figure when it has this seed and date."""
        if s == base.TEST_START and mode == cli_seed and k == args.num_seeds and fk_order == args.fk_order:
            return nodes[i], hops[i]
        rng = np.random.default_rng(i)
        if chain:
            return walk_of(i, mode, k, cutoff, K, 1000, rng, fk_order)
        n1, n2 = neighbourhood(G, roots, visible)
        pool, hop = np.r_[n1, n2], np.r_[np.ones(len(n1), np.int8), np.full(len(n2), 2, np.int8)]
        perm = rng.permutation(len(pool))
        if fk_order == "recent":  # newest first, ties random
            perm = perm[np.argsort(-G.time[pool[perm]], kind="stable")]
        pick = perm[:K - len(roots)]
        return np.r_[roots, pool[pick]], np.r_[np.zeros(len(roots), np.int8), hop[pick]]

    @lru_cache(maxsize=32)
    def user_view(date, i, view, mode, k, colour, fk_order):
        s_time, cutoff, week_day, visible = state_at(date)
        roots = seed_set(G, seeds[i], visible, mode, k)
        tok, hp = tokens_of(i, mode, k, roots, cutoff, s_time, visible, fk_order)
        if view != "all":
            extra = tok, hp
        elif chain:  # the same walk with room for everything; drop the random rows of an empty walk
            n, h = walk_of(i, mode, k, cutoff, args.max_nodes, args.max_nodes, np.random.default_rng(0), fk_order)
            extra = n[h <= args.max_hops], h[h <= args.max_hops]
        else:
            extra = all_nodes(G, roots, visible)
        v, h = np.r_[tok, extra[0]], np.r_[hp, extra[1]]
        o = np.lexsort((h, v))
        first = np.r_[True, v[o][1:] != v[o][:-1]]  # each node once, at its smallest hop
        shown, hop = v[o][first], h[o][first]
        x, y = radial_layout(G, shown, hop)
        pairs = edge_pairs(G, shown, "none", seeds[i], visible)
        pairs["action"] = edge_pairs(G, np.unique(tok), args.action_edges, seeds[i], visible)["action"]
        n_edges = sum(map(len, pairs.values()))
        root_set = set(roots.tolist())
        pairs["seed"] = {p for p in pairs["fk"] if root_set & set(p)}  # faint: they would hide the centre
        pairs["fk"] -= pairs["seed"]

        fig = go.Figure()
        edge_styles = [("seed", "foreign key to a seed", dict(color="#e4e3df", width=0.4)),
                       ("fk", "foreign key", dict(color="#c9c8c3", width=0.6)),
                       ("owner", "action to its user", dict(color=OWNER_EDGE, width=1.2)),
                       ("chain", "prev / next same type", dict(color="#52514e", width=1.2)),
                       ("action", f"next action, {args.action_edges}", dict(color="#52514e", width=1, dash="dash"))]
        for kind, label, line in edge_styles:
            if pairs[kind]:
                a, b = np.searchsorted(shown, np.array(sorted(pairs[kind]))).T
                nan = np.full(len(a), np.nan)
                # WebGL lines have no dash styles, so a dashed kind needs an SVG trace.
                trace = go.Scatter if "dash" in line else go.Scattergl
                fig.add_trace(trace(x=np.c_[x[a], x[b], nan].ravel(), y=np.c_[y[a], y[b], nan].ravel(),
                                    mode="lines", line=line, hoverinfo="skip",
                                    name=f"{label} ({len(a):,})"))

        by_age = colour == "age"
        age = week_day - G.time[shown]  # days from the row to the cut-off date
        custom = np.c_[shown, hop, age]
        text = np.array([logins[n - user0] if n == seeds[i] else node_label(n) for n in shown], dtype=object)
        is_root = np.isin(shown, roots)
        sampled = np.isin(shown, tok) & ~is_root
        marks = [(~sampled & ~is_root, "not sampled", dict(size=6, opacity=0.8 if by_age else 0.45)),
                 (sampled, "sampled token", dict(size=11, line=dict(color=INK, width=1.2)))]
        for mask, status, marker in marks:
            for t in TABLES:
                m = mask & (G.ntype[shown] == TABLES.index(t))
                if m.any():
                    action = t in R.LAG_KEY
                    fill = dict(color=COLOR[t])
                    if by_age:
                        fill = (dict(color=log_age(age[m]), coloraxis="coloraxis", symbol=SYMBOL[t]) if action
                                else dict(color=NO_AGE, symbol=SYMBOL[t]))
                    fig.add_trace(go.Scattergl(
                        x=x[m], y=y[m], mode="markers", name=t, legendgroup=t, showlegend=False,
                        marker=dict(**fill, **marker), text=text[m].tolist(), customdata=custom[m],
                        hovertemplate=f"<b>%{{text}}</b><br>{t}, {status}<br>hop %{{customdata[1]:.0f}}<br>"
                                      f"{'' if action else 'first seen '}%{{customdata[2]:.1f}} days before the cut-off"
                                      f"<extra></extra>"))
        for t in TABLES:  # legend keys, one per table; clicking toggles the whole table
            n = int((G.ntype[shown] == TABLES.index(t)).sum())
            if n:
                key = dict(color=NO_AGE, symbol=SYMBOL[t]) if by_age else dict(color=COLOR[t])
                fig.add_trace(go.Scattergl(x=[None], y=[None], mode="markers", name=f"{t} ({n:,})", legendgroup=t,
                                           marker=dict(size=10, **key)))
        s = np.searchsorted(shown, roots)  # no customdata: clicking a seed selects nobody
        root_is_action = roots[0] != seeds[i]  # the seeds are all actions, or the user alone
        if not by_age:
            star = dict(color=[COLOR[TABLES[G.ntype[n]]] for n in roots])
        else:
            star = dict(color=log_age(age[s]), coloraxis="coloraxis") if root_is_action else dict(color=NO_AGE)
        fig.add_trace(go.Scattergl(
            x=x[s], y=y[s], mode="markers", name="seed" if len(roots) == 1 else f"seeds ({len(roots)})", marker=dict(
                symbol="star", size=22 if len(roots) == 1 else 14, line=dict(color=INK, width=1.2), **star),
            text=[(logins[n - user0] if n == seeds[i] else node_label(n)) + f"<br>seed {rank + 1} of {len(roots)}"
                  + ("" if n == seeds[i] else f", {a:.1f} days before the cut-off")
                  for rank, (n, a) in enumerate(zip(roots, age[s]))],
            hovertemplate="<b>%{text}</b><extra></extra>"))
        people = np.flatnonzero(G.ntype[shown] == TABLES.index("Users"))
        if chain and len(people) <= MAX_USER_LABELS:  # says at a glance whose rows these are
            fig.add_trace(go.Scattergl(
                x=x[people], y=y[people], mode="text", text=logins[shown[people] - user0].tolist(), legendgroup="Users",
                textposition="top center", textfont=dict(color=INK, size=11), showlegend=False, hoverinfo="skip"))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x")
        fig.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=8, b=56 if by_age else 8),
                          uirevision=f"{date}-{i}-{view}-{mode}-{k}-{fk_order}",
                          legend=dict(font=dict(color=INK, size=12), itemsizing="constant"),
                          hoverlabel=dict(bgcolor="white", font=dict(color=INK)), dragmode="pan")
        if by_age:
            fig.update_layout(coloraxis=dict(
                colorscale=AGE_SCALE, cmin=log_age(AGE_TICKS[0]), cmax=log_age(AGE_TICKS[-1]),
                colorbar=dict(orientation="h", x=0, xanchor="left", y=-0.01, yanchor="top", len=0.45, thickness=12,
                              title=dict(text="Age of the action, days before the cut-off", side="top",
                                         font=dict(color=INK, size=12)),
                              tickvals=log_age(np.array(AGE_TICKS)).tolist(), ticktext=[f"{d}" for d in AGE_TICKS],
                              tickfont=dict(color=INK, size=11))))
        o = own_rows(G, seeds[i], visible)
        table_age = cutoff[G.ntype[o]] - G.time[o]
        seen = np.isin(o, tok)
        info = (f"Cut-off {s_time:%Y-%m-%d %H:%M}. {visible_until(cutoff)} "
                f"{len(shown):,} nodes shown, {n_edges:,} edges; {len(roots)} seed{'s' * (len(roots) > 1)} and "
                f"{int(sampled.sum())} sampled tokens. Own visible rows: {len(o):,}. Among the tokens, of each table's "
                f"last 7 visible days: {int((seen & (table_age < 7)).sum())} of {int((table_age < 7).sum())}; last 30: "
                f"{int((seen & (table_age < 30)).sum())} of {int((table_age < 30).sum())}.")
        if not len(o):
            info += " This user has no visible actions at this cut-off" + (
                ", so the user is the seed." if mode == "latest" else ".")
        elif mode == "latest" and len(roots) < k:
            info += f" This user has only {len(roots)} visible actions, so {len(roots)} seeds instead of {k}."
        if view == "all" and chain and len(np.unique(extra[0])) >= args.max_nodes:
            info += f" Not every node: the walk stopped at --max-nodes {args.max_nodes:,}."
        return fig, info

    app = Dash(__name__, title="RelGT token graph")
    opts = f"graph: {args.graph_structure}, action edges: {args.action_edges}" + (
        f", lookback {args.lookback_days:g} days" if args.lookback_days else "") + (
        ", no pushes" if args.exclude_push else "")
    app.layout = html.Div([
        html.H2("RelGT input for one test user", style={"margin": "0 0 4px"}),
        html.P(f"K = {K} tokens, seeds included ({opts}). Stars are the seeds, rings are hops from them, "
               f"large outlined marks are the sampled tokens. Scroll to zoom, click a legend entry to hide a table. "
               f"Latest-action seeds train in run.py with --graph-structure chain --chain-seed latest --num-seeds k; "
               f"seeds other than the command line's, or other dates than the default, draw their own tokens.",
               style={"margin": "0 0 12px", "color": MUTED}),
        html.Div([
            dcc.Dropdown(id="user", options=options, value=example, clearable=False,
                         placeholder="Type a login or GitHub id", style={"width": "380px"}),
            html.Label(["Cut-off date ", dcc.DatePickerSingle(
                id="cutoff", date=base.TEST_START.date(), min_date_allowed=base.DATA_START.date(),
                max_date_allowed=base.TEST_START.date(), display_format="YYYY-MM-DD")]),
            dcc.RadioItems(id="colour", value="age", inline=True, labelStyle={"marginRight": "16px"}, options=[
                {"label": "Colour: action age", "value": "age"}, {"label": "Colour: table", "value": "table"}]),
            dcc.RadioItems(id="view", value="all", inline=True, labelStyle={"marginRight": "16px"}, options=[
                {"label": "All visible nodes", "value": "all"}, {"label": "Sampled tokens only", "value": "tokens"}]),
            dcc.RadioItems(id="fk-order", value=args.fk_order, inline=True, labelStyle={"marginRight": "16px"}, options=[
                {"label": "FK sampling: random", "value": "random"}, {"label": "FK sampling: newest first", "value": "recent"}]),
            dcc.RadioItems(id="seed-mode", value=cli_seed, inline=True, labelStyle={"marginRight": "16px"}, options=[
                {"label": "Seed: user", "value": "user"}, {"label": "Seed: latest actions", "value": "latest"}]),
            html.Label(["Number of seeds (k) ", dcc.Input(id="seed-k", type="number", value=args.num_seeds, min=1, max=K - 1, step=1,
                                                          debounce=True, style={"width": "70px"})],
                       id="seed-k-box", style=K_BOX[cli_seed]),
        ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "20px"}),
        html.P(id="info", style={"margin": "12px 0", "color": INK}),
        dcc.Loading(dcc.Graph(id="graph", style={"height": "78vh"},
                              config=dict(displaylogo=False, scrollZoom=True)), type="dot", color=COLOR["Users"]),
    ], style={"fontFamily": "system-ui, sans-serif", "color": INK, "padding": "16px"})

    @app.callback(Output("graph", "figure"), Output("info", "children"), Input("user", "value"), Input("view", "value"),
                  Input("seed-mode", "value"), Input("seed-k", "value"), Input("cutoff", "date"), Input("colour", "value"),
                  Input("fk-order", "value"))
    def draw(i, view, mode, k, date, colour, fk_order):
        if not date or (mode == "latest" and not k):  # k is None while out of range
            return no_update, no_update
        return user_view(date, i, view, mode, 1 if mode == "user" else k, colour, fk_order)

    @app.callback(Output("seed-k-box", "style"), Input("seed-mode", "value"))
    def show_k(mode):
        return K_BOX[mode]

    @app.callback(Output("user", "value"), Input("graph", "clickData"), prevent_initial_call=True)
    def click_user(click):
        custom = click["points"][0].get("customdata")
        return test_of.get(int(custom[0]), no_update) if custom else no_update

    print(f"serving {len(seeds):,} test users at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph-structure", choices=["fk", "chain"], default="fk")
    ap.add_argument("--max-hops", type=int, default=50)
    ap.add_argument("--chain-seed", choices=["user", "latest"], default="user")
    ap.add_argument("--num-seeds", type=int, default=1, help="with --graph-structure chain --chain-seed latest")
    ap.add_argument("--action-edges", choices=list(R.ACTION_CHAINS), default="none")
    ap.add_argument("--lookback-days", type=float, default=None)
    ap.add_argument("--exclude-push", action="store_true")
    ap.add_argument("--fk-order", choices=["random", "recent"], default="random")
    ap.add_argument("--interactive", action="store_true", help="serve a Dash app instead of writing the figure")
    ap.add_argument("--max-nodes", type=int, default=50000, help="--interactive, chain graph: walk size limit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8051)
    args = ap.parse_args()
    chain = args.graph_structure == "chain"
    if args.num_seeds != 1 and (not chain or args.chain_seed != "latest"):
        ap.error("--num-seeds needs --graph-structure chain --chain-seed latest")
    suffix = (("" if args.num_seeds == 1 else f"_s{args.num_seeds}") + ("_chain" if chain else "") + ("_latest" if chain and args.chain_seed == "latest" else "") + ("" if args.action_edges == "none" else f"_{args.action_edges}")
              + ("" if not args.lookback_days else f"_lb{args.lookback_days:g}")
              + ("_nopush" if args.exclude_push else "") + ("_recent" if args.fk_order == "recent" else ""))
    out = OUT.with_name(OUT.stem + suffix + OUT.suffix)

    cohort, ev, own, data_end = base.load()
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    cutoff = table_cutoffs(base.TEST_START, lags)
    G = R.Graph(args.graph_structure)
    visible = R.visible_nodes(G, cutoff, args.lookback_days, args.exclude_push)

    test = pd.read_csv(base.EVAL_DIR / "test_input.csv")
    cidx = pd.Series(np.arange(len(cohort)), index=cohort.github_user_id.values).loc[test.UserID].values
    seeds = G.offset[TABLES.index("Users")] + cohort.user_id.values[cidx]
    rng = np.random.default_rng(0)
    if chain:
        nodes, hops, _ = R.sample_walk(G, seeds, cutoff, K, rng, args.max_hops, lookback=args.lookback_days,
                                    start=args.chain_seed, exclude_push=args.exclude_push,
                                    num_seeds=args.num_seeds, fk_order=args.fk_order)
    else:
        nodes, hops, _ = R.sample_tokens(G, seeds, cutoff, K, rng, lookback=args.lookback_days,
                                         exclude_push=args.exclude_push, fk_order=args.fk_order)
    fallback = args.max_hops + 1 if chain else R.MAX_HOP
    print(f"users with only random fallback tokens: {(hops == fallback).any(1).mean():.1%}; "
          f"Pushes tokens: {(G.ntype[nodes[:, 1:]] == TABLES.index('Pushes')).mean():.1%}")
    actor = G.actor[nodes]
    other_rows = (actor >= 0) & (actor != seeds[:, None])
    other_users = (G.ntype[nodes] == TABLES.index("Users")) & (nodes != seeds[:, None])
    print(f"tokens: other users' rows {other_rows.mean():.1%}, other users {other_users.mean():.1%}, "
          f"own rows {((actor >= 0) & ~other_rows).mean():.1%}")

    # Coverage over all test users: the share of the user's own recent rows that become tokens.
    rows = []
    for i, sd in enumerate(seeds):
        o = own_rows(G, sd, visible)
        age = cutoff[G.ntype[o]] - G.time[o]  # days before the table's cutoff
        tok = np.isin(o, nodes[i])
        n1, n2 = (0, 0) if chain else map(len, neighbourhood(G, sd, visible))
        rows.append(dict(own=len(o), n1=n1, n2=n2, types=len(np.unique(G.ntype[o])),
                         distinct=len(np.unique(nodes[i, 1:])), max_hop=int(hops[i].max()),
                         recent30=int((age < 30).sum()), recent30_sampled=int((tok & (age < 30)).sum()),
                         recent7=int((age < 7).sum()), recent7_sampled=int((tok & (age < 7)).sum())))
    st = pd.DataFrame(rows)
    st["total"] = st.n1 + st.n2
    print(f"test users: {len(st)}; own visible rows, median {st.own.median():.0f}, max {st.own.max()}")
    if chain:
        print(f"distinct tokens per user, median {st.distinct.median():.0f}; walk depth, median "
              f"{st.max_hop.median():.0f}, max {st.max_hop.max()}")
    else:
        print(f"visible 1+2-hop neighbours, median {st.total.median():.0f}, 75th pct {st.total.quantile(.75):.0f}, "
              f"max {st.total.max()}; users with > {K - 1}: {(st.total > K - 1).mean():.0%}")
    for w in (7, 30):
        a, b = st[f"recent{w}"].sum(), st[f"recent{w}_sampled"].sum()
        print(f"own rows in the last {w:2d} days: {a:,}, sampled as tokens {b:,} ({b / max(a, 1):.0%})")

    # Example: an active, varied user near the 75th percentile of own rows.
    cand = st[(st.types >= 4) & (st.recent30 >= 10)]
    cand = cand if len(cand) else st
    i = int((cand.own - st.own.quantile(.75)).abs().idxmin())
    if args.interactive:
        serve(args, G, lags, test, seeds, nodes, hops, i)
        return
    sd, tok, hp = nodes[i, 0], nodes[i], hops[i]  # sd: token 0, the user or its latest action
    o = own_rows(G, seeds[i], visible)
    r = st.loc[i]
    print(f"example user github id {test.UserID.values[i]}: {r.own} own visible rows")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.patch.set_facecolor("white")

    # Left: token subgraph. Dark edges: same-type chain (chain structure), dashed: --action-edges.
    pairs = edge_pairs(G, tok, args.action_edges, seeds[i], visible)
    g = nx.Graph()
    uniq = np.unique(tok)
    g.add_nodes_from(uniq.tolist())
    g.add_edges_from(pairs["fk"] | pairs["owner"] | pairs["chain"] | pairs["action"])
    pos = nx.kamada_kawai_layout(g)
    hop_of = {int(v): int(h) for v, h in zip(tok[::-1], hp[::-1])}  # smallest index wins
    nx.draw_networkx_edges(g, pos, edgelist=list(pairs["fk"]), ax=ax, edge_color="#c9c8c3", width=0.8)
    nx.draw_networkx_edges(g, pos, edgelist=list(pairs["owner"]), ax=ax, edge_color=OWNER_EDGE, width=1.2)
    nx.draw_networkx_edges(g, pos, edgelist=list(pairs["chain"]), ax=ax, edge_color="#52514e", width=1.4)
    nx.draw_networkx_edges(g, pos, edgelist=list(pairs["action"]), ax=ax, edge_color="#52514e", width=1.0,
                           style="dashed")
    for h in sorted(set(hop_of.values()), reverse=True):
        vs = [v for v in uniq.tolist() if hop_of[v] == h]
        nx.draw_networkx_nodes(g, pos, nodelist=vs, ax=ax, node_size=420 if h == 0 else max(40, 140 - 12 * (h - 1)),
                               node_shape="*" if h == 0 else "o",
                               node_color=[COLOR[TABLES[G.ntype[v]]] for v in vs], edgecolors="white", linewidths=1.2)
    seed_label = ("seed: user" if G.ntype[sd] == TABLES.index("Users")
                  else f"seed: latest action ({TABLES[G.ntype[sd]]})")
    ax.annotate(seed_label, pos[int(sd)], xytext=(8, 10), textcoords="offset points", color=INK, fontsize=10)
    if sd != seeds[i] and int(seeds[i]) in pos:
        ax.annotate("user", pos[int(seeds[i])], xytext=(8, 10), textcoords="offset points", color=INK, fontsize=10)
    present = [t for t in TABLES if (G.ntype[uniq] == TABLES.index(t)).any()]
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[t], markersize=8, label=t) for t in present]
    handles += [plt.Line2D([], [], color="#c9c8c3", lw=1.5, label=f"foreign key ({len(pairs['fk'])})")]
    if chain:
        handles += [plt.Line2D([], [], color=OWNER_EDGE, lw=1.5, label=f"action to its user ({len(pairs['owner'])})"),
                    plt.Line2D([], [], color="#52514e", lw=1.5, label=f"prev / next same type ({len(pairs['chain'])})")]
    if args.action_edges != "none":
        handles += [plt.Line2D([], [], color="#52514e", lw=1.2, ls="--",
                               label=f"next action, {args.action_edges} ({len(pairs['action'])})")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=4, frameon=False, fontsize=9)
    seen_from = f"{r.own:,} own rows, walk depth {int(hp.max())}" if chain else f"{r.total:,} visible neighbours"
    ax.set_title(f"What RelGT sees: {int((hp == 0).sum())} seed(s) + {len(uniq) - int((hp == 0).sum())} sampled rows ({seen_from})\n"
                 f"(dot size shrinks with hops from the seed)",
                 loc="left", fontsize=11, color=INK)
    ax.axis("off")

    # Right: the user's own rows over time, all vs sampled.
    age = cutoff[G.ntype[o]] - G.time[o]
    sampled = np.isin(o, tok)
    jitter = np.random.default_rng(1)
    for row, (mask, label) in enumerate([(np.ones(len(o), bool), f"all visible own rows ({len(o):,})"),
                                         (sampled, f"sampled as tokens ({sampled.sum()})")]):
        for t in TABLES:
            m = mask & (G.ntype[o] == TABLES.index(t))
            if m.any():
                bx.scatter(-age[m], row + jitter.uniform(-0.28, 0.28, m.sum()), s=12, color=COLOR[t],
                           alpha=0.75, linewidths=0)
        bx.text(-age.max() if len(age) else -90, row + 0.42, label, color=INK, fontsize=10)
    bx.axvspan(-30, 0, color=GRID, alpha=0.5, lw=0, zorder=0)
    for w, y in [(30, -0.55), (7, -0.85)]:
        n_all, n_s = int((age < w).sum()), int((sampled & (age < w)).sum())
        bx.axvline(-w, color=MUTED, lw=1, ls="--")
        bx.text(-w - 3, y, f"last {w}d: {n_all} rows, {n_s} sampled", color=INK, fontsize=9, ha="right",
                va="center")
    bx.set_ylim(-1.15, 1.6)
    bx.set_yticks([])
    bx.set_xlabel("days before the table's cutoff (0 = latest visible row)", color=INK)
    bx.spines[["top", "right", "left"]].set_visible(False)
    bx.grid(axis="x", color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    if chain:
        right = "The walk starts at the latest action of each type\nand goes back in time along the chains"
    elif args.lookback_days:
        right = f"Only the last {args.lookback_days:g} days are visible: which own rows were sampled"
    else:
        right = ("Uniform sampling spreads the 99 tokens over the whole history,\n"
                 "so only a few of the most recent rows (the strongest signal) are seen")
    bx.set_title(right, loc="left", fontsize=11, color=INK)

    opts = ", ".join([f"graph: {args.graph_structure}" + (f", seed: {args.chain_seed}" if chain else ""), f"action edges: {args.action_edges}"] +
                     ([f"lookback {args.lookback_days:g} days"] if args.lookback_days else []) +
                     (["no pushes"] if args.exclude_push else []))
    fig.suptitle(f"RelGT input for one test user at the test week (K = {K} tokens; {opts})", x=0.01, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
