"""Step-by-step demo of how one example's graph is built: chain graph, latest-action seeds, recency fill.

This is --graph-structure chain --chain-seed latest --num-seeds k --fk-order recent of experiments/relgt/run.py,
drawn on one real test user with the real sampler (R.sample_walk) at the test cut-off. Six panels:

  1  the user's visible rows, one lane per table, linked in time order: the chains
  2  hop 0: the k latest actions (any table) are the seeds
  3  hop 1: the user, and the latest row of each chain that is not a seed, plus the rows the seeds point to
  4  hop 2: neighbours of the hop-1 rows (FK links, previous row in a chain, other users' rows), newest first
  5  hop 3+ and the final token graph: FK / chain edges among the K tokens
  6  recency fill: of everything the walk could reach (large K), which rows made it into the K tokens?

Run from the repo root:
    python analysis/chain_build_demo.py [--num-seeds 4] [--k 40] [--user 839357836028682] [--frames]
--frames also writes one PNG per step to figures/chain_build/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.relgt import run as R  # noqa: E402
from analysis.relgt_graph import table_cutoffs  # noqa: E402
from analysis.fk_vs_chain_figures import ACTIONS, SHORT, COLOR, own_rows  # noqa: E402
from analysis.multi_seed_figure import pick_user, FULL_COLOR, EDGE, USER_EDGE  # noqa: E402

base, TABLES = R.base, R.TABLES
OUT = ROOT / "figures" / "chain_build_real.png"
STEP_TITLES = [
    "1  Chains: each table's rows in time order",
    "2  Seeds: the k latest actions (hop 0)",
    "3  Hop 1: user + latest row of each chain",
    "4  Hop 2: neighbours of hop 1, newest first",
    "5  Hop 3+ and the edges among the tokens",
    "6  Recency fill: what the budget K keeps",
]


def walk(G, seed, cutoff, K, num_seeds, max_hops):
    nodes, hops, _ = R.sample_walk(G, np.array([seed]), cutoff, K, np.random.default_rng(0), max_hops,
                                   start="latest", num_seeds=num_seeds, fk_order="recent")
    ids, first = np.unique(nodes[0], return_index=True)  # a walk that cannot fill K repeats random tokens
    keep = np.sort(first)
    return nodes[0][keep], hops[0][keep]


def layout(G, ids, hop):
    """Concentric rings by hop (hop 0 on the inner ring); nodes clustered by table, newest first."""
    pos = {}
    for h in np.unique(hop):
        m = ids[hop == h]
        m = m[np.lexsort((-G.time[m], G.ntype[m]))]
        r = 0.9 + 1.9 * h
        a = np.linspace(0, 2 * np.pi, len(m), endpoint=False) + 0.3 * h
        for v, t in zip(m, a):
            pos[v] = r * np.array([np.cos(t), np.sin(t)])
    return pos


def node_color(G, v):
    return FULL_COLOR.get(TABLES[G.ntype[v]], "#777")


def panel_chains(ax, G, rows, cutoff):
    """Lanes by table; rows linked oldest -> newest (the chain)."""
    for i, t in enumerate(ACTIONS):
        r = rows[G.ntype[rows] == TABLES.index(t)]
        if not len(r):
            continue
        r = r[np.argsort(G.time[r])]
        age = cutoff[TABLES.index(t)] - G.time[r]
        ax.plot(age, np.full(len(r), i), color=COLOR[t], lw=1.0, alpha=0.7, zorder=1)
        ax.scatter(age, np.full(len(r), i), s=22, c=COLOR[t], zorder=2)
        ax.scatter(age[-1], i, s=150, facecolors="none", edgecolors="black", linewidths=1.6, zorder=3)
        ax.text(age[-1], i + 0.42, "latest", fontsize=7, ha="center", va="center")
    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels([SHORT[t] for t in ACTIONS], fontsize=8)
    ax.set_ylim(len(ACTIONS) - 0.4, -0.6)
    ax.set_xscale("symlog", linthresh=30)
    ax.set_xlim(left=max(ax.get_xlim()[0], -1))
    ax.invert_xaxis()
    ax.set_xlabel("days before the table's cut-off (older to the left)", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee")


def panel_graph(ax, G, ids, hop, pos, edges, seed, upto, k_label=True):
    """Tokens with hop <= upto, and the edges among them."""
    shown = set(ids[hop <= upto].tolist())
    done = set()
    for a, b in edges:
        key = (min(a, b), max(a, b))
        if a == b or key in done or a not in shown or b not in shown:
            continue
        done.add(key)
        chain = (G.actor[a] >= 0 and G.actor[a] == G.actor[b] and G.ntype[a] == G.ntype[b])
        user = seed in key
        col = COLOR[TABLES[G.ntype[a]]] if chain else (USER_EDGE if user else EDGE)
        ax.plot(*zip(pos[a], pos[b]), color=col, lw=1.8 if chain else (1.3 if user else 0.7),
                alpha=0.9 if chain or user else 0.8, zorder=1)
    for v, h in zip(ids, hop):
        if v not in shown:
            ax.scatter(*pos[v], s=14, c="#e6e6e6", zorder=0)  # not yet reached
            continue
        t = TABLES[G.ntype[v]]
        ax.scatter(*pos[v], s=260 if h == 0 else 90, c=node_color(G, v), zorder=3, marker="s" if t == "Users" else "o",
                   edgecolors="black" if h == 0 else "white", linewidths=2.2 if h == 0 else 0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_recency(ax, G, ids, hop, big_ids, big_hop, cutoff, K):
    """Age of every activity row the walk can reach (large K), by hop; filled = kept in the K tokens."""
    kept = set(ids.tolist())
    act = np.array([G.actor[v] >= 0 for v in big_ids])
    for h in range(int(big_hop.max()) + 1):
        m = act & (big_hop == h)
        if not m.any():
            continue
        v = big_ids[m]
        age = cutoff[G.ntype[v]] - G.time[v]
        k = np.array([x in kept for x in v])
        jit = (np.random.default_rng(h).random(len(v)) - 0.5) * 0.5
        ax.scatter(age[~k], h + jit[~k], s=16, facecolors="none", edgecolors="#aaa", linewidths=0.8)
        ax.scatter(age[k], h + jit[k], s=26, c=[COLOR[TABLES[G.ntype[x]]] for x in v[k]], zorder=3)
        if (~k).any() and k.any():
            ax.axvline(age[k].max(), ymin=0, ymax=0)  # no-op, keeps limits tidy
    ax.set_yticks(range(int(big_hop.max()) + 1))
    ax.set_yticklabels([f"hop {h}" for h in range(int(big_hop.max()) + 1)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=30)
    ax.set_xlim(left=max(ax.get_xlim()[0], -1))
    ax.invert_xaxis()
    ax.set_xlabel("days before the table's cut-off (older to the left)", fontsize=8)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color="#555", label=f"kept (K={K})"),
                       Line2D([0], [0], marker="o", ls="", mfc="none", mec="#aaa", label="reachable, dropped")],
              frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee")


def draw_step(i, ax, ctx):
    G, rows, cutoff, ids, hop, pos, edges, seed, K, big = (ctx[k] for k in (
        "G", "rows", "cutoff", "ids", "hop", "pos", "edges", "seed", "K", "big"))
    if i == 0:
        panel_chains(ax, G, rows, cutoff)
    elif i == 5:
        panel_recency(ax, G, ids, hop, *big, cutoff, K)
    else:
        upto = [0, 1, 2, int(hop.max())][i - 1]
        panel_graph(ax, G, ids, hop, pos, edges, seed, upto)
        n = int((hop <= upto).sum())
        ax.text(0.5, -0.02, f"{n} of {len(ids)} tokens reached", transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#444")
    ax.set_title(STEP_TITLES[i], fontsize=10.5, loc="left")


def legend_handles():
    return ([Line2D([0], [0], marker="o", ls="", color=COLOR[t], label=SHORT[t]) for t in ACTIONS]
            + [Line2D([0], [0], marker="s", ls="", color="#222222", label="user"),
               Line2D([0], [0], marker="o", ls="", color="#9a9a9a", label="repository / other"),
               Line2D([0], [0], marker="o", ls="", mfc="white", mec="black", mew=2, label="seed (hop 0)"),
               Line2D([0], [0], color="#555", lw=1.8, label="chain edge (same table, same user)"),
               Line2D([0], [0], color=USER_EDGE, lw=1.5, label="edge to the user"),
               Line2D([0], [0], color=EDGE, lw=1.2, label="FK edge")])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-seeds", type=int, default=4, help="k latest actions used as seeds (16 in the real model)")
    ap.add_argument("--k", type=int, default=40, help="tokens per example (100 in the real model); small = legible")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--big-k", type=int, default=600, help="budget for the 'everything reachable' run (panel 6)")
    ap.add_argument("--user", type=int, default=None, help="a test UserID; default picks one")
    ap.add_argument("--user-rank", type=int, default=0)
    ap.add_argument("--frames", action="store_true", help="also write one PNG per step to figures/chain_build/")
    args = ap.parse_args()

    cohort, ev, own, data_end = base.load()
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    cutoff = table_cutoffs(base.TEST_START, lags)
    G = R.Graph("chain")
    visible = R.visible_nodes(G, cutoff)
    test = pd.read_csv(base.TEST_INPUT)
    cidx = pd.Series(np.arange(len(cohort)), index=cohort.github_user_id.values).loc[test.UserID].values
    seeds = G.offset[TABLES.index("Users")] + cohort.user_id.values[cidx]
    j = (int(np.flatnonzero(test.UserID.values == args.user)[0]) if args.user
         else pick_user(G, seeds, visible, test, args.user_rank))
    seed, uid = int(seeds[j]), test.UserID.values[j]
    rows = own_rows(G, seed, visible)
    print(f"user {uid}: {len(rows)} visible activity rows")

    ids, hop = walk(G, seed, cutoff, args.k, args.num_seeds, args.max_hops)
    big = walk(G, seed, cutoff, args.big_k, args.num_seeds, args.max_hops)
    r_, i_, j_, _ = R.subgraph_edges(G, ids[None, :], users=np.array([seed]), visible=visible)
    edges = list(zip(ids[i_], ids[j_]))
    ctx = dict(G=G, rows=rows, cutoff=cutoff, ids=ids, hop=hop, pos=layout(G, ids, hop), edges=edges, seed=seed,
               K=args.k, big=big)
    print("tokens per hop:", {int(h): int((hop == h).sum()) for h in np.unique(hop)})

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for i, ax in enumerate(axes.ravel()):
        draw_step(i, ax, ctx)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle(f"How the chain graph with latest-action seeds and recency fill is built, real test user {uid}\n"
                 f"(--chain-seed latest --num-seeds {args.num_seeds} --fk-order recent, K={args.k})", fontsize=12.5)
    fig.subplots_adjust(bottom=0.12, top=0.9, hspace=0.25, wspace=0.12)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print("saved", OUT)

    if args.frames:
        d = OUT.parent / "chain_build"
        d.mkdir(exist_ok=True)
        for i in range(6):
            f, ax = plt.subplots(figsize=(7, 6.5))
            draw_step(i, ax, ctx)
            f.savefig(d / f"step{i + 1}.png", dpi=170, bbox_inches="tight")
            plt.close(f)
        print("saved frames to", d)


if __name__ == "__main__":
    main()
