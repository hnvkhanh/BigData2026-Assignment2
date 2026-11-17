"""Multiple latest-action seeds on the chain graph, from the real data.

experiments/relgt/run.py --graph-structure chain --chain-seed latest --num-seeds k seeds the K-token walk at
the user's k latest visible actions (hop 0) instead of at the user. This draws, for one real test user and
with the real sampler (R.sample_walk) at the test cut-off:

  top left   --chain-seed user: token 0 is the user, hop 1 holds the latest row of each of the user's chains
  top right  --chain-seed latest --num-seeds k: the k latest actions are the seeds (hop 0), the user is a
             hop-1 token; edges are the FK / chain edges among the tokens (R.subgraph_edges)
  bottom     the user's actions in time, with the k seeds ringed: they are the k newest rows, whatever table

Run from the repo root:
    python analysis/multi_seed_figure.py [--num-seeds 4] [--k 40] [--user 839357836028682] [--user-rank 0]
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

base, TABLES = R.base, R.TABLES
OUT = ROOT / "figures" / "multi_seed_real.png"
USER_COLOR, EDGE, USER_EDGE = "#222222", "#c9c9c9", "#c2410c"
FULL_COLOR = {**COLOR, "Users": USER_COLOR, "Repositories": "#9a9a9a"}


def pick_user(G, seeds, visible, test, rank):
    """A user with several tables and a modest history: score = sum over tables of min(rows, 5)."""
    score = np.full(len(seeds), -1.0)
    for j, s in enumerate(seeds):
        o = own_rows(G, s, visible)
        if 40 <= len(o) <= 400:
            c = np.bincount(G.ntype[o], minlength=len(TABLES))
            score[j] = np.minimum(c, 5).sum() - len(o) / 1000
    return int(np.argsort(-score)[rank])


def sample(G, seed, cutoff, K, mode, num_seeds, max_hops):
    nodes, hops, _ = R.sample_walk(G, np.array([seed]), cutoff, K, np.random.default_rng(0), max_hops,
                                   start=mode, num_seeds=num_seeds)
    return nodes, hops


def draw_tokens(ax, G, nodes, hops, seed, visible, title):
    """Concentric rings by hop; unique tokens only (a walk that cannot fill K repeats random ones)."""
    ids, first = np.unique(nodes[0], return_index=True)
    ids, hop = nodes[0][np.sort(first)], hops[0][np.sort(first)]
    pos = {}
    n0 = (hop == 0).sum()
    r0 = 0.0 if n0 == 1 else 0.9
    for h in np.unique(hop):
        m = ids[hop == h]
        m = m[np.lexsort((-G.time[m], G.ntype[m]))]  # cluster by table, newest first
        r = r0 + 1.9 * h if h > 0 or n0 == 1 else r0
        if h == 0 and n0 == 1:
            pos[m[0]] = np.zeros(2)
            continue
        a = np.linspace(0, 2 * np.pi, len(m), endpoint=False) + 0.3 * h
        for v, t in zip(m, a):
            pos[v] = r * np.array([np.cos(t), np.sin(t)])
    # Edges among the tokens (FK edges plus chain edges, and actor edges to other users).
    r_, i, j, _ = R.subgraph_edges(G, nodes, users=np.array([seed]), visible=visible)
    seen = set()
    for a, b in zip(nodes[0][i], nodes[0][j]):
        key = (min(a, b), max(a, b))
        if a == b or key in seen:
            continue
        seen.add(key)
        hit_user = seed in key
        ax.plot(*zip(pos[a], pos[b]), color=USER_EDGE if hit_user else EDGE, lw=1.3 if hit_user else 0.7,
                alpha=0.9 if hit_user else 0.8, zorder=1)
    for v, p in pos.items():
        t = TABLES[G.ntype[v]]
        is_seed = hop[ids == v][0] == 0
        ax.scatter(*p, s=260 if is_seed else 90, c=FULL_COLOR.get(t, "#777"),
                   edgecolors="black" if is_seed else "white", linewidths=2.2 if is_seed else 0.8, zorder=3,
                   marker="s" if t == "Users" else "o")
    ax.set_title(title, fontsize=10.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return len(ids), (hop == 0).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-seeds", type=int, default=4, help="k latest actions used as seeds")
    ap.add_argument("--k", type=int, default=40, help="tokens per example, small so the graph stays legible")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--user", type=int, default=None, help="a test UserID; default picks one")
    ap.add_argument("--user-rank", type=int, default=0)
    args = ap.parse_args()

    cohort, ev, own, data_end = base.load()
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    cutoff = table_cutoffs(base.TEST_START, lags)
    G = R.Graph("chain")
    visible = R.visible_nodes(G, cutoff)
    test = pd.read_csv(base.TEST_INPUT)
    cidx = pd.Series(np.arange(len(cohort)), index=cohort.github_user_id.values).loc[test.UserID].values
    seeds = G.offset[TABLES.index("Users")] + cohort.user_id.values[cidx]
    j = int(np.flatnonzero(test.UserID.values == args.user)[0]) if args.user else pick_user(G, seeds, visible, test, args.user_rank)
    seed, uid = int(seeds[j]), test.UserID.values[j]
    rows = own_rows(G, seed, visible)
    print(f"user {uid}: {len(rows)} visible activity rows")

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.15], hspace=0.12)
    a1, a2, tl = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    nu, hu = sample(G, seed, cutoff, args.k, "user", 1, args.max_hops)
    nl, hl = sample(G, seed, cutoff, args.k, "latest", args.num_seeds, args.max_hops)
    n1, s1 = draw_tokens(a1, G, nu, hu, seed, visible,
                         f"--chain-seed user\ntoken 0 = the user ({n1_label(nu)} distinct of K={args.k} tokens)")
    n2, s2 = draw_tokens(a2, G, nl, hl, seed, visible,
                         f"--chain-seed latest --num-seeds {args.num_seeds}\n"
                         f"hop 0 = the {s2_label(hl)} latest actions, the user is a hop-1 token\n"
                         f"({n1_label(nl)} distinct of K={args.k} tokens)")

    # Timeline: age of each of the user's rows, seeds ringed.
    seed_rows = set(nl[0][hl[0] == 0].tolist())
    y = {t: i for i, t in enumerate(ACTIONS)}
    for v in rows:
        t = TABLES[G.ntype[v]]
        age = cutoff[G.ntype[v]] - G.time[v]
        tl.scatter(age, y[t], s=30, c=COLOR[t], alpha=0.8, zorder=2)
        if v in seed_rows:
            tl.scatter(age, y[t], s=190, facecolors="none", edgecolors="black", linewidths=2, zorder=3)
    tl.set_yticks(list(y.values()))
    tl.set_yticklabels([SHORT[t] for t in y])
    tl.invert_yaxis()
    tl.set_xscale("symlog", linthresh=30)
    tl.set_xlim(left=max(tl.get_xlim()[0], -1))
    tl.invert_xaxis()
    tl.set_xlabel("days before the table's cut-off (older to the left)")
    tl.set_title(f"The user's {len(rows)} visible actions in time; ringed = the {args.num_seeds} latest, "
                 f"the seeds of --num-seeds {args.num_seeds} (they can come from any table)", fontsize=10.5)
    for s in ("top", "right"):
        tl.spines[s].set_visible(False)
    tl.grid(axis="x", color="#eee")

    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", color=COLOR[t], label=SHORT[t]) for t in ACTIONS]
               + [Line2D([0], [0], marker="s", ls="", color=USER_COLOR, label="user"),
                  Line2D([0], [0], marker="o", ls="", color="#9a9a9a", label="repository / other"),
                  Line2D([0], [0], marker="o", ls="", mfc="white", mec="black", mew=2, label="seed (hop 0)"),
                  Line2D([0], [0], color=USER_EDGE, lw=1.5, label="edge to the user"),
                  Line2D([0], [0], color=EDGE, lw=1.2, label="FK / chain edge")],
               loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle(f"Multiple latest-action seeds, real test user {uid}", fontsize=12.5, y=0.98)
    fig.subplots_adjust(bottom=0.12, top=0.9)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("saved", OUT)


def n1_label(nodes):
    return len(np.unique(nodes[0]))


def s2_label(hops):
    return int((hops[0] == 0).sum())


if __name__ == "__main__":
    main()
