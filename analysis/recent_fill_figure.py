"""Chain + latest seeds, random fill vs recent fill, from the real data.

experiments/relgt/run.py --graph-structure chain --chain-seed latest --num-seeds k seeds the K-token walk at the
user's k latest actions. Where nodes compete for the K slots (the cap on a node's neighbours, the budget of a hop,
the final tokens), --fk-order random keeps a random subset and --fk-order recent keeps the newest rows. This draws,
for one real test user, the same seeds, K and cut-off, with the real sampler (R.sample_walk):

  left   --fk-order random
  right  --fk-order recent

Each panel is the token graph in rings by hop (hop 0 = the seeds, ringed in black). An activity row is coloured by
its age, days before its table's cut-off, on one colour scale shared by both panels (bright = new, dark = old); its
shape is its table. Users are black stars, repositories grey hexagons.

Run from the repo root:
    python analysis/recent_fill_figure.py [--num-seeds 16] [--k 100] [--user 839357836028682] [--user-rank 0] [--recent-days 30]
Without --user it picks the test user with the most actions in the last --recent-days days before the cut-off.
The output is figures/recent_fill_real.png (--out to change it). Default K is 100, as in training; the walk only
differs between the two modes if the user has more candidates than free slots, so a user with a long history shows it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.relgt import run as R  # noqa: E402
from analysis.relgt_graph import table_cutoffs  # noqa: E402
from analysis.fk_vs_chain_figures import ACTIONS, SHORT, own_rows  # noqa: E402
from analysis.multi_seed_figure import USER_EDGE, EDGE  # noqa: E402

base, TABLES = R.base, R.TABLES
SHAPE = dict(zip(ACTIONS, ["^", "D", "o", "P", "v", "s"]))  # one marker per activity table
CMAP = "viridis_r"  # new = yellow, old = dark purple


def sample(G, seed, cutoff, K, num_seeds, max_hops, fk_order):
    nodes, hops, _ = R.sample_walk(G, np.array([seed]), cutoff, K, np.random.default_rng(0), max_hops,
                                   start="latest", num_seeds=num_seeds, fk_order=fk_order)
    ids, first = np.unique(nodes[0], return_index=True)
    first = np.sort(first)
    return nodes, nodes[0][first], hops[0][first]


def age_of(G, cutoff, v):
    return float(cutoff[G.ntype[v]] - G.time[v])


def pick_recent_user(G, seeds, visible, cutoff, days, rank):
    """The test user with the most non-push activity rows in the last `days` days before the cut-off.

    Pushes are left out of the count: a few users have thousands, which would hide the users whose recent
    PRs / issues / comments / reviews compete for the K slots."""
    score = np.zeros(len(seeds))
    for j, s in enumerate(seeds):
        o = own_rows(G, s, visible)
        o = o[G.ntype[o] != TABLES.index("Pushes")]
        if len(o):
            score[j] = (cutoff[G.ntype[o]] - G.time[o] <= days).sum() + len(o) * 1e-6
    return int(np.argsort(-score)[rank]), score


def layout(G, ids, hop):
    """Concentric rings by hop, clustered by table and newest first inside a ring."""
    pos, n0 = {}, (hop == 0).sum()
    r0 = 0.0 if n0 == 1 else 0.9
    for h in np.unique(hop):
        m = ids[hop == h]
        m = m[np.lexsort((-G.time[m], G.ntype[m]))]
        if h == 0 and n0 == 1:
            pos[m[0]] = np.zeros(2)
            continue
        r = r0 + 1.9 * h if h > 0 or n0 == 1 else r0
        a = np.linspace(0, 2 * np.pi, len(m), endpoint=False) + 0.3 * h
        for v, t in zip(m, a):
            pos[v] = r * np.array([np.cos(t), np.sin(t)])
    return pos


def draw_graph(ax, G, cutoff, nodes, ids, hop, seed, visible, norm, title):
    pos = layout(G, ids, hop)
    r_, i, j, _ = R.subgraph_edges(G, nodes, users=np.array([seed]), visible=visible)
    seen = set()
    for a, b in zip(nodes[0][i], nodes[0][j]):
        key = (min(a, b), max(a, b))
        if a == b or key in seen:
            continue
        seen.add(key)
        hit_user = seed in key
        ax.plot(*zip(pos[a], pos[b]), color=USER_EDGE if hit_user else EDGE, lw=1.3 if hit_user else 0.6,
                alpha=0.9 if hit_user else 0.7, zorder=1)
    seed_set = set(ids[hop == 0].tolist())
    cmap = plt.get_cmap(CMAP)
    for v, p in pos.items():
        t = TABLES[G.ntype[v]]
        seeded = v in seed_set
        if t in SHAPE:
            ax.scatter(*p, s=300 if seeded else 120, c=[cmap(norm(age_of(G, cutoff, v) + 1))], marker=SHAPE[t],
                       edgecolors="black" if seeded else "white", linewidths=2.4 if seeded else 0.8, zorder=3)
        elif t == "Users":
            ax.scatter(*p, s=320, c="black", marker="*", edgecolors="white", linewidths=0.8, zorder=4)
        else:
            ax.scatter(*p, s=90, c="#b5b5b5", marker="h", edgecolors="white", linewidths=0.8, zorder=2)
    ax.set_title(title, fontsize=10.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-seeds", type=int, default=16, help="k latest actions used as seeds")
    ap.add_argument("--k", type=int, default=100, help="tokens per example")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--user", type=int, default=None, help="a test UserID; default picks one")
    ap.add_argument("--user-rank", type=int, default=0, help="0 = the user with the most recent actions, 1 = next...")
    ap.add_argument("--recent-days", type=float, default=30, help="window (days before cut-off) for 'recent' when picking")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "recent_fill_real.png")
    args = ap.parse_args()

    cohort, ev, own, data_end = base.load()
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    cutoff = table_cutoffs(base.TEST_START, lags)
    G = R.Graph("chain")
    visible = R.visible_nodes(G, cutoff)
    test = pd.read_csv(base.TEST_INPUT)
    cidx = pd.Series(np.arange(len(cohort)), index=cohort.github_user_id.values).loc[test.UserID].values
    seeds = G.offset[TABLES.index("Users")] + cohort.user_id.values[cidx]
    if args.user:
        j = int(np.flatnonzero(test.UserID.values == args.user)[0])
    else:
        j, score = pick_recent_user(G, seeds, visible, cutoff, args.recent_days, args.user_rank)
        print(f"picked by most non-push rows in the last {args.recent_days} d: {int(score[j])} rows")
    seed, uid = int(seeds[j]), test.UserID.values[j]
    rows = own_rows(G, seed, visible)
    print(f"user {uid}: {len(rows)} visible activity rows")

    runs, act = {}, [TABLES.index(t) for t in ACTIONS]
    for order in ("random", "recent"):
        nodes, ids, hop = sample(G, seed, cutoff, args.k, args.num_seeds, args.max_hops, order)
        ages = np.array([age_of(G, cutoff, v) for v in ids[np.isin(G.ntype[ids], act)]])
        runs[order] = (nodes, ids, hop, ages)
        print(f"fk-order {order}: {len(ids)} tokens, {len(ages)} activity rows, median age {np.median(ages):.1f} d, "
              f"mean {ages.mean():.1f} d, oldest {ages.max():.1f} d")
    both = set(runs["random"][1]) & set(runs["recent"][1])
    print("activity rows in both:", sum(G.ntype[v] in act for v in both))

    top = max(a.max() for *_, a in runs.values()) + 1
    norm = LogNorm(vmin=1, vmax=max(top, 2))  # age + 1 days, log: the newest rows stay apart
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6))
    for ax, order in zip(axes, ("random", "recent")):
        nodes, ids, hop, ages = runs[order]
        draw_graph(ax, G, cutoff, nodes, ids, hop, seed, visible, norm,
                   f"--fk-order {order}\n{len(ages)} activity rows: median age {np.median(ages):.0f} d, "
                   f"oldest {ages.max():.0f} d")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035, pad=0.02, aspect=50)
    cb.set_label("age of an action: days before its table's cut-off (log scale; bright = new, dark = old)")
    ticks = [t for t in (1, 3, 10, 30, 100, 300) if t <= norm.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t - 1) for t in ticks])

    fig.legend(handles=[Line2D([0], [0], marker=SHAPE[t], ls="", mfc="#777", mec="white", ms=9, label=SHORT[t])
                        for t in ACTIONS]
               + [Line2D([0], [0], marker="*", ls="", color="black", ms=12, label="user"),
                  Line2D([0], [0], marker="h", ls="", color="#b5b5b5", ms=9, label="repository"),
                  Line2D([0], [0], marker="o", ls="", mfc="white", mec="black", mew=2.2, ms=10, label="seed (hop 0)"),
                  Line2D([0], [0], color=USER_EDGE, lw=1.5, label="edge to the user"),
                  Line2D([0], [0], color=EDGE, lw=1.2, label="FK / chain edge")],
               loc="lower center", ncol=11, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Chain graph, latest-action seeds (k={args.num_seeds}), K={args.k}: random vs recent fill, "
                 f"real test user {uid}", fontsize=12.5)
    fig.subplots_adjust(top=0.88, bottom=0.2, wspace=0.02)
    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
