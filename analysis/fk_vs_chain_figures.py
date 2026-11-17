"""Figures from the real data: the FK hub problem and the chain graph.

Uses experiments/relgt/run.py's Graph on the real tables, at the test cut-off (the same visible rows
as analysis/relgt_graph.py), and takes the test user with the most visible activity rows.

  figures/fk_hub_real.png      the user's real rows as an FK hub, and what a 1-hop sample of them is made of
                               (this user, and the average over all test users)
  figures/chain_graph_real.png the same user's real chain: user -> latest row per table -> previous rows,
                               each labelled with its age in days before the table's cut-off

Run from the repo root:  python analysis/fk_vs_chain_figures.py [--sample 20] [--user-rank 0]
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

base, TABLES = R.base, R.TABLES
ACTIONS = [t for t in TABLES if t in R.ACTOR_COL]  # the six activity tables
SHORT = {"PullRequests": "PR", "Issues": "issue", "IssueComments": "comment", "PRReviews": "review",
         "PRReviewComments": "review comment", "Pushes": "push"}
COLOR = {"Pushes": "#d95f02", "IssueComments": "#1b9e77", "PullRequests": "#7570b3",
         "Issues": "#e7298a", "PRReviews": "#66a61e", "PRReviewComments": "#1f78b4"}
OUT_HUB, OUT_CHAIN = ROOT / "figures" / "fk_hub_real.png", ROOT / "figures" / "chain_graph_real.png"


def own_rows(G, seed, visible):
    """The user's visible activity rows, grouped by table then ordered by time (oldest first)."""
    rows = G.aadj[G.aptr[seed]:G.aptr[seed + 1]]
    return rows[visible[rows]]


def counts(G, rows):
    return np.array([(G.ntype[rows] == TABLES.index(t)).sum() for t in ACTIONS])


def draw_hub(ax, G, rows, sample_idx, max_drawn=220, rng=None):
    """Hub of the user's rows. Drawn rows are a proportional subsample when there are more than max_drawn."""
    n = len(rows)
    # Draw every non-push row, fill up with random pushes, add the sampled rows, then mix the order.
    push = G.ntype[rows] == TABLES.index("Pushes")
    rest = np.flatnonzero(push)
    room = max(max_drawn - int((~push).sum()), 0)
    keep = set(np.flatnonzero(~push).tolist()) | set(rng.choice(rest, min(room, len(rest)), replace=False).tolist())
    keep = np.array(sorted(keep | set(sample_idx)), int)
    rng.shuffle(keep)
    ang = np.linspace(0, 2 * np.pi, len(keep), endpoint=False)
    r = 1.0 + 0.25 * rng.random(len(keep))
    chosen = set(sample_idx)
    for i, a, rr in zip(keep, ang, r):
        x, y = rr * np.cos(a), rr * np.sin(a)
        c = COLOR[TABLES[G.ntype[rows[i]]]]
        pick = i in chosen
        ax.plot([0, x], [0, y], color="#222" if pick else "#bbb", lw=1.6 if pick else 0.5,
                alpha=0.9 if pick else 0.6, zorder=1)
        ax.scatter(x, y, s=75 if pick else 26, c=c, edgecolors="black" if pick else "none",
                   linewidths=1.2, alpha=1 if pick else 0.55, zorder=2 + pick)
    ax.scatter(0, 0, s=2200, c="#222", zorder=5)
    ax.text(0, 0, "user", color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")


def hub_figure(G, rows, per_user_share, k, rng, uid):
    n = len(rows)
    k = min(k, n)
    sample_idx = rng.choice(n, k, replace=False).tolist()
    full = counts(G, rows)
    samp = counts(G, rows[sample_idx])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    draw_hub(ax, G, rows, sample_idx, rng=rng)
    note = "" if n <= 220 else "; all non-push rows and a random subset of pushes drawn"
    ax.set_title(f"Plain FK graph, real test user {uid}: every activity row links to the user\n"
                 f"({n:,} visible rows{note}; bold = a random {k}-row 1-hop sample)", fontsize=11)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=COLOR[t], label=f"{SHORT[t]} ({c:,})")
                       for t, c in zip(ACTIONS, full)],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=9)

    y = np.arange(len(ACTIONS))
    h = 0.27
    cols = [COLOR[t] for t in ACTIONS]
    bx.barh(y - h, per_user_share * 100, height=h, color=cols, alpha=0.3, label="mean over all test users")
    bx.barh(y, full / n * 100, height=h, color=cols, alpha=0.6, label="this user, all rows")
    bx.barh(y + h, samp / k * 100, height=h, color=cols, label=f"this user, {k}-row 1-hop sample")
    bx.set_yticks(y)
    bx.set_yticklabels([SHORT[t] for t in ACTIONS])
    bx.invert_yaxis()
    bx.set_xlabel("% of the user's 1-hop neighbours")
    bx.set_title("What a 1-hop sample around the user is made of", fontsize=11)
    bx.legend(frameon=False, loc="upper right", fontsize=8)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_HUB, dpi=200, bbox_inches="tight")
    print("saved", OUT_HUB)


def chain_figure(G, rows, cutoff, uid, depth=4):
    fig, ax = plt.subplots(figsize=(10, 9))
    present = [t for t in ACTIONS if (G.ntype[rows] == TABLES.index(t)).any()]
    angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, len(present), endpoint=False)
    extent = [np.zeros(2)]
    for t, ang in zip(present, angles):
        tr = rows[G.ntype[rows] == TABLES.index(t)]  # oldest first
        chain = tr[::-1][:depth]  # newest first
        age = cutoff[TABLES.index(t)] - G.time[chain]
        d = np.array([np.cos(ang), np.sin(ang)])
        pts = [(1.7 + 1.0 * k) * d for k in range(len(chain))]
        ax.annotate("", xy=pts[0], xytext=(0, 0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=COLOR[t], lw=2.6, shrinkA=24, shrinkB=12))
        for k in range(len(pts) - 1):
            ax.annotate("", xy=pts[k + 1], xytext=pts[k], zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=COLOR[t], lw=1.4, shrinkA=12, shrinkB=12))
        for k, p in enumerate(pts):
            ax.scatter(*p, s=620, c=COLOR[t], edgecolors="white", linewidths=1, zorder=4)
            ax.text(*p, f"{age[k]:.0f}d", ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", zorder=5)
        end = pts[-1] + 0.8 * d
        if len(tr) > len(chain):
            ax.plot(*zip(pts[-1] + 0.3 * d, end), color=COLOR[t], ls=":", lw=1.3)
            ax.text(*(end + 0.35 * d), f"{len(tr) - len(chain):,} older", ha="center", va="center",
                    fontsize=8, color="#444")
        extent += pts + [end + 0.35 * d]
        ax.text(*(pts[0] + np.array([-d[1], d[0]]) * 0.75), SHORT[t], ha="center", va="center", fontsize=9,
                color=COLOR[t], fontweight="bold")
    ax.scatter(0, 0, s=2400, c="#222", zorder=5)
    ax.text(0, 0, "user", color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)
    ax.set_title(f"Chain graph, real test user {uid}\nuser -> latest row per table, each row -> the user's "
                 f"previous row of that table\n(labels: days between the row and the table's cut-off)", fontsize=11)
    ax.set_aspect("equal")
    e = np.array(extent)
    (x0, y0), (x1, y1) = e.min(0) - 1.2, e.max(0) + 1.2
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches(9, 9 * (y1 - y0) / (x1 - x0) + 1.2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_CHAIN, dpi=200, bbox_inches="tight")
    print("saved", OUT_CHAIN)


def latest_rows(G, rows, depth):
    """Per activity table, the user's newest `depth` visible rows, newest first."""
    out = {}
    for t in ACTIONS:
        tr = rows[G.ntype[rows] == TABLES.index(t)]
        if len(tr):
            out[t] = tr[::-1][:depth]
    return out


def fk_targets(G, n, seed, visible):
    """(node, column) of the FKs a row keeps in the chain graph: all but its actor FK.
    An FK to a user other than the seed is skipped; an assignee equal to the seed stays."""
    out = []
    for e in range(G.fptr[n], G.fptr[n + 1]):
        tbl, col = R.RELATIONS[G.frel[e]]
        dst = int(G.fadj[e])
        if col == R.ACTOR_COL.get(tbl) or not visible[dst]:
            continue
        if G.ntype[dst] == TABLES.index("Users") and dst != seed:
            continue
        out.append((dst, col))
    return out


def fk_example_score(G, seed, rows, visible, depth):
    """How well a user shows the chain + FK picture: many tables, FK links to parent rows, shared repos."""
    shown = latest_rows(G, rows, depth)
    if len(shown) < 4:
        return -1
    seen, repos, parents = {}, {}, set()
    for rs in shown.values():
        for n in rs:
            for dst, col in fk_targets(G, n, seed, visible):
                if col == "repo_id":
                    repos[dst] = repos.get(dst, 0) + 1
                else:
                    parents.add(dst)
    return 2 * len(shown) + len(parents) + 3 * sum(c > 1 for c in repos.values()) - 1.5 * len(repos)


def chain_fk_figure(G, rows, seed, cutoff, visible, uid, depth=3):
    """The chain graph around one user with the FKs the rows keep: repository, issue / PR parents."""
    import networkx as nx
    shown = latest_rows(G, rows, depth)
    ang = dict(zip(shown, np.linspace(0.5 * np.pi, 2.5 * np.pi, len(shown), endpoint=False)))
    H = nx.DiGraph()
    pos, chain_nodes, role = {seed: np.zeros(2)}, {seed}, {}
    H.add_node(seed)
    chain_edges, fk_edges = [], []
    for t, rs in shown.items():
        d = np.array([np.cos(ang[t]), np.sin(ang[t])])
        prev = seed
        for k, n in enumerate(rs):
            n = int(n)
            pos[n] = (2.0 + 1.4 * k) * d
            chain_nodes.add(n)
            role[n] = ("chain", t, cutoff[TABLES.index(t)] - G.time[n])
            chain_edges.append((prev, n))
            prev = n
    for n in list(chain_nodes - {seed}):
        for dst, col in fk_targets(G, n, seed, visible):
            fk_edges.append((n, dst, col))
            if dst not in chain_nodes:
                role[dst] = ("fk", TABLES[G.ntype[dst]], None)
    H.add_edges_from([(a, b) for a, b in chain_edges] + [(a, b) for a, b, _ in fk_edges])
    free = [n for n in H.nodes if n not in chain_nodes]
    rng = np.random.default_rng(0)
    init = dict(pos)
    for n in free:
        nb = [a for a, b, _ in fk_edges if b == n] + [b for a, b, _ in fk_edges if a == n]
        init[n] = np.mean([pos[m] for m in nb if m in pos], 0) * 1.1 + rng.normal(0, 0.3, 2)
    pos = nx.spring_layout(H.to_undirected(), pos=init, fixed=list(chain_nodes), k=1.6, iterations=300, seed=0)

    fig, ax = plt.subplots(figsize=(11, 10))
    for a, b, col in fk_edges:
        ax.annotate("", xy=pos[b], xytext=pos[a], zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0, ls=(0, (4, 3)), shrinkA=14, shrinkB=14,
                                    connectionstyle="arc3,rad=0.08"))
        if col != "repo_id":
            m = (np.array(pos[a]) + np.array(pos[b])) / 2
            ax.text(*m, col, fontsize=7, color="#444", ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8), zorder=2)
    for a, b in chain_edges:
        t = role[b][1]
        ax.annotate("", xy=pos[b], xytext=pos[a], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=COLOR[t], lw=2.6 if a == seed else 1.6,
                                    shrinkA=24 if a == seed else 15, shrinkB=15))
    for n, (kind, t, age) in role.items():
        x, y = pos[n]
        if kind == "chain":
            ax.scatter(x, y, s=700, c=COLOR[t], edgecolors="white", linewidths=1, zorder=4)
            ax.text(x, y, f"{age:.0f}d", ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=5)
            if n == int(shown[t][0]):
                ax.text(x, y + 0.55, SHORT[t], ha="center", fontsize=9, color=COLOR[t], fontweight="bold")
        elif t == "Repositories":
            ax.scatter(x, y, s=650, marker="s", c="#d9d9d9", edgecolors="#555", linewidths=1, zorder=4)
            ax.text(x, y, "repo", ha="center", va="center", fontsize=7, zorder=5)
        else:
            ax.scatter(x, y, s=700, c="white", edgecolors=COLOR.get(t, "#555"), linewidths=2.2, zorder=4)
            ax.text(x, y, SHORT.get(t, t), ha="center", va="center", fontsize=7, zorder=5)
    ax.scatter(0, 0, s=2400, c="#222", zorder=6)
    ax.text(0, 0, "user", color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=7)
    ax.legend(handles=[Line2D([0], [0], color="#222", lw=2.5, label="chain link (user -> latest row -> previous row)"),
                       Line2D([0], [0], color="#666", lw=1.2, ls="--", label="foreign key the row keeps (row -> repo / parent row)"),
                       Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", mew=2, label="row not by this user (FK target)"),
                       Line2D([0], [0], marker="s", ls="", color="#d9d9d9", mec="#555", label="repository")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=9)
    ax.set_title(f"Chain graph with the foreign keys it keeps, real test user {uid}\n"
                 f"(the user's {depth} latest rows per table; labels: days before the table's cut-off)", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = ROOT / "figures" / "chain_graph_fk_real.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fk-user-rank", type=int, default=0,
                    help="chain + FK figure: 0 = best-scoring test user, 1 = next ...")
    ap.add_argument("--fk-depth", type=int, default=2, help="chain + FK figure: latest rows per table")
    ap.add_argument("--sample", type=int, default=20, help="1-hop sample size drawn around the user")
    ap.add_argument("--user-rank", type=int, default=0, help="0 = test user with the most visible rows, 1 = next ...")
    ap.add_argument("--depth", type=int, default=4, help="chain links drawn per table")
    args = ap.parse_args()

    cohort, ev, own, data_end = base.load()
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    cutoff = table_cutoffs(base.TEST_START, lags)
    G = R.Graph("fk")  # the chain is each user's own rows in time order, which the FK graph already stores
    visible = R.visible_nodes(G, cutoff)

    test = pd.read_csv(base.TEST_INPUT)
    cidx = pd.Series(np.arange(len(cohort)), index=cohort.github_user_id.values).loc[test.UserID].values
    seeds = G.offset[TABLES.index("Users")] + cohort.user_id.values[cidx]
    all_rows = [own_rows(G, s, visible) for s in seeds]
    n_rows = np.array([len(r) for r in all_rows])
    share = np.array([counts(G, r) / len(r) for r in all_rows if len(r)]).mean(0)
    print(f"{len(seeds):,} test users, {(n_rows == 0).mean():.1%} without visible rows; "
          f"mean share of rows by table: " + ", ".join(f"{SHORT[t]} {s:.1%}" for t, s in zip(ACTIONS, share)))

    i = int(np.argsort(-n_rows)[args.user_rank])
    rows, uid = all_rows[i], test.UserID.values[i]
    print(f"user {uid}: {len(rows):,} visible rows: " + ", ".join(f"{SHORT[t]} {c:,}" for t, c in zip(ACTIONS, counts(G, rows))))
    rng = np.random.default_rng(0)
    hub_figure(G, rows, share, args.sample, rng, uid)
    chain_figure(G, rows, cutoff, uid, args.depth)

    # A user whose chain rows also show FK links: several tables, parent issues / PRs, shared repositories.
    cand = [j for j in range(len(seeds)) if 4 <= n_rows[j] <= 400]
    score = np.array([fk_example_score(G, seeds[j], all_rows[j], visible, args.fk_depth) for j in cand])
    j = cand[int(np.argsort(-score)[args.fk_user_rank])]
    print(f"chain + FK example: user {test.UserID.values[j]}, {n_rows[j]} rows, score {score.max():.1f}")
    chain_fk_figure(G, all_rows[j], seeds[j], cutoff, visible, test.UserID.values[j], args.fk_depth)


if __name__ == "__main__":
    main()
