"""Visualize the chain graph next to the plain FK graph.

Plain FK graph: every action row (push, comment, PR, issue, review) links to its user.
Chain graph  : the user links only to their LATEST row of each table, and each row links
               to the same actor's previous row of that table (a linked list per table).
               Another user's sampled row brings that user in as a token (shown dashed).

Counts are illustrative. Usage: python plot_chain_graph.py -> docs/figures/chain_graph.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COUNTS = {"push": 9, "comment": 6, "pull_request": 4, "issue": 3, "review": 4}  # rows drawn per table
COLORS = {
    "push": "#d95f02",
    "comment": "#1b9e77",
    "pull_request": "#7570b3",
    "issue": "#e7298a",
    "review": "#66a61e",
}
OUT = os.path.join("docs", "figures", "chain_graph.png")
tables = list(COUNTS)
T = len(tables)
base_angles = np.linspace(0.5 * np.pi, 0.5 * np.pi + 2 * np.pi, T, endpoint=False)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7))


def hub(ax):
    ax.scatter(0, 0, s=2200, c="#222", zorder=5)
    ax.text(0, 0, "user", color="white", ha="center", va="center", fontsize=10,
            fontweight="bold", zorder=6)


def node(ax, x, y, t, label=None, s=150):
    ax.scatter(x, y, s=s, c=COLORS[t], edgecolors="white", linewidths=1, zorder=4)
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold", zorder=5)


# ---- Left: plain FK graph (all rows hang off the user) -------------------
rng = np.random.default_rng(1)
for t, ang in zip(tables, base_angles):
    n = COUNTS[t]
    spread = np.linspace(-0.38, 0.38, n)
    for k, d in enumerate(spread):
        r = 1.5 + 0.9 * (k % 3) / 2
        x, y = r * np.cos(ang + d), r * np.sin(ang + d)
        a1.plot([0, x], [0, y], color="#aaa", lw=0.9, zorder=1)
        node(a1, x, y, t, s=90)
hub(a1)
a1.set_title("Plain FK graph\nevery row -> user  (hub with 26 neighbours)", fontsize=12)

# ---- Right: chain graph ----------------------------------------------------
for t, ang in zip(tables, base_angles):
    n = COUNTS[t]
    shown = min(n, 4)  # latest rows, then a truncation marker
    dx, dy = np.cos(ang), np.sin(ang)
    px, py = -dy, dx  # perpendicular, for a small zig-zag so chains stay readable
    pts = []
    for k in range(shown):
        r = 1.5 + 0.8 * k
        pts.append((r * dx + 0.0 * px, r * dy + 0.0 * py))
    # user -> latest row (solid, thick = the only user link of this table)
    a2.annotate("", xy=pts[0], xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=COLORS[t], lw=2.4, shrinkA=22, shrinkB=9), zorder=2)
    # row -> previous row of the same actor and table
    for k in range(shown - 1):
        a2.annotate("", xy=pts[k + 1], xytext=pts[k],
                    arrowprops=dict(arrowstyle="-|>", color=COLORS[t], lw=1.4, shrinkA=8, shrinkB=8), zorder=2)
    for k, (x, y) in enumerate(pts):
        node(a2, x, y, t, label="new" if k == 0 else f"-{k}", s=420)
    if n > shown:
        ex, ey = (1.5 + 0.8 * shown) * dx, (1.5 + 0.8 * shown) * dy
        a2.annotate("", xy=(ex, ey), xytext=pts[-1],
                    arrowprops=dict(arrowstyle="-", color=COLORS[t], lw=1.2, ls=":", shrinkA=8), zorder=2)
        a2.text(ex + 0.12 * dx, ey + 0.12 * dy, f"... {n - shown} older", ha="center", va="center",
                fontsize=7, color="#444")

# another user's row pulled in as a token (dashed link from the latest PR)
t_idx = tables.index("pull_request")
ang = base_angles[t_idx]
lx, ly = 1.5 * np.cos(ang), 1.5 * np.sin(ang)
ox, oy = lx + 1.6 * np.cos(ang + 0.9), ly + 1.6 * np.sin(ang + 0.9)
a2.annotate("", xy=(ox, oy), xytext=(lx, ly),
            arrowprops=dict(arrowstyle="-", color="#555", lw=1.3, ls="--", shrinkA=8, shrinkB=12), zorder=2)
a2.scatter(ox, oy, s=420, c="#555", zorder=4)
a2.text(ox, oy, "other\nuser", color="white", ha="center", va="center", fontsize=6.5, zorder=5)

hub(a2)
a2.set_title("Chain graph\nuser -> latest row per table, row -> previous row (same table)", fontsize=12)

for ax in (a1, a2):
    ax.set_aspect("equal")
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-5.2, 5.2)
    ax.axis("off")

fig.legend(
    handles=[Line2D([0], [0], marker="o", ls="", color=c, label=t, markersize=9) for t, c in COLORS.items()]
    + [Line2D([0], [0], color="#555", ls="--", label="other user's sampled row"),
       Line2D([0], [0], ls="", label="new = latest row, -k = k rows earlier")],
    loc="lower center", ncol=4, frameon=False, fontsize=10,
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", OUT)
