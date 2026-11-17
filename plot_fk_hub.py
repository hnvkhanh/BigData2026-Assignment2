"""Illustrate why a plain FK graph makes heavy users into hubs dominated by pushes.

Left : a heavy user with every action row linked directly to them (star / hub).
Right: composition of a random 1-hop sample (k nodes) around that user.

Usage: python plot_fk_hub.py  ->  docs/figures/fk_hub.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Illustrative row counts for one heavy user (pushes are by far the most frequent table).
COUNTS = {"push": 120, "comment": 18, "pull_request": 8, "issue": 6, "review": 4}
COLORS = {
    "push": "#d95f02",
    "comment": "#1b9e77",
    "pull_request": "#7570b3",
    "issue": "#e7298a",
    "review": "#66a61e",
}
SAMPLE_K = 20  # 1-hop neighbour sample size
OUT = os.path.join("docs", "figures", "fk_hub.png")

rng = np.random.default_rng(0)

# Rows in a flat list, grouped by table
rows = [t for t, n in COUNTS.items() for _ in range(n)]
rng.shuffle(rows)
N = len(rows)

# 1-hop sample (uniform over neighbours) -> mostly pushes
sampled_idx = set(rng.choice(N, size=SAMPLE_K, replace=False).tolist())
sample_counts = {t: sum(rows[i] == t for i in sampled_idx) for t in COUNTS}

fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.5, 1]}
)

# ---- Left: hub graph ------------------------------------------------------
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
radius = 1.0 + 0.25 * rng.random(N)
xs, ys = radius * np.cos(angles), radius * np.sin(angles)

for i, (x, y) in enumerate(zip(xs, ys)):
    picked = i in sampled_idx
    ax.plot(
        [0, x], [0, y],
        color="#222" if picked else "#bbb",
        lw=1.6 if picked else 0.5,
        alpha=0.9 if picked else 0.6,
        zorder=1,
    )
for i, (x, y) in enumerate(zip(xs, ys)):
    picked = i in sampled_idx
    ax.scatter(
        x, y, s=70 if picked else 28, c=COLORS[rows[i]],
        edgecolors="black" if picked else "none", linewidths=1.2, zorder=2,
        alpha=1.0 if picked else 0.55,
    )
ax.scatter(0, 0, s=2200, c="#222", zorder=3)
ax.text(0, 0, "heavy\nuser", color="white", ha="center", va="center",
        fontsize=9, fontweight="bold", zorder=4)
ax.set_title(f"Plain FK graph: every action row links directly to the user\n"
             f"({N} rows; bold = {SAMPLE_K}-node 1-hop sample)", fontsize=11)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(
    handles=[Line2D([0], [0], marker="o", ls="", color=c, label=f"{t} ({COUNTS[t]})")
             for t, c in COLORS.items()],
    loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=9,
)

# ---- Right: sample composition -------------------------------------------
names = list(COUNTS)
full = np.array([COUNTS[t] for t in names]) / N * 100
samp = np.array([sample_counts[t] for t in names]) / SAMPLE_K * 100
y = np.arange(len(names))
bx.barh(y + 0.2, full, height=0.4, color=[COLORS[t] for t in names], alpha=0.35,
        label="all neighbours")
bx.barh(y - 0.2, samp, height=0.4, color=[COLORS[t] for t in names],
        label=f"1-hop sample (k={SAMPLE_K})")
bx.set_yticks(y)
bx.set_yticklabels(names)
bx.invert_yaxis()
bx.set_xlabel("% of sampled neighbours")
bx.set_title("Sample is dominated by the most frequent table: push", fontsize=11)
bx.legend(frameon=False, loc="lower right")
for s in ("top", "right"):
    bx.spines[s].set_visible(False)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", OUT)
