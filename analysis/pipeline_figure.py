"""Schematic of the full prediction pipeline (steps 1-9), plus the train / validation / test week split.

No data needed. Run from the repo root:
    python analysis/pipeline_figure.py            # writes figures/pipeline.png and figures/pipeline.svg
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

# phase colours: data / model / decision / final
C = {"data": "#2b6cb0", "model": "#c2410c", "post": "#2f855a", "final": "#6b46c1"}
BOX_W, BOX_H = 5.6, 3.1

STEPS = [
    # (number, title, phase, lines)
    (1, "Collect data", "data",
     ["GitHub event logs of the cohort", "Tables: users, repositories, PRs,", "issues, comments, reviews, pushes",
      "Weeks with missing logs: handled"]),
    (2, "Build weekly examples", "data",
     ["One example per eligible user-week", "Input: everything before the week", "Label: did the user do each of the",
      "6 activities during the week?"]),
    (3, "Build the graph", "data",
     ["Rows = nodes, foreign keys = edges", "Each user's actions of one type", "are also chained in time order"]),
    (4, "RelGT encodes examples", "model",
     ["Seeds: the user's 16 latest actions", "Around each seed ~100 nearby rows", "become tokens (newest kept)",
      "Transformer over tokens + global", "summary -> 6 probabilities"]),
    (5, "Train", "model",
     ["Training weeks: past weeks,", "15 epochs at most", "Validation weeks: most recent,",
      "held back, decide when to stop"]),
    (6, "Probabilities -> yes / no", "post",
     ["Per activity: rank users by prob.", "Top share = positive", "Share maximises F1 on validation weeks"]),
    (7, "Repeat with 3 seeds", "post",
     ["Run steps 4-6 three times", "with different random seeds", "-> 3 rank lists + 3 shares per activity"]),
    (8, "Bag", "post",
     ["Per activity: average the 3 rank lists", "Top share = positive, share =", "mean of the 3 seeds' shares"]),
    (9, "Final fit and predict", "final",
     ["Redo steps 5-8 on ALL labelled weeks", "(nothing held out)", "Predict the test week"]),
]

# snake layout on a 3 x 3 grid: row 0 left->right, row 1 right->left, row 2 left->right
ORDER = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (2, 1), 5: (1, 1), 6: (0, 1), 7: (0, 2), 8: (1, 2), 9: (2, 2)}
GX, GY = 6.6, 4.4  # grid pitch


def centre(col, row):
    return col * GX + BOX_W / 2, -row * GY - BOX_H / 2


def draw_step(ax, n, title, phase, lines):
    col, row = ORDER[n]
    cx, cy = centre(col, row)
    x0, y0 = cx - BOX_W / 2, cy - BOX_H / 2
    ax.add_patch(FancyBboxPatch((x0, y0), BOX_W, BOX_H, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc="white", ec=C[phase], lw=1.8, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, y0 + BOX_H - 0.75), BOX_W, 0.75, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=C[phase], ec=C[phase], lw=1.8, zorder=3))
    ax.add_patch(Rectangle((x0, y0 + BOX_H - 0.75), BOX_W, 0.3, fc=C[phase], ec="none", zorder=3))
    ax.text(x0 + 0.2, y0 + BOX_H - 0.38, f"{n}  {title}", color="white", fontsize=11.5, fontweight="bold",
            va="center", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x0 + 0.25, y0 + BOX_H - 1.15 - i * 0.42, line, fontsize=9.5, color="#222", va="center", zorder=4)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#555", zorder=1, **kw))


def connect(ax, n, m):
    (c1, r1), (c2, r2) = ORDER[n], ORDER[m]
    (x1, y1), (x2, y2) = centre(c1, r1), centre(c2, r2)
    if r1 == r2:  # horizontal hop
        s = 1 if c2 > c1 else -1
        arrow(ax, (x1 + s * BOX_W / 2, y1), (x2 - s * BOX_W / 2, y2))
    else:  # vertical hop down
        arrow(ax, (x1, y1 - BOX_H / 2), (x2, y2 + BOX_H / 2))


def draw_timeline(ax, y):
    """Weeks axis: training | validation | test, for the model-selection fit (steps 5-8) and the final fit (step 9)."""
    x0, w, h = 0.0, 18.4, 0.55
    n_train, n_val, n_test = 14, 4, 1
    unit = w / (n_train + n_val + n_test)
    ax.text(x0, y + 2.2, "Weeks: which are used for what", fontsize=11.5, fontweight="bold", color="#222")
    rows = [
        ("Steps 5-8: select epochs and shares", [(n_train, "train", C["model"]), (n_val, "validation", C["post"]),
                                                 (n_test, "test", "#999")]),
        ("Step 9: final fit", [(n_train + n_val, "train on all labelled weeks", C["final"]),
                               (n_test, "predict", "#222")]),
    ]
    for r, (label, segs) in enumerate(rows):
        yy = y + 0.9 - r * 1.1
        ax.text(x0 - 0.15, yy + h / 2, label, ha="right", va="center", fontsize=9.5, color="#222")
        x = x0
        for k, name, col in segs:
            ax.add_patch(Rectangle((x, yy), k * unit, h, fc=col, ec="white", lw=1.5, alpha=0.9))
            ax.text(x + k * unit / 2, yy + h / 2, name, ha="center", va="center", fontsize=8.5, color="white",
                    fontweight="bold")
            x += k * unit
    arrow(ax, (x0, y - 0.55), (x0 + w, y - 0.55))
    ax.text(x0 + w, y - 0.95, "time", ha="right", fontsize=9, color="#555")


def main():
    fig, ax = plt.subplots(figsize=(15, 13.5))
    for step in STEPS:
        draw_step(ax, *step)
    for n in range(1, 9):
        connect(ax, n, n + 1)

    # seed loop around step 7 -> back to step 4
    (c4, r4), (c7, r7) = ORDER[4], ORDER[7]
    x7, y7 = centre(c7, r7)
    ax.text(x7, y7 - BOX_H / 2 - 0.4, "3 independent runs of steps 4-6", ha="center", fontsize=9, color="#555",
            style="italic")

    # legend
    for i, (key, name) in enumerate([("data", "Data"), ("model", "Model"), ("post", "Decision rule"),
                                     ("final", "Final fit")]):
        ax.add_patch(Rectangle((0.0 + i * 3.4, 1.0), 0.35, 0.35, fc=C[key], ec="none"))
        ax.text(0.5 + i * 3.4, 1.18, name, va="center", fontsize=10)

    draw_timeline(ax, -3 * GY - 2.4)
    ax.set_xlim(-6.0, 3 * GX - 1.0 + 0.2)
    ax.set_ylim(-3 * GY - 5.0, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Weekly GitHub activity prediction: pipeline", fontsize=15, fontweight="bold", loc="left", x=0.03)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"pipeline.{ext}", dpi=160, bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "pipeline.png")


if __name__ == "__main__":
    main()
