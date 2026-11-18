"""Two questions about the best model's graph, one figure each (figures/):

  ctx_effect.png       Does leaving out other users' actions help?       (--other-actions context)
  seedtype_effect.png  Does seeding at the latest action of each type
                       instead of the 16 latest actions help?             (--chain-seed latest-per-type)

Each figure compares the variant with its baseline twice, with random fill and with recent fill (--fk-order recent):
  left    holdout score (x100): final model, mean of 3 seeds with min-max bar, and the 3-seed bag (diamond)
  middle  holdout score of the validation model (mean of seeds), i.e. before the refit through the tune week
  right   change in F1 per activity (final model, mean of seeds): variant minus baseline

Per-seed scores come from experiments/relgt/outputs/<cfg>_<suffix>_s<seed>/metrics.json when all configs have runs;
otherwise (or with --from-docs) the numbers copied from docs/experiments/relgt_ctx.md and relgt_seedtype.md are used.
Bag scores always come from those docs (computed by experiments/ensemble).

Run from the repo root: python analysis/ctx_seedtype_figures.py [--suffix ens0926] [--from-docs] [--out figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RELGT_OUT = ROOT / "experiments" / "relgt" / "outputs"
ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
SHORT = ["PR", "issue", "comment", "review", "review\ncomment", "push"]

# cfg: (final mean, min, max, validation mean, bag, F1 per activity of the final model)   (docs/experiments)
DOCS = {
    "relgt_chain": (52.59, 51.72, 53.04, 52.05, 54.42, [0.402, 0.462, 0.407, 0.447, 0.443, 0.681]),
    "relgt_chain_recent": (53.41, 52.63, 53.90, 52.33, 54.06, [0.419, 0.451, 0.442, 0.467, 0.430, 0.683]),
    "relgt_chain_ctx": (52.58, 52.01, 53.57, 51.18, 54.06, [0.415, 0.462, 0.378, 0.462, 0.431, 0.683]),
    "relgt_chain_ctx_recent": (52.91, 52.58, 53.12, 52.21, 52.58, [0.415, 0.466, 0.413, 0.463, 0.409, 0.682]),
    "relgt_chain_type": (52.48, 52.32, 52.64, 49.99, 53.59, [0.400, 0.430, 0.407, 0.445, 0.438, 0.687]),
    "relgt_chain_type_recent": (52.53, 51.73, 53.17, 50.02, 52.84, [0.414, 0.449, 0.384, 0.458, 0.411, 0.686]),
}
# (file, title, verdict, variant label, baseline label, {fill: (baseline cfg, variant cfg)})
QUESTIONS = [
    ("ctx_effect.png", "Does leaving out other users' actions help?",
     "No: level with random fill, worse with recent fill; the comment activities lose.",
     "other users' actions as context only", "all actions",
     {"random fill": ("relgt_chain", "relgt_chain_ctx"),
      "recent fill": ("relgt_chain_recent", "relgt_chain_ctx_recent")}),
    ("seedtype_effect.png", "Does seeding at the latest action of each type help?",
     "No: final models level, bags lower, validation models clearly worse.",
     "latest action of each type (up to 6 seeds)", "16 latest actions",
     {"random fill": ("relgt_chain", "relgt_chain_type"),
      "recent fill": ("relgt_chain_recent", "relgt_chain_type_recent")}),
]
TEXT, MUTED, BASE, VAR = "#222222", "#9a9a9a", "#2b6cb0", "#c2410c"
GOOD, BAD = "#2f855a", "#b91c1c"


def from_runs(suffix, seeds):
    """Same tuples as DOCS (bag taken from DOCS) built from the run folders; {} unless every config has runs."""
    out = {}
    for cfg, doc in DOCS.items():
        fin, val, f1 = [], [], []
        for s in seeds:
            f = RELGT_OUT / f"{cfg}_{suffix}_s{s}" / "metrics.json"
            if not f.exists():
                continue
            h = json.loads(f.read_text()).get("holdout")
            if isinstance(h, list) and h:
                fin.append(100 * h[0]["final model"]["score"])
                val.append(100 * h[0]["validation model"]["score"])
                pe = h[0]["final model"]["per_event"]
                f1.append([pe[a] for a in ACTS])
        if not fin:
            return {}
        out[cfg] = (float(np.mean(fin)), min(fin), max(fin), float(np.mean(val)), doc[4], list(np.mean(f1, axis=0)))
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#555", labelsize=9)
    ax.yaxis.grid(True, color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)


def figure(q, data, out):
    fname, title, verdict, vlab, blab, fills = q
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.1, 0.8, 1.5]})
    x = np.arange(len(fills))
    w = 0.34
    for j, (b, v) in enumerate(fills.values()):
        for k, (cfg, col) in enumerate(((b, BASE), (v, VAR))):
            m, lo, hi, val, bag, _ = data[cfg]
            xx = j + (k - 0.5) * w
            ax1.bar(xx, m, w * 0.92, color=col, alpha=0.85)
            ax1.plot([xx, xx], [lo, hi], color=TEXT, lw=1.6)
            ax1.text(xx, 51.1, f"{m:.2f}", ha="center", va="bottom", fontsize=9, color="white")
            ax1.plot(xx, bag, "D", color="white", mec=TEXT, mew=1.4, ms=7, zorder=3)
            ax2.bar(xx, val, w * 0.92, color=col, alpha=0.85)
            ax2.text(xx, val - 0.08, f"{val:.2f}", ha="center", va="top", fontsize=8.5, color="white")
    for ax, ylim in ((ax1, (51.0, 54.8)), (ax2, (49.0, 53.0))):
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(list(fills), fontsize=9.5, color=TEXT)
        style(ax)
    ax1.set_ylabel("holdout score (x100)", fontsize=9.5, color=TEXT)
    ax1.set_title("final model (line = min-max of 3 seeds,\ndiamond = 3-seed bag)", fontsize=10, color=TEXT, loc="left")
    ax2.set_title("validation model (before the refit),\nmean of seeds", fontsize=10, color=TEXT, loc="left")

    xs = np.arange(len(ACTS))
    for k, (b, v) in enumerate(fills.values()):
        d = np.array(data[v][5]) - np.array(data[b][5])
        ax3.bar(xs + (k - 0.5) * 0.36, d, 0.33, color=[GOOD if t >= 0 else BAD for t in d],
                alpha=1.0 if k else 0.5, hatch="" if k else "//", edgecolor="white")
    ax3.axhline(0, color=TEXT, lw=0.8)
    ax3.set_xticks(xs)
    ax3.set_xticklabels(SHORT, fontsize=9, color=TEXT)
    ax3.set_ylabel("F1 change (variant - baseline)", fontsize=9.5, color=TEXT)
    ax3.set_title("per activity, final model, mean of seeds", fontsize=10, color=TEXT, loc="left")
    ax3.legend(handles=[Patch(facecolor=MUTED, hatch="//", edgecolor="white", label="random fill"),
                        Patch(facecolor=MUTED, label="recent fill"),
                        Patch(color=GOOD, label="better"), Patch(color=BAD, label="worse")],
               frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    style(ax3)

    fig.legend(handles=[Patch(color=BASE, alpha=0.85, label=blab), Patch(color=VAR, alpha=0.85, label=vlab)],
               frameon=False, fontsize=9.5, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 0.93))
    fig.suptitle(title, x=0.01, y=1.02, ha="left", fontsize=12.5, color=TEXT, fontweight="bold")
    fig.text(0.01, 0.935, verdict, ha="left", fontsize=10.5, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / fname, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="ens0926")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--from-docs", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {} if args.from_docs else from_runs(args.suffix, args.seeds)
    print("scores from:", "run folders" if data else "docs/experiments (embedded)")
    data = data or DOCS
    for cfg, d in data.items():
        print(f"{cfg:26s} final {d[0]:.2f} ({d[1]:.2f}-{d[2]:.2f})  validation {d[3]:.2f}  bag {d[4]:.2f}")
    for q in QUESTIONS:
        figure(q, data, out)
        print("wrote", out / q[0])


if __name__ == "__main__":
    main()
