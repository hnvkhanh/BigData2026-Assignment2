"""Effect of each component of the best model (relgt_chain_recent) on the holdout score.

Three figures in figures/:
  component_ladder.png     the six ablation configs, mean of the seeds with min-max range, and the 3-seed bag
  component_effects.png    score change from adding each component (latest seed, recent fill, chain graph)
  component_negatives.png  variants that were not adopted (other-actions context, seed type, hop cap), bag change
                           against relgt_chain_recent with the 95% CI where the docs give one

Final-model scores per seed are read from experiments/relgt/outputs/<cfg>_<suffix>_s<seed>/metrics.json when
present; otherwise (or with --from-docs) the numbers copied from docs/experiments/*.md are used. Bag scores and
CIs always come from the docs (they are computed by experiments/ensemble).

Run from the repo root: python analysis/component_effects_figure.py [--suffix ens0926] [--from-docs]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RELGT_OUT = ROOT / "experiments" / "relgt" / "outputs"

# name: (chain, latest seed, recent fill, label, final mean, min, max, bag)  -- docs/experiments/relgt_ablation.md
ABLATION = {
    "relgt_fk": (0, 0, 0, "FK graph", 50.25, 50.15, 50.31, 52.66),
    "relgt_fk_recent": (0, 0, 1, "FK graph", 52.44, 52.20, 52.65, 54.10),
    "relgt_chain_user": (1, 0, 0, "chain graph", 51.62, 51.01, 52.09, 53.31),
    "relgt_chain_user_recent": (1, 0, 1, "chain graph", 52.04, 51.62, 52.32, 53.17),
    "relgt_chain": (1, 1, 0, "chain graph", 52.59, 51.72, 53.04, 54.42),
    "relgt_chain_recent": (1, 1, 1, "chain graph", 53.41, 52.63, 53.90, 54.06),
}
BEST = "relgt_chain_recent"
REF_BAG = ABLATION[BEST][7]
# (label, bag, delta vs best bag, ci low, ci high)  -- relgt_ctx.md, relgt_seedtype.md, relgt_hops.md
NEGATIVES = [
    ("other users' actions\nas context only", 54.06, +0.01, -0.75, +0.76),
    ("context only\n+ recent fill", 52.58, 52.58 - REF_BAG, None, None),
    ("latest seed per type", 53.59, -0.47, -1.35, +0.43),
    ("latest seed per type\n+ recent fill", 52.84, -1.21, -2.07, -0.35),
    ("max 3 hops", 53.12, -0.94, -1.64, -0.22),
    ("max 5 hops", 53.50, -0.56, -1.35, +0.20),
    ("max 10 hops", 52.91, -1.15, -2.02, -0.31),
]
MODEL, DATA, TEXT, MUTED = "#c2410c", "#2b6cb0", "#222222", "#9a9a9a"
GOOD, BAD = "#2f855a", "#b91c1c"


def load_runs(suffix, seeds):
    """{cfg: scores of the final model per seed} from the run folders; configs without runs are left out."""
    out = {}
    for cfg in ABLATION:
        sc = []
        for s in seeds:
            f = RELGT_OUT / f"{cfg}_{suffix}_s{s}" / "metrics.json"
            if f.exists():
                h = json.loads(f.read_text()).get("holdout")
                if isinstance(h, list) and h:
                    sc.append(100 * h[0]["final model"]["score"])
        if sc:
            out[cfg] = sc
    return out if len(out) == len(ABLATION) else {}


def table(runs):
    """Ablation rows (cfg -> mean, min, max, bag); from the runs when all six are there, else the docs."""
    t = {}
    for cfg, (_, _, _, _, m, lo, hi, bag) in ABLATION.items():
        if runs:
            sc = runs[cfg]
            m, lo, hi = float(np.mean(sc)), min(sc), max(sc)
        t[cfg] = (m, lo, hi, bag)
    return t


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#555", labelsize=9)
    ax.yaxis.grid(True, color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)


def ladder(t, out):
    names = list(ABLATION)
    x = np.arange(len(names))
    fig = plt.figure(figsize=(8.5, 5.6))
    ax = fig.add_axes([0.09, 0.30, 0.89, 0.62])
    for i, n in enumerate(names):
        m, lo, hi, bag = t[n]
        c = MODEL if n == BEST else DATA
        ax.plot([i, i], [lo, hi], color=c, lw=6, alpha=0.30, solid_capstyle="round")
        ax.plot(i, m, "o", color=c, ms=9, zorder=3)
        ax.text(i + 0.14, m, f"{m:.2f}", va="center", fontsize=8.5, color=TEXT)
        ax.plot(i, bag, "D", color="white", mec=c, mew=1.6, ms=7, zorder=3)
    ax.plot([], [], "o", color=DATA, label="final model, mean of 3 seeds (bar = min-max)")
    ax.plot([], [], "D", color="white", mec=DATA, mew=1.6, label="3-seed bag")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_xlim(-0.6, len(names) - 0.3)
    ax.set_xticks([])
    ax.set_ylabel("holdout score (x100)", fontsize=9.5, color=TEXT)
    ax.set_title("Ablation of the three components; the best model (orange) has all three",
                 fontsize=11, color=TEXT, loc="left")
    style(ax)
    grid = fig.add_axes([0.09, 0.05, 0.89, 0.20], sharex=ax)
    rows = ["chain graph", "latest seed", "recent fill"]
    for r, lab in enumerate(rows):
        grid.text(-0.65, r, lab, ha="right", va="center", fontsize=9, color=TEXT)
        for i, n in enumerate(names):
            on = ABLATION[n][r]
            c = MODEL if n == BEST else DATA
            grid.plot(i, r, "o", ms=9, color=c if on else "white", mec=c if on else "#cfcfcf", mew=1.2)
    grid.set_ylim(2.6, -0.6)
    grid.set_yticks([])
    grid.set_xticks([])
    for s in grid.spines.values():
        s.set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def effects(t, out):
    d = lambda a, b, k: t[b][k] - t[a][k]  # noqa: E731  (k: 0 final mean, 3 bag)
    comps = [
        ("latest seed", [("without recent fill", "relgt_chain_user", "relgt_chain"),
                         ("with recent fill", "relgt_chain_user_recent", "relgt_chain_recent")]),
        ("recent fill", [("on FK graph", "relgt_fk", "relgt_fk_recent"),
                         ("on chain graph", "relgt_chain", "relgt_chain_recent")]),
        ("chain graph", [("without recent fill", "relgt_fk", "relgt_chain_user"),
                         ("with recent fill", "relgt_fk_recent", "relgt_chain_user_recent")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    axes[0].invert_yaxis()  # shared y: invert once
    for ax, (k, title) in zip(axes, ((0, "final model, mean of 3 seeds"), (3, "3-seed bag"))):
        y = 0
        ticks, labels = [], []
        for comp, conds in comps:
            for cond, a, b in conds:
                v = d(a, b, k)
                ax.barh(y, v, color=GOOD if v >= 0 else BAD, height=0.65)
                ax.text(v + (0.06 if v >= 0 else -0.06), y, f"{v:+.2f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=9, color=TEXT)
                ticks.append(y)
                labels.append(f"{comp}\n{cond}")
                y += 1
            y += 0.6
        ax.axvline(0, color=TEXT, lw=0.8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=9, color=TEXT)
        ax.set_xlim(-1.6, 2.9)
        ax.set_xlabel("score change from adding the component (x100)", fontsize=9, color=TEXT)
        ax.set_title(title, fontsize=10.5, color=TEXT, loc="left")
        style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color="#e5e5e5", lw=0.8)
    fig.suptitle("Effect of adding each component (latest seed helps in every setting; the others depend on context)", x=0.01, ha="left",
                 fontsize=11.5, color=TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def negatives(out):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(NEGATIVES))
    for i, (lab, _, dv, lo, hi) in enumerate(NEGATIVES):
        sig = hi is not None and hi < 0
        c = BAD if sig else MUTED
        ax.barh(i, dv, color=c, height=0.62)
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=TEXT, lw=1.4)
        ax.text(1.28, i, f"{dv:+.2f}", va="center", ha="right", fontsize=9, color=TEXT)
    ax.axvline(0, color=TEXT, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([n[0] for n in NEGATIVES], fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("bag score change vs relgt_chain_recent (x100); whiskers = 95% CI", fontsize=9, color=TEXT)
    ax.set_title("Variants that were not adopted: none clearly beats the best model", fontsize=11, color=TEXT, loc="left")
    ax.set_xlim(-2.3, 1.3)
    ax.legend(handles=[Patch(color=BAD, label="CI excludes 0"), Patch(color=MUTED, label="CI includes 0 / not available")],
              frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color="#e5e5e5", lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="ens0926", help="run folder suffix")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--from-docs", action="store_true", help="use the numbers copied from docs/experiments")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {} if args.from_docs else load_runs(args.suffix, args.seeds)
    print("scores from:", "run folders" if runs else "docs/experiments (embedded)")
    t = table(runs)
    for n, (m, lo, hi, bag) in t.items():
        print(f"{n:26s} final {m:.2f} ({lo:.2f}-{hi:.2f})  bag {bag:.2f}")
    ladder(t, out / "component_ladder.png")
    effects(t, out / "component_effects.png")
    negatives(out / "component_negatives.png")
    print("wrote", *(out / f for f in ("component_ladder.png", "component_effects.png", "component_negatives.png")),
          sep="\n  ")


if __name__ == "__main__":
    main()
