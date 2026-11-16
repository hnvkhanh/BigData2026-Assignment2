"""Holdout scores of multi-seed runs, one row per config (mean over seeds, range in brackets).

Reads experiments/gnn/outputs/<cfg>_<suffix>_s<seed>/metrics.json and, for RelGT runs,
experiments/relgt/outputs/<cfg>_<suffix>_s<seed>/metrics.json (two rows: its validation model and
its final model). A config name may contain * to match several configs. Used by the scripts in runs/.

    python scripts/summarize_runs.py --suffix ho0926 het het_notime all_ego_uf
    python scripts/summarize_runs.py --suffix ho0926 --segments all_ego_uf ego_uf_qprev
    python scripts/summarize_runs.py --suffix ho0523 --sort "smoke_*"
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {"gnn": ROOT / "experiments" / "gnn" / "outputs", "relgt": ROOT / "experiments" / "relgt" / "outputs"}
RELGT_VARIANTS = ("validation model", "final model")


def runs(model, cfg, suffix):
    out = []
    for f in sorted(glob.glob(str(OUTPUTS[model] / f"{glob.escape(cfg)}_{suffix}_s*" / "metrics.json"))):
        if re.fullmatch(rf"{re.escape(cfg)}_{re.escape(suffix)}_s\d+", Path(f).parent.name):
            out.append(json.loads(Path(f).read_text()))
    return out


def expand(pattern, suffix):
    """Config names matching a pattern with *, from the run folders of both models."""
    if "*" not in pattern:
        return [pattern]
    names = set()
    for d in OUTPUTS.values():
        for p in glob.glob(str(d / f"{pattern}_{suffix}_s*")):
            m = re.fullmatch(rf"(.+)_{re.escape(suffix)}_s\d+", Path(p).name)
            if m:
                names.add(m.group(1))
    return sorted(names)


def rows(cfg, suffix, segments):
    """(name, scores, best epochs, mean F1 per activity, extra text, push-rule line) per row of cfg."""
    out = []
    ms = []
    for m in runs("gnn", cfg, suffix):
        if isinstance(m.get("holdout"), dict) and "score" in m["holdout"]:
            ms.append(m)
    if ms:
        hs = [m["holdout"] for m in ms]
        extra = ""
        if segments:
            seg = {s: np.mean([h["per_segment"][s]["f1_avg"] for h in hs]) for s in ("quiet", "active")}
            extra = f"  | F1 quiet {seg['quiet']:.3f} active {seg['active']:.3f}"
        push = None
        hps = [m.get("holdout_push") for m in ms if m.get("holdout_push")]
        if hps:
            push = "6-activity score by push rule: " + "  ".join(
                f"{o} {np.mean([100 * h[o]['score'] for h in hps]):.2f} "
                f"(share {np.mean([h[o]['share'] for h in hps]):.3f})"
                for o in hps[0] if isinstance(hps[0][o], dict))
        out.append((cfg, [100 * h["score"] for h in hs], [m.get("best_epoch") for m in ms],
                    {k: np.mean([h["f1"][k] for h in hs]) for k in hs[0]["f1"]}, extra, push))
    rel = [m for m in runs("relgt", cfg, suffix) if isinstance(m.get("holdout"), list) and m["holdout"]]
    for v in RELGT_VARIANTS if rel else ():
        hs = [m["holdout"][0][v] for m in rel]
        out.append((f"{cfg} ({v})", [100 * h["score"] for h in hs], [m.get("best_epoch") for m in rel],
                    {k: np.mean([h["per_event"][k] for h in hs]) for k in hs[0]["per_event"]}, "", None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="config names (the run folders without _<suffix>_s<seed>)")
    ap.add_argument("--suffix", required=True, help="holdout suffix of the run folders, e.g. ho0926 or ho")
    ap.add_argument("--segments", action="store_true", help="add the mean F1 of quiet and active users")
    ap.add_argument("--sort", action="store_true", help="best score first")
    args = ap.parse_args()

    table = [r for p in args.configs for c in expand(p, args.suffix) for r in rows(c, args.suffix, args.segments)]
    if args.sort:
        table.sort(key=lambda r: -np.mean(r[1]))
    w = max([16] + [len(r[0]) + 1 for r in table])
    for name, sc, eps, f1, extra, push in table:
        print(f"{name:{w}s} n={len(sc)} score {np.mean(sc):.2f} ({min(sc):.2f}-{max(sc):.2f}) best_ep {eps}  "
              + "  ".join(f"{k} {v:.3f}" for k, v in f1.items()) + extra)
        if push:
            print(f"{'':{w}s} {push}")


if __name__ == "__main__":
    main()
