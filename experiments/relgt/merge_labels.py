"""Merge the single-label RelGT runs (experiments/relgt/run.py --labels <label>) into one 6-label run.

For every seed the 6 runs <cfg>_only_<label>_<suffix>_s<seed> are joined: column j of every probability, decision
and rate array comes from the run trained for label j. The result is written as a run folder
<cfg>_perlabel_<suffix>_s<seed> with the files of a normal run (holdout.npz, test.npz, output.csv, metrics.json), so
scripts/summarize_runs.py and experiments/ensemble/run.py read it like any other config.

    python experiments/relgt/merge_labels.py --cfg relgt_chain_recent --suffix ens0926 --seeds 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ghactivity as base  # noqa: E402

OUT = ROOT / "experiments" / "relgt" / "outputs"
EVENT_COLS = base.EVENT_COLS
VARIANTS = ("final", "validation")


def merge_seed(cfg, suffix, seed, prefix=""):
    names = [f"{prefix}{cfg}_only_{k}_{suffix}_s{seed}" for k in EVENT_COLS]
    for n in names:
        if not (OUT / n / "holdout.npz").exists():
            sys.exit(f"missing {OUT / n / 'holdout.npz'}")
    hs = [np.load(OUT / n / "holdout.npz", allow_pickle=True) for n in names]
    ts = [np.load(OUT / n / "test.npz", allow_pickle=True) for n in names]
    ms = [json.loads((OUT / n / "metrics.json").read_text()) for n in names]

    def same(key, arrs):
        for a, n in zip(arrs, names):
            if not np.array_equal(a[key], arrs[0][key]):
                sys.exit(f"{n}: {key} differs from {names[0]}")

    def pick(arrs, key):  # column j from the run of label j
        return np.stack([a[key][:, j] for j, a in enumerate(arrs)], 1)

    # tune and holdout arrays: the same users in every run, columns taken per label
    dump = {"tune_weeks": hs[0]["tune_weeks"]}
    same("tune_weeks", hs)
    for tw in hs[0]["tune_weeks"]:
        for part in ("gids", "truth"):
            same(f"tune_{tw}_{part}", hs)
            dump[f"tune_{tw}_{part}"] = hs[0][f"tune_{tw}_{part}"]
        dump[f"tune_{tw}_prob"] = pick(hs, f"tune_{tw}_prob")
    weeks = sorted(k[:-5] for k in hs[0].files if k.endswith("_gids") and not k.startswith("tune_"))
    holdout = []
    for w in weeks:
        for part in ("gids", "truth"):
            same(f"{w}_{part}", hs)
            dump[f"{w}_{part}"] = hs[0][f"{w}_{part}"]
        for v in VARIANTS:
            dump[f"{w}_{v}"] = pick(hs, f"{w}_{v}")
            dump[f"{w}_prob_{v}"] = pick(hs, f"{w}_prob_{v}")
        gids, truth = dump[f"{w}_gids"], dump[f"{w}_truth"]
        truth = base.to_output(gids, {k: truth[:, j] for j, k in enumerate(EVENT_COLS)})
        r = {"week": w, "users": int(len(gids))}
        for name, v in (("final model", "final"), ("validation model", "validation")):
            pred = base.to_output(gids, {k: dump[f"{w}_{v}"][:, j] for j, k in enumerate(EVENT_COLS)})
            r[name] = base.scores(pred, truth)
        holdout.append(r)
        print(f"seed {seed} holdout {w}: " + "  ".join(f"{v} {100 * r[v]['score']:.2f}"
                                                       for v in ("final model", "validation model")))

    out = OUT / f"{prefix}{cfg}_perlabel_{suffix}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "holdout.npz", **dump)
    for t in ts:
        if not np.array_equal(t["user_id"], ts[0]["user_id"]):
            sys.exit("test users differ between the runs")
    rates = np.array([t["rates"][j] for j, t in enumerate(ts)])
    np.savez(out / "test.npz", user_id=ts[0]["user_id"], prob=pick(ts, "prob").astype(np.float32), rates=rates,
             labels=np.array(EVENT_COLS))
    pd.DataFrame({"UserID": ts[0]["user_id"],
                  **{k: pd.read_csv(OUT / n / "output.csv")[k].values for k, n in zip(EVENT_COLS, names)}}
                 ).to_csv(out / "output.csv", index=False)
    metrics = {"args": {**ms[0]["args"], "labels": None, "name": out.name, "merged_from": names},
               "best_epoch": {k: m["best_epoch"] for k, m in zip(EVENT_COLS, ms)},
               "per_label": {k: m["per_label"][k] for k, m in zip(EVENT_COLS, ms)}, "holdout": holdout}
    if holdout:
        metrics["holdout_score_mean"] = {v: float(np.mean([r[v]["score"] for r in holdout]))
                                         for v in ("final model", "validation model")}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="config name, e.g. relgt_chain_recent")
    ap.add_argument("--suffix", required=True, help="run suffix, e.g. ens0926")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--prefix", default="", help="run name prefix (smoke_ for the smoke runs)")
    args = ap.parse_args()
    for s in args.seeds:
        merge_seed(args.cfg, args.suffix, s, args.prefix)


if __name__ == "__main__":
    main()
