"""Seed-bagged submission from several RelGT runs (the best model: the seed bag).

Each run of experiments/relgt/run.py (with --holdout-weeks 0) writes test.npz: the final model's probabilities for
the test users and the validation share of positives per label. This script turns each run's probabilities into
ranks in (0, 1] per activity (as experiments/ensemble/run.py does), averages them over the runs, and predicts
positive for the top share of users per activity, the share being the mean of the runs' validation shares.

    python experiments/relgt/bag_submit.py relgt_chain_recent_test_s0 relgt_chain_recent_test_s1 relgt_chain_recent_test_s2 \
        --name relgt_chain_recent_bag

Writes experiments/relgt/outputs/<name>/output.csv (UserID + the 6 activity columns).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "relgt" / "outputs"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders under experiments/relgt/outputs (each with test.npz)")
    ap.add_argument("--name", default="relgt_bag", help="output sub-directory")
    args = ap.parse_args()

    runs = [np.load(OUT / r / "test.npz", allow_pickle=True) for r in args.runs]
    ids, labels = runs[0]["user_id"], list(runs[0]["labels"])
    for r, n in zip(runs, args.runs):
        if not (np.array_equal(r["user_id"], ids) and list(r["labels"]) == labels):
            sys.exit(f"{n}: users or labels differ from {args.runs[0]}")
    n = len(ids)
    score = np.mean([np.stack([pd.Series(r["prob"][:, j]).rank().values / n for j in range(len(labels))], 1) for r in runs], 0)
    rates = np.mean([r["rates"] for r in runs], 0)
    out = pd.DataFrame({"UserID": ids})
    for j, k in enumerate(labels):
        out[k] = 0 if rates[j] <= 0 else (score[:, j] >= np.quantile(score[:, j], 1 - rates[j])).astype(int)
        print(f"{k:18s} share predicted positive {out[k].mean():.3f}  (validation share {rates[j]:.3f})")
    d = OUT / args.name
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "output.csv", index=False)
    print(f"bagged {len(runs)} runs -> {d / 'output.csv'}")


if __name__ == "__main__":
    main()
