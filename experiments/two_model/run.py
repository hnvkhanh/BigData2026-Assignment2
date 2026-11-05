"""Two RelGT models: one predicts push, the other the 5 remaining activities.

push model:  the full chain graph, all 6 labels trained, only its push column is used.
rest model:  --exclude-push (no Pushes rows in the graph) and --exclude-push-label (push is not
             trained or predicted); its 5 other columns are used.

Both runs are experiments/relgt/run.py with the same chain / latest-seed / 16-seed setup. Each
saves its holdout predictions (holdout.npz); this script joins them per label (push from one,
the rest from the other), scores the joined predictions on the holdout weeks with the
assignment's metric and writes the joined test output.

Run from the repo root:
    python experiments/two_model/run.py
    python experiments/two_model/run.py --skip-run        # reuse outputs of an earlier run
Extra arguments after -- go to both relgt runs, e.g. -- --epochs 10
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RELGT = HERE.parent / "relgt"
_spec = importlib.util.spec_from_file_location("baseline_run", HERE.parent / "baseline" / "run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

EVENT_COLS = base.EVENT_COLS
IP = EVENT_COLS.index("push")
COMMON = ["--test-pre-june", "--graph-structure", "chain", "--chain-seed", "latest", "--num-seeds", "16",
          "--epochs", "15", "--holdout-weeks", "1"]  # both models need the same holdout weeks to be joined
VARIANTS = ["final", "validation"]  # the two ways relgt decides on a holdout week


def run_relgt(name, extra, passthrough):
    cmd = [sys.executable, str(RELGT / "run.py"), *COMMON, "--name", name, *extra, *passthrough]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push-name", default="two_model_push", help="relgt output name of the push model")
    ap.add_argument("--rest-name", default="two_model_rest", help="relgt output name of the other model")
    ap.add_argument("--skip-run", action="store_true", help="do not train, join existing relgt outputs")
    ap.add_argument("--name", default="default", help="output sub-directory")
    args, passthrough = ap.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]
    if not args.skip_run:
        run_relgt(args.push_name, [], passthrough)
        run_relgt(args.rest_name, ["--exclude-push", "--exclude-push-label"], passthrough)

    push_dir, rest_dir = RELGT / "outputs" / args.push_name, RELGT / "outputs" / args.rest_name
    out_dir = HERE / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    A, B = np.load(push_dir / "holdout.npz"), np.load(rest_dir / "holdout.npz")
    weeks = sorted({k.split("_")[0] for k in A.files})
    metrics = {"push_model": args.push_name, "rest_model": args.rest_name, "holdout": []}
    for w in weeks:
        gids, truth = A[f"{w}_gids"], A[f"{w}_truth"]
        assert (gids == B[f"{w}_gids"]).all(), "the two models scored different users"
        if np.isnan(truth).any():
            continue
        truth_df = base.to_output(gids, dict(zip(EVENT_COLS, truth.T)))
        r = {"week": w, "users": int(len(gids))}
        for v in VARIANTS:
            pa, pb = A[f"{w}_{v}"], B[f"{w}_{v}"]
            joined = pb.copy()
            joined[:, IP] = pa[:, IP]
            rest_only = pb.copy()
            rest_only[:, IP] = 0
            for label, p in [("two models", joined), ("push model alone", pa), ("rest model alone", rest_only)]:
                r.setdefault(v, {})[label] = base.scores(base.to_output(gids, dict(zip(EVENT_COLS, p.T))), truth_df)
        metrics["holdout"].append(r)
        print(f"holdout {w}: " + "  ".join(f"{v}: two {100 * r[v]['two models']['score']:.2f} "
                                          f"(push alone {100 * r[v]['push model alone']['score']:.2f}, "
                                          f"rest alone {100 * r[v]['rest model alone']['score']:.2f})"
                                          for v in VARIANTS))
    if metrics["holdout"]:
        metrics["holdout_score_mean"] = {v: {label: float(np.mean([r[v][label]["score"] for r in metrics["holdout"]]))
                                             for label in metrics["holdout"][0][v]} for v in VARIANTS}
        for v, s in metrics["holdout_score_mean"].items():
            print(f"mean holdout score, {v} model: " + "  ".join(f"{k} {100 * x:.2f}" for k, x in s.items()))

    push = pd.read_csv(push_dir / "output.csv").set_index("UserID")
    rest = pd.read_csv(rest_dir / "output.csv").set_index("UserID")
    out = rest.copy()
    out["push"] = push.loc[out.index, "push"]
    out = out.reset_index()[base.OUTPUT_COLUMNS]
    out.to_csv(out_dir / "output.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {out_dir / 'output.csv'}")


if __name__ == "__main__":
    main()
