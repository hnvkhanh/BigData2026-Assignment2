"""Ensemble of holdout runs: combine the saved per-user probabilities of several models and seeds.

The graph models tie on the holdout week 09-27 -> 10-04 (52.3-53.1, seed ranges overlapping;
docs/experiments/holdout_0927.md) but differ per activity: RelGT is best on open_issue, the ego GNN on
open_pr, the hetero GNN on pr_review and the comment activities. This experiment trains nothing new.
It combines the probabilities the runs saved (holdout.npz, written by --holdout runs of
experiments/gnn/run.py and by experiments/relgt/run.py with --holdout-weeks):

1. Members: configs <cfg>, each with runs <cfg>_<suffix>_s<seed> under experiments/gnn/outputs or
   experiments/relgt/outputs. A RelGT run gives its final model (retrained through the tune week,
   --relgt-variant final) or its validation model; its tune-week probabilities are always the
   validation model's (the final model saw that week).
2. Per run and week the probabilities become ranks in (0, 1] per activity (--normalize rank), so models
   with different probability scales mix evenly (--normalize prob averages the raw values).
3. Variants: seed:<run> = one run; bag:<cfg> = the mean over the seeds of a config; mean = the
   (--weights) mean over the configs' bags; select = per activity, the bag with the best tune-week F1.
4. Decision rule, per activity: the share of users predicted positive that maximises F1 on the tune week
   of the same combination (--rule f1), or the shares after a coordinate ascent of the score over the
   tune week (--rule joint). It is applied to the holdout week as a quantile, as every runner does. Push:
   that share (--push-rule tune) or the share of users with a push in the last 7 days of push data before
   the week (--push-rule prev-week, the submission's rule; read from a GNN member's holdout.npz).
5. The holdout week is scored with the official formula over the activities that have labels. A paired
   bootstrap over users (--bootstrap resamples, the same draws for every variant) gives the 95% interval
   of each variant's difference to the reference: the bag with the best tune-week score.
The runs' own holdout decisions (their saved `pred`; the numbers in the docs) are listed for reference.

    python experiments/ensemble/run.py --suffix ens0926 all_ego_uf het relgt_chain
    python experiments/ensemble/run.py --suffix ens0926 relgt_chain relgt_chain_hist --relgt-variant validation
    python experiments/ensemble/run.py --suffix ens0926 all_ego_uf het relgt_chain --push-rule prev-week
Writes experiments/ensemble/outputs/<name>/metrics.json; name = the configs joined by +, then the options.
"""
import argparse
import glob
import json
import re
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ghactivity as base  # noqa: E402
from experiments.gnn_review import run as review  # noqa: E402  (joint thresholds)

EVENT_COLS = list(base.EVENT_COLS)
IP = EVENT_COLS.index("push")
OUTPUTS = {"gnn": ROOT / "experiments" / "gnn" / "outputs", "relgt": ROOT / "experiments" / "relgt" / "outputs"}
OUT = Path(__file__).resolve().parent / "outputs"


class Run:
    """One seed of one config: per-user probabilities and labels of the holdout and tune weeks."""

    def __init__(self, name, model, **arrays):
        self.name, self.model = name, model
        for k, v in arrays.items():
            setattr(self, k, v)


def load_gnn(path, name, variant="validation"):
    d = np.load(path)
    labels = [str(x) for x in d["labels"]]
    if labels != EVENT_COLS:
        raise SystemExit(f"{name}: an all-activity run is needed (labels {labels})")
    if variant == "final" and "prob_final" not in d.files:
        raise SystemExit(f"{name}: no final model in {path} (run experiments/gnn/run.py with --holdout-final)")
    prob, pred = (d["prob_final"], d["pred_final"]) if variant == "final" else (d["prob"], d["pred"])
    return Run(name, "gnn", gids=d["gids"], prob=prob.astype(np.float64), truth=d["truth"], pred=pred,
               tune_gids=d["tune_gids"], tune_prob=d["tune_prob"].astype(np.float64), tune_truth=d["tune_truth"],
               push_prev=float(d["push_prev_week_share"]), tune_push_prev=float(d["tune_push_prev_week_share"]),
               week=str(d["week"]), tune_week=str(d["tune_week"]))


def load_relgt(path, name, variant):
    d = np.load(path)
    weeks = sorted({k[:10] for k in d.files if re.fullmatch(r"\d{4}-\d{2}-\d{2}_gids", k)})
    if len(weeks) != 1:
        raise SystemExit(f"{name}: one holdout week expected, found {weeks} (--holdout-weeks 1)")
    w = weeks[0]
    if f"{w}_prob_{variant}" not in d.files or "tune_weeks" not in d.files:
        raise SystemExit(f"{name}: no probabilities in {path}; rerun with the current experiments/relgt/run.py")
    tw = [str(x) for x in d["tune_weeks"]]
    if len(tw) != 1:
        raise SystemExit(f"{name}: one tune week expected, found {tw}")
    return Run(name, "relgt", gids=d[f"{w}_gids"], prob=d[f"{w}_prob_{variant}"].astype(np.float64),
               truth=d[f"{w}_truth"].astype(np.float32), pred=d[f"{w}_{variant}"],
               tune_gids=d[f"tune_{tw[0]}_gids"], tune_prob=d[f"tune_{tw[0]}_prob"].astype(np.float64),
               tune_truth=d[f"tune_{tw[0]}_truth"], push_prev=None, tune_push_prev=None, week=w, tune_week=tw[0])


def find_runs(cfg, suffix, variant, gnn_variant="validation"):
    out = []
    for model, d in OUTPUTS.items():
        for f in sorted(glob.glob(str(d / f"{glob.escape(cfg)}_{suffix}_s*" / "holdout.npz"))):
            name = Path(f).parent.name
            if re.fullmatch(rf"{re.escape(cfg)}_{re.escape(suffix)}_s\d+", name):
                out.append(load_gnn(f, name, gnn_variant) if model == "gnn" else load_relgt(f, name, variant))
    if not out:
        raise SystemExit(f"no holdout.npz for {cfg}_{suffix}_s* (runs made before probabilities were saved "
                         f"have only metrics.json: rerun them, e.g. bash runs/ensemble_ho0927.sh)")
    return out


def align(runs, gid_attr, attrs):
    """Reorder every run's rows to the sorted intersection of the runs' user ids; returns those ids."""
    common = reduce(np.intersect1d, [getattr(r, gid_attr) for r in runs])
    for r in runs:
        g = getattr(r, gid_attr)
        if len(g) != len(common):
            print(f"  {r.name}: {len(g)} users, {len(common)} shared by every run ({gid_attr})")
        pos = pd.Series(np.arange(len(g)), index=g).loc[common].values
        for a in attrs:
            setattr(r, a, getattr(r, a)[pos])
        setattr(r, gid_attr, common)
    return common


def ranks(p):
    """Per column: average ranks scaled to (0, 1]."""
    out = np.empty(p.shape)
    for j in range(p.shape[1]):
        out[:, j] = pd.Series(p[:, j]).rank(method="average").values / len(p)
    return out


def f1_cols(pred, y):
    tp = (pred & (y == 1)).sum(0)
    denom = pred.sum(0) + (y == 1).sum(0)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0), tp, denom


def score(pred, y):
    """(macro-F1 + micro-F1) / 2 of bool predictions and 0/1 labels (rows x scored activities)."""
    f1, tp, denom = f1_cols(pred, y)
    return (f1.mean() + 2 * tp.sum() / max(denom.sum(), 1)) / 2, f1


def shares_f1(y, p):
    """Per column, the share of rows predicted positive at the F1-maximising threshold."""
    out = np.zeros(p.shape[1])
    for j in range(p.shape[1]):
        thr, _ = base.best_threshold(y[:, j], p[:, j])
        out[j] = float((p[:, j] >= thr).mean())
    return out


def shares_joint(y, p):
    thr = np.array([base.best_threshold(y[:, j], p[:, j])[0] for j in range(p.shape[1])])
    joint, _ = review.joint_thresholds(y.astype(bool), p, thr, False)
    return np.array([float((p[:, j] >= joint[j]).mean()) for j in range(p.shape[1])])


def apply_shares(p, shares):
    pred = np.zeros(p.shape, bool)
    for j, s in enumerate(shares):
        if s > 0:
            pred[:, j] = p[:, j] >= np.quantile(p[:, j], 1 - s)
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="member configs (run folders without _<suffix>_s<seed>)")
    ap.add_argument("--suffix", required=True, help="suffix of the run folders, e.g. ens0926")
    ap.add_argument("--weights", type=float, nargs="+", default=None, help="one weight per config (default: equal)")
    ap.add_argument("--normalize", choices=["rank", "prob"], default="rank")
    ap.add_argument("--rule", choices=["f1", "joint"], default="f1", help="how the tune week sets the shares")
    ap.add_argument("--push-rule", choices=["tune", "prev-week"], default="tune")
    ap.add_argument("--relgt-variant", choices=["final", "validation"], default="final",
                    help="which RelGT model's holdout probabilities to use")
    ap.add_argument("--gnn-variant", choices=["validation", "final"], default="validation",
                    help="which GNN model's holdout probabilities to use: the one trained to the tune week "
                         "(validation), or the refit through it (final; runs made with --holdout-final). The "
                         "tune-week probabilities are always the validation model's")
    ap.add_argument("--bootstrap", type=int, default=2000, help="paired bootstrap resamples (0: none)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.weights and len(args.weights) != len(args.configs):
        ap.error("--weights needs one value per config")
    weights = np.array(args.weights or [1.0] * len(args.configs))
    weights = weights / weights.sum()
    name = args.name or ("+".join(args.configs) + f"_{args.suffix}" + ("" if args.normalize == "rank" else "_prob")
                         + ("" if args.rule == "f1" else "_joint") + ("" if args.push_rule == "tune" else "_pprev")
                         + ("" if args.relgt_variant == "final" else "_val")
                         + ("" if args.gnn_variant == "validation" else "_gfinal"))
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    members = {c: find_runs(c, args.suffix, args.relgt_variant, args.gnn_variant) for c in args.configs}
    runs = [r for rs in members.values() for r in rs]
    weeks = {(r.week[:10], r.tune_week[:10]) for r in runs}
    if len(weeks) != 1:
        raise SystemExit(f"the runs score different weeks: {sorted(weeks)}")
    (week, tune_week), = weeks
    for c, rs in members.items():
        print(f"{c}: {len(rs)} runs ({rs[0].model}): " + ", ".join(r.name for r in rs))
    gids = align(runs, "gids", ["prob", "truth", "pred"])
    align(runs, "tune_gids", ["tune_prob", "tune_truth"])
    truth, tune_truth = runs[0].truth, runs[0].tune_truth
    for r in runs[1:]:
        if not (np.array_equal(np.nan_to_num(r.truth, nan=-1), np.nan_to_num(truth, nan=-1))
                and np.array_equal(np.nan_to_num(r.tune_truth, nan=-1), np.nan_to_num(tune_truth, nan=-1))):
            raise SystemExit(f"{r.name}: labels differ from {runs[0].name}'s")
    scored = [j for j in range(len(EVENT_COLS)) if not np.isnan(truth[:, j]).any()]
    tune_scored = [j for j in range(len(EVENT_COLS)) if not np.isnan(tune_truth[:, j]).any()]
    acts = [EVENT_COLS[j] for j in scored]
    print(f"holdout week {week}: {len(gids):,} users, scored {acts}; tune week {tune_week}: "
          f"{len(runs[0].tune_gids):,} users, labelled {[EVENT_COLS[j] for j in tune_scored]}")
    y, ty = truth[:, scored], tune_truth[:, scored]
    tune_ok = np.array([j in tune_scored for j in scored])  # shares can be tuned for these

    push_prev = next((r.push_prev for r in runs if r.push_prev is not None), None)
    tune_push_prev = next((r.tune_push_prev for r in runs if r.tune_push_prev is not None), None)
    if args.push_rule == "prev-week" and push_prev is None:
        ap.error("--push-rule prev-week needs a GNN member (its holdout.npz has the previous week's push share)")
    if IP in scored and not tune_ok[scored.index(IP)] and args.push_rule != "prev-week":
        print("push is scored but the tune week has no push labels: push is predicted for nobody "
              "(use --push-rule prev-week)")

    norm = ranks if args.normalize == "rank" else (lambda p: p)
    for r in runs:
        r.h, r.t = norm(r.prob)[:, scored], norm(r.tune_prob)[:, scored]

    def decide(h, t):
        """Shares from the tune week (push by --push-rule), the holdout predictions and the tune score."""
        shares = np.zeros(len(scored))
        if tune_ok.any():
            cols = np.flatnonzero(tune_ok)
            fn = shares_joint if args.rule == "joint" else shares_f1
            shares[cols] = fn(ty[:, cols], t[:, cols])
        tune_shares = shares.copy()
        if IP in scored and args.push_rule == "prev-week":
            shares[scored.index(IP)] = push_prev
            tune_shares[scored.index(IP)] = tune_push_prev
        pred = apply_shares(h, shares)
        tune_pred = apply_shares(t, tune_shares)
        tune_score = score(tune_pred[:, tune_ok], ty[:, tune_ok])[0] if tune_ok.any() else float("nan")
        return pred, shares, tune_score

    def result(pred, shares, tune_score, **extra):
        s, f1 = score(pred, y)
        return {"score": float(s), "macro_f1": float(f1.mean()), "micro_f1": float(2 * s - f1.mean()),
                "f1": {a: float(v) for a, v in zip(acts, f1)}, "shares": {a: float(v) for a, v in zip(acts, shares)},
                "tune_score": float(tune_score), **extra}

    variants, preds = {}, {}
    # The runs' own decisions, for reference (the numbers in the docs).
    own = {}
    for c, rs in members.items():
        for r in rs:
            own[r.name] = result(r.pred[:, scored].astype(bool), np.full(len(scored), np.nan), float("nan"))
        own[f"own:{c}"] = {"score": float(np.mean([own[r.name]["score"] for r in rs])),
                           "f1": {a: float(np.mean([own[r.name]["f1"][a] for r in rs])) for a in acts},
                           "runs": [r.name for r in rs]}
    # Single seeds through the ensemble's pipeline (normalisation + the tune-week share rule).
    for r in runs:
        variants[f"seed:{r.name}"] = result(*decide(r.h, r.t))
    bags = {}
    for c, rs in members.items():
        bags[c] = (np.mean([r.h for r in rs], 0), np.mean([r.t for r in rs], 0))
        pred, shares, ts = decide(*bags[c])
        variants[f"bag:{c}"] = result(pred, shares, ts, runs=len(rs))
        preds[f"bag:{c}"] = pred
    if len(members) > 1:
        h = sum(w * bags[c][0] for w, c in zip(weights, args.configs))
        t = sum(w * bags[c][1] for w, c in zip(weights, args.configs))
        pred, shares, ts = decide(h, t)
        variants["mean"] = result(pred, shares, ts, weights=dict(zip(args.configs, weights.tolist())))
        preds["mean"] = pred
        # select: per activity, the bag with the best tune-week F1 at its own share.
        choice = []
        for i, j in enumerate(scored):
            if not tune_ok[i]:
                choice.append(args.configs[0])
                continue
            best = None
            for c in args.configs:
                tp = apply_shares(bags[c][1][:, [i]], shares_f1(ty[:, [i]], bags[c][1][:, [i]]))
                f1 = f1_cols(tp, ty[:, [i]])[0][0]
                if best is None or f1 > best[0]:
                    best = (f1, c)
            choice.append(best[1])
        h = np.stack([bags[c][0][:, i] for i, c in enumerate(choice)], 1)
        t = np.stack([bags[c][1][:, i] for i, c in enumerate(choice)], 1)
        pred, shares, ts = decide(h, t)
        variants["select"] = result(pred, shares, ts, chosen=dict(zip(acts, choice)))
        preds["select"] = pred

    # Official scorer as a cross-check of the fast one, on the main variant.
    main_v = "mean" if "mean" in variants else next(k for k in variants if k.startswith("bag:"))
    official = base.scores(base.to_output(gids, {k: (preds[main_v][:, scored.index(j)] if j in scored
                                                    else np.zeros(len(gids), bool)) for j, k in enumerate(EVENT_COLS)}),
                           base.to_output(gids, {k: np.nan_to_num(truth[:, j]) for j, k in enumerate(EVENT_COLS)}),
                           events=acts)
    assert abs(official["score"] - variants[main_v]["score"]) < 1e-9, (official["score"], variants[main_v]["score"])

    # Paired bootstrap over users: each variant against the bag with the best tune-week score.
    boot = {}
    ref = max((k for k in preds if k.startswith("bag:")), key=lambda k: variants[k]["tune_score"])
    if args.bootstrap:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(gids), (args.bootstrap, len(gids)))
        sc = {k: np.array([score(p[i], y[i])[0] for i in idx]) for k, p in preds.items()}
        for k in preds:
            d = sc[k] - sc[ref]
            boot[k] = {"vs": ref, "diff": float(variants[k]["score"] - variants[ref]["score"]),
                       "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                       "p_better": float((d > 0).mean())}

    w = max(len(k) for k in list(variants) + list(own)) + 1
    print(f"\n{'variant':{w}s} {'score':>6s} {'tune':>6s}  " + "  ".join(f"{a[:9]:>9s}" for a in acts)
          + ("  vs reference [95% CI]" if boot else ""))
    for k in [k for k in own if k.startswith("own:")] + list(variants):
        v = own[k] if k in own else variants[k]
        has_tune = "tune_score" in v and not np.isnan(v["tune_score"])
        line = (f"{k:{w}s} {100 * v['score']:6.2f} " + (f"{100 * v['tune_score']:6.2f}" if has_tune else "     -")
                + "  " + "  ".join(f"{v['f1'][a]:9.3f}" for a in acts))
        if k in boot:
            b = boot[k]
            line += f"  {100 * b['diff']:+5.2f} [{100 * b['ci95'][0]:+5.2f}, {100 * b['ci95'][1]:+5.2f}]"
        print(line)
    print(f"reference for the bootstrap: {ref} (best tune-week score). "
          f"Official scorer on {main_v}: {100 * official['score']:.2f}")
    if "select" in variants:
        print("select picked: " + ", ".join(f"{a} <- {c}" for a, c in variants["select"]["chosen"].items()))

    metrics = {"args": vars(args), "week": week, "tune_week": tune_week, "users": int(len(gids)), "scored": acts,
               "members": {c: [r.name for r in rs] for c, rs in members.items()},
               "push_prev_week_share": push_prev, "own": own, "variants": variants, "bootstrap": boot,
               "official_check": {main_v: official["score"]}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
