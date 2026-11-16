"""Baseline: per-user activity-history features + one LightGBM classifier per activity.

One training sample = (cohort user, week start s). Labels are whether the user performs each
activity in [s, s + 7d). Features use only rows that were available before a per-table cutoff
s - lag, where lag is the distance between the official test week start and the end of that
table's data. This reproduces at training time the same staleness the final prediction has, so
it follows the date the GH Archive files were downloaded.

Run from the repo root:
    python experiments/baseline/run.py
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ghactivity import (EVENT_COLS, PARAMS, TEST_INPUT, TEST_START, add_period_args,  # noqa: E402
                        best_threshold, build_week, load, scores, split_weeks, to_output, training_weights)

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    add_period_args(ap)
    ap.add_argument("--gap-mode", default="none",
                    choices=["none", "mask", "drop", "downweight", "recent"],
                    help="how training weeks hit by collection outages are handled: none = keep, "
                         "mask = drop a week only for labels whose label window has an outage, "
                         "drop = drop weeks whose label window or feature window has an outage, "
                         "downweight = give those weeks --gap-weight, "
                         "recent = train only on weeks after the summer outage")
    ap.add_argument("--gap-weight", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    params = {**PARAMS, "seed": args.seed}
    out_dir = OUT_DIR / args.gap_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, ev, own, data_end = load()
    n = len(cohort)
    lags = {k: TEST_START - v for k, v in data_end.items()}
    for k in data_end:
        print(f"{k:18s} data until {data_end[k]}  lag {lags[k]}")

    # Week starts aligned to the test week (same weekday and hour), last one with any label.
    weeks, train_weeks, val_weeks, final_weeks = split_weeks(args, data_end)
    print(f"train weeks {train_weeks[0]} .. {train_weeks[-1]} ({len(train_weeks)}), "
          f"validation weeks {[str(w) for w in val_weeks]}, final fit {len(final_weeks)} weeks")

    data = pd.concat([build_week(w, ev, own, lags, data_end, n) for w in weeks], ignore_index=True)
    data = data[data.eligible].reset_index(drop=True)
    feats = [c for c in data.columns if c not in {"eligible", "cidx", "week"} and not c.startswith("y_")]
    print(f"{len(data)} samples, {len(feats)} features")

    # Validation rows are never reweighted, so every gap mode is scored on the same samples.
    week_w = training_weights(args.gap_mode, weeks, lags, args.gap_weight)
    weight = {k: data.week.map(week_w[k]).values for k in EVENT_COLS}
    tr = data.week.isin(train_weeks).values
    va = data.week.isin(val_weeks).values
    val_prob, thresholds, rounds = {}, {}, {}
    metrics = {"gap_mode": args.gap_mode, "gap_weight": args.gap_weight, "seed": args.seed,
               "per_label": {}}
    for k in EVENT_COLS:
        y = data[f"y_{k}"].values
        mtr, mva = tr & ~np.isnan(y) & (weight[k] > 0), va & ~np.isnan(y)
        dtr = lgb.Dataset(data.loc[mtr, feats], y[mtr], weight=weight[k][mtr])
        dva = lgb.Dataset(data.loc[mva, feats], y[mva], reference=dtr)
        model = lgb.train(params, dtr, 2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds[k] = model.best_iteration
        p = model.predict(data[feats], num_iteration=model.best_iteration)
        val_prob[k] = p
        thresholds[k], f1 = best_threshold(y[mva], p[mva])
        metrics["per_label"][k] = {"rounds": rounds[k], "threshold": thresholds[k], "val_f1": f1,
                                   "train_weeks": int(sum(week_w[k][w] > 0 for w in train_weeks)),
                                   "val_pos_rate": float(y[mva].mean()),
                                   "val_pred_rate": float((p[mva] >= thresholds[k]).mean())}
        print(f"{k:18s} train weeks {metrics['per_label'][k]['train_weeks']:2d}  "
              f"rounds {rounds[k]:4d}  val F1 {f1:.4f}  val positives {y[mva].mean():.3f}  "
              f"predicted {(p[mva] >= thresholds[k]).mean():.3f}")

    # Score validation weeks that have all 6 labels, like the official scorer does.
    weekly = []
    for w in val_weeks:
        m = (data.week == w).values
        if data.loc[m, [f"y_{k}" for k in EVENT_COLS]].isna().any().any():
            continue
        gids = cohort.github_user_id.values[data.cidx.values[m]]
        truth = to_output(gids, {k: data.loc[m, f"y_{k}"].values for k in EVENT_COLS})
        model_pred = to_output(gids, {k: val_prob[k][m] >= thresholds[k] for k in EVENT_COLS})
        recent_pred = to_output(gids, {k: data.loc[m, f"{k}_cnt_30d"].values > 0 for k in EVENT_COLS})
        for name, pred in [("lightgbm", model_pred), ("active_last_30d", recent_pred)]:
            r = scores(pred, truth)
            r.update(week=str(w), method=name, users=int(m.sum()), f1=r.pop("per_event"))
            weekly.append(r)
    metrics["validation"] = weekly
    for name in ["lightgbm", "active_last_30d"]:
        rs = [r for r in weekly if r["method"] == name]
        print(f"validation {name:16s} " + "  ".join(
            f"{r['week'][:10]}: {100 * r['score']:.2f}" for r in rs)
            + f"  mean score {100 * np.mean([r['score'] for r in rs]):.2f}")

    # Final models: all labelled weeks, same number of rounds. Retraining on the newer (less
    # active) weeks shifts the probability scale, so the validation threshold is carried over as
    # the share of users predicted positive rather than as a raw probability.
    test = pd.read_csv(TEST_INPUT)
    cidx_of = pd.Series(np.arange(n), index=cohort.github_user_id.values)
    cidx = cidx_of.loc[test.UserID].values
    X_test = build_week(TEST_START, ev, own, lags, data_end, n, with_labels=False).iloc[cidx]
    pred = {}
    for k in EVENT_COLS:
        y = data[f"y_{k}"].values
        m = ~np.isnan(y) & (weight[k] > 0) & data.week.isin(final_weeks).values
        model = lgb.train(params, lgb.Dataset(data.loc[m, feats], y[m], weight=weight[k][m]),
                          max(rounds[k], 50))
        p = model.predict(X_test[feats])
        rate = metrics["per_label"][k]["val_pred_rate"]
        pred[k] = p >= np.quantile(p, 1 - rate)
        metrics["per_label"][k]["test_pos_rate"] = float(pred[k].mean())
    out = to_output(test.UserID.values, pred)
    out.to_csv(out_dir / "output.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print("predicted positive rate on test:", {k: round(float(out[k].mean()), 3) for k in EVENT_COLS})
    print(f"wrote {out_dir / 'output.csv'}")


if __name__ == "__main__":
    main()
