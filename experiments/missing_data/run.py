"""Ways of handling GH Archive's missing events, compared on the latest weeks.

GH Archive drops events: non-push activity of the cohort is ~100% of its usual level Dec-May,
3-25% from June to early September, 100-250% in a burst on Sep 4-17, then 10-30% in October and
~12% in November; pushes are ~0-4% since 10-27. The answer key of the test week is built from the
same archive, so no approach fills in events. They change which weeks the model learns from, or
how the features are scaled, so that training looks like the current, thinned data.

Protocol, shared by every approach so that they can be compared:
  * the features, labels, per-table lags and eligibility rule of experiments/baseline/run.py;
  * training weeks: labels ending by the tuning week (week starts up to 10-10);
  * tuning week 10-17: early stopping and the decision rule (share of users predicted positive
    that maximises F1, carried to other weeks as in the baseline's final step);
  * holdout weeks 10-24 and 10-31, unseen by everything above: Score = (macro-F1 + micro-F1) / 2
    of the official scorer over the 5 non-push activities. Push data ends 10-27, so no recent
    week has push labels; push is looked at separately (push_scenario).

Approaches (all run by default):
  baseline             every training week, as in the baseline
  rule_from_train      baseline models, threshold maximising F1 on the training weeks instead
  mask_burst           non-push labels of weeks overlapping the Sep 4-17 burst get weight 0
  gaps_mask / gaps_drop / gaps_downweight
                       the baseline's --gap-mode options with GAPS re-measured on this data
  recent               only weeks from 09-19 on
  relative             counts divided by the cohort mean of the same week, plus cohort volumes
  relative_no_volume   relative without the cohort volumes
  relative_mask_burst  relative and mask_burst together
  rank                 counts replaced by their percentile rank within the week
  thin                 weeks before June rebuilt from events kept at random with probability p
                       (p = events per eligible user-week, recent weeks / spring weeks)
  thin_x3              three such random draws, weight 1/3 each
  thin_x3_mask_burst   thin_x3 and mask_burst together

Run from the repo root (~13 min with 8 LightGBM threads):
    python experiments/missing_data/run.py
    python experiments/missing_data/run.py --approaches baseline thin --seeds 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ghactivity as B  # noqa: E402

EVENT_COLS, WEEK = B.EVENT_COLS, B.WEEK
NONPUSH = [k for k in EVENT_COLS if k != "push"]
START = pd.Timestamp("2026-01-14 16:00")   # the baseline's --start
TUNE = pd.Timestamp("2026-10-17 16:00")
HOLDOUT = [pd.Timestamp("2026-10-24 16:00"), pd.Timestamp("2026-10-31 16:00")]
PUSH_WEEK = pd.Timestamp("2026-11-07 16:00")  # push scenario; its labels end with the data (11-14 ~14:00)
THIN_END = pd.Timestamp("2026-06-01")          # weeks whose labels end by then are thinned
BURST = (pd.Timestamp("2026-09-04"), pd.Timestamp("2026-09-18"))
RECENT_START = pd.Timestamp("2026-09-19 16:00")
SPRING = (pd.Timestamp("2026-01-17 16:00"), pd.Timestamp("2026-05-02 16:00"))  # before the May sampling
REF = (RECENT_START, TUNE + WEEK)  # recent weeks p is measured on, all before the holdout
# Re-measured on the GH Archive data (the baseline's GAPS describe the old course data).
NEW_GAPS = {"nonpush": [("2026-06-01", "2026-09-18")],
            "push": [("2026-08-18", "2026-09-17"), ("2026-10-23", "2026-10-24")]}
COUNT_KEYS = B.COUNT_KEYS

APPROACHES = {  # name: (thinned draws, feature view, week weighting)
    "baseline": (0, "base", None),
    "rule_from_train": (0, "base", None),
    "mask_burst": (0, "base", "mask_burst"),
    "gaps_mask": (0, "base", "gaps:mask"),
    "gaps_drop": (0, "base", "gaps:drop"),
    "gaps_downweight": (0, "base", "gaps:downweight"),
    "recent": (0, "base", "recent"),
    "relative": (0, "relative", None),
    "relative_no_volume": (0, "relative_no_volume", None),
    "relative_mask_burst": (0, "relative", "mask_burst"),
    "rank": (0, "rank", None),
    "thin": (1, "base", None),
    "thin_x3": (3, "base", None),
    "thin_x3_mask_burst": (3, "base", "mask_burst"),
}


def week_starts(first, last):
    out, s = [], B.TEST_START - WEEK
    while s >= first:
        if s <= last:
            out.append(s)
        s -= WEEK
    return sorted(out)


def add_views(df, base_feats):
    """Capture-independent versions of the count features, computed over all cohort users of the week."""
    rel, vol = B.relative_views(df, base_feats)  # shared with experiments/gnn/run.py --relative-features
    x = df[[c for c in base_feats if any(k in c for k in COUNT_KEYS)]].astype(float)
    rank = x.rank(pct=True).where(x > 0, 0).add_prefix("rank_")
    return pd.concat([df, rel, rank, vol], axis=1)


def build(weeks, ev, own, lags, data_end, n, draw):
    frames = []
    for s in weeks:
        df = B.build_week(s, ev, own, lags, data_end, n)
        base_feats = [c for c in df.columns
                      if c not in {"eligible", "cidx", "week"} and not c.startswith("y_")]
        df = add_views(df, base_feats)
        df = df[df.eligible].copy()
        df["draw"] = draw
        frames.append(df)
    return pd.concat(frames, ignore_index=True), base_feats


def events_per_user_week(e, data, wks, n):
    tot = users = 0
    for s in wks:
        c = data.cidx.values[(data.week == s).values & (data.draw == -1).values]
        m = (e.t.values >= np.datetime64(s)) & (e.t.values < np.datetime64(s + WEEK))
        tot += np.bincount(e.uid.values[m], minlength=n)[c].sum()
        users += len(c)
    return tot / max(users, 1)


def thin_tables(tables, p, rng):
    """Drop each non-push event before THIN_END with probability 1 - p (per activity).

    A PR or issue whose opening is dropped keeps its row (it may still be the target of reviews or
    comments) but loses its author and created_at, as happens when GH Archive misses the event."""
    t = dict(tables)

    def dropped(times, prob):
        return (times.notna() & (times < THIN_END)).to_numpy() & (rng.random(len(times)) >= prob)

    prs = tables["prs"].copy()
    d = dropped(prs.created_at, p["open_pr"]) & prs.author_id.notna().to_numpy()
    prs.loc[d, "author_id"], prs.loc[d, "created_at"] = pd.NA, pd.NaT
    t["prs"] = prs
    iss = tables["issues"].copy()
    d = (dropped(iss.created_at, p["open_issue"]) & iss.author_id.notna().to_numpy()
         & ~iss.is_pull_request.fillna(False).to_numpy(bool))
    iss.loc[d, "author_id"], iss.loc[d, "created_at"] = pd.NA, pd.NaT
    t["issues"] = iss
    for key, col, k in [("comments", "created_at", "issue_comment"),
                        ("reviews", "submitted_at", "pr_review"),
                        ("rcomments", "created_at", "pr_review_comment")]:
        t[key] = tables[key][~dropped(tables[key][col], p[k])]
    return t


def overlaps(s, a, b):
    return s < b and s + WEEK > a


def week_weights(rule, data, weeks, lags):
    w = {k: np.ones(len(data)) for k in EVENT_COLS}
    wk = data.week
    if rule == "mask_burst":
        hit = wk.map(lambda s: overlaps(s, *BURST)).values
        for k in NONPUSH:
            w[k][hit] = 0
    elif rule == "recent":
        for k in EVENT_COLS:
            w[k][(wk < RECENT_START).values] = 0
    elif rule and rule.startswith("gaps:"):
        ww = B.training_weights(rule.split(":")[1], weeks, lags, 0.2, gaps=NEW_GAPS)
        for k in EVENT_COLS:
            w[k] = wk.map(ww[k]).values.astype(float)
    return w


def top_share(p, share):
    return p >= np.quantile(p, 1 - share) if share > 0 else np.zeros(len(p), bool)


def score(gids, truth, pred, events):
    zero = np.zeros(len(gids))
    t = B.to_output(gids, {k: truth.get(k, zero) for k in EVENT_COLS})
    q = B.to_output(gids, {k: pred.get(k, zero) for k in EVENT_COLS})
    r = B.scores(q, t, events=events)
    return {"score": r["score"], "macro_f1": r["macro_f1"], "micro_f1": r["micro_f1"], "f1": r["per_event"]}


def run(name, data, feats, weights, train_weeks, seed, gids_of, threads, extra=None):
    params = {**B.PARAMS, "seed": seed, "num_threads": threads}
    base = data.draw.values == -1
    tr = data.week.isin(train_weeks).values
    tune = base & (data.week == TUNE).values
    hold = {w: base & (data.week == w).values for w in HOLDOUT}
    labels = NONPUSH + (["push"] if extra is not None else [])
    pred_share = {w: {} for w in HOLDOUT}
    pred_thr = {w: {} for w in HOLDOUT}
    info, models = {}, {}
    for k in labels:
        y = data[f"y_{k}"].values
        mtr = tr & ~np.isnan(y) & (weights[k] > 0)
        dtr = lgb.Dataset(data.loc[mtr, feats], y[mtr], weight=weights[k][mtr])
        dva = lgb.Dataset(data.loc[tune, feats], y[tune], reference=dtr)
        model = lgb.train(params, dtr, 2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        models[k] = model
        p_tune = model.predict(data.loc[tune, feats], num_iteration=model.best_iteration)
        thr, f1 = B.best_threshold(y[tune], p_tune)
        if name == "rule_from_train":
            p_tr = model.predict(data.loc[mtr, feats], num_iteration=model.best_iteration)
            thr = B.best_threshold(y[mtr], p_tr)[0]
        share = float((p_tune >= thr).mean())
        info[k] = {"rounds": model.best_iteration, "train_rows": int(mtr.sum()), "tune_f1": f1,
                   "threshold": thr, "share": share, "tune_pos_rate": float(y[tune].mean())}
        if k == "push":
            continue
        for w, m in hold.items():
            p = model.predict(data.loc[m, feats], num_iteration=model.best_iteration)
            pred_thr[w][k] = p >= thr
            pred_share[w][k] = p >= thr if name == "rule_from_train" else top_share(p, share)
    out = {"labels": info, "holdout": {}, "holdout_threshold": {}, "_pred": pred_share, "_truth": {}}
    for w, m in hold.items():
        gids = gids_of(data.cidx.values[m])
        truth = {k: data.loc[m, f"y_{k}"].values for k in NONPUSH}
        out["_truth"][w] = truth
        out["holdout"][str(w)] = score(gids, truth, pred_share[w], NONPUSH)
        out["holdout_threshold"][str(w)] = score(gids, truth, pred_thr[w], NONPUSH)
    out["score"] = float(np.mean([r["score"] for r in out["holdout"].values()]))
    out["score_threshold"] = float(np.mean([r["score"] for r in out["holdout_threshold"].values()]))
    if extra is not None:
        out["push_scenario"] = push_scenario(models, info, feats, extra, gids_of)
    return out


def paired_bootstrap(preds, truth, ref, n_boot=2000, seed=0):
    """Holdout score difference of each approach to `ref`, on users resampled with replacement
    within each holdout week (the same draws for every approach), averaged over seeds.

    preds: approach -> list (one per seed) of {week: {activity: bool array}}; truth: {week: {activity: array}}.
    Returns approach -> mean difference, 95% interval and share of draws where it is above 0."""
    rng = np.random.default_rng(seed)
    counts = {w: rng.multinomial(len(t[NONPUSH[0]]), np.full(len(t[NONPUSH[0]]), 1 / len(t[NONPUSH[0]])),
                                 size=n_boot) for w, t in truth.items()}

    def draws(pred):  # (n_boot,) score of one seed's predictions, mean over the holdout weeks
        per_week = []
        for w, c in counts.items():
            y = {k: truth[w][k] == 1 for k in NONPUSH}
            tp = np.stack([c @ (pred[w][k] & y[k]) for k in NONPUSH], 1).astype(float)
            fp = np.stack([c @ (pred[w][k] & ~y[k]) for k in NONPUSH], 1).astype(float)
            fn = np.stack([c @ (~pred[w][k] & y[k]) for k in NONPUSH], 1).astype(float)
            den = 2 * tp + fp + fn
            f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 1.0)
            T, D = tp.sum(1), den.sum(1)
            micro = np.where(D > 0, 2 * T / np.maximum(D, 1), 1.0)
            per_week.append((f1.mean(1) + micro) / 2)
        return np.mean(per_week, 0)

    base = np.mean([draws(p) for p in preds[ref]], 0)
    out = {}
    for name, runs in preds.items():
        d = 100 * (np.mean([draws(p) for p in runs], 0) - base)
        out[name] = {"diff": float(d.mean()), "lo": float(np.quantile(d, 0.025)),
                     "hi": float(np.quantile(d, 0.975)), "p_above_0": float((d > 0).mean())}
    return out


def push_scenario(models, info, feats, extra, gids_of):
    """Week PUSH_WEEK (push ~1% of its usual volume), scored on all 6 activities as if push counted.

    Non-push: shares from the tuning week. Push options: the tuning-week share (what the pipeline
    does), nobody, the push rate observed in the previous week (a rule usable at test time), and
    the best share in hindsight."""
    x, rate_prev = extra
    gids = gids_of(x.cidx.values)
    truth = {k: x[f"y_{k}"].values for k in EVENT_COLS}
    pred = {k: top_share(models[k].predict(x[feats], num_iteration=models[k].best_iteration),
                         info[k]["share"]) for k in NONPUSH}
    p_push = models["push"].predict(x[feats], num_iteration=models["push"].best_iteration)
    best_share = max(np.linspace(0, 0.2, 81),
                     key=lambda s: score(gids, truth, {"push": top_share(p_push, s)}, ["push"])["score"])
    out = {"push_pos_rate": float(truth["push"].mean()), "prev_week_push_rate": rate_prev,
           "nonpush_only": score(gids, truth, pred, NONPUSH)}
    for opt, share in [("tune_share", info["push"]["share"]), ("none", 0.0),
                       ("prev_week_rate", rate_prev), ("best_in_hindsight", float(best_share))]:
        r = score(gids, truth, {**pred, "push": top_share(p_push, share)}, EVENT_COLS)
        out[opt] = {"share": share, "score": r["score"], "push_f1": r["f1"]["push"],
                    "micro_f1": r["micro_f1"], "macro_f1": r["macro_f1"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--approaches", nargs="+", default=list(APPROACHES), choices=list(APPROACHES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--threads", type=int, default=8,
                    help="LightGBM threads (all cores can be ~100x slower, e.g. under WSL)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tables = B.read_tables()
    cohort, ev, own, data_end = B.load(tables)
    n = len(cohort)
    lags = {k: B.TEST_START - v for k, v in data_end.items()}
    gids_of = lambda c: cohort.github_user_id.values[c]  # noqa: E731
    weeks = week_starts(START, HOLDOUT[-1])
    train_weeks = [w for w in weeks if w + WEEK <= TUNE]
    thin_weeks = [w for w in weeks if w + WEEK <= THIN_END]
    print(f"{len(train_weeks)} training weeks {train_weeks[0]} .. {train_weeks[-1]}, tune {TUNE}, "
          f"holdout {[str(w) for w in HOLDOUT]}; {len(thin_weeks)} weeks thinned by the thin approaches")

    base, base_feats = build(weeks, ev, own, lags, data_end, n, draw=-1)
    count_feats = [c for c in base_feats if any(k in c for k in COUNT_KEYS)]
    other = [c for c in base_feats if c not in count_feats]
    feat_sets = {"base": base_feats,
                 "relative": other + [f"rel_{c}" for c in count_feats]
                 + [c for c in base.columns if c.startswith("vol_")],
                 "relative_no_volume": other + [f"rel_{c}" for c in count_feats],
                 "rank": other + [f"rank_{c}" for c in count_feats]}
    print(f"built {len(base):,} examples in {time.time() - t0:.0f}s")

    # Capture probability p per non-push activity: events per eligible user-week, recent / spring.
    spring_wks = [w for w in weeks if w >= SPRING[0] and w + WEEK <= SPRING[1]]
    ref_wks = [w for w in weeks if w >= REF[0] and w + WEEK <= REF[1]]
    p = {k: min(1.0, events_per_user_week(ev[k], base, ref_wks, n)
                / events_per_user_week(ev[k], base, spring_wks, n)) for k in NONPUSH}
    print("capture p:", {k: round(v, 3) for k, v in p.items()})

    # Push scenario week (labels to the end of the data) and last week's observed push rate.
    big_end = {k: pd.Timestamp.max for k in data_end}
    x = B.build_week(PUSH_WEEK, ev, own, lags, big_end, n)
    x = x[x.eligible].reset_index(drop=True)
    prev = HOLDOUT[-1]
    c = base.cidx.values[(base.week == prev).values]
    e = ev["push"]
    m = (e.t.values >= np.datetime64(prev)) & (e.t.values < np.datetime64(prev + WEEK))
    rate_prev = float((np.bincount(e.uid.values[m], minlength=n)[c] > 0).mean())

    thinned = {}  # (seed, draw) -> examples of the thinned weeks

    def thinned_draw(seed, r):
        if (seed, r) not in thinned:
            t1 = time.time()
            rng = np.random.default_rng([seed, r])
            _, ev_r, own_r, _ = B.load(thin_tables(tables, p, rng))
            thinned[(seed, r)] = build(thin_weeks, ev_r, own_r, lags, data_end, n, draw=r)[0]
            print(f"  thinned draw seed {seed} #{r}: {len(thinned[(seed, r)]):,} examples "
                  f"({time.time() - t1:.0f}s)")
        return thinned[(seed, r)]

    # Check: positive rates of the thinned spring weeks should be close to those of the recent weeks.
    if any(APPROACHES[a][0] for a in args.approaches):
        th = thinned_draw(args.seeds[0], 0)
        rate = lambda d, wks, k: float(d.loc[d.week.isin(wks), f"y_{k}"].mean())  # noqa: E731
        check = {k: {"spring": rate(base, spring_wks, k), "spring_thinned": rate(th, spring_wks, k),
                     "recent": rate(base, ref_wks, k)} for k in NONPUSH}
        print("positive rate (spring / thinned spring / recent):",
              {k: " / ".join(f"{100 * v:.1f}" for v in c.values()) for k, c in check.items()})

    results = {"p": p, "approaches": {}}
    preds, truth = {}, None
    for name in args.approaches:
        draws, view, rule = APPROACHES[name]
        runs = []
        for seed in args.seeds:
            t1 = time.time()
            if draws:
                keep = base[~base.week.isin(thin_weeks)]
                th = [thinned_draw(seed, r) for r in range(draws)]
                data = pd.concat([keep, *th], ignore_index=True)
            else:
                data = base
            weights = week_weights(rule, data, weeks, lags)
            if draws > 1:
                for k in EVENT_COLS:
                    weights[k] = weights[k] * np.where(data.draw.values >= 0, 1 / draws, 1)
            r = run(name, data, feat_sets[view], weights, train_weeks, seed, gids_of, args.threads,
                    extra=(x, rate_prev) if name == "baseline" else None)
            r["seed"] = seed
            preds.setdefault(name, []).append(r.pop("_pred"))
            truth = r.pop("_truth")
            runs.append(r)
            print(f"{name:20s} seed {seed}: holdout score {100 * r['score']:.2f} "
                  f"(threshold transfer {100 * r['score_threshold']:.2f})  {time.time() - t1:.0f}s")
        results["approaches"][name] = runs
        (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))

    ref = "baseline" if "baseline" in preds else next(iter(preds))
    boot = results["bootstrap_vs_" + ref] = paired_bootstrap(preds, truth, ref)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))

    rows = []
    for name, runs in results["approaches"].items():
        s = np.array([r["score"] for r in runs]) * 100
        row = {"approach": name, "score": s.mean(), "min": s.min(), "max": s.max(),
               "score_threshold": 100 * np.mean([r["score_threshold"] for r in runs]),
               f"diff_vs_{ref}": boot[name]["diff"], "diff_lo": boot[name]["lo"],
               "diff_hi": boot[name]["hi"], "p_above_0": boot[name]["p_above_0"]}
        for w in HOLDOUT:
            row[str(w)[:10]] = 100 * np.mean([r["holdout"][str(w)]["score"] for r in runs])
        for k in NONPUSH:
            row[k] = np.mean([np.mean([h["f1"][k] for h in r["holdout"].values()]) for r in runs])
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("approach").sort_values("score", ascending=False)
    summary.to_csv(args.out / "summary.csv")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    if "baseline" in results["approaches"]:
        print("push scenario:", json.dumps([r["push_scenario"] for r in results["approaches"]["baseline"]],
                                           indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s, wrote {args.out}")


if __name__ == "__main__":
    main()
