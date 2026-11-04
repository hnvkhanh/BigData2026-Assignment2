"""Baseline: per-user activity-history features + one LightGBM classifier per activity.

One training sample = (cohort user, week start s). Labels are whether the user performs each
activity in [s, s + 7d). Features use only rows that were available before a per-table cutoff
s - lag, where lag is the distance between the official test week start and the end of that
table's data. This reproduces at training time the same staleness the final prediction has
(the data ends ~10 days before the test week, pushes ~16 days).

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
DB = ROOT / "assignment2" / "data" / "db"
EVAL_DIR = ROOT / "assignment2" / "evaluation"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

sys.path.insert(0, str(EVAL_DIR))
from evaluate import EVENT_COLS, OUTPUT_COLUMNS, f1_per_event, scores  # noqa: E402

TEST_START = pd.Timestamp("2026-11-11 16:00")      # 2026-11-12 00:00 Taiwan time
SELECTION_TIME = pd.Timestamp("2026-10-28 12:00")  # test users: >= 5 records in the 90 days before
DATA_START = pd.Timestamp("2025-12-01")
WEEK = pd.Timedelta(days=7)
DAY = pd.Timedelta(days=1)
WINDOWS = [7, 14, 30, 90]
MIN_RECORDS = 5

# Collection outages, found by inspecting daily event counts of the cohort: non-push activity
# drops to ~5-20% of its usual level from June to early September (ending in a backfill burst on
# 9/4-9/6), and pushes are missing for several days in August-October. [start, end) in UTC.
GAPS = {
    "nonpush": [("2026-06-01", "2026-09-07")],
    "push": [("2026-08-18", "2026-09-10"), ("2026-10-07", "2026-10-08"),
             ("2026-10-15", "2026-10-19"), ("2026-10-23", "2026-10-24")],
}
GAP_FEATURE_DAYS = 30     # feature window checked for outages
GAP_FEATURE_SHARE = 0.25  # a feature window counts as corrupted above this share of outage days
RECENT_START = pd.Timestamp("2026-09-09 16:00")  # first week start after the summer outage

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=0)


def load():
    """Read the tables and turn them into per-source event lists of cohort users.

    Each event has uid (cohort index), t (when it happened), avail (when the row is visible in
    the data, so features at cutoff c may only use avail < c) and repo.
    """
    users = pd.read_parquet(DB / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"])
    prs = pd.read_parquet(DB / "PullRequests.parquet",
                          columns=["repo_id", "author_id", "created_at", "closed_at", "first_seen_at"])
    issues = pd.read_parquet(DB / "Issues.parquet",
                             columns=["repo_id", "author_id", "is_pull_request", "created_at",
                                      "closed_at", "first_seen_at"])
    comments = pd.read_parquet(DB / "IssueComments.parquet",
                               columns=["issue_id", "repo_id", "author_id", "created_at"])
    reviews = pd.read_parquet(DB / "PRReviews.parquet",
                              columns=["pr_id", "repo_id", "reviewer_id", "submitted_at", "first_seen_at"])
    rcomments = pd.read_parquet(DB / "PRReviewComments.parquet",
                                columns=["pr_id", "repo_id", "author_id", "created_at"])
    pushes = pd.read_parquet(DB / "Pushes.parquet", columns=["user_id", "repo_id", "created_at"])

    cohort = users[users.is_cohort].reset_index(drop=True)
    to_cidx = np.full(len(users), -1, np.int64)
    to_cidx[cohort.user_id.values] = np.arange(len(cohort))

    def ids(s):
        return s.to_numpy(dtype="float64", na_value=np.nan)

    def events(uid, t, avail, repo):
        d = pd.DataFrame({"uid": ids(uid), "t": t.values, "avail": avail, "repo": ids(repo)})
        d = d[d.uid.notna() & d.t.notna() & (d.t >= DATA_START)]
        d["uid"] = to_cidx[d.uid.astype(np.int64).values]
        return d[d.uid >= 0].sort_values("t").reset_index(drop=True)

    def later(*cols):
        return np.maximum.reduce([c.values for c in cols])

    plain = issues[~issues.is_pull_request]
    ev = {
        "open_pr": events(prs.author_id, prs.created_at, later(prs.created_at, prs.first_seen_at),
                          prs.repo_id),
        "open_issue": events(plain.author_id, plain.created_at,
                             later(plain.created_at, plain.first_seen_at), plain.repo_id),
        "issue_comment": events(comments.author_id, comments.created_at, comments.created_at.values,
                                comments.repo_id),
        "pr_review": events(reviews.reviewer_id, reviews.submitted_at,
                            later(reviews.submitted_at, reviews.first_seen_at), reviews.repo_id),
        "pr_review_comment": events(rcomments.author_id, rcomments.created_at,
                                    rcomments.created_at.values, rcomments.repo_id),
        "push": events(pushes.user_id, pushes.created_at, pushes.created_at.values, pushes.repo_id),
    }

    # Relational: activity other users direct at the user's own issues and PRs
    # (comment -> issue -> author, review / review comment -> PR -> author).
    parent_issue = issues[["author_id", "first_seen_at"]].rename(
        columns={"author_id": "owner", "first_seen_at": "parent_seen"})
    parent_pr = prs[["author_id", "first_seen_at"]].rename(
        columns={"author_id": "owner", "first_seen_at": "parent_seen"})
    recv = [
        comments.rename(columns={"author_id": "actor"}).merge(
            parent_issue, left_on="issue_id", right_index=True, how="left"),
        reviews.rename(columns={"reviewer_id": "actor", "submitted_at": "created_at"}).merge(
            parent_pr, left_on="pr_id", right_index=True, how="left"),
        rcomments.rename(columns={"author_id": "actor"}).merge(
            parent_pr, left_on="pr_id", right_index=True, how="left"),
    ]
    parts = []
    for d in recv:
        d = d[d.owner.notna() & d.actor.notna()]
        d = d[ids(d.owner) != ids(d.actor)]
        seen = [d.created_at, d.parent_seen] + ([d.first_seen_at] if "first_seen_at" in d else [])
        parts.append(events(d.owner, d.created_at, later(*seen), d.repo_id))
    ev["received"] = pd.concat(parts).sort_values("t").reset_index(drop=True)

    # Authored PRs / issues with open/closed state (closed_at is only used when it is before the cutoff).
    own = {}
    for k, d in {"pr": prs, "issue": plain}.items():
        d = d[d.author_id.notna() & d.created_at.notna()]
        own[k] = pd.DataFrame({"uid": to_cidx[ids(d.author_id).astype(np.int64)],
                               "avail": later(d.created_at, d.first_seen_at),
                               "closed_at": d.closed_at.values})
        own[k] = own[k][own[k].uid >= 0].reset_index(drop=True)

    # Last moment each table has data; the lag of a source = TEST_START - that moment.
    data_end = {
        "open_pr": prs.first_seen_at.max(),
        "open_issue": issues.first_seen_at.max(),
        "issue_comment": comments.created_at.max(),
        "pr_review": reviews.first_seen_at.max(),
        "pr_review_comment": rcomments.created_at.max(),
        "push": pushes.created_at.max(),
    }
    data_end["received"] = min(data_end[k] for k in ["issue_comment", "pr_review", "pr_review_comment"])
    return cohort, ev, own, data_end


def source_features(e, cutoff, n):
    e = e[e.avail.values < np.datetime64(cutoff)]
    t, u = e.t.values, e.uid.values
    out = {}
    for w in WINDOWS:
        m = t >= np.datetime64(cutoff - w * DAY)
        out[f"cnt_{w}d"] = np.bincount(u[m], minlength=n)
    out["cnt_all"] = np.bincount(u, minlength=n)
    since = np.full(n, 999.0)
    last = pd.Series(t).groupby(u).max()
    since[last.index.values] = (cutoff - pd.DatetimeIndex(last.values)) / DAY
    out["days_since"] = since
    m = t >= np.datetime64(cutoff - 30 * DAY)
    for name, col in [("active_days_30d", t[m].astype("datetime64[D]")), ("repos_30d", e.repo.values[m])]:
        v = np.zeros(n)
        g = pd.Series(col).groupby(u[m]).nunique()
        v[g.index.values] = g.values
        out[name] = v
    return out


def open_count(d, cutoff, n):
    c = np.datetime64(cutoff)
    m = (d.avail.values < c) & ~(d.closed_at.values < c)  # NaT compares False -> still open
    return np.bincount(d.uid.values[m], minlength=n)


def build_week(s, ev, own, lags, data_end, n, with_labels=True):
    """Features (and labels) for every cohort user for the week starting at s."""
    f = {}
    for name, e in ev.items():
        for k, v in source_features(e, s - lags[name], n).items():
            f[f"{name}_{k}"] = v
    c = s - lags["open_pr"]
    f["open_prs_now"] = open_count(own["pr"], c, n)
    f["open_issues_now"] = open_count(own["issue"], c, n)
    df = pd.DataFrame(f)
    df["any_days_since"] = df[[f"{k}_days_since" for k in EVENT_COLS]].min(axis=1)
    df["n_types_30d"] = (df[[f"{k}_cnt_30d" for k in EVENT_COLS]] > 0).sum(axis=1)

    # Same rule the TAs used to pick test users, shifted to this week.
    sel = s - (TEST_START - SELECTION_TIME)
    rec = np.zeros(n, np.int64)
    for k in EVENT_COLS:
        c = min(sel, s - lags[k])
        e = ev[k]
        m = (e.avail.values < np.datetime64(c)) & (e.t.values >= np.datetime64(sel - 90 * DAY))
        rec += np.bincount(e.uid.values[m], minlength=n)
    df["eligible"] = rec >= MIN_RECORDS
    df["cidx"] = np.arange(n)
    df["week"] = s

    if with_labels:
        for k in EVENT_COLS:
            if s + WEEK > data_end[k]:
                df[f"y_{k}"] = np.nan
                continue
            e = ev[k]
            m = (e.t.values >= np.datetime64(s)) & (e.t.values < np.datetime64(s + WEEK))
            df[f"y_{k}"] = (np.bincount(e.uid.values[m], minlength=n) > 0).astype(float)
    return df


def group_of(k):
    return "push" if k == "push" else "nonpush"


def gap_share(group, start, end):
    """Share of [start, end) that falls inside the collection outages of a source group."""
    covered = pd.Timedelta(0)
    for a, b in GAPS[group]:
        lo, hi = max(start, pd.Timestamp(a)), min(end, pd.Timestamp(b))
        if hi > lo:
            covered += hi - lo
    return covered / (end - start)


def training_weights(mode, weeks, lags, gap_weight):
    """Per label, a weight for each training week (0 drops it) under the chosen outage handling."""
    w = {k: {} for k in EVENT_COLS}
    for s in weeks:
        label_gap = {g: gap_share(g, s, s + WEEK) > 0 for g in GAPS}
        feat_gap = {}
        for g, src in [("nonpush", "open_pr"), ("push", "push")]:
            c = s - lags[src]
            feat_gap[g] = gap_share(g, c - GAP_FEATURE_DAYS * DAY, c) > GAP_FEATURE_SHARE
        corrupted = any(label_gap.values()) or any(feat_gap.values())
        for k in EVENT_COLS:
            if mode == "none":
                w[k][s] = 1.0
            elif mode == "mask":
                w[k][s] = 0.0 if label_gap[group_of(k)] else 1.0
            elif mode == "drop":
                w[k][s] = 0.0 if corrupted else 1.0
            elif mode == "downweight":
                w[k][s] = gap_weight if corrupted else 1.0
            elif mode == "recent":
                w[k][s] = 1.0 if s >= RECENT_START else 0.0
    return w


def best_threshold(y, p):
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    i = int(np.argmax(f1))
    return float(ps[i]), float(f1[i])


def to_output(gids, pred):
    out = pd.DataFrame({"UserID": gids})
    for k in EVENT_COLS:
        out[k] = pred[k].astype(int)
    return out[OUTPUT_COLUMNS]


def add_period_args(ap):
    """Options choosing the data period, for all stages or for one stage (shared by all runners).

    A week (label window [s, s + 7d)) belongs to a period when it lies fully inside
    [start, end]. Dates are UTC; weeks start Wednesday 16:00. A stage option overrides the
    global one for that stage only.
    """
    g = ap.add_argument_group("data period (UTC, e.g. 2026-03-01 or '2026-03-01 16:00')")
    g.add_argument("--start", default="2026-01-14 16:00", help="first moment of data, all stages")
    g.add_argument("--end", default=None, help="last moment of data, all stages (default: all labelled weeks)")
    g.add_argument("--train-start", default=None, help="override --start for training weeks")
    g.add_argument("--train-end", default=None, help="override --end for training weeks")
    g.add_argument("--val-start", default=None,
                   help="validation period start; with --val-end it replaces --val-weeks")
    g.add_argument("--val-end", default=None, help="validation period end")
    g.add_argument("--test-start", default=None,
                   help="override --start for the weeks the final (test) models are fitted on")
    g.add_argument("--test-end", default=None, help="override --end for the final (test) fit weeks")
    g.add_argument("--val-weeks", type=int, default=3,
                   help="without --val-start/--val-end: number of most recent weeks used for "
                        "validation / thresholds")


def split_weeks(args, data_end):
    """Return (weeks, train_weeks, val_weeks, final_weeks) for the chosen data periods.

    weeks = every labelled week inside the global period (the ones features are built for).
    Training labels must end before validation starts, so train weeks overlapping validation
    are dropped.
    """
    last = max(data_end[k] for k in EVENT_COLS) - WEEK
    ts = lambda x, d: pd.Timestamp(x) if x else d  # noqa: E731
    start, end = pd.Timestamp(args.start), ts(args.end, pd.Timestamp.max)
    weeks, s = [], TEST_START - WEEK
    while s >= start:
        if s <= last and s + WEEK <= end:
            weeks.append(s)
        s -= WEEK
    weeks = sorted(weeks)

    def inside(a, b):
        a, b = ts(a, start), ts(b, end)
        return [w for w in weeks if w >= a and w + WEEK <= b]

    if args.val_start or args.val_end:
        val_weeks = inside(args.val_start, args.val_end)
    else:
        val_weeks = weeks[-args.val_weeks:]
    if not val_weeks:
        sys.exit("no validation weeks in the selected period")
    train_weeks = [w for w in inside(args.train_start, args.train_end) if w + WEEK <= val_weeks[0]]
    final_weeks = inside(args.test_start, args.test_end)
    if not train_weeks or not final_weeks:
        sys.exit("no training / final-fit weeks in the selected period")
    return weeks, train_weeks, val_weeks, final_weeks


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
            r.update(week=str(w), method=name, users=int(m.sum()),
                     f1=dict(zip(EVENT_COLS, f1_per_event(pred, truth))))
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
    test = pd.read_csv(EVAL_DIR / "test_input.csv")
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
