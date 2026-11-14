"""Cross-table relation features, screened with the baseline's LightGBM.

Each group turns one relationship between tables (or a column the baseline ignores) into
per-user features, with the same no-leakage rule as the baseline: a source table only
contributes rows visible before s - lag of that table. Groups:

  review_habit  PRReviews <-> PRReviewComments: how often the user's reviews come with line
                comments, review states, reply / unlinked comment shares (experiments/gnn_review)
  push_branch   Pushes.branch: pushes to non-default branches (feature work) and to the head
                branch of the user's own open PRs
  own_pr        PullRequests state of the user's own PRs: open (same repo / fork), merged or
                closed unmerged recently
  maintainer    PullRequests.merged_by_id: merging other people's PRs
  assigned      Issues.assignee_id: open issues / PRs assigned to the user
  demand        new PRs / issues by others in the repos where the user reviews, comments or
                pushes (work arriving at the user)
  comment_mix   IssueComments -> Issues: comments on own vs others' PRs / issues, via an app
  org           Repositories.org_id: share of the user's activity in organisation repos

For every group, one LightGBM per activity is trained on the baseline features plus that group
(and on baseline + all groups), on the baseline's weeks and split; the validation F1 per
activity and the score are compared with the baseline features alone.

Run from the repo root:
    python experiments/relations/run.py
    python experiments/relations/run.py --groups own_pr demand --seeds 0 1 2
"""
import argparse
import importlib.util
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load("baseline_run", HERE.parent / "baseline" / "run.py")
gnn_review = _load("gnn_review_run", HERE.parent / "gnn_review" / "run.py")

EVENT_COLS, DAY, DB = base.EVENT_COLS, base.DAY, base.DB
NONPUSH = [k for k in EVENT_COLS if k != "push"]
DEFAULT_BRANCHES = ["aaaa", "aaaaaa"]  # main / master once masked; other 4- and 6-letter names look the same
GROUPS = ["review_habit", "push_branch", "own_pr", "maintainer", "assigned", "demand", "comment_mix", "org"]
OUT_DIR = HERE / "outputs"


def load_extra(cohort):
    """Columns of the tables the baseline does not read, with ids mapped to cohort indices (-1 = not cohort)."""
    users = pd.read_parquet(DB / "Users.parquet", columns=["user_id"])
    to_c = np.full(len(users) + 1, -1, np.int64)
    to_c[cohort.user_id.values] = np.arange(len(cohort))

    def cidx(s):  # FK column -> cohort index, -1 for null or non-cohort
        v = s.to_numpy(dtype="float64", na_value=np.nan)
        return np.where(np.isnan(v), -1, to_c[np.nan_to_num(v, nan=len(users)).astype(np.int64)])

    prs = pd.read_parquet(DB / "PullRequests.parquet",
                          columns=["repo_id", "author_id", "merged_by_id", "merged_at", "closed_at",
                                   "created_at", "first_seen_at", "is_cross_repo", "head_ref"])
    iss = pd.read_parquet(DB / "Issues.parquet",
                          columns=["repo_id", "author_id", "is_pull_request", "assignee_id",
                                   "created_at", "first_seen_at", "closed_at"])
    ic = pd.read_parquet(DB / "IssueComments.parquet",
                         columns=["issue_id", "repo_id", "author_id", "is_pull_request", "via_app", "created_at"])
    pu = pd.read_parquet(DB / "Pushes.parquet", columns=["user_id", "repo_id", "branch", "created_at"])
    repos = pd.read_parquet(DB / "Repositories.parquet", columns=["repo_id", "org_id"])
    is_org = np.zeros(repos.repo_id.max() + 1, bool)
    is_org[repos.repo_id.values] = repos.org_id.notna().values

    X = {"is_org": is_org}
    X["pr"] = pd.DataFrame({
        "author": cidx(prs.author_id), "merger": cidx(prs.merged_by_id), "repo": prs.repo_id.values,
        "avail": np.maximum(prs.created_at.values, prs.first_seen_at.values),
        "merged_at": prs.merged_at.values, "closed_at": prs.closed_at.values,
        "fork": prs.is_cross_repo.fillna(False).values.astype(bool), "head_ref": prs.head_ref.values})
    X["issue"] = pd.DataFrame({
        "author": cidx(iss.author_id), "assignee": cidx(iss.assignee_id), "repo": iss.repo_id.values,
        "is_pr": iss.is_pull_request.values.astype(bool),
        "avail": np.maximum(iss.created_at.values, iss.first_seen_at.values), "closed_at": iss.closed_at.values})
    parent_author = iss.author_id.to_numpy(dtype="float64", na_value=np.nan)[ic.issue_id.values]
    X["comment"] = pd.DataFrame({
        "uid": cidx(ic.author_id), "t": ic.created_at.values, "repo": ic.repo_id.values,
        "on_pr": ic.is_pull_request.values.astype(bool), "via_app": ic.via_app.notna().values,
        "own": parent_author == ic.author_id.to_numpy(dtype="float64", na_value=np.nan)})
    X["comment"] = X["comment"][X["comment"].uid >= 0].reset_index(drop=True)
    pu = pd.DataFrame({"uid": cidx(pu.user_id), "t": pu.created_at.values, "repo": pu.repo_id.values,
                       "branch": pu.branch.values})
    X["push"] = pu[pu.uid >= 0].reset_index(drop=True)
    X["R"], X["C"] = gnn_review.load_review_detail(cohort)
    return X


def _cut(s, lags, *tables):
    return np.datetime64(min(s - lags[k] for k in tables))


def _ago(c, days):
    return c - np.timedelta64(days, "D")


def _count(uid, n, weights=None):
    return np.bincount(uid, weights=weights, minlength=n).astype(float)


def _pairs_count(users_repos, items, n):
    """Per user: number of items (repo, author) in the user's repos, excluding the user's own."""
    j = users_repos.merge(items, on="repo")
    j = j[j.u != j.a]
    out = np.zeros(n)
    g = j.groupby("u").size()
    out[g.index.values] = g.values
    return out


def group_features(group, s, X, ev, lags, n):
    f = {}
    if group == "review_habit":
        return gnn_review.habit_features(s, X["R"], X["C"], lags, n)

    if group == "push_branch":
        c = _cut(s, lags, "push")
        p = X["push"]
        vis = (p.t.values < c) & (p.t.values >= _ago(c, 30))
        nd = vis & ~np.isin(p.branch.values, DEFAULT_BRANCHES)
        rec = nd & (p.t.values >= _ago(c, 7))
        f["pb_nondefault_7d"] = _count(p.uid.values[rec], n)
        f["pb_nondefault_30d"] = _count(p.uid.values[nd], n)
        f["pb_nondefault_share_30d"] = f["pb_nondefault_30d"] / np.maximum(_count(p.uid.values[vis], n), 1)
        f["pb_branches_30d"] = np.zeros(n)
        g = pd.Series(p.branch.values[nd]).groupby(p.uid.values[nd]).nunique()
        f["pb_branches_30d"][g.index.values] = g.values
        # pushes to the head branch of one of the user's own PRs still open at the cutoff
        cp = _cut(s, lags, "push", "open_pr")
        pr = X["pr"]
        op = pr[(pr.author.values >= 0) & (pr.avail.values < cp) & ~(pr.closed_at.values < cp)]
        mp = (p.t.values < cp) & (p.t.values >= _ago(cp, 14))
        j = pd.DataFrame({"u": p.uid.values[mp], "repo": p.repo.values[mp], "b": p.branch.values[mp]}).merge(
            pd.DataFrame({"u": op.author.values, "b": op.head_ref.values}).dropna().drop_duplicates(), on=["u", "b"])
        f["pb_push_to_open_pr_14d"] = _count(j.u.values.astype(int), n)
        return pd.DataFrame(f)

    if group == "own_pr":
        c = _cut(s, lags, "open_pr")
        pr = X["pr"]
        mine = (pr.author.values >= 0) & (pr.avail.values < c)
        is_open = mine & ~(pr.closed_at.values < c)
        recent = pr.avail.values >= _ago(c, 30)
        for name, m in [("op_open_30d", is_open & recent), ("op_open_fork_30d", is_open & recent & pr.fork.values),
                        ("op_open_all", is_open)]:
            f[name] = _count(pr.author.values[m], n)
        merged = mine & (pr.merged_at.values < c) & (pr.merged_at.values >= _ago(c, 30))
        closed = mine & (pr.closed_at.values < c) & (pr.closed_at.values >= _ago(c, 30)) & ~(pr.merged_at.values < c)
        f["op_merged_30d"] = _count(pr.author.values[merged], n)
        f["op_closed_unmerged_30d"] = _count(pr.author.values[closed], n)
        done = f["op_merged_30d"] + f["op_closed_unmerged_30d"]
        f["op_merge_rate_30d"] = f["op_merged_30d"] / np.maximum(done, 1)
        f["op_fork_share_90d"] = (_count(pr.author.values[mine & pr.fork.values & (pr.avail.values >= _ago(c, 90))], n)
                                  / np.maximum(_count(pr.author.values[mine & (pr.avail.values >= _ago(c, 90))], n), 1))
        return pd.DataFrame(f)

    if group == "maintainer":
        c = _cut(s, lags, "open_pr")
        pr = X["pr"]
        m = (pr.merger.values >= 0) & (pr.merger.values != pr.author.values) & (pr.merged_at.values < c)
        for d in [30, 90]:
            md = m & (pr.merged_at.values >= _ago(c, d))
            f[f"mt_merged_others_{d}d"] = _count(pr.merger.values[md], n)
        md = m & (pr.merged_at.values >= _ago(c, 90))
        f["mt_merge_repos_90d"] = np.zeros(n)
        g = pd.Series(pr.repo.values[md]).groupby(pr.merger.values[md]).nunique()
        f["mt_merge_repos_90d"][g.index.values] = g.values
        return pd.DataFrame(f)

    if group == "assigned":
        c = _cut(s, lags, "open_issue")
        it = X["issue"]
        m = (it.assignee.values >= 0) & (it.avail.values < c) & ~(it.closed_at.values < c)
        for name, mm in [("as_open_issue_60d", m & ~it.is_pr.values & (it.avail.values >= _ago(c, 60))),
                         ("as_open_pr_60d", m & it.is_pr.values & (it.avail.values >= _ago(c, 60))),
                         ("as_open_all", m)]:
            f[name] = _count(it.assignee.values[mm], n)
        f["as_assigned_by_other_90d"] = _count(it.assignee.values[(it.assignee.values >= 0) & (it.assignee.values != it.author.values)
                                                                 & (it.avail.values < c) & (it.avail.values >= _ago(c, 90))], n)
        return pd.DataFrame(f)

    if group == "demand":
        pr, it = X["pr"], X["issue"]
        for src, k in [("rev", "pr_review"), ("cmt", "issue_comment"), ("push", "push")]:
            c = _cut(s, lags, k)
            e = ev[k]
            m = (e.avail.values < c) & (e.t.values >= _ago(c, 90)) & ~np.isnan(e.repo.values)
            ur = pd.DataFrame({"u": e.uid.values[m], "repo": e.repo.values[m].astype(np.int64)}).drop_duplicates()
            cp = _cut(s, lags, "open_pr")
            for d in [7, 30]:
                mp = (pr.avail.values < cp) & (pr.avail.values >= _ago(cp, d))
                f[f"dm_newpr_in_{src}_repos_{d}d"] = _pairs_count(
                    ur, pd.DataFrame({"repo": pr.repo.values[mp], "a": pr.author.values[mp]}), n)
            ci = _cut(s, lags, "open_issue")
            mi = (it.avail.values < ci) & (it.avail.values >= _ago(ci, 7)) & ~it.is_pr.values
            f[f"dm_newissue_in_{src}_repos_7d"] = _pairs_count(
                ur, pd.DataFrame({"repo": it.repo.values[mi], "a": it.author.values[mi]}), n)
        return pd.DataFrame(f)

    if group == "comment_mix":
        c = _cut(s, lags, "issue_comment")
        cm = X["comment"]
        m = (cm.t.values < c) & (cm.t.values >= _ago(c, 90))
        tot = np.maximum(_count(cm.uid.values[m], n), 1)
        for name, mm in [("own_pr", cm.on_pr.values & cm.own.values), ("other_pr", cm.on_pr.values & ~cm.own.values),
                         ("own_issue", ~cm.on_pr.values & cm.own.values), ("other_issue", ~cm.on_pr.values & ~cm.own.values),
                         ("via_app", cm.via_app.values)]:
            f[f"cm_{name}_share_90d"] = _count(cm.uid.values[m & mm], n) / tot
        return pd.DataFrame(f)

    if group == "org":
        tot, org = np.zeros(n), np.zeros(n)
        for k in NONPUSH:
            c = _cut(s, lags, k)
            e = ev[k]
            m = (e.avail.values < c) & (e.t.values >= _ago(c, 90)) & ~np.isnan(e.repo.values)
            tot += _count(e.uid.values[m], n)
            org += _count(e.uid.values[m], n, X["is_org"][e.repo.values[m].astype(np.int64)])
            f[f"org_share_{k}_90d"] = (_count(e.uid.values[m], n, X["is_org"][e.repo.values[m].astype(np.int64)])
                                       / np.maximum(_count(e.uid.values[m], n), 1))
        f["org_share_all_90d"] = org / np.maximum(tot, 1)
        return pd.DataFrame(f)
    raise ValueError(group)


def evaluate(data, feats, train_weeks, val_weeks, seed):
    """baseline-style LightGBM per activity: validation F1 per activity and mean weekly score."""
    params = {**base.PARAMS, "seed": seed}
    tr, va = data.week.isin(train_weeks).values, data.week.isin(val_weeks).values
    prob, thr, f1s = {}, {}, {}
    for k in EVENT_COLS:
        y = data[f"y_{k}"].values
        mtr, mva = tr & ~np.isnan(y), va & ~np.isnan(y)
        dtr = lgb.Dataset(data.loc[mtr, feats], y[mtr])
        model = lgb.train(params, dtr, 2000, valid_sets=[lgb.Dataset(data.loc[mva, feats], y[mva], reference=dtr)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        prob[k] = model.predict(data[feats], num_iteration=model.best_iteration)
        thr[k], f1s[k] = base.best_threshold(y[mva], prob[k][mva])
    weekly = []
    for w in val_weeks:
        m = (data.week == w).values
        if data.loc[m, [f"y_{k}" for k in EVENT_COLS]].isna().any().any():
            continue
        truth = base.to_output(data.cidx.values[m], {k: data.loc[m, f"y_{k}"].values for k in EVENT_COLS})
        pred = base.to_output(data.cidx.values[m], {k: prob[k][m] >= thr[k] for k in EVENT_COLS})
        weekly.append(base.scores(pred, truth)["score"])
    return f1s, float(np.mean(weekly))


def main():
    ap = argparse.ArgumentParser()
    base.add_period_args(ap)
    ap.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort, ev, own, data_end = base.load()
    X = load_extra(cohort)
    n = len(cohort)
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    weeks, train_weeks, val_weeks, _ = base.split_weeks(args, data_end)
    print(f"{len(train_weeks)} train weeks, validation {[str(w)[:10] for w in val_weeks]}")

    parts = []
    for w in weeks:
        t = base.build_week(w, ev, own, lags, data_end, n)
        extra = [group_features(g, w, X, ev, lags, n).add_prefix(f"{g}__") for g in args.groups]
        parts.append(pd.concat([t] + extra, axis=1)[t.eligible.values])
    data = pd.concat(parts, ignore_index=True)
    base_feats = [c for c in data.columns if "__" not in c and c not in {"eligible", "cidx", "week"}
                  and not c.startswith("y_")]
    print(f"{len(data)} samples, {len(base_feats)} baseline features")

    configs = {"baseline": base_feats}
    for g in args.groups:
        configs[f"+{g}"] = base_feats + [c for c in data.columns if c.startswith(f"{g}__")]
    configs["+all"] = base_feats + [c for c in data.columns if "__" in c]

    rows = {}
    for name, feats in configs.items():
        res = [evaluate(data, feats, train_weeks, val_weeks, s) for s in args.seeds]
        rows[name] = {**{k: np.mean([r[0][k] for r in res]) for k in EVENT_COLS},
                      "score": np.mean([r[1] for r in res]), "n_feats": len(feats)}
        print(f"{name:14s} " + "  ".join(f"{k[:10]} {rows[name][k]:.4f}" for k in EVENT_COLS)
              + f"  score {100 * rows[name]['score']:.2f}", flush=True)

    table = pd.DataFrame(rows).T
    delta = table[EVENT_COLS + ["score"]] - table.loc["baseline", EVENT_COLS + ["score"]]
    print("\nchange vs baseline features (F1 per activity, score in points):")
    delta["score"] *= 100
    print(delta.round(4).to_string())
    (OUT_DIR / "screen.json").write_text(json.dumps({"seeds": args.seeds, "results": table.to_dict("index")},
                                                    indent=2, default=float))


if __name__ == "__main__":
    main()
