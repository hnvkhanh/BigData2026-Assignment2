"""Per-user features and labels of one week, and their capture-independent (relative) views."""
import numpy as np
import pandas as pd

from .config import DAY, MIN_RECORDS, SELECTION_TIME, TEST_START, WEEK, WINDOWS
from .evaluation import EVENT_COLS


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


# Count-like features: their scale follows how much of the week GH Archive captured.
COUNT_KEYS = ("_cnt_", "_active_days_30d", "_repos_30d", "open_prs_now", "open_issues_now")


def relative_views(df, feats):
    """Capture-independent views of the count features of one week's table (all cohort rows of the week).

    rel_<c> = c / cohort mean of c in that week; vol_<c> = log1p(cohort total of c) for the 7d / 30d counts.
    Returns (rel, vol) DataFrames on df's index."""
    cnt = [c for c in feats if any(k in c for k in COUNT_KEYS)]
    x = df[cnt].astype(float)
    rel = x.div(x.mean().replace(0, np.nan)).fillna(0).add_prefix("rel_")
    vol = pd.DataFrame({f"vol_{c}": np.log1p(x[c].sum()) for c in cnt if c.endswith(("_cnt_7d", "_cnt_30d"))},
                       index=df.index)
    return rel, vol


def relative_features(df, feats):
    """The `relative` feature set of experiments/missing_data/run.py: non-count features, rel_ counts and
    vol_ cohort volumes. Returns (df with the new columns, new feature list)."""
    rel, vol = relative_views(df, feats)
    other = [c for c in feats if not any(k in c for k in COUNT_KEYS)]
    return pd.concat([df, rel, vol], axis=1), other + list(rel.columns) + list(vol.columns)
