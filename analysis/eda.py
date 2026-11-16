"""Interactive exploratory data analysis of the cohort activity data (Dash + Plotly).

Activities are defined exactly like the labels: one event per row of each table, at the time
the activity happened (`created_at`, `submitted_at` for reviews), for the 9,000 cohort users.
Weeks are aligned to the test week (Saturday 16:00 UTC), as in experiments/baseline/run.py.

Run from the repo root:
    python analysis/eda.py                    # then open http://127.0.0.1:8050
    python analysis/eda.py --port 8051 --debug
"""
import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, dash_table, dcc, html
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ghactivity import DB, EVENT_COLS, TEST_INPUT  # noqa: E402

# Same constants as ghactivity/config.py.
TEST_START = pd.Timestamp("2026-11-21 16:00")      # 2026-11-22 00:00 Taiwan time
SELECTION_TIME = pd.Timestamp("2026-11-04 12:00")  # test users: >= 3 records in the 90 days before
DATA_START = pd.Timestamp("2025-12-01")  # placeholder; Data() resets it to the first hour of the data
WEEK = pd.Timedelta(days=7)
DAY = pd.Timedelta(days=1)
MIN_RECORDS = 3
GAPS = {
    "nonpush": [("2026-06-01", "2026-09-07")],
    "push": [("2026-08-18", "2026-09-13"), ("2026-10-07", "2026-10-08"),
             ("2026-10-15", "2026-10-19"), ("2026-10-23", "2026-10-24")],
}
FEED_END = {"push": pd.Timestamp("2026-10-27")}  # GH Archive PushEvents ~1% of usual volume after

# Quiet users tab. An example (eligible user, week) is quiet with fewer than --active-min-events
# visible rows in the --ego-lookback days before the cutoffs (experiments/gnn/run.py).
MODEL_START = pd.Timestamp("2026-01-14 16:00")  # first week of the models (baseline --start)
QUIET_WINDOWS = (7, 14, 21, 30)
# The cohort was drawn from users active in May: 8,997 of the 9,000 are active in this window
# (daily active users peak 05-15 -> 05-28), so quiet users return far more often in its weeks.
SAMPLING = (pd.Timestamp("2026-05-04"), pd.Timestamp("2026-05-31"))
SUMMER_OUTAGE = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-09-13"))  # both GAPS, combined
# Validation F1 per activity of the ego GNN + user features (README). Predicting a group of quiet
# users positive raises an activity's F1 only if the group's hit rate is above F1 / 2.
REF_F1 = {"open_pr": 0.403, "open_issue": 0.464, "issue_comment": 0.347, "pr_review": 0.376,
          "pr_review_comment": 0.368, "push": 0.628}
QUIET_DRIVERS = {
    "since": "Days since the latest visible row",
    "streak": "Consecutive quiet weeks, this one included",
    "own": "Own visible rows of the activity, last 90 days",
    "volume": "Visible rows of all activities, last 90 days",
    "types": "Activity types with a visible row, last 90 days",
    "last": "Activity of the latest visible row",
}
# Bucket lower edges and labels.
SINCE_BINS = ([0, 7, 14, 21, 28, 42, 60, 90], ["0-6", "7-13", "14-20", "21-27", "28-41", "42-59",
                                                "60-89", "90+"])
OWN_BINS = ([0, 1, 2, 4, 8, 16], ["0", "1", "2-3", "4-7", "8-15", "16+"])
VOLUME_BINS = ([0, 3, 6, 11, 21, 51, 101], ["0-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101+"])
MIN_GROUP = 50  # smallest group drawn in the quiet breakdowns

TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments", "PRReviews",
          "PRReviewComments", "Pushes"]
POPULATIONS = {"all": "All cohort users", "test": "Test users", "other": "Cohort, not in test"}
LABELS = {"open_pr": "Open PR", "open_issue": "Open issue", "issue_comment": "Issue comment",
          "pr_review": "PR review", "pr_review_comment": "PR review comment", "push": "Push"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --- Look: reference palette of the dataviz guidelines, light mode. -------------------------
# Each activity keeps its categorical slot everywhere, whatever is filtered.
COLORS = dict(zip(EVENT_COLS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
SURFACE, PAGE, INK, INK2, MUTED = "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BORDER = "#e1e0d9", "#c3c2b7", "rgba(11,11,11,0.10)"
OUTAGE, TEST_FILL = "rgba(137,135,129,0.16)", "rgba(42,120,214,0.10)"
SAMPLING_FILL = "rgba(237,161,0,0.12)"
# Segments: quiet takes categorical slot 7 (no activity uses it), active is the gray reference.
QUIET, ACTIVE = "#6250d6", MUTED
SEQUENTIAL = ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
              "#0d366b"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_axis = dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED),
             title_font=dict(color=INK2), showline=True)
pio.templates["eda"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=INK, size=13),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=list(COLORS.values()),
    xaxis=_axis, yaxis=_axis,
    title=dict(font=dict(size=15, color=INK), x=0.01, xanchor="left", y=0.98, yanchor="top"),
    legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(color=INK2)),
    hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
    margin=dict(l=64, r=24, t=96, b=48),
    bargap=0.25,
))
pio.templates.default = "plotly_white+eda"


# --- Data ------------------------------------------------------------------------------------

class Data:
    """Everything the app plots, computed once at startup."""

    def __init__(self):
        t0 = time.time()
        users = pd.read_parquet(DB / "Users.parquet",
                                columns=["user_id", "github_user_id", "login", "is_bot", "is_cohort"])
        prs = pd.read_parquet(DB / "PullRequests.parquet",
                              columns=["pr_id", "repo_id", "author_id", "created_at", "first_seen_at"])
        issues = pd.read_parquet(DB / "Issues.parquet",
                                 columns=["issue_id", "repo_id", "author_id", "is_pull_request",
                                          "created_at", "first_seen_at"])
        comments = pd.read_parquet(DB / "IssueComments.parquet",
                                   columns=["issue_id", "repo_id", "author_id", "created_at"])
        reviews = pd.read_parquet(DB / "PRReviews.parquet",
                                  columns=["pr_id", "repo_id", "reviewer_id", "submitted_at",
                                           "first_seen_at"])
        rcomments = pd.read_parquet(DB / "PRReviewComments.parquet",
                                    columns=["pr_id", "repo_id", "author_id", "created_at"])
        pushes = pd.read_parquet(DB / "Pushes.parquet", columns=["user_id", "repo_id", "created_at"])
        self.repos = pd.read_parquet(DB / "Repositories.parquet",
                                     columns=["repo_id", "name", "org_id", "first_seen_at"])

        cohort = users[users.is_cohort].reset_index(drop=True)
        test_ids = set(pd.read_csv(TEST_INPUT).UserID)
        cohort["is_test"] = cohort.github_user_id.isin(test_ids)
        self.cohort = cohort
        self.n = n = len(cohort)
        to_cidx = np.full(len(users), -1, np.int64)
        to_cidx[cohort.user_id.values] = np.arange(n)

        # activity -> (actor, time it happened, time the row appears in the data, repo, time the
        # PR / issue / repository it acts on appears in the data)
        plain = issues[~issues.is_pull_request]
        pr_seen = prs.set_index("pr_id").first_seen_at
        issue_seen = issues.set_index("issue_id").first_seen_at
        repo_seen = self.repos.set_index("repo_id").first_seen_at
        sources = {
            "open_pr": (prs.author_id, prs.created_at, prs.first_seen_at, prs.repo_id,
                        prs.first_seen_at.values),
            "open_issue": (plain.author_id, plain.created_at, plain.first_seen_at, plain.repo_id,
                           plain.first_seen_at.values),
            "issue_comment": (comments.author_id, comments.created_at, comments.created_at,
                              comments.repo_id, issue_seen.reindex(comments.issue_id).values),
            "pr_review": (reviews.reviewer_id, reviews.submitted_at, reviews.first_seen_at,
                          reviews.repo_id, pr_seen.reindex(reviews.pr_id).values),
            "pr_review_comment": (rcomments.author_id, rcomments.created_at, rcomments.created_at,
                                  rcomments.repo_id, pr_seen.reindex(rcomments.pr_id).values),
            "push": (pushes.user_id, pushes.created_at, pushes.created_at, pushes.repo_id,
                     repo_seen.reindex(pushes.repo_id).values),
        }
        # Last moment of each table as experiments/baseline/run.py defines it (its lags set the
        # per-table cutoffs of the models; open_issue ends with the last issue of any kind).
        table_end = {k: min(seen.max(), FEED_END.get(k, pd.Timestamp.max))
                     for k, (_, _, seen, _, _) in sources.items()}
        table_end["open_issue"] = issues.first_seen_at.max()
        parts, self.data_end = [], {}
        for k, (actor, t, seen, repo, entity_seen) in sources.items():
            self.data_end[k] = min(seen.max(), FEED_END.get(k, pd.Timestamp.max))
            d = pd.DataFrame({"uid": actor.to_numpy("float64", na_value=np.nan), "t": t.values,
                              "seen": seen.values, "repo": repo.to_numpy("float64", na_value=np.nan),
                              "entity_seen": entity_seen})
            d = d[d.uid.notna() & d.t.notna()]
            d["uid"] = to_cidx[d.uid.astype(np.int64).values]
            d = d[d.uid >= 0].assign(activity=k)
            parts.append(d)
        ev = pd.concat(parts, ignore_index=True)
        # The span starts at the first hour of the GH Archive files: the earliest time a row
        # appears in the data. Older `created_at` values exist (e.g. issues from 2011 that got a
        # comment in the window) but are not activity inside the data, so they are dropped,
        # as in experiments/baseline/run.py.
        global DATA_START
        DATA_START = ev.seen.min().floor("D")
        ev = ev[ev.t >= DATA_START].sort_values("t", ignore_index=True)
        ev["activity"] = pd.Categorical(ev.activity, categories=EVENT_COLS)
        ev["uid"] = ev.uid.astype(np.int32)
        ev["is_test"] = cohort.is_test.values[ev.uid.values]
        ev["day"] = ev.t.dt.floor("D")
        for tz, shift in [("utc", 0), ("tw", 8)]:
            local = ev.t + pd.Timedelta(hours=shift)
            ev[f"how_{tz}"] = (local.dt.dayofweek * 24 + local.dt.hour).astype(np.int16)

        # Weeks [s, s + 7d) aligned to the test week; week 0 is the first full week of data.
        self.n_weeks = w = int((TEST_START - DATA_START) // WEEK)
        self.week_starts = pd.DatetimeIndex([TEST_START - WEEK * i for i in range(w, 0, -1)])
        ev["week"] = ((ev.t - self.week_starts[0]) // WEEK).astype(np.int32)
        self.ev = ev

        # Per activity: events per (user, week); a week is labelled only if its data is complete.
        self.counts, self.valid = {}, {}
        for k in EVENT_COLS:
            d = ev[(ev.activity == k) & (ev.week >= 0)]
            flat = np.bincount(d.uid.values.astype(np.int64) * w + d.week.values, minlength=n * w)
            self.counts[k] = flat.reshape(n, w).astype(np.int32)
            self.valid[k] = np.asarray(self.week_starts + WEEK <= self.data_end[k])

        # Test selection rule shifted to every week (>= 3 records in the 90 days before
        # s - 17d4h). Approximation: it ignores that some rows enter the data later.
        tv, uv = ev.t.values, ev.uid.values
        self.eligible = np.zeros((n, w), bool)
        for j, s in enumerate(self.week_starts):
            sel = s - (TEST_START - SELECTION_TIME)
            a, b = np.searchsorted(tv, [np.datetime64(sel - 90 * DAY), np.datetime64(sel)])
            self.eligible[:, j] = np.bincount(uv[a:b], minlength=n) >= MIN_RECORDS

        self.ex = self._examples(table_end)
        test_now = self._eligible_at(TEST_START)
        print(f"Loaded {len(ev):,} cohort events in {time.time() - t0:.1f}s. Selection rule at the "
              f"test week: {test_now.sum():,} users, {(test_now == cohort.is_test.values).mean():.2%} "
              f"agree with test_input.csv")
        q = self.ex[self.ex.j == w]
        print(f"{len(self.ex):,} model examples; test week: {len(q):,} users, "
              f"{(q.vis_14d == 0).sum():,} quiet (no visible activity in the 14 days before the cutoffs)")

    def _examples(self, table_end):
        """The examples (eligible user, week) of the models, with the inputs of the Quiet users tab.

        Same rules as experiments/baseline/run.py and the quiet segment of experiments/gnn/run.py
        (`ego_tokens`, `--segment activity`). Activity k is visible at week s up to its own cutoff
        c_k = s - lag_k (lag_k = TEST_START - the last time of that table), and a row counts only if it
        and the PR / issue / repository it acts on have appeared in the data before c_k. Eligibility is
        the baseline's selection rule shifted to s (it counts rows by when they appear); the test week
        holds the test users. One row per example: cidx, j (week index; n_weeks = the test week),
        vis_<L>d (visible rows of the L days before the cutoffs), since (days from the latest visible
        row to its cutoff, inf beyond 90), last (its activity index, -1 none), c90_<k> (visible rows
        of k in the 90 days before c_k), blind (rows with c_k <= t < s, i.e. activity that happened
        but is not visible yet; -1 when the data does not reach s) and y_<k> (label, NaN unknown).
        """
        n, w, ev = self.n, self.n_weeks, self.ev
        self.lag = {k: TEST_START - end for k, end in table_end.items()}
        never = np.iinfo(np.int64).max

        def ns(x):  # times as int64 ns; NaT (never visible) becomes the largest value
            a = np.asarray(x, dtype="datetime64[ns]")
            return np.where(np.isnat(a), never, a.astype(np.int64))

        src = {}
        for k in EVENT_COLS:
            d = ev[ev.activity == k]  # sorted by t
            t = ns(d.t.values)
            avail = np.maximum(t, ns(d.seen.values))  # when the row appears (baseline's avail)
            entity = np.where(d.repo.isna().values, never, ns(d.entity_seen.values))
            src[k] = (d.uid.values.astype(np.int64), t, avail, np.maximum(avail, entity))

        starts = list(self.week_starts) + [TEST_START]
        cols = {c: [] for c in ["cidx", "j", "since", "last", "blind"]
                + [f"vis_{L}d" for L in QUIET_WINDOWS] + [f"c90_{k}" for k in EVENT_COLS]
                + [f"y_{k}" for k in EVENT_COLS]}
        day = 86400 * 10**9
        for j, s in enumerate(starts):
            if s < MODEL_START:
                continue
            sel = s - (TEST_START - SELECTION_TIME)
            rec = np.zeros(n, np.int64)
            vis = {L: np.zeros(n, np.int64) for L in QUIET_WINDOWS}
            since, last = np.full(n, np.inf), np.full(n, -1, np.int8)
            blind, blind_known = np.zeros(n, np.int64), True
            c90 = {}
            for i, k in enumerate(EVENT_COLS):
                uid, t, avail, visible_at = src[k]
                c = s - self.lag[k]
                cut, sel_cut = ns(c).item(), ns(min(sel, c)).item()
                a, b = np.searchsorted(t, [ns(sel - 90 * DAY).item(), sel_cut])
                rec += np.bincount(uid[a:b][avail[a:b] < sel_cut], minlength=n)
                a, b = np.searchsorted(t, [cut - 90 * day, cut])
                m = visible_at[a:b] < cut
                u, tm = uid[a:b][m], t[a:b][m]
                c90[k] = np.bincount(u, minlength=n)
                for L in QUIET_WINDOWS:
                    vis[L] += np.bincount(u[tm >= cut - L * day], minlength=n)
                age = np.full(n, np.inf)
                np.minimum.at(age, u, (cut - tm) / day)
                last[age < since] = i
                since = np.minimum(since, age)
                if s <= table_end[k]:
                    a, b = np.searchsorted(t, [cut, ns(s).item()])
                    blind += np.bincount(uid[a:b], minlength=n)
                else:
                    blind_known = False
            users = np.flatnonzero(self.cohort.is_test.values if j == w else rec >= MIN_RECORDS)
            cols["cidx"].append(users)
            cols["j"].append(np.full(len(users), j))
            cols["since"].append(since[users])
            cols["last"].append(last[users])
            cols["blind"].append(blind[users] if blind_known else np.full(len(users), -1))
            for L in QUIET_WINDOWS:
                cols[f"vis_{L}d"].append(vis[L][users])
            for k in EVENT_COLS:
                cols[f"c90_{k}"].append(c90[k][users])
                y = np.full(len(users), np.nan)
                if j < w and self.valid[k][j]:
                    y = (self.counts[k][users, j] > 0).astype(float)
                cols[f"y_{k}"].append(y)
        ex = pd.DataFrame({c: np.concatenate(v) for c, v in cols.items()})
        small = {c: np.int32 for c in ex.columns if c.startswith(("vis_", "c90_"))}
        return ex.astype({**small, "cidx": np.int32, "j": np.int16, "blind": np.int32,
                          "since": np.float32})

    def _eligible_at(self, s):
        sel = s - (TEST_START - SELECTION_TIME)
        tv = self.ev.t.values
        a, b = np.searchsorted(tv, [np.datetime64(sel - 90 * DAY), np.datetime64(sel)])
        return np.bincount(self.ev.uid.values[a:b], minlength=self.n) >= MIN_RECORDS

    def user_mask(self, pop):
        if pop == "test":
            return self.cohort.is_test.values
        if pop == "other":
            return ~self.cohort.is_test.values
        return np.ones(self.n, bool)

    def week_mask(self, start, end):
        return np.asarray((self.week_starts >= start) & (self.week_starts < end))


@lru_cache(maxsize=2)
def load_table(name):
    return pd.read_parquet(DB / f"{name}.parquet")


@lru_cache(maxsize=None)
def table_summary(name):
    df = load_table(name)
    rows = []
    for c in df.columns:
        s = df[c]
        nonnull = s.dropna()
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "null %": round(100 * s.isna().mean(), 2),
            "unique": int(nonnull.nunique()),
            "example": str(nonnull.iloc[0])[:80] if len(nonnull) else "",
        })
    return pd.DataFrame(rows)


# --- Figure helpers --------------------------------------------------------------------------

def empty_fig(message, height=320):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.update_layout(height=height)


def add_context(fig, acts, row=None, col=None, test_label=True):
    """Shade collection outages of the given activities and mark the test week."""
    groups = {"push" if k == "push" else "nonpush" for k in acts}
    for g in sorted(groups):
        for a, b in GAPS[g]:
            fig.add_vrect(x0=a, x1=b, fillcolor=OUTAGE, line_width=0, layer="below", row=row, col=col)
    fig.add_vrect(x0=TEST_START, x1=TEST_START + WEEK, fillcolor=TEST_FILL, line_width=0,
                  layer="below", row=row, col=col)
    if test_label:
        fig.add_annotation(x=TEST_START + WEEK / 2, y=1, yref="paper", text="test week",
                           showarrow=False, yanchor="bottom", font=dict(size=11, color=MUTED))


def log_bins(values, edges=(0, 1, 2, 3, 4, 6, 11, 21, 51, 101, 201, 501, 1001)):
    """Counts of integer values in human-readable buckets: 0, 1, 2, 3, 4-5, 6-10, ..., 1001+."""
    edges = list(edges) + [np.inf]
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        hi = b - 1
        labels.append(f"{a}" if hi == a else (f"{a}+" if np.isinf(b) else f"{a}-{int(hi)}"))
    hist, _ = np.histogram(values, bins=edges)
    return labels, hist


def pct(x):
    return f"{100 * x:.1f}%"


def card(*children, **style):
    return html.Div(children, className="card", style=style)


def graph(gid, height=None):
    return dcc.Loading(dcc.Graph(id=gid, config={"displaylogo": False},
                                 style={"height": f"{height}px"} if height else {}),
                       type="dot", color=COLORS["open_pr"])


def note(text):
    return html.P(text, className="note")


def control(label, component):
    return html.Div([html.Label(label, className="ctl-label"), component], className="ctl")


# --- App -------------------------------------------------------------------------------------

def build_app(D):
    ev = D.ev
    app = Dash(__name__, title="GitHub activity EDA")
    app.index_string = INDEX_HTML

    @lru_cache(maxsize=16)
    def events(acts, pop, start, end):
        m = ev.activity.isin(acts).values & (ev.t.values >= np.datetime64(pd.Timestamp(start)))
        m &= ev.t.values < np.datetime64(pd.Timestamp(end))
        if pop != "all":
            m &= ev.is_test.values == (pop == "test")
        return ev[m]

    def filters(acts, pop, start, end):
        acts = tuple(k for k in EVENT_COLS if k in (acts or []))
        return acts, pop, pd.Timestamp(start), pd.Timestamp(end) + DAY  # end date is inclusive

    # --- Header: stat tiles and global filters ---
    tiles = [
        ("Cohort users", f"{D.n:,}"),
        ("Test users", f"{D.cohort.is_test.sum():,}"),
        ("Cohort events", f"{len(ev):,}"),
        ("Data span", f"{DATA_START:%Y-%m-%d} → {max(D.data_end.values()):%m-%d}"),
        ("Test week (UTC)", f"{TEST_START:%m-%d %H:%M} + 7d"),
    ]
    header = html.Div([
        html.H1("GitHub user activity: exploratory analysis"),
        html.P("Activity of the 9,000 cohort users, defined the same way as the 6 prediction "
               "labels. Gray bands mark collection outages, the blue band marks the test week.",
               className="sub"),
        html.Div([html.Div([html.Div(v, className="tile-value"), html.Div(k, className="tile-label")],
                           className="tile") for k, v in tiles], className="tiles"),
        html.Div([
            control("Date range", dcc.DatePickerRange(
                id="dates", start_date=DATA_START.date(), end_date=(TEST_START + WEEK).date(),
                min_date_allowed=DATA_START.date(), max_date_allowed=(TEST_START + WEEK).date(),
                display_format="YYYY-MM-DD")),
            control("Users", dcc.Dropdown(
                id="pop", options=[{"label": v, "value": k} for k, v in POPULATIONS.items()],
                value="all", clearable=False, style={"width": "220px"})),
            control("Activities", dcc.Checklist(
                id="acts", value=list(EVENT_COLS), inline=True, className="acts",
                options=[{"label": html.Span([html.Span(className="swatch",
                                                        style={"background": COLORS[k]}), LABELS[k]]),
                          "value": k} for k in EVENT_COLS])),
        ], className="filters"),
        html.P("Filters apply to every tab except the table and user explorers.", className="note"),
    ], className="header")

    act_options = [{"label": LABELS[k], "value": k} for k in EVENT_COLS]
    # Test users first, then by number of events.
    totals = sum(c.sum(1) for c in D.counts.values())
    order = np.lexsort((-totals, ~D.cohort.is_test.values))
    user_options = [{"label": f"{D.cohort.login[i]} · {D.cohort.github_user_id[i]}"
                              + (" · test" if D.cohort.is_test[i] else ""), "value": int(i)}
                    for i in order]

    tabs = dcc.Tabs(id="tabs", value="time", className="tabs", children=[
        dcc.Tab(label="Activity over time", value="time", children=[card(
            html.Div([
                control("Measure", dcc.RadioItems(id="ts-metric", inline=True, value="events", options=[
                    {"label": "Events", "value": "events"},
                    {"label": "Active users", "value": "users"}])),
                control("Granularity", dcc.RadioItems(id="ts-gran", inline=True, value="day", options=[
                    {"label": "Day", "value": "day"}, {"label": "Week", "value": "week"}])),
                control("Scale", dcc.Checklist(id="ts-log", inline=True, value=[],
                                               options=[{"label": "Log y", "value": "log"}])),
            ], className="row"),
            note("One panel per activity, each on its own y-scale. Vertical line: last timestamp in "
                 "that table; the gap to the test week is how stale features are at prediction time."),
            graph("ts-fig"),
        )]),
        dcc.Tab(label="Weekly labels", value="labels", children=[
            card(
                html.Div([control("Rows", dcc.Checklist(
                    id="lb-elig", inline=True, value=["elig"],
                    options=[{"label": "Only users who pass the test selection rule that week",
                              "value": "elig"}]))], className="row"),
                note("Positive rate = share of users who perform the activity in the week [s, s+7d), "
                     "i.e. the label. Weeks whose data is incomplete are left out. In the first "
                     "weeks the selection rule has < 90 days of history, so it keeps only very "
                     "active users."),
                graph("lb-rate"),
            ),
            html.Div([card(graph("lb-base")), card(graph("lb-ntypes"))], className="grid2"),
            card(note("Read across a row: of the user-weeks with the row's activity, the share that "
                      "also have the column's activity in the same week."), graph("lb-cooc")),
        ]),
        dcc.Tab(label="Persistence", value="persist", children=[
            card(note("How predictive is the past? P(active in a week) by the number of weeks since "
                      "the user last did the same activity. Uses the eligibility toggle of the "
                      "Weekly labels tab."), graph("ps-recency")),
            card(note("P(column activity this week | row activity last week). The last row is the "
                      "unconditional rate, for comparison."), graph("ps-trans")),
        ]),
        dcc.Tab(label="Quiet users", value="quiet", children=[
            card(
                html.Div([
                    control("Quiet window", dcc.RadioItems(
                        id="qt-window", inline=True, value=14,
                        options=[{"label": f"{L} days", "value": L} for L in QUIET_WINDOWS])),
                    control("Active needs", dcc.RadioItems(
                        id="qt-min", inline=True, value=1,
                        options=[{"label": f"≥ {v} visible row{'s' if v > 1 else ''}", "value": v}
                                 for v in (1, 2, 3, 5)])),
                    control("Leave out weeks distorted by", dcc.Checklist(
                        id="qt-drop", inline=True, value=["sampling", "outage"], options=[
                            {"label": "cohort sampling (05-04 → 05-31)", "value": "sampling"},
                            {"label": "summer outage (06-01 → 09-13)", "value": "outage"}])),
                ], className="row"),
                note(f"Examples are the models' (eligible user, week s) pairs from "
                     f"{MODEL_START:%Y-%m-%d}. An example is quiet when it has fewer visible "
                     "activity rows than 'Active needs' in the window before the cutoffs: the "
                     "segment of experiments/gnn/run.py --segment activity (defaults: 14 days, "
                     "≥ 1 row, i.e. no ego token). Each table is visible up to its own cutoff "
                     f"s − lag: {D.lag['open_pr'] / DAY:.1f} days before s, "
                     f"{D.lag['push'] / DAY:.1f} days for push. Quiet uses all 6 activities; the "
                     "Activities filter picks the labels shown. 'Leave out' applies to every chart "
                     "but the timeline, which shades those weeks instead."),
                html.Div(id="qt-tiles", className="tiles"),
            ),
            card(note("Every model week, at its start, for the selected users. Up to March the "
                      "selection rule sees < 90 days of data, so it keeps mostly very active users. "
                      "Yellow: the cohort was sampled among users active in May, so quiet users "
                      "of those weeks come back far more often. Gray: outages. From June to "
                      "September non-push rows are missing, so most examples look quiet. The "
                      "bottom panel is the share of the push quiet window that falls in a push "
                      "outage, which inflates the quiet share in the weeks after it, including "
                      "the test week."), graph("qt-time")),
            html.Div([card(graph("qt-rates")), card(graph("qt-share"))], className="grid2"),
            card(
                html.Div([control("Break down quiet examples by", dcc.Dropdown(
                    id="qt-driver", value="since", clearable=False, style={"width": "380px"},
                    options=[{"label": v, "value": k} for k, v in QUIET_DRIVERS.items()]))],
                    className="row"),
                note("Share of quiet examples that do the activity in the week, per group. Groups "
                     "with fewer than 50 examples are not drawn. Dashed line: break-even hit "
                     "rate, half the activity's validation F1 for the ego GNN + user features "
                     "(README). Predicting a group positive raises that activity's F1 only if "
                     "the group's hit rate is above the line."),
                graph("qt-drivers"),
            ),
            html.Div([
                card(note("Rows that happen between a table's cutoff and the week start exist "
                          "but are not visible to the model yet. Weeks whose data does not "
                          "reach their start are left out."), graph("qt-blind")),
                card(note("Activity of the latest visible row (each against its own cutoff), "
                          "for quiet and active examples."), graph("qt-last")),
            ], className="grid2"),
            html.Div([
                card(note("Users with at least 5 examples in the selected weeks."), graph("qt-users")),
                card(note("Days from the latest visible row to its cutoff, for the quiet users of "
                          "the test week and of the selected labelled weeks. At the test week, "
                          "42-59 days before the push cutoff is the summer push outage, so few "
                          "users have their latest visible row there."), graph("qt-test")),
            ], className="grid2"),
        ]),
        dcc.Tab(label="Users", value="users", children=[
            html.Div([card(graph("us-reach")), card(graph("us-lorenz"))], className="grid2"),
            card(html.Div([control("Activity", dcc.Dropdown(
                id="us-act", options=act_options, value="push", clearable=False,
                style={"width": "240px"}))], className="row"), graph("us-hist")),
        ]),
        dcc.Tab(label="Time of day", value="tod", children=[
            card(html.Div([
                control("Activity", dcc.Dropdown(
                    id="tod-act", options=[{"label": "All selected", "value": "all"}] + act_options,
                    value="all", clearable=False, style={"width": "240px"})),
                control("Time zone", dcc.RadioItems(id="tod-tz", inline=True, value="utc", options=[
                    {"label": "UTC", "value": "utc"}, {"label": "Taiwan (UTC+8)", "value": "tw"}])),
            ], className="row"), graph("tod-heat")),
            card(graph("tod-hour")),
        ]),
        dcc.Tab(label="Repositories", value="repos", children=[
            card(html.Div([
                control("Activity", dcc.Dropdown(id="rp-act", options=act_options, value="push",
                                                 clearable=False, style={"width": "240px"})),
                control("Rank by", dcc.RadioItems(id="rp-rank", inline=True, value="users", options=[
                    {"label": "Distinct users", "value": "users"},
                    {"label": "Events", "value": "events"}])),
            ], className="row"), graph("rp-top")),
            html.Div([card(graph("rp-per-user")), card(graph("rp-org"))], className="grid2"),
        ]),
        dcc.Tab(label="Data quality", value="quality", children=[
            html.Div([card(graph("dq-lag")), card(graph("dq-delay"))], className="grid2"),
            card(note("Missing values per column of every table (cells are null %)."),
                 graph("dq-nulls")),
        ]),
        dcc.Tab(label="Table explorer", value="tables", children=[card(
            html.Div([
                control("Table", dcc.Dropdown(id="tx-table", options=TABLES, value="PullRequests",
                                              clearable=False, style={"width": "240px"})),
                control("Column", dcc.Dropdown(id="tx-col", clearable=False, style={"width": "260px"})),
            ], className="row"),
            note("Whole tables, not only cohort rows. Text columns with many distinct values are "
                 "shown as a length distribution."),
            graph("tx-fig"),
            dcc.Loading(html.Div(id="tx-summary"), type="dot"),
        )]),
        dcc.Tab(label="User explorer", value="user", children=[card(
            html.Div([control("User", dcc.Dropdown(id="ux-user", options=user_options,
                                                   value=user_options[0]["value"],
                                                   placeholder="Type a login or GitHub id",
                                                   style={"width": "420px"}))],
                     className="row"),
            html.Div(id="ux-info", className="note"),
            graph("ux-heat"),
            graph("ux-repos"),
        )]),
    ])

    app.layout = html.Div([header, tabs], className="page")
    common = [Input("acts", "value"), Input("pop", "value"), Input("dates", "start_date"),
              Input("dates", "end_date")]

    # --- Activity over time ---
    @app.callback(Output("ts-fig", "figure"), *common, Input("ts-metric", "value"),
                  Input("ts-gran", "value"), Input("ts-log", "value"))
    def time_series(acts, pop, start, end, metric, gran, log):
        acts, pop, start, end = filters(acts, pop, start, end)
        if not acts:
            return empty_fig("Select at least one activity")
        e = events(acts, pop, start, end)
        if gran == "week":
            e = e[e.week >= 0]
            key = D.week_starts[e.week.values]
        else:
            key = e.day.values
        g = e.groupby([e.activity.values, key], observed=True)
        s = g.size() if metric == "events" else g["uid"].nunique()
        fig = make_subplots(rows=len(acts), cols=1, shared_xaxes=True, vertical_spacing=0.045,
                            subplot_titles=[LABELS[k] for k in acts])
        unit = "events" if metric == "events" else "active users"
        for i, k in enumerate(acts, 1):
            sk = s[k] if k in s.index.get_level_values(0) else pd.Series(dtype=float)
            fig.add_trace(go.Scatter(
                x=sk.index, y=sk.values, mode="lines", line=dict(color=COLORS[k], width=2),
                name=LABELS[k], hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:,}} {unit}<extra>{LABELS[k]}</extra>",
            ), row=i, col=1)
            add_context(fig, [k], row=i, col=1, test_label=False)
            fig.add_vline(x=D.data_end[k], line=dict(color=MUTED, width=1), row=i, col=1)
        fig.update_layout(height=190 * len(acts) + 60, showlegend=False, hovermode="x",
                          margin=dict(t=40))
        fig.update_annotations(font=dict(size=13, color=INK2), xanchor="left", x=0)
        fig.update_yaxes(type="log" if "log" in (log or []) else "linear", rangemode="tozero")
        fig.update_xaxes(range=[start, end])
        return fig

    # --- Weekly labels ---
    def label_matrix(k, users, weeks, elig):
        """Boolean labels and a mask of the (user, week) cells that count."""
        y = D.counts[k] > 0
        m = np.outer(users, weeks & D.valid[k])
        if elig:
            m &= D.eligible
        return y, m

    @app.callback(Output("lb-rate", "figure"), Output("lb-base", "figure"),
                  Output("lb-ntypes", "figure"), Output("lb-cooc", "figure"),
                  *common, Input("lb-elig", "value"))
    def weekly_labels(acts, pop, start, end, elig):
        acts, pop, start, end = filters(acts, pop, start, end)
        if not acts:
            return (empty_fig("Select at least one activity"),) * 4
        users, weeks, elig = D.user_mask(pop), D.week_mask(start, end), bool(elig)

        rate = go.Figure()
        base = {}
        for k in acts:
            y, m = label_matrix(k, users, weeks, elig)
            num, den = (y & m).sum(0), m.sum(0)
            r = np.where(den > 0, num / np.maximum(den, 1), np.nan)
            base[k] = (y & m).sum() / max(m.sum(), 1)
            rate.add_trace(go.Scatter(
                x=D.week_starts, y=r, mode="lines+markers", name=LABELS[k],
                line=dict(color=COLORS[k], width=2), marker=dict(size=6),
                customdata=np.stack([num, den], 1),
                hovertemplate="week of %{x|%Y-%m-%d}<br>%{y:.1%} (%{customdata[0]:,} of "
                              "%{customdata[1]:,} users)<extra>" + LABELS[k] + "</extra>"))
        add_context(rate, acts)
        rate.update_layout(title="Weekly positive rate", height=440, hovermode="x unified")
        rate.update_yaxes(tickformat=".0%", rangemode="tozero")
        rate.update_xaxes(range=[start, end])

        bars = go.Figure(go.Bar(
            x=[LABELS[k] for k in acts], y=[base[k] for k in acts],
            marker=dict(color=[COLORS[k] for k in acts], cornerradius=4),
            text=[pct(base[k]) for k in acts], textposition="outside",
            hovertemplate="%{x}: %{y:.2%} of user-weeks<extra></extra>"))
        bars.update_layout(title="Average positive rate (class balance)", height=380)
        bars.update_yaxes(tickformat=".0%")

        # Number of activity types per user-week, on weeks where all selected labels are valid.
        all_valid = weeks.copy()
        for k in acts:
            all_valid &= D.valid[k]
        m = np.outer(users, all_valid) & (D.eligible if elig else True)
        ntypes = sum((D.counts[k] > 0).astype(np.int8) for k in acts)[m]
        hist = np.bincount(ntypes, minlength=len(acts) + 1)
        share = hist / max(hist.sum(), 1)
        nt = go.Figure(go.Bar(
            x=list(range(len(acts) + 1)), y=share, marker=dict(color=COLORS["open_pr"], cornerradius=4),
            customdata=hist, hovertemplate="%{x} activity types: %{y:.1%} "
                                           "(%{customdata:,} user-weeks)<extra></extra>"))
        nt.update_layout(title="Activity types per user-week", height=380)
        nt.update_xaxes(title="number of selected activities done in the week", dtick=1)
        nt.update_yaxes(tickformat=".0%")

        # Same-week co-occurrence P(col | row).
        mats = {k: label_matrix(k, users, weeks, elig) for k in acts}
        z = np.full((len(acts), len(acts)), np.nan)
        for i, a in enumerate(acts):
            ya, ma = mats[a]
            for j, b in enumerate(acts):
                yb, mb = mats[b]
                cond = ya & ma & mb
                z[i, j] = (yb & cond).sum() / cond.sum() if cond.sum() else np.nan
        cooc = heatmap(z, [LABELS[k] for k in acts], [LABELS[k] for k in acts],
                       "Same-week co-occurrence: P(column | row)",
                       "of weeks with %{y}, %{z:.1%} also have %{x}")
        return rate, bars, nt, cooc

    def heatmap(z, rows, cols, title, hover):
        fig = go.Figure(go.Heatmap(
            z=z, x=cols, y=rows, colorscale=SEQUENTIAL, zmin=0, zmax=1, xgap=2, ygap=2,
            texttemplate="%{z:.0%}", textfont=dict(size=12), hovertemplate=hover + "<extra></extra>",
            colorbar=dict(tickformat=".0%", outlinewidth=0, thickness=12)))
        fig.update_layout(title=title, height=120 + 52 * len(rows))
        fig.update_yaxes(autorange="reversed", showgrid=False, showline=False)
        fig.update_xaxes(showgrid=False, showline=False, side="bottom")
        return fig

    # --- Persistence ---
    @app.callback(Output("ps-recency", "figure"), Output("ps-trans", "figure"),
                  *common, Input("lb-elig", "value"))
    def persistence(acts, pop, start, end, elig):
        acts, pop, start, end = filters(acts, pop, start, end)
        if not acts:
            return (empty_fig("Select at least one activity"),) * 2
        users, weeks, elig = D.user_mask(pop), D.week_mask(start, end), bool(elig)
        cats = [str(i) for i in range(1, 13)] + ["13+", "never"]

        rec = go.Figure()
        for k in acts:
            y, m = label_matrix(k, users, weeks, elig)
            last = np.full(D.n, -1)
            pos, tot = np.zeros(len(cats)), np.zeros(len(cats))
            for j in range(D.n_weeks):
                gap = np.where(last >= 0, np.minimum(j - last, 13) - 1, 13)
                mj = m[:, j]
                if j > 0 and mj.any():
                    pos += np.bincount(gap[mj], weights=y[mj, j].astype(float), minlength=len(cats))
                    tot += np.bincount(gap[mj], minlength=len(cats))
                last[y[:, j]] = j
            r = np.where(tot > 0, pos / np.maximum(tot, 1), np.nan)
            rec.add_trace(go.Scatter(
                x=cats, y=r, mode="lines+markers", name=LABELS[k], line=dict(color=COLORS[k], width=2),
                marker=dict(size=8), customdata=tot,
                hovertemplate="last active %{x} week(s) ago: %{y:.1%} active "
                              "(n=%{customdata:,.0f})<extra>" + LABELS[k] + "</extra>"))
        rec.update_layout(title="P(active this week) by weeks since last active", height=440,
                          hovermode="x unified")
        rec.update_xaxes(title="weeks since the user last did this activity", type="category")
        rec.update_yaxes(tickformat=".0%", rangemode="tozero")

        mats = {k: label_matrix(k, users, weeks, elig) for k in acts}
        z = np.full((len(acts) + 1, len(acts)), np.nan)
        for j, b in enumerate(acts):
            yb, mb = mats[b]
            prev_ok = np.zeros_like(mb)
            prev_ok[:, 1:] = mb[:, 1:]  # cell (u, w) with week w-1 inside the data
            for i, a in enumerate(acts):
                ya, ma = mats[a]
                cond = np.zeros_like(mb)
                cond[:, 1:] = ya[:, :-1] & ma[:, :-1] & prev_ok[:, 1:]
                z[i, j] = (yb & cond).sum() / cond.sum() if cond.sum() else np.nan
            z[-1, j] = (yb & prev_ok).sum() / max(prev_ok.sum(), 1)
        rows = [f"{LABELS[k]} last week" for k in acts] + ["any user (base rate)"]
        trans = heatmap(z, rows, [LABELS[k] for k in acts],
                        "Week-to-week transitions: P(column this week | row last week)",
                        "%{y} → %{x} this week: %{z:.1%}")
        return rec, trans

    # --- Quiet users ---
    X = D.ex
    ex_u, ex_j = X.cidx.values.astype(np.int64), X.j.values.astype(np.int64)
    j_test = D.n_weeks  # week index of the test week in D.ex
    starts = D.week_starts.append(pd.DatetimeIndex([TEST_START]))
    Y = {k: X[f"y_{k}"].values for k in EVENT_COLS}
    last_kind, since, blind = X["last"].values.astype(np.int64), X.since.values, X.blind.values
    c90 = np.column_stack([X[f"c90_{k}"].values for k in EVENT_COLS])
    is_example = np.zeros((D.n, j_test + 1), bool)
    is_example[ex_u, ex_j] = True
    distorted = {key: np.asarray((starts < b) & (starts + WEEK > a))  # label week overlaps
                 for key, (a, b) in [("sampling", SAMPLING), ("outage", SUMMER_OUTAGE)]}

    @lru_cache(maxsize=8)
    def segment(window, min_rows):
        """Quiet flag of every example, its run of consecutive quiet weeks up to and including it,
        and the (user, week) grid of quiet examples."""
        quiet = X[f"vis_{window}d"].values < min_rows
        grid = np.zeros_like(is_example)
        grid[ex_u[quiet], ex_j[quiet]] = True
        run = np.zeros(grid.shape, np.int16)
        for j in range(grid.shape[1]):
            run[:, j] = np.where(grid[:, j], (run[:, j - 1] if j else 0) + 1, 0)
        return quiet, run[ex_u, ex_j], grid

    def quiet_rows(pop, start, end, drop):
        """Examples of the selected users and labelled weeks (never the test week)."""
        weeks = np.append(D.week_mask(start, end), False)
        for key in drop or []:
            weeks &= ~distorted[key]
        return weeks[ex_j] & D.user_mask(pop)[ex_u]

    def any_label(acts):
        """1 if the example does any of acts in its week; NaN unless all of them are labelled."""
        return np.max(np.column_stack([Y[k] for k in acts]), axis=1)

    def rate(y, m):
        m = m & ~np.isnan(y)
        return (float(y[m].mean()) if m.any() else np.nan), int(m.sum())

    def share_text(x):
        return "–" if x is None or np.isnan(x) else pct(x)

    def push_window_lost(s, window):
        """Share of the push quiet window [c - window, c) that falls in a push outage."""
        c = s - D.lag["push"]
        a0, lost = c - window * DAY, pd.Timedelta(0)
        for a, b in GAPS["push"]:
            lost += max(pd.Timedelta(0), min(pd.Timestamp(b), c) - max(pd.Timestamp(a), a0))
        return lost / (window * DAY)

    def driver_groups(driver, key, run):
        """Group code of every example and the group labels, for one panel (key = activity or
        'any')."""
        if driver == "since":
            edges, labels = SINCE_BINS
            return np.digitize(since, edges[1:]), labels
        if driver == "streak":
            return np.clip(run, 1, 8) - 1, [str(i) for i in range(1, 8)] + ["8+"]
        if driver == "own" and key != "any":
            edges, labels = OWN_BINS
            return np.digitize(c90[:, EVENT_COLS.index(key)], edges[1:]), labels
        if driver in ("own", "volume"):
            edges, labels = VOLUME_BINS
            return np.digitize(c90.sum(1), edges[1:]), labels
        if driver == "types":
            return (c90 > 0).sum(1), [str(i) for i in range(7)]
        return np.where(last_kind < 0, len(EVENT_COLS), last_kind), [LABELS[k] for k in EVENT_COLS] + ["none"]

    @app.callback(Output("qt-tiles", "children"), Output("qt-time", "figure"),
                  Output("qt-rates", "figure"), Output("qt-share", "figure"),
                  Output("qt-blind", "figure"), Output("qt-last", "figure"),
                  Output("qt-users", "figure"), Output("qt-test", "figure"),
                  *common, Input("qt-window", "value"), Input("qt-min", "value"),
                  Input("qt-drop", "value"))
    def quiet_tab(acts, pop, start, end, window, min_rows, drop):
        acts, pop, start, end = filters(acts, pop, start, end)
        if not acts:
            return [], *(empty_fig("Select at least one activity"),) * 7
        quiet, run, grid = segment(window, min_rows)
        m = quiet_rows(pop, start, end, drop)
        if not m.any():
            return [], *(empty_fig("No labelled weeks in the selection"),) * 7
        ya = any_label(acts)
        users = D.user_mask(pop)[ex_u]
        tm = (ex_j == j_test) & users

        # Tiles
        rq, _ = rate(ya, m & quiet)
        ra, _ = rate(ya, m & ~quiet)
        pos = sum(int(((Y[k] == 1) & m).sum()) for k in acts)
        pos_q = sum(int(((Y[k] == 1) & m & quiet).sum()) for k in acts)
        cur = np.flatnonzero(m & quiet)
        nxt = cur[is_example[ex_u[cur], ex_j[cur] + 1]]
        stay = grid[ex_u[nxt], ex_j[nxt] + 1].mean() if len(nxt) else np.nan
        tiles = [
            (share_text(quiet[m].mean()), f"of {m.sum():,} examples are quiet"),
            (f"{quiet[tm].sum():,} / {tm.sum():,}", "test users quiet at the test week"),
            (f"{share_text(rq)} vs {share_text(ra)}", "do a selected activity in the week: quiet vs active"),
            (share_text(pos_q / pos if pos else np.nan), "of the positive labels are quiet examples"),
            (share_text(stay), "of quiet examples are quiet again the next week"),
        ]
        tiles = [html.Div([html.Div(v, className="tile-value"), html.Div(k, className="tile-label")],
                          className="tile") for v, k in tiles]

        # Timeline: every model week, distorted ones shaded rather than dropped
        n_weeks = len(starts)
        lab = users & ~np.isnan(ya)

        def weekly(mask, values):
            """Per week: mean of values over the examples in mask, and their number."""
            c = np.bincount(ex_j[mask], minlength=n_weeks)
            s = np.bincount(ex_j[mask], weights=values[mask], minlength=n_weeks)
            return np.where(c > 0, s / np.maximum(c, 1), np.nan), c

        share_w, n_w = weekly(users, quiet.astype(float))
        used = n_w > 0
        x = starts[used]
        time_fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.36, 0.40, 0.24],
            subplot_titles=["Quiet share of the week's examples",
                            "Share that does a selected activity in the week",
                            f"Share of the push quiet window ({window} days) inside a push outage"])
        time_fig.update_annotations(font=dict(size=13, color=INK2), xanchor="left", x=0)
        q_count = np.bincount(ex_j[users & quiet], minlength=n_weeks)
        time_fig.add_trace(go.Scatter(
            x=x, y=share_w[used], mode="lines+markers", line=dict(color=QUIET, width=2),
            marker=dict(size=7), showlegend=False, customdata=np.stack([q_count[used], n_w[used]], -1),
            hovertemplate="week of %{x|%Y-%m-%d}: %{y:.1%} quiet "
                          "(%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>"), row=1, col=1)
        for name, seg, color in [("Quiet", quiet, QUIET), ("Active", ~quiet, ACTIVE)]:
            r, c = weekly(lab & seg, ya)
            time_fig.add_trace(go.Scatter(
                x=x, y=r[used], mode="lines+markers", name=name, line=dict(color=color, width=2),
                marker=dict(size=7), customdata=c[used],
                hovertemplate=f"week of %{{x|%Y-%m-%d}}: %{{y:.1%}} of %{{customdata:,}} "
                              f"{name.lower()} examples<extra></extra>"), row=2, col=1)
        lost = np.array([push_window_lost(s, window) for s in x])
        time_fig.add_trace(go.Scatter(
            x=x, y=lost, mode="lines+markers", line=dict(color=COLORS["push"], width=2),
            marker=dict(size=7), showlegend=False,
            hovertemplate="week of %{x|%Y-%m-%d}: %{y:.0%} of the push window missing<extra></extra>"),
            row=3, col=1)
        for r in (1, 2, 3):
            add_context(time_fig, acts, row=r, col=1, test_label=r == 1)
            time_fig.add_vrect(x0=SAMPLING[0], x1=SAMPLING[1], fillcolor=SAMPLING_FILL, line_width=0,
                               layer="below", row=r, col=1)
        for (a, b), text in [(SAMPLING, "cohort sampled"), (SUMMER_OUTAGE, "summer outage")]:
            time_fig.add_annotation(x=a + (b - a) / 2, y=1, yref="paper", text=text, showarrow=False,
                                    yanchor="bottom", font=dict(size=11, color=MUTED))
        row2_top = time_fig.layout.yaxis2.domain[1]  # legend beside the title of the panel it keys
        time_fig.update_layout(height=760, hovermode="x", margin=dict(t=72),
                               legend=dict(y=row2_top, yanchor="bottom", x=1, xanchor="right"))
        time_fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        time_fig.update_xaxes(range=[max(start, MODEL_START - DAY), end])

        # Label rate per activity and segment
        rates_fig = go.Figure()
        for name, seg, color in [("Quiet", quiet, QUIET), ("Active", ~quiet, ACTIVE)]:
            stats = [rate(Y[k], m & seg) for k in acts]
            rates_fig.add_trace(go.Bar(
                x=[LABELS[k] for k in acts], y=[r for r, _ in stats], name=name, marker_color=color,
                text=[share_text(r) for r, _ in stats] if name == "Quiet" else None,
                textposition="outside", textfont=dict(color=INK2, size=12), cliponaxis=False,
                customdata=[c for _, c in stats],
                hovertemplate=f"{name}: %{{y:.1%}} of %{{customdata:,}} examples<extra>%{{x}}</extra>"))
        rates_fig.update_layout(title="Label rate: quiet vs active examples", barmode="group",
                                bargroupgap=0.08, barcornerradius=4, height=420)
        rates_fig.update_yaxes(tickformat=".0%", rangemode="tozero")

        # Share of the positives held by quiet examples
        held, totals = [], []
        for k in acts:
            p = (Y[k] == 1) & m
            held.append((p & quiet).sum() / max(p.sum(), 1))
            totals.append(int(p.sum()))
        ref = quiet[m].mean()
        share_fig = go.Figure(go.Bar(
            y=[LABELS[k] for k in acts], x=held, orientation="h", marker_color=QUIET,
            text=[pct(v) for v in held], textposition="outside", textfont=dict(color=INK2, size=12),
            cliponaxis=False, customdata=totals,
            hovertemplate="%{x:.1%} of %{customdata:,} positives<extra>%{y}</extra>"))
        share_fig.add_vline(x=ref, line=dict(color=INK2, width=1.5, dash="dash"))
        share_fig.add_annotation(x=ref, y=1, yref="paper", yanchor="bottom", xanchor="right",
                                 showarrow=False, text=f"quiet share of examples: {pct(ref)} ",
                                 font=dict(size=11, color=INK2))
        share_fig.update_layout(title="Share of the positive labels held by quiet examples",
                                height=420, showlegend=False, barcornerradius=4)
        share_fig.update_yaxes(autorange="reversed", showgrid=False)
        share_fig.update_xaxes(tickformat=".0%", range=[0, max(max(held), ref) * 1.3])

        # Hidden activity between the cutoffs and the week start
        known = m & (blind >= 0)
        groups = [("Active examples", known & ~quiet, ACTIVE), ("Quiet examples", known & quiet, QUIET),
                  ("Quiet, active after the cutoff", known & quiet & (blind > 0), QUIET),
                  ("Quiet, silent until the week starts", known & quiet & (blind == 0), QUIET)]
        n_quiet = max((known & quiet).sum(), 1)
        stats = [rate(ya, g) for _, g, _ in groups]
        names = [name if i < 2 else f"{name} ({pct(g.sum() / n_quiet)} of quiet)"
                 for i, (name, g, _) in enumerate(groups)]
        if any(c for _, c in stats):
            blind_fig = go.Figure(go.Bar(
                y=names, x=[r for r, _ in stats], orientation="h",
                marker_color=[color for *_, color in groups], text=[share_text(r) for r, _ in stats],
                textposition="outside", textfont=dict(color=INK2, size=12), cliponaxis=False,
                customdata=[c for _, c in stats],
                hovertemplate="%{x:.1%} of %{customdata:,} examples<extra>%{y}</extra>"))
            blind_fig.update_layout(title="Share that does a selected activity in the week",
                                    height=420, showlegend=False, barcornerradius=4)
            blind_fig.update_yaxes(autorange="reversed", showgrid=False)
            blind_fig.update_xaxes(tickformat=".0%", range=[0, np.nanmax([r for r, _ in stats]) * 1.25])
        else:
            blind_fig = empty_fig("No selected week has data up to its start", height=420)

        # Activity of the latest visible row
        last_fig = go.Figure()
        for name, seg, color in [("Quiet", quiet, QUIET), ("Active", ~quiet, ACTIVE)]:
            sel = m & seg & (last_kind >= 0)
            share = np.bincount(last_kind[sel], minlength=len(EVENT_COLS)) / max(sel.sum(), 1)
            last_fig.add_trace(go.Bar(
                y=[LABELS[k] for k in EVENT_COLS], x=share, orientation="h", name=name,
                marker_color=color, text=[pct(v) for v in share] if name == "Quiet" else None,
                textposition="outside", textfont=dict(color=INK2, size=12), cliponaxis=False,
                hovertemplate=f"{name}: %{{x:.1%}}<extra>%{{y}}</extra>"))
        last_fig.update_layout(title="Activity of the latest visible row", barmode="group",
                               bargroupgap=0.08, barcornerradius=4, height=420)
        last_fig.update_yaxes(autorange="reversed", showgrid=False)
        last_fig.update_xaxes(tickformat=".0%", rangemode="tozero")

        # How often each user is quiet
        n_u = np.bincount(ex_u[m], minlength=D.n)
        q_u = np.bincount(ex_u[m & quiet], minlength=D.n)
        ok = n_u >= 5
        if ok.any():
            s = q_u[ok] / n_u[ok]
            codes = np.where(s == 0, 0, np.where(s == 1, 11, np.minimum(np.ceil(s * 10), 10))).astype(int)
            labels = ["never"] + [f"{10 * i}-{10 * (i + 1)}%" for i in range(10)] + ["always"]
            users_fig = go.Figure(go.Bar(
                x=labels, y=np.bincount(codes, minlength=12), marker_color=QUIET,
                hovertemplate="quiet in %{x} of their examples: %{y:,} users<extra></extra>"))
            users_fig.update_layout(title=f"How often each user is quiet ({ok.sum():,} users)",
                                    height=420, barcornerradius=4)
            users_fig.update_xaxes(title="share of the user's examples that are quiet", type="category")
            users_fig.update_yaxes(title="users")
        else:
            users_fig = empty_fig("No user has 5 examples in the selection", height=420)

        # Recency of the test week's quiet users vs the selected weeks'
        edges, labels = SINCE_BINS
        codes = np.digitize(since, edges[1:])
        t_hist = np.bincount(codes[tm & quiet], minlength=len(labels))
        h_hist = np.bincount(codes[m & quiet], minlength=len(labels))
        if t_hist.sum() and h_hist.sum():
            lab_q = m & quiet & ~np.isnan(ya)
            hc = np.bincount(codes[lab_q], minlength=len(labels))
            hit = np.bincount(codes[lab_q], weights=ya[lab_q], minlength=len(labels))
            w = t_hist * (hc >= MIN_GROUP)
            est = (w * hit / np.maximum(hc, 1)).sum() / w.sum() if w.sum() else np.nan
            keep = (t_hist + h_hist) > 0
            test_fig = go.Figure()
            for name, hist, color in [("Test week", t_hist, QUIET), ("Selected weeks", h_hist, ACTIVE)]:
                test_fig.add_trace(go.Bar(
                    x=np.array(labels)[keep], y=(hist / hist.sum())[keep], name=name, marker_color=color,
                    customdata=hist[keep],
                    hovertemplate=f"{name}: %{{y:.1%}} (%{{customdata:,}} quiet examples)"
                                  "<extra>%{x} days</extra>"))
            test_fig.update_layout(
                title=dict(text="Days since the latest visible row, quiet examples", subtitle=dict(
                    text=f"At the test week's mix, the selected weeks' return rates give "
                         f"{share_text(est)} active (observed in those weeks: "
                         f"{share_text(rate(ya, m & quiet)[0])})",
                    font=dict(size=12, color=INK2))),
                barmode="group", bargroupgap=0.08, barcornerradius=4, height=420, margin=dict(t=110))
            test_fig.update_xaxes(title="days", type="category")
            test_fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        else:
            test_fig = empty_fig("No quiet test users in this population", height=420)
        return (tiles, time_fig, rates_fig, share_fig, blind_fig, last_fig, users_fig, test_fig)

    @app.callback(Output("qt-drivers", "figure"), *common, Input("qt-window", "value"),
                  Input("qt-min", "value"), Input("qt-drop", "value"), Input("qt-driver", "value"))
    def quiet_drivers(acts, pop, start, end, window, min_rows, drop, driver):
        acts, pop, start, end = filters(acts, pop, start, end)
        if not acts:
            return empty_fig("Select at least one activity")
        quiet, run, _ = segment(window, min_rows)
        m = quiet_rows(pop, start, end, drop) & quiet
        keys = ["any"] + list(acts)
        cols = min(4, len(keys))
        rows = -(-len(keys) // cols)
        fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.06,
                            vertical_spacing=(0.5 if driver == "last" else 0.36) / rows,
                            subplot_titles=["Any selected activity"] + [LABELS[k] for k in acts])
        fig.update_annotations(font=dict(size=13, color=INK2))
        ya = any_label(acts)
        panels = []
        for key in keys:
            y = ya if key == "any" else Y[key]
            codes, labels = driver_groups(driver, key, run)
            sel = m & ~np.isnan(y)
            n = np.bincount(codes[sel], minlength=len(labels))
            hit = np.bincount(codes[sel], weights=y[sel], minlength=len(labels))
            panels.append((key, np.array(labels), n, hit))
        # Panels with the same groups show the same ones: those large enough in at least one of
        # them; a group too small in a panel is a gap there.
        shown_by = {}
        for _, labels, n, _ in panels:
            key = tuple(labels)
            shown_by[key] = shown_by.get(key, np.zeros(len(labels), bool)) | (n >= MIN_GROUP)
        for i, (key, labels, n, hit) in enumerate(panels):
            r, c = i // cols + 1, i % cols + 1
            shown = shown_by[tuple(labels)]
            many = shown.sum() > 5
            rates = np.where(n >= MIN_GROUP, hit / np.maximum(n, 1), np.nan)[shown]
            labels, n = labels[shown], n[shown]
            color = INK2 if key == "any" else COLORS[key]
            hover = "%{x}: %{y:.1%} (n=%{customdata:,})<extra>" + (
                "any selected" if key == "any" else LABELS[key]) + "</extra>"
            if driver == "last":
                fig.add_trace(go.Bar(x=labels, y=rates, marker_color=color, customdata=n,
                                     hovertemplate=hover, showlegend=False), row=r, col=c)
            else:
                fig.add_trace(go.Scatter(x=labels, y=rates, mode="lines+markers", customdata=n,
                                         line=dict(color=color, width=2), marker=dict(size=8),
                                         hovertemplate=hover, showlegend=False), row=r, col=c)
            top = np.nanmax(rates) if np.isfinite(rates).any() else 0.05
            if key != "any":
                fig.add_hline(y=REF_F1[key] / 2, line=dict(color=MUTED, width=1.5, dash="dash"),
                              row=r, col=c)
                top = max(top, REF_F1[key] / 2)
            fig.update_yaxes(range=[0, top * 1.15], tickformat=".0%", row=r, col=c)
            fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels,
                             tickangle=-40 if driver == "last" or many else 0, row=r, col=c)
        fig.update_layout(title=f"Share of quiet examples active in the week, by "
                                f"{QUIET_DRIVERS[driver][0].lower() + QUIET_DRIVERS[driver][1:]}",
                          height=(330 if driver == "last" else 270) * rows + 90, margin=dict(t=90),
                          barcornerradius=4)
        return fig

    # --- Users ---
    @app.callback(Output("us-reach", "figure"), Output("us-lorenz", "figure"),
                  Output("us-hist", "figure"), *common, Input("us-act", "value"))
    def users_tab(acts, pop, start, end, act):
        acts, pop, start, end = filters(acts, pop, start, end)
        if not acts:
            return (empty_fig("Select at least one activity"),) * 3
        e = events(acts, pop, start, end)
        n_users = int(D.user_mask(pop).sum())
        per_user = {k: np.bincount(e.uid.values[(e.activity == k).values], minlength=D.n)[D.user_mask(pop)]
                    for k in acts}

        reach = [(per_user[k] > 0).mean() for k in acts]
        fig1 = go.Figure(go.Bar(
            x=[LABELS[k] for k in acts], y=reach, marker=dict(color=[COLORS[k] for k in acts],
                                                              cornerradius=4),
            text=[pct(r) for r in reach], textposition="outside",
            hovertemplate="%{x}: %{y:.1%} of " + f"{n_users:,} users<extra></extra>"))
        fig1.update_layout(title="Users with at least one event in the date range", height=400)
        fig1.update_yaxes(tickformat=".0%", range=[0, 1.08])

        fig2 = go.Figure()
        for k in acts:
            c = np.sort(per_user[k])[::-1]
            if c.sum() == 0:
                continue
            x = np.arange(1, len(c) + 1) / len(c)
            y = np.cumsum(c) / c.sum()
            idx = np.unique(np.linspace(0, len(c) - 1, 400).astype(int))
            fig2.add_trace(go.Scatter(
                x=x[idx], y=y[idx], mode="lines", name=LABELS[k], line=dict(color=COLORS[k], width=2),
                hovertemplate="top %{x:.1%} of users make %{y:.1%} of events<extra>"
                              + LABELS[k] + "</extra>"))
        fig2.update_layout(title="Concentration: share of events from the most active users",
                           height=400)
        ticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
        fig2.update_xaxes(title="top share of users (log scale)", type="log", tickvals=ticks,
                          ticktext=[f"{100 * t:g}%" for t in ticks])
        fig2.update_yaxes(title="share of events", tickformat=".0%", range=[0, 1.02])

        if act not in acts:
            return fig1, fig2, empty_fig(f"{LABELS[act]} is filtered out")
        labels, hist = log_bins(per_user[act])
        fig3 = go.Figure(go.Bar(
            x=labels, y=hist, marker=dict(color=COLORS[act], cornerradius=4),
            customdata=hist / max(hist.sum(), 1),
            hovertemplate="%{x} events: %{y:,} users (%{customdata:.1%})<extra></extra>"))
        fig3.update_layout(title=f"{LABELS[act]}: events per user in the date range", height=400)
        fig3.update_xaxes(title="events per user", type="category")
        fig3.update_yaxes(title="users")
        return fig1, fig2, fig3

    # --- Time of day ---
    @app.callback(Output("tod-heat", "figure"), Output("tod-hour", "figure"),
                  *common, Input("tod-act", "value"), Input("tod-tz", "value"))
    def time_of_day(acts, pop, start, end, act, tz):
        acts, pop, start, end = filters(acts, pop, start, end)
        if not acts:
            return (empty_fig("Select at least one activity"),) * 2
        e = events(acts, pop, start, end)
        col, tzname = f"how_{tz}", "UTC" if tz == "utc" else "Taiwan time"
        sel = e if act == "all" else e[(e.activity == act).values]
        grid = np.bincount(sel[col].values, minlength=168).reshape(7, 24)
        what = "selected activities" if act == "all" else LABELS[act]
        heat = go.Figure(go.Heatmap(
            z=grid, x=[f"{h:02d}" for h in range(24)], y=WEEKDAYS, xgap=2, ygap=2,
            colorscale=SEQUENTIAL, colorbar=dict(outlinewidth=0, thickness=12, title="events"),
            hovertemplate="%{y} %{x}:00: %{z:,} events<extra></extra>"))
        heat.update_layout(title=f"When do users act? {what}, {tzname}", height=380)
        heat.update_yaxes(autorange="reversed", showgrid=False, showline=False)
        heat.update_xaxes(title=f"hour ({tzname})", showgrid=False, showline=False)

        hours = go.Figure()
        for k in acts:
            h = np.bincount(e[col].values[(e.activity == k).values] % 24, minlength=24)
            if h.sum() == 0:
                continue
            hours.add_trace(go.Scatter(
                x=list(range(24)), y=h / h.sum(), mode="lines", name=LABELS[k],
                line=dict(color=COLORS[k], width=2, shape="spline", smoothing=0.4),
                hovertemplate="%{x}:00: %{y:.1%} of events<extra>" + LABELS[k] + "</extra>"))
        hours.update_layout(title=f"Share of each activity by hour ({tzname})", height=380,
                            hovermode="x unified")
        hours.update_xaxes(title="hour", dtick=2)
        hours.update_yaxes(tickformat=".0%", rangemode="tozero")
        return heat, hours

    # --- Repositories ---
    repo_names = D.repos.set_index("repo_id").name
    repo_is_org = D.repos.set_index("repo_id").org_id.notna()

    @app.callback(Output("rp-top", "figure"), Output("rp-per-user", "figure"),
                  Output("rp-org", "figure"), *common, Input("rp-act", "value"),
                  Input("rp-rank", "value"))
    def repos_tab(acts, pop, start, end, act, rank):
        acts, pop, start, end = filters(acts, pop, start, end)
        if not acts:
            return (empty_fig("Select at least one activity"),) * 3
        e = events(acts, pop, start, end)
        e = e[e.repo.notna()]

        org = go.Figure()
        share = [repo_is_org.reindex(e.repo.values[(e.activity == k).values].astype(int)).mean()
                 for k in acts]
        org.add_trace(go.Bar(
            x=[LABELS[k] for k in acts], y=share, marker=dict(color=[COLORS[k] for k in acts],
                                                              cornerradius=4),
            text=[pct(s) if s == s else "" for s in share], textposition="outside",
            hovertemplate="%{x}: %{y:.1%} of events in organisation repos<extra></extra>"))
        org.update_layout(title="Share of events in organisation-owned repositories", height=400)
        org.update_yaxes(tickformat=".0%", range=[0, 1.08])

        if act not in acts:
            msg = empty_fig(f"{LABELS[act]} is filtered out")
            return msg, msg, org
        a = e[(e.activity == act).values]
        g = a.groupby("repo").agg(events=("uid", "size"), users=("uid", "nunique"))
        top = g.sort_values([rank, "events" if rank == "users" else "users"], ascending=False).head(20)
        top = top.iloc[::-1]
        names = repo_names.reindex(top.index.astype(int)).fillna("?").values
        fig = go.Figure(go.Bar(
            y=names, x=top[rank], orientation="h", marker=dict(color=COLORS[act], cornerradius=4),
            customdata=top[["events", "users"]].values,
            hovertemplate="%{y}<br>%{customdata[0]:,} events by %{customdata[1]:,} cohort users"
                          "<extra></extra>"))
        fig.update_layout(title=f"{LABELS[act]}: top 20 repositories by "
                                f"{'distinct cohort users' if rank == 'users' else 'events'}",
                          height=620, margin=dict(l=280))
        fig.update_yaxes(showgrid=False)

        per_user = a.groupby("uid").repo.nunique().reindex(
            np.flatnonzero(D.user_mask(pop)), fill_value=0).values
        labels, hist = log_bins(per_user, edges=(0, 1, 2, 3, 4, 6, 11, 21, 51))
        pu = go.Figure(go.Bar(
            x=labels, y=hist, marker=dict(color=COLORS[act], cornerradius=4),
            hovertemplate="%{x} repositories: %{y:,} users<extra></extra>"))
        pu.update_layout(title=f"{LABELS[act]}: distinct repositories per user", height=400)
        pu.update_xaxes(title="repositories", type="category")
        pu.update_yaxes(title="users")
        return fig, pu, org

    # --- Data quality (static) ---
    @app.callback(Output("dq-lag", "figure"), Output("dq-delay", "figure"),
                  Output("dq-nulls", "figure"), Input("tabs", "value"))
    def quality(tab):
        if tab != "quality":
            return (go.Figure(),) * 3
        lag = [(TEST_START - D.data_end[k]) / DAY for k in EVENT_COLS]
        fig1 = go.Figure(go.Bar(
            x=[LABELS[k] for k in EVENT_COLS], y=lag,
            marker=dict(color=[COLORS[k] for k in EVENT_COLS], cornerradius=4),
            text=[f"{v:.1f} d" for v in lag], textposition="outside",
            customdata=[f"{D.data_end[k]:%Y-%m-%d %H:%M}" for k in EVENT_COLS],
            hovertemplate="%{x}: data ends %{customdata} UTC, %{y:.1f} days before the test week"
                          "<extra></extra>"))
        fig1.update_layout(title="Gap between the end of the data and the test week", height=420)
        fig1.update_yaxes(title="days")

        fig2 = go.Figure()
        grid = np.geomspace(1 / 1440, 365, 300)
        for k in ["open_pr", "open_issue", "pr_review"]:
            d = ev[(ev.activity == k).values]
            delay = np.sort(((d.seen - d.t) / DAY).clip(lower=0).values)
            y = np.searchsorted(delay, grid, side="right") / len(delay)
            fig2.add_trace(go.Scatter(
                x=grid, y=y, mode="lines", name=LABELS[k], line=dict(color=COLORS[k], width=2),
                hovertemplate="%{y:.1%} appear within %{x:.3g} days<extra>" + LABELS[k] + "</extra>"))
        fig2.update_layout(title="How late rows appear: first_seen_at − event time", height=420,
                           hovermode="x unified")
        ticks = [1 / 1440, 1 / 24, 1, 7, 30, 365]
        fig2.update_xaxes(title="delay (log scale)", type="log", tickvals=ticks,
                          ticktext=["1 min", "1 h", "1 d", "1 w", "30 d", "1 y"])
        fig2.update_yaxes(title="share of cohort rows", tickformat=".0%")

        cols, rows, z = [], [], []
        for name in TABLES:
            s = table_summary(name).set_index("column")["null %"]
            rows.append(name)
            z.append(s)
        z = pd.DataFrame(z, index=rows)
        z = z.loc[:, z.max() > 0]
        fig3 = go.Figure(go.Heatmap(
            z=z.values / 100, x=z.columns, y=z.index, colorscale=SEQUENTIAL, zmin=0, zmax=1,
            xgap=2, ygap=2, hoverongaps=False, colorbar=dict(tickformat=".0%", outlinewidth=0, thickness=12),
            hovertemplate="%{y}.%{x}: %{z:.1%} null<extra></extra>"))
        fig3.update_layout(title="Null share of columns that have missing values", height=440,
                           margin=dict(b=140))
        fig3.update_yaxes(autorange="reversed", showgrid=False, showline=False)
        fig3.update_xaxes(showgrid=False, showline=False, tickangle=-45)
        return fig1, fig2, fig3

    # --- Table explorer ---
    @app.callback(Output("tx-col", "options"), Output("tx-col", "value"),
                  Output("tx-summary", "children"), Input("tx-table", "value"))
    def table_columns(name):
        s = table_summary(name)
        df = load_table(name)
        table = dash_table.DataTable(
            data=s.to_dict("records"), columns=[{"name": c, "id": c} for c in s.columns],
            sort_action="native", style_as_list_view=True, page_size=25,
            style_cell={"fontFamily": FONT, "fontSize": 13, "padding": "6px 10px", "textAlign": "left",
                        "backgroundColor": SURFACE, "color": INK, "maxWidth": "420px",
                        "overflow": "hidden", "textOverflow": "ellipsis"},
            style_header={"fontWeight": 600, "color": INK2, "borderBottom": f"1px solid {AXIS}"})
        info = html.P(f"{name}: {len(df):,} rows, {len(df.columns)} columns", className="note")
        default = next((c for c in df.columns if df[c].dtype.kind == "M"), df.columns[0])
        return list(df.columns), default, [info, table]

    @app.callback(Output("tx-fig", "figure"), Input("tx-table", "value"), Input("tx-col", "value"))
    def table_column(name, col):
        df = load_table(name)
        if col not in df.columns:
            return empty_fig("Pick a column")
        s = df[col].dropna()
        color = COLORS["open_pr"]
        if not len(s):
            return empty_fig(f"{col} is always null")
        title = f"{name}.{col}"
        if s.dtype.kind == "M":
            c = s.dt.floor("D").value_counts().sort_index()
            fig = go.Figure(go.Scatter(x=c.index, y=c.values, mode="lines", line=dict(color=color, width=2),
                                       hovertemplate="%{x|%Y-%m-%d}: %{y:,} rows<extra></extra>"))
            fig.update_layout(title=f"{title}: rows per day", hovermode="x")
            if s.min() < DATA_START:
                fig.update_xaxes(range=[DATA_START, TEST_START + WEEK])
                title += " (axis starts at the data window; older values exist)"
                fig.update_layout(title=f"{title}: rows per day")
        elif s.dtype.kind in "iuf" and s.nunique() > 30:
            v = s.to_numpy(dtype=float)
            hi = np.quantile(v, 0.995)
            counts, edges = np.histogram(np.minimum(v, hi), bins=60)
            fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges) * 0.9,
                                   marker=dict(color=color, cornerradius=2),
                                   hovertemplate="≈%{x:,.1f}: %{y:,} rows<extra></extra>"))
            fig.update_layout(title=f"{title}: distribution (values above the 99.5th percentile, "
                                    f"{hi:,.0f}, are clipped)")
        elif s.dtype.kind not in "biuf" and s.nunique() > 200:
            lens = s.astype(str).str.len()
            counts, edges = np.histogram(lens.clip(upper=lens.quantile(0.995)), bins=60)
            fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges) * 0.9,
                                   marker=dict(color=color, cornerradius=2),
                                   hovertemplate="≈%{x:,.0f} chars: %{y:,} rows<extra></extra>"))
            fig.update_layout(title=f"{title}: text length ({s.nunique():,} distinct values)")
            fig.update_xaxes(title="characters")
        else:
            c = s.astype(str).value_counts().head(25).iloc[::-1]
            fig = go.Figure(go.Bar(y=c.index, x=c.values, orientation="h",
                                   marker=dict(color=color, cornerradius=4),
                                   customdata=c.values / len(df),
                                   hovertemplate="%{y}: %{x:,} rows (%{customdata:.1%})<extra></extra>"))
            fig.update_layout(title=f"{title}: most common values (of {len(df):,} rows, "
                                    f"{df[col].isna().mean():.1%} null)",
                              height=max(320, 26 * len(c) + 120), margin=dict(l=220))
            fig.update_yaxes(showgrid=False, type="category")
        return fig

    # --- User explorer ---
    @app.callback(Output("ux-heat", "figure"), Output("ux-repos", "figure"), Output("ux-info", "children"),
                  Input("ux-user", "value"))
    def user_view(uid):
        if uid is None:
            return empty_fig("Pick a user"), empty_fig(""), ""
        u = D.cohort.iloc[uid]
        z = np.stack([D.counts[k][uid] for k in EVENT_COLS]).astype(float)
        for i, k in enumerate(EVENT_COLS):
            z[i, ~D.valid[k]] = np.nan
        heat = go.Figure(go.Heatmap(
            z=np.log1p(z), x=D.week_starts, y=[LABELS[k] for k in EVENT_COLS], customdata=z,
            colorscale=SEQUENTIAL, xgap=2, ygap=2, hoverongaps=False, showscale=False,
            hovertemplate="week of %{x|%Y-%m-%d}<br>%{y}: %{customdata:,.0f} events<extra></extra>"))
        heat.update_layout(title=f"{u.login}: events per week (darker = more, log scale)", height=360)
        heat.update_yaxes(autorange="reversed", showgrid=False, showline=False)
        heat.update_xaxes(showgrid=False, showline=False)

        e = ev[ev.uid.values == uid]
        g = e[e.repo.notna()].groupby(["repo", "activity"], observed=True).size().unstack(fill_value=0)
        if g.empty:
            repos = empty_fig("No repository activity")
        else:
            g = g.loc[g.sum(1).sort_values(ascending=False).index[:15]].iloc[::-1]
            names = repo_names.reindex(g.index.astype(int)).fillna("?").values
            repos = go.Figure()
            for k in [k for k in EVENT_COLS if k in g.columns]:
                repos.add_trace(go.Bar(y=names, x=g[k], orientation="h", name=LABELS[k],
                                       marker=dict(color=COLORS[k], line=dict(color=SURFACE, width=2)),
                                       hovertemplate="%{y}: %{x:,} events<extra>" + LABELS[k] + "</extra>"))
            repos.update_layout(barmode="stack", title=f"{u.login}: top repositories",
                                height=max(220, 32 * len(g) + 150), margin=dict(l=280),
)
            repos.update_yaxes(showgrid=False)
        last = f"{e.t.max():%Y-%m-%d %H:%M} UTC" if len(e) else "never"
        elig = D._eligible_at(TEST_START)[uid]
        info = (f"GitHub id {u.github_user_id} · {'bot · ' if u.is_bot else ''}"
                f"{'in the test set' if u.is_test else 'not in the test set'} · "
                f"{len(e):,} events · last event {last} · "
                f"passes the selection rule at the test week: {'yes' if elig else 'no'}")
        return heat, repos, info

    return app


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body { margin: 0; background: """ + PAGE + """; color: """ + INK + """; font-family: """ + FONT + """; }
  .page { max-width: 1280px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 24px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: """ + INK2 + """; margin: 0 0 16px; }
  .note { color: """ + INK2 + """; font-size: 13px; margin: 4px 0 8px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;
           margin-bottom: 16px; }
  .tile, .card { background: """ + SURFACE + """; border: 1px solid """ + BORDER + """;
                 border-radius: 8px; }
  .tile { padding: 12px 16px; }
  .tile-value { font-size: 22px; font-weight: 600; }
  .tile-label { color: """ + MUTED + """; font-size: 12px; margin-top: 2px; }
  .card { padding: 12px 16px; margin-top: 16px; }
  .filters, .row { display: flex; flex-wrap: wrap; gap: 16px 28px; align-items: flex-end; }
  .ctl { display: flex; flex-direction: column; gap: 4px; }
  .ctl-label { font-size: 12px; color: """ + MUTED + """; text-transform: uppercase;
               letter-spacing: .04em; }
  .ctl input[type=radio], .ctl input[type=checkbox] { margin-right: 5px; }
  .ctl label { margin-right: 14px; font-size: 14px; cursor: pointer; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 0 16px; }
  .tabs { margin-top: 20px; }
  @media (max-width: 560px) { .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app = build_app(Data())
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
