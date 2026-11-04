"""Interactive exploratory data analysis of the cohort activity data (Dash + Plotly).

Activities are defined exactly like the labels: one event per row of each table, at the time
the activity happened (`created_at`, `submitted_at` for reviews), for the 9,000 cohort users.
Weeks are aligned to the test week (Wednesday 16:00 UTC), as in experiments/baseline/run.py.

Run from the repo root:
    python eda.py                    # then open http://127.0.0.1:8050
    python eda.py --port 8051 --debug
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

ROOT = Path(__file__).resolve().parent
DB = ROOT / "assignment2" / "data" / "db"
EVAL_DIR = ROOT / "assignment2" / "evaluation"

sys.path.insert(0, str(EVAL_DIR))
from evaluate import EVENT_COLS  # noqa: E402

# Same constants as experiments/baseline/run.py.
TEST_START = pd.Timestamp("2026-11-11 16:00")      # 2026-11-12 00:00 Taiwan time
SELECTION_TIME = pd.Timestamp("2026-10-28 12:00")  # test users: >= 5 records in the 90 days before
DATA_START = pd.Timestamp("2025-12-01")
WEEK = pd.Timedelta(days=7)
DAY = pd.Timedelta(days=1)
MIN_RECORDS = 5
GAPS = {
    "nonpush": [("2026-06-01", "2026-09-07")],
    "push": [("2026-08-18", "2026-09-10"), ("2026-10-07", "2026-10-08"),
             ("2026-10-15", "2026-10-19"), ("2026-10-23", "2026-10-24")],
}

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
                              columns=["repo_id", "author_id", "created_at", "first_seen_at"])
        issues = pd.read_parquet(DB / "Issues.parquet",
                                 columns=["repo_id", "author_id", "is_pull_request", "created_at",
                                          "first_seen_at"])
        comments = pd.read_parquet(DB / "IssueComments.parquet",
                                   columns=["repo_id", "author_id", "created_at"])
        reviews = pd.read_parquet(DB / "PRReviews.parquet",
                                  columns=["repo_id", "reviewer_id", "submitted_at", "first_seen_at"])
        rcomments = pd.read_parquet(DB / "PRReviewComments.parquet",
                                    columns=["repo_id", "author_id", "created_at"])
        pushes = pd.read_parquet(DB / "Pushes.parquet", columns=["user_id", "repo_id", "created_at"])
        self.repos = pd.read_parquet(DB / "Repositories.parquet", columns=["repo_id", "name", "org_id"])

        cohort = users[users.is_cohort].reset_index(drop=True)
        test_ids = set(pd.read_csv(EVAL_DIR / "test_input.csv").UserID)
        cohort["is_test"] = cohort.github_user_id.isin(test_ids)
        self.cohort = cohort
        self.n = n = len(cohort)
        to_cidx = np.full(len(users), -1, np.int64)
        to_cidx[cohort.user_id.values] = np.arange(n)

        # activity -> (actor, time it happened, time the row appears in the data, repo)
        plain = issues[~issues.is_pull_request]
        sources = {
            "open_pr": (prs.author_id, prs.created_at, prs.first_seen_at, prs.repo_id),
            "open_issue": (plain.author_id, plain.created_at, plain.first_seen_at, plain.repo_id),
            "issue_comment": (comments.author_id, comments.created_at, comments.created_at,
                              comments.repo_id),
            "pr_review": (reviews.reviewer_id, reviews.submitted_at, reviews.first_seen_at,
                          reviews.repo_id),
            "pr_review_comment": (rcomments.author_id, rcomments.created_at, rcomments.created_at,
                                  rcomments.repo_id),
            "push": (pushes.user_id, pushes.created_at, pushes.created_at, pushes.repo_id),
        }
        parts, self.data_end = [], {}
        for k, (actor, t, seen, repo) in sources.items():
            self.data_end[k] = seen.max()
            d = pd.DataFrame({"uid": actor.to_numpy("float64", na_value=np.nan), "t": t.values,
                              "seen": seen.values, "repo": repo.to_numpy("float64", na_value=np.nan)})
            d = d[d.uid.notna() & d.t.notna() & (d.t >= DATA_START)]
            d["uid"] = to_cidx[d.uid.astype(np.int64).values]
            d = d[d.uid >= 0].assign(activity=k)
            parts.append(d)
        ev = pd.concat(parts, ignore_index=True).sort_values("t", ignore_index=True)
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

        # Test selection rule shifted to every week (>= 5 records in the 90 days before
        # s - 14d4h). Approximation: it ignores that some rows enter the data later.
        tv, uv = ev.t.values, ev.uid.values
        self.eligible = np.zeros((n, w), bool)
        for j, s in enumerate(self.week_starts):
            sel = s - (TEST_START - SELECTION_TIME)
            a, b = np.searchsorted(tv, [np.datetime64(sel - 90 * DAY), np.datetime64(sel)])
            self.eligible[:, j] = np.bincount(uv[a:b], minlength=n) >= MIN_RECORDS

        test_now = self._eligible_at(TEST_START)
        print(f"Loaded {len(ev):,} cohort events in {time.time() - t0:.1f}s. Selection rule at the "
              f"test week: {test_now.sum():,} users, {(test_now == cohort.is_test.values).mean():.2%} "
              f"agree with test_input.csv")

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
