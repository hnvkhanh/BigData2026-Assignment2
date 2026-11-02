# Global trend / action-transition explorer + user-group comparison
# (runs from the terminal, opens in browser)
# pip install pandas pyarrow plotly dash scipy   (scipy is optional: only for the significance tests)
#
# Usage (from the repository root):
#   python global_trends_dash.py
# then open http://127.0.0.1:8051

# ---------------------------------------------------------------- CONFIG
DATA_PATH = "assignment2/data/db"  # repository Parquet database, file, or glob
USER_COL = None       # e.g. "user_id"   (None = auto-detect)
TIME_COL = None       # e.g. "timestamp" (None = auto-detect)
ACTION_COL = None     # e.g. "action"    (None = auto-detect)
ITEM_COL = None       # unused here, kept for parity with the per-user explorer
TIME_UNIT = None      # "s" or "ms" if timestamps are numeric (None = guess)
PORT = 8051

# ---------------------------------------------------------------- LOAD
import glob
import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, dcc, html, dash_table

try:
    from scipy.stats import kruskal
except Exception:  # scipy missing: skip significance tests
    kruskal = None

GITHUB_ACTIVITY_TABLES = (
    ("PullRequests", "author_id", "created_at", "open_pr"),
    ("Issues", "author_id", "created_at", "open_issue"),
    ("IssueComments", "author_id", "created_at", "issue_comment"),
    ("PRReviews", "reviewer_id", "submitted_at", "pr_review"),
    ("PRReviewComments", "author_id", "created_at", "pr_review_comment"),
    ("Pushes", "user_id", "created_at", "push"),
)


def read_github_activity_db(db_dir):
    """Convert the assignment's activity tables into a normalized event table."""
    db_dir = Path(db_dir)
    required = ["Users.parquet"] + [
        f"{table}.parquet" for table, _, _, _ in GITHUB_ACTIVITY_TABLES
    ]
    missing = [name for name in required if not (db_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required GitHub activity table(s) in {db_dir}: "
            + ", ".join(missing)
        )

    users = pd.read_parquet(
        db_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"],
    )
    cohort_users = users.loc[users["is_cohort"]].set_index("user_id")[
        "github_user_id"
    ]
    events = []
    for table, actor_col, time_col, action in GITHUB_ACTIVITY_TABLES:
        columns = [actor_col, "repo_id", time_col]
        if table == "Issues":
            columns.append("is_pull_request")
        raw = pd.read_parquet(db_dir / f"{table}.parquet", columns=columns)
        if table == "Issues":
            raw = raw.loc[~raw["is_pull_request"].fillna(False)]
        rows = raw.loc[
            raw[actor_col].isin(cohort_users.index) & raw[time_col].notna()
        ]
        events.append(
            pd.DataFrame(
                {
                    "user_id": rows[actor_col].map(cohort_users),
                    "timestamp": rows[time_col],
                    "action": action,
                    "item_id": rows["repo_id"],
                }
            )
        )
    result = pd.concat(events, ignore_index=True)
    return result.dropna(subset=["user_id", "timestamp"])


def read_any(path):
    path = str(path)
    if Path(path).is_dir():
        return read_github_activity_db(path)

    files = sorted(glob.glob(path))
    if not files and Path(path).is_file():
        files = [path]
    if not files:
        raise FileNotFoundError(
            f"No input files matched DATA_PATH={path!r}. Set DATA_PATH to an "
            "existing file, glob, or assignment Parquet database directory."
        )

    dfs = []
    for f in files:
        suffix = Path(f).suffix.lower()
        if suffix == ".parquet":
            dfs.append(pd.read_parquet(f))
        elif suffix in (".json", ".jsonl"):
            dfs.append(pd.read_json(f, lines=suffix == ".jsonl"))
        else:
            dfs.append(pd.read_csv(f, sep=None, engine="python"))
    return pd.concat(dfs, ignore_index=True)


def guess(cols, keys):
    low = {c.lower(): c for c in cols}
    for k in keys:
        for lc, c in low.items():
            if k in lc:
                return c
    return None


df = read_any(DATA_PATH)
if df.empty:
    raise ValueError(f"No activity rows were loaded from DATA_PATH={DATA_PATH!r}.")
print("Columns:", list(df.columns), "| rows:", len(df))

USER_COL = USER_COL or guess(df.columns, ["user", "uid", "member", "customer"])
TIME_COL = TIME_COL or guess(df.columns, ["time", "date", "ts", "datetime"])
ACTION_COL = ACTION_COL or guess(df.columns, ["action", "event", "behavior", "type", "act"])
print(f"user={USER_COL}  time={TIME_COL}  action={ACTION_COL}")
if not USER_COL or not TIME_COL:
    raise ValueError("Could not identify user and time columns. Set them in CONFIG.")
missing_columns = [c for c in (USER_COL, TIME_COL) if c not in df]
if missing_columns:
    raise ValueError(f"Configured column(s) not found in the input: {missing_columns}")

t = df[TIME_COL]
if pd.api.types.is_numeric_dtype(t):
    unit = TIME_UNIT or ("ms" if t.median() > 1e11 else "s")
    df["_t"] = pd.to_datetime(t, unit=unit)
else:
    df["_t"] = pd.to_datetime(t, errors="coerce")
df = df.dropna(subset=["_t"])
if ACTION_COL is None:
    ACTION_COL = "_action"
    df[ACTION_COL] = "event"
df[ACTION_COL] = df[ACTION_COL].astype(str)

# keep only what we need; sort by user then time so consecutive rows are a user's sequence
df["_u"] = df[USER_COL].astype(str)
srt = (df[["_u", "_t", ACTION_COL]]
       .sort_values(["_u", "_t"]).reset_index(drop=True))
srt["_uc"] = pd.factorize(srt["_u"])[0]
actions_all = sorted(srt[ACTION_COL].unique())
K = len(actions_all)
CODE = {a: i for i, a in enumerate(actions_all)}
srt["_a"] = srt[ACTION_COL].map(CODE).astype(np.int32)
# users x actions count matrix (used for per-user distributions)
ua = (pd.crosstab(srt["_u"], srt[ACTION_COL])
        .reindex(columns=actions_all, fill_value=0))
print(f"{len(srt):,} events | {srt['_uc'].nunique():,} users | {K} actions: {actions_all}")

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B"]
CMAP = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(actions_all)}
BINS = ["1h", "6h", "1D", "1W", "1MS"]
GAPS = {"any": np.inf, "1h": 3600, "6h": 6 * 3600, "1D": 86400,
        "7D": 7 * 86400, "30D": 30 * 86400}
GAP_LABELS = {"any": "No limit", "1h": "within 1 hour", "6h": "within 6 hours",
              "1D": "within 1 day", "7D": "within 7 days", "30D": "within 30 days"}
METRICS = {"first": "Actions in first N days",
           "peak": "Actions in busiest N-day block",
           "total": "Total actions (whole period)"}
MIN_SUPPORT = 30   # min 'from' events in a group before a transition probability is shown
GROUP_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
NORMS = {"count": "Counts", "prob": "P(next | previous), row %",
         "lift": "Lift = P(next | previous) / P(next)"}

# ---------------------------------------------------------------- HELPERS


def empty_fig(msg):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", annotations=[
        dict(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(size=18))])
    return fig


def fmt_secs(s):
    if not np.isfinite(s):
        return ""
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.0f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


def bucket_labels(edges):
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = hi - 1
        labels.append(f"{int(lo)}" if hi <= lo else f"{int(lo)}-{int(hi)}")
    return labels


def ordered(acts):
    return [a for a in actions_all if a in set(acts or [])]


# ---------------------------------------------------------------- APP
app = Dash(__name__)
app.title = "Global action trends"

TBL_STYLE = dict(style_table={"overflowX": "auto"},
                 style_cell={"textAlign": "left", "fontSize": 13})

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "maxWidth": "1300px", "margin": "0 auto",
           "padding": "16px"},
    children=[
        html.H2("Global action trends"),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            html.Div(style={"width": "420px"}, children=[
                html.Label("Actions"),
                dcc.Dropdown(id="actions",
                             options=[{"label": a, "value": a} for a in actions_all],
                             value=actions_all, multi=True)]),
            html.Div(style={"width": "140px"}, children=[
                html.Label("Time bin (trend)"),
                dcc.Dropdown(id="bin", options=BINS, value="1D", clearable=False)]),
            html.Div(style={"width": "340px"}, children=[
                html.Label("Transition matrix shows"),
                dcc.Dropdown(id="norm",
                             options=[{"label": v, "value": k} for k, v in NORMS.items()],
                             value="prob", clearable=False)]),
            html.Div(style={"width": "200px"}, children=[
                html.Label("Next action happens"),
                dcc.Dropdown(id="gap",
                             options=[{"label": v, "value": k} for k, v in GAP_LABELS.items()],
                             value="any", clearable=False)]),
        ]),
        html.Pre(id="summary", style={"whiteSpace": "pre-wrap", "marginTop": "12px"}),
        dcc.Graph(id="overview"),
        html.H3("Which action leads to which?"),
        html.Div(id="note", style={"color": "#555", "marginBottom": "6px"}),
        dcc.Graph(id="trans"),
        html.H4("Top transitions (previous -> next)"),
        dash_table.DataTable(
            id="bigram", page_size=15, sort_action="native",
            columns=[{"name": c, "id": c} for c in
                     ["from", "to", "count", "P(to|from) %", "lift", "median time to next"]],
            **TBL_STYLE),
        html.H4("Top 3-step sequences"),
        dash_table.DataTable(
            id="trigram", page_size=15, sort_action="native",
            columns=[{"name": c, "id": c} for c in
                     ["step 1", "step 2", "step 3", "count", "% of 3-step sequences"]],
            **TBL_STYLE),

        # ---------------- user-group comparison
        html.Hr(),
        html.H3("Do light and heavy users behave differently?"),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            html.Div(style={"width": "300px"}, children=[
                html.Label("Split users by"),
                dcc.Dropdown(id="g-metric",
                             options=[{"label": v, "value": k} for k, v in METRICS.items()],
                             value="first", clearable=False)]),
            html.Div(style={"width": "140px"}, children=[
                html.Label("N (days)"),
                dcc.Dropdown(id="g-window",
                             options=[{"label": f"{w} day{'s' if w > 1 else ''}", "value": w}
                                      for w in (1, 7, 30)],
                             value=7, clearable=False)]),
            html.Div(style={"width": "160px"}, children=[
                html.Label("Number of groups"),
                dcc.Dropdown(id="g-q", options=[2, 3, 4, 5], value=3, clearable=False)]),
            html.Div(style={"width": "300px"}, children=[
                html.Label("Transition heatmaps show"),
                dcc.Dropdown(id="g-view",
                             options=[{"label": "P(next | previous) per group", "value": "prob"},
                                      {"label": "Difference vs all users (pp)", "value": "diff"}],
                             value="diff", clearable=False)]),
        ]),
        html.Div(id="g-note", style={"color": "#555", "margin": "8px 0", "whiteSpace": "pre-wrap"}),
        html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
            dcc.Graph(id="g-mix", style={"flex": "1 1 560px"}),
            dcc.Graph(id="g-pool", style={"flex": "1 1 560px"}),
        ]),
        html.H4("Action mix by group (per-user share of actions)"),
        dash_table.DataTable(id="g-stats", page_size=15, sort_action="native", **TBL_STYLE),
        html.H4("Transitions by group"),
        dcc.Graph(id="g-trans"),
        html.Div(id="g-tnote", style={"color": "#555", "marginBottom": "6px"}),
        html.H4("Transitions that differ most between groups"),
        dash_table.DataTable(
            id="g-tdiff", page_size=15, sort_action="native",
            columns=[{"name": c, "id": c} for c in
                     ["from", "to", "highest group", "lowest group", "spread (pp)",
                      "P highest %", "P lowest %"]],
            **TBL_STYLE),
    ],
)


@app.callback(
    Output("summary", "children"),
    Output("overview", "figure"),
    Input("actions", "value"),
    Input("bin", "value"),
)
def overview(acts, bin_):
    acts = ordered(acts)
    if not acts:
        return "Select at least one action.", empty_fig("Select at least one action")
    d = srt[srt[ACTION_COL].isin(acts)]
    if d.empty:
        return "No events.", empty_fig("No events")

    totals = ua[acts].sum(axis=1)
    totals = totals[totals > 0]
    vc = d[ACTION_COL].value_counts().reindex(acts).fillna(0)
    top = vc.sort_values(ascending=False)
    summary = (f"{len(d):,} events | {d['_uc'].nunique():,} users | "
               f"{d['_t'].min()} -> {d['_t'].max()}\n"
               f"Actions per user: median {totals.median():.0f}, mean {totals.mean():.1f}, "
               f"max {totals.max():.0f}\n"
               "Most common: " + ", ".join(f"{a} ({n / len(d):.1%})"
                                          for a, n in top.head(3).items()))

    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]],
        subplot_titles=("Events per action", "Distinct users per action",
                        "Events over time", "Actions per user (histogram)",
                        "Hour-of-day x day-of-week (all users)",
                        "Per-user action mix (share of each user's actions)"),
        row_heights=[0.2, 0.22, 0.25, 0.25], vertical_spacing=0.08)

    # 1. events per action
    fig.add_trace(go.Bar(
        x=vc.index, y=vc.values, marker_color=[CMAP[a] for a in vc.index],
        text=[f"{v / vc.sum():.1%}" for v in vc.values], textposition="outside",
        showlegend=False), row=1, col=1)

    # 2. distinct users per action
    nu = d.groupby(ACTION_COL)["_uc"].nunique().reindex(acts).fillna(0)
    nall = d["_uc"].nunique()
    fig.add_trace(go.Bar(
        x=nu.index, y=nu.values, marker_color=[CMAP[a] for a in nu.index],
        text=[f"{v / nall:.0%}" for v in nu.values], textposition="outside",
        showlegend=False), row=1, col=2)

    # 3. trend
    g = (d.groupby([ACTION_COL, pd.Grouper(key="_t", freq=bin_)]).size()
           .unstack(0).fillna(0))
    for a in acts:
        if a in g.columns:
            fig.add_trace(go.Bar(x=g.index, y=g[a], name=a, marker_color=CMAP[a]),
                          row=2, col=1)

    # 4. actions-per-user histogram (log-spaced buckets)
    mx = int(totals.max())
    edges = np.unique(np.round(np.logspace(0, np.log10(mx + 1), 20)).astype(int))
    if len(edges) < 2:
        edges = np.array([1, mx + 1])
    elif edges[-1] <= mx:
        edges = np.append(edges, mx + 1)
    h, _ = np.histogram(totals, bins=edges)
    fig.add_trace(go.Bar(x=bucket_labels(edges), y=h, marker_color="#4C78A8",
                         showlegend=False,
                         hovertemplate="%{x} actions: %{y} users<extra></extra>"),
                  row=3, col=1)

    # 5. heatmap
    hm = (d.assign(h=d["_t"].dt.hour, w=d["_t"].dt.dayofweek)
            .groupby(["w", "h"]).size().unstack(fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0))
    fig.add_trace(go.Heatmap(z=hm.values, x=list(range(24)),
                             y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                             colorscale="Viridis", showscale=False), row=3, col=2)

    # 6. per-user mix
    sub = ua.loc[totals.index, acts]
    share = sub.div(sub.sum(axis=1), axis=0)
    for a in acts:
        fig.add_trace(go.Box(y=share[a], name=a, marker_color=CMAP[a],
                             boxpoints=False, boxmean=True, showlegend=False),
                      row=4, col=1)

    fig.update_yaxes(title_text="events", row=1, col=1)
    fig.update_yaxes(title_text="users", row=1, col=2)
    fig.update_yaxes(title_text="share", tickformat=".0%", row=4, col=1)
    fig.update_xaxes(title_text="actions per user", row=3, col=1)
    fig.update_layout(barmode="stack", height=1450, template="plotly_white",
                      legend=dict(orientation="h", y=1.03))
    return summary, fig


@app.callback(
    Output("trans", "figure"),
    Output("note", "children"),
    Output("bigram", "data"),
    Output("trigram", "data"),
    Input("actions", "value"),
    Input("norm", "value"),
    Input("gap", "value"),
)
def transitions(acts, norm, gap_key):
    acts = ordered(acts)
    if not acts:
        return empty_fig("Select at least one action"), "", [], []
    d = srt[srt[ACTION_COL].isin(acts)]
    if len(d) < 2:
        return empty_fig("Not enough events"), "", [], []

    a = d["_a"].to_numpy()
    u = d["_uc"].to_numpy()
    tt = d["_t"].to_numpy()
    same = u[1:] == u[:-1]
    gap = (tt[1:] - tt[:-1]) / np.timedelta64(1, "s")
    ok = same & (gap <= GAPS[gap_key])
    if ok.sum() == 0:
        return empty_fig("No transitions with this filter"), "", [], []

    a0, a1, g_ok = a[:-1][ok], a[1:][ok], gap[ok]
    pair = a0.astype(np.int64) * K + a1
    M = np.bincount(pair, minlength=K * K).reshape(K, K).astype(float)
    med = pd.Series(g_ok).groupby(pair).median()
    G = np.full(K * K, np.nan)
    G[med.index.to_numpy()] = med.to_numpy()
    G = G.reshape(K, K)

    idx = [CODE[x] for x in acts]
    C = M[np.ix_(idx, idx)]
    Gs = G[np.ix_(idx, idx)]
    rs = C.sum(axis=1, keepdims=True)
    cs = C.sum(axis=0, keepdims=True) / C.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(rs > 0, C / rs, np.nan)
        L = np.where(cs > 0, P / cs, np.nan)

    if norm == "count":
        z, text = C, [[f"{v:,.0f}" for v in r] for r in C]
        cs_kw = dict(colorscale="Blues", colorbar=dict(title="count"))
    elif norm == "prob":
        z, text = P * 100, [["" if np.isnan(v) else f"{v * 100:.1f}%" for v in r] for r in P]
        cs_kw = dict(colorscale="Blues", zmin=0, zmax=100, colorbar=dict(title="%"))
    else:
        z = L
        text = [["" if np.isnan(v) else f"{v:.2f}x" for v in r] for r in L]
        cs_kw = dict(colorscale="RdBu", zmin=0, zmax=2,
                     colorbar=dict(title="lift (clipped at 2)"))

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.18,
        subplot_titles=(NORMS[norm], "Median time from previous to next action"))
    fig.add_trace(go.Heatmap(
        z=z, x=acts, y=acts, text=text, texttemplate="%{text}",
        hovertemplate="%{y} -> %{x}<br>%{text}<extra></extra>", **cs_kw,
        colorbar_x=0.43), row=1, col=1)
    zg = np.log10(np.maximum(Gs, 1))
    tg = [[fmt_secs(v) for v in r] for r in Gs]
    fig.add_trace(go.Heatmap(
        z=zg, x=acts, y=acts, text=tg, texttemplate="%{text}",
        colorscale="YlOrRd", showscale=False,
        hovertemplate="%{y} -> %{x}<br>median %{text}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(autorange="reversed", title_text="previous action")
    fig.update_xaxes(title_text="next action")
    fig.update_layout(template="plotly_white", height=520 + 25 * len(acts))

    limit = GAP_LABELS[gap_key]
    note = (f"{int(ok.sum()):,} of {int(same.sum()):,} consecutive same-user event pairs "
            f"used ({limit}). Rows = what the user did, columns = what they did next.")

    # bigram table
    rows = []
    for i, fa in enumerate(acts):
        for j, ta in enumerate(acts):
            if C[i, j] > 0:
                rows.append({"from": fa, "to": ta, "count": int(C[i, j]),
                             "P(to|from) %": round(float(P[i, j] * 100), 2),
                             "lift": None if np.isnan(L[i, j]) else round(float(L[i, j]), 2),
                             "median time to next": fmt_secs(Gs[i, j])})
    rows.sort(key=lambda r: -r["count"])

    # trigram table
    tri = []
    if len(a) >= 3:
        ok2 = ok[:-1] & ok[1:]
        if ok2.any():
            codes = (a[:-2].astype(np.int64) * K + a[1:-1]) * K + a[2:]
            vc = pd.Series(codes[ok2]).value_counts()
            total = int(ok2.sum())
            for c, n in vc.head(25).items():
                c0, rem = divmod(int(c), K * K)
                c1, c2 = divmod(rem, K)
                tri.append({"step 1": actions_all[c0], "step 2": actions_all[c1],
                            "step 3": actions_all[c2], "count": int(n),
                            "% of 3-step sequences": round(100 * n / total, 2)})
    return fig, note, rows, tri


# ---------------------------------------------------------------- USER GROUPS


def metric_label(metric, window):
    return {"first": f"actions in first {window} days",
            "peak": f"actions in busiest {window}-day block",
            "total": "total actions"}[metric]


@lru_cache(maxsize=16)
def user_groups(acts_t, metric, window, q):
    """Per-user activity metric and quantile group (0 = lightest) for the chosen actions."""
    d = srt[srt[ACTION_COL].isin(acts_t)]
    if metric == "total":
        m = d.groupby("_uc").size()
    else:
        first = d.groupby("_uc")["_t"].transform("min")
        blk = np.floor((d["_t"] - first) / np.timedelta64(window, "D")).astype(np.int64)
        if metric == "first":
            m = (blk == 0).groupby(d["_uc"]).sum()
        else:  # busiest block of N days, blocks aligned to each user's first event
            m = blk.groupby([d["_uc"], blk]).size().groupby(level=0).max()
    m = m.astype(float)
    try:
        grp = pd.qcut(m, q, labels=False, duplicates="drop")
    except ValueError:  # all values identical
        grp = pd.Series(0, index=m.index)
    grp = grp.astype(int)
    labels = []
    for g in sorted(grp.unique()):
        v = m[grp == g]
        lo, hi = v.min(), v.max()
        labels.append(f"G{g + 1} ({lo:g})" if lo == hi else f"G{g + 1} ({lo:g}-{hi:g})")
    return m, grp, labels


@app.callback(
    Output("g-note", "children"),
    Output("g-mix", "figure"),
    Output("g-pool", "figure"),
    Output("g-stats", "data"),
    Output("g-stats", "columns"),
    Input("actions", "value"),
    Input("g-metric", "value"),
    Input("g-window", "value"),
    Input("g-q", "value"),
)
def group_mix(acts, metric, window, q):
    acts = ordered(acts)
    if not acts:
        e = empty_fig("Select at least one action")
        return "Select at least one action.", e, e, [], []
    m, grp, labels = user_groups(tuple(acts), metric, window, q)
    Gn = len(labels)
    d = srt[srt[ACTION_COL].isin(acts)]
    cnt = (pd.crosstab(d["_uc"], d[ACTION_COL])
             .reindex(columns=acts, fill_value=0))
    grp = grp.reindex(cnt.index)
    share = cnt.div(cnt.sum(axis=1), axis=0)
    sizes = [int((grp == g).sum()) for g in range(Gn)]

    note = (f"Users split into {Gn} group(s) by {metric_label(metric, window)} "
            f"(quantiles; G1 = lightest). Group sizes: "
            + "; ".join(f"{labels[g]} n={sizes[g]:,}" for g in range(Gn)))
    if Gn < q:
        note += f"\n(only {Gn} groups possible: many users share the same value)"
    if kruskal is None:
        note += "\nInstall scipy (pip install scipy) to get significance tests."
    else:
        note += ("\nKruskal-Wallis tests whether the per-user share differs across groups; "
                 "with many users p-values get tiny, so also read the effect size "
                 "(eps^2 ~ 0.01 small, 0.06 medium, 0.14 large).")

    # left: mean per-user share with 95% CI
    fig1 = go.Figure()
    for g in range(Gn):
        sub = share[grp == g]
        fig1.add_trace(go.Bar(
            x=acts, y=sub.mean().values, name=f"{labels[g]}, n={sizes[g]:,}",
            marker_color=GROUP_COLORS[g % len(GROUP_COLORS)],
            error_y=dict(type="data", array=(1.96 * sub.sem().fillna(0)).values)))
    fig1.update_layout(barmode="group", template="plotly_white", height=460,
                       title="Mean share of each user's actions (±95% CI)",
                       yaxis=dict(tickformat=".0%"), legend=dict(orientation="h", y=-0.2))

    # right: pooled events per group, 100% stacked
    pooled = cnt.groupby(grp).sum()
    pshare = pooled.div(pooled.sum(axis=1), axis=0)
    xl = [labels[i] for i in pshare.index]
    fig2 = go.Figure()
    for a in acts:
        fig2.add_trace(go.Bar(x=xl, y=pshare[a].values, name=a, marker_color=CMAP[a]))
    fig2.update_layout(barmode="stack", template="plotly_white", height=460,
                       title="All events pooled per group (heavy users dominate)",
                       yaxis=dict(tickformat=".0%"), legend=dict(orientation="h", y=-0.2))

    rows = []
    n_users = len(share)
    for a in acts:
        row = {"action": a}
        means = []
        for g in range(Gn):
            mu = float(share.loc[grp == g, a].mean())
            means.append(mu)
            row[f"G{g + 1} mean %"] = round(100 * mu, 2)
        row["spread (pp)"] = round(100 * (max(means) - min(means)), 2)
        p = eps = None
        if kruskal is not None and Gn >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    H, p = kruskal(*[share.loc[grp == g, a].to_numpy() for g in range(Gn)])
                eps = H / (n_users - 1)
                if not (np.isfinite(H) and np.isfinite(p)):  # all values identical
                    p = eps = None
            except ValueError:
                p = eps = None
        row["Kruskal p"] = None if p is None else float(f"{p:.3g}")
        row["effect size (eps^2)"] = None if eps is None else round(float(eps), 3)
        row["signif. (Bonferroni)"] = "" if p is None else ("yes" if p < 0.05 / len(acts) else "no")
        rows.append(row)
    cols = [{"name": c, "id": c} for c in rows[0].keys()]
    return note, fig1, fig2, rows, cols


@app.callback(
    Output("g-trans", "figure"),
    Output("g-tnote", "children"),
    Output("g-tdiff", "data"),
    Input("actions", "value"),
    Input("gap", "value"),
    Input("g-metric", "value"),
    Input("g-window", "value"),
    Input("g-q", "value"),
    Input("g-view", "value"),
)
def group_trans(acts, gap_key, metric, window, q, view):
    acts = ordered(acts)
    if not acts:
        return empty_fig("Select at least one action"), "", []
    m, grp, labels = user_groups(tuple(acts), metric, window, q)
    Gn = len(labels)
    if Gn < 2:
        return empty_fig("Only one group: pick a different metric / fewer groups"), "", []
    d = srt[srt[ACTION_COL].isin(acts)]
    if len(d) < 2:
        return empty_fig("Not enough events"), "", []

    a = d["_a"].to_numpy()
    u = d["_uc"].to_numpy()
    tt = d["_t"].to_numpy()
    same = u[1:] == u[:-1]
    gap = (tt[1:] - tt[:-1]) / np.timedelta64(1, "s")
    ok = same & (gap <= GAPS[gap_key])
    if ok.sum() == 0:
        return empty_fig("No transitions with this filter"), "", []

    gmap = np.full(int(srt["_uc"].max()) + 1, -1, dtype=np.int64)
    gmap[grp.index.to_numpy()] = grp.to_numpy()
    gu = gmap[u[:-1]][ok]
    pair = a[:-1][ok].astype(np.int64) * K + a[1:][ok]
    Mg = np.bincount(gu * K * K + pair, minlength=Gn * K * K).reshape(Gn, K, K).astype(float)
    idx = [CODE[x] for x in acts]
    Cg = Mg[:, idx][:, :, idx]                      # group x from x to
    rs = Cg.sum(axis=2)                              # group x from
    nu = [int((grp == g).sum()) for g in range(Gn)]
    with np.errstate(divide="ignore", invalid="ignore"):
        Pg = np.where(rs[:, :, None] > 0, Cg / rs[:, :, None], np.nan)
        Call = Cg.sum(axis=0)
        rsa = Call.sum(axis=1, keepdims=True)
        Pall = np.where(rsa > 0, Call / rsa, np.nan)
    Pg = np.where((rs >= MIN_SUPPORT)[:, :, None], Pg, np.nan)

    if view == "prob":
        Z = Pg * 100
        fmt = lambda v: "" if np.isnan(v) else f"{v:.1f}%"
        kw = dict(colorscale="Blues", zmin=0, zmax=100, colorbar=dict(title="%"))
    else:
        Z = (Pg - Pall[None]) * 100
        fin = np.abs(Z[np.isfinite(Z)])
        zmax = float(max(10, min(50, np.percentile(fin, 99)))) if fin.size else 10.0
        fmt = lambda v: "" if np.isnan(v) else f"{v:+.1f}"
        kw = dict(colorscale="RdBu", zmin=-zmax, zmax=zmax, colorbar=dict(title="pp vs all"))

    ncols = min(Gn, 3)
    nrows = math.ceil(Gn / ncols)
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"{labels[g]}, {nu[g]:,} users" for g in range(Gn)],
                        horizontal_spacing=0.12, vertical_spacing=0.18)
    for g in range(Gn):
        fig.add_trace(go.Heatmap(
            z=Z[g], x=acts, y=acts, text=[[fmt(v) for v in r] for r in Z[g]],
            texttemplate="%{text}", showscale=(g == 0), **kw,
            hovertemplate="%{y} -> %{x}<br>%{text}<extra></extra>"),
            row=g // ncols + 1, col=g % ncols + 1)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=250 + 330 * nrows)

    note = (f"Rows = previous action, columns = next action ({GAP_LABELS[gap_key]}). "
            f"Cells where the group has fewer than {MIN_SUPPORT} 'previous' events are blank. "
            + ("Values are the group's P(next | previous) in %."
               if view == "prob" else
               "Values are percentage points above (+, blue) or below (-, red) all users."))

    rows = []
    k = len(acts)
    for i in range(k):
        for j in range(k):
            vals = [(g, Pg[g, i, j] * 100) for g in range(Gn) if not np.isnan(Pg[g, i, j])]
            if len(vals) < 2:
                continue
            hi = max(vals, key=lambda t: t[1])
            lo = min(vals, key=lambda t: t[1])
            rows.append({"from": acts[i], "to": acts[j],
                         "highest group": labels[hi[0]], "lowest group": labels[lo[0]],
                         "spread (pp)": round(float(hi[1] - lo[1]), 2),
                         "P highest %": round(float(hi[1]), 2),
                         "P lowest %": round(float(lo[1]), 2)})
    rows.sort(key=lambda r: -r["spread (pp)"])
    return fig, note, rows[:30]


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)