# Interactive single-user action explorer (runs from the terminal, opens in browser)
# pip install pandas pyarrow plotly dash
#
# Usage (from the repository root):
#   python user_explorer_dash.py
# then open http://127.0.0.1:8050 (Ctrl+click the link in the VS Code terminal).

# ---------------------------------------------------------------- CONFIG
DATA_PATH = "assignment2/data/db"  # repository Parquet database, file, or glob
USER_COL = None       # e.g. "user_id"   (None = auto-detect)
TIME_COL = None       # e.g. "timestamp" (None = auto-detect)
ACTION_COL = None     # e.g. "action"    (None = auto-detect)
ITEM_COL = None       # e.g. "item_id"   (optional, auto-detect)
TIME_UNIT = None      # "s" or "ms" if timestamps are numeric (None = guess)
PORT = 8050

# ---------------------------------------------------------------- LOAD
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, dcc, html, dash_table

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
ITEM_COL = ITEM_COL or guess(df.columns, ["item", "product", "movie", "song", "sku", "page"])
print(f"user={USER_COL}  time={TIME_COL}  action={ACTION_COL}  item={ITEM_COL}")
if not USER_COL or not TIME_COL:
    raise ValueError(
        "Could not identify user and time columns. Set USER_COL and TIME_COL "
        "in the CONFIG block."
    )
missing_columns = [c for c in (USER_COL, TIME_COL) if c not in df]
if missing_columns:
    raise ValueError(f"Configured column(s) not found in the input: {missing_columns}")

t = df[TIME_COL]
if pd.api.types.is_numeric_dtype(t):
    unit = TIME_UNIT or ("ms" if t.median() > 1e11 else "s")
    df["_t"] = pd.to_datetime(t, unit=unit)
else:
    df["_t"] = pd.to_datetime(t, errors="coerce")
df = df.dropna(subset=["_t"]).sort_values("_t")
if ACTION_COL is None:
    ACTION_COL = "_action"
    df[ACTION_COL] = "event"
df[ACTION_COL] = df[ACTION_COL].astype(str)

df["_u"] = df[USER_COL].astype(str)
user_counts = df["_u"].value_counts()  # most active first
if user_counts.empty:
    raise ValueError("No users have valid timestamps in the loaded activity data.")
print(f"{len(user_counts)} users; median {int(user_counts.median())} actions/user")

actions_all = sorted(df[ACTION_COL].unique())
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B"]
CMAP = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(actions_all)}
BINS = ["1min", "5min", "15min", "1h", "1D", "1W"]

# group once so selecting a user is fast
by_user = {u: g for u, g in df.groupby("_u", sort=False)}

# ---------------------------------------------------------------- APP
app = Dash(__name__)
app.title = "User action explorer"

user_options = [{"label": f"{u}  ({n} actions)", "value": u}
                for u, n in user_counts.items()]

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "0 auto",
           "padding": "16px"},
    children=[
        html.H2("User action explorer"),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            html.Div(style={"width": "380px"}, children=[
                html.Label("User (type to search)"),
                dcc.Dropdown(id="user", options=user_options,
                             value=user_options[0]["value"], clearable=False),
            ]),
            html.Div(style={"width": "380px"}, children=[
                html.Label("Actions"),
                dcc.Dropdown(id="actions",
                             options=[{"label": a, "value": a} for a in actions_all],
                             value=actions_all, multi=True),
            ]),
            html.Div(style={"width": "160px"}, children=[
                html.Label("Bin"),
                dcc.Dropdown(id="bin", options=BINS, value="1h", clearable=False),
            ]),
        ]),
        html.Pre(id="summary", style={"whiteSpace": "pre-wrap", "marginTop": "12px"}),
        dcc.Graph(id="graph"),
        html.H4("Raw events (first 200)"),
        dash_table.DataTable(id="table", page_size=20,
                             style_table={"overflowX": "auto"},
                             style_cell={"textAlign": "left", "fontSize": 13}),
    ],
)


def empty_fig(msg):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", annotations=[
        dict(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(size=18))])
    return fig


@app.callback(
    Output("summary", "children"),
    Output("graph", "figure"),
    Output("table", "data"),
    Output("table", "columns"),
    Input("user", "value"),
    Input("actions", "value"),
    Input("bin", "value"),
)
def update(u, acts, bin_):
    if u is None or u not in by_user:
        return "No user selected.", empty_fig("No user selected"), [], []
    d = by_user[u]
    d = d[d[ACTION_COL].isin(acts or [])]
    if d.empty:
        return (f"User {u}: no rows for this action filter.",
                empty_fig("No rows for this user / action filter"), [], [])

    span = d["_t"].max() - d["_t"].min()
    gaps = d["_t"].diff().dropna()
    summary = (f"User {u}: {len(d)} actions | {d['_t'].min()} -> {d['_t'].max()} "
               f"({span}) | median gap {gaps.median() if len(gaps) else 'n/a'}")
    if ITEM_COL:
        summary += f"\nDistinct items: {d[ITEM_COL].nunique()}"

    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"colspan": 2}, None], [{"colspan": 2}, None], [{}, {}], [{}, {}]],
        subplot_titles=("Timeline (hover for details)", "Activity volume over time",
                        "Action counts", "Hour-of-day x day-of-week",
                        "Action transitions (from -> to)", "Cumulative actions"),
        row_heights=[0.28, 0.2, 0.26, 0.26], vertical_spacing=0.09)

    # 1. timeline
    for a in sorted(d[ACTION_COL].unique()):
        s = d[d[ACTION_COL] == a]
        hover = s[ITEM_COL].astype(str) if ITEM_COL else s[ACTION_COL]
        fig.add_trace(go.Scattergl(
            x=s["_t"], y=[a] * len(s), mode="markers", name=a, text=hover,
            marker=dict(color=CMAP[a], size=8, opacity=.75),
            hovertemplate="%{x}<br>%{y}<br>%{text}<extra></extra>"), row=1, col=1)

    # 2. volume stacked by action
    piv = (d.set_index("_t").groupby(ACTION_COL)
             .resample(bin_).size().unstack(0).fillna(0))
    for a in piv.columns:
        fig.add_trace(go.Bar(x=piv.index, y=piv[a], name=a, marker_color=CMAP[a],
                             showlegend=False), row=2, col=1)

    # 3. counts
    vc = d[ACTION_COL].value_counts()
    fig.add_trace(go.Bar(x=vc.index, y=vc.values,
                         marker_color=[CMAP[a] for a in vc.index],
                         showlegend=False), row=3, col=1)

    # 4. heatmap
    hm = (d.assign(h=d["_t"].dt.hour, w=d["_t"].dt.dayofweek)
            .groupby(["w", "h"]).size().unstack(fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0))
    fig.add_trace(go.Heatmap(z=hm.values, x=list(range(24)),
                             y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                             colorscale="Viridis", showscale=False), row=3, col=2)

    # 5. transitions
    nxt = d[ACTION_COL].shift(-1)
    tr = pd.crosstab(d[ACTION_COL][:-1].values, nxt.dropna().values)
    fig.add_trace(go.Heatmap(z=tr.values, x=list(tr.columns), y=list(tr.index),
                             colorscale="Blues", showscale=False,
                             hovertemplate="%{y} -> %{x}: %{z}<extra></extra>"),
                  row=4, col=1)

    # 6. cumulative
    fig.add_trace(go.Scatter(x=d["_t"], y=np.arange(1, len(d) + 1), mode="lines",
                             line=dict(color="#333"), showlegend=False), row=4, col=2)

    fig.update_layout(barmode="stack", height=1100, title=f"User {u}",
                      template="plotly_white", legend=dict(orientation="h", y=1.04))

    cols = ["_t", ACTION_COL] + ([ITEM_COL] if ITEM_COL else [])
    extra = [c for c in d.columns if c not in cols + [USER_COL, TIME_COL, "_u"]][:4]
    tbl = d[cols + extra].head(200).copy()
    tbl["_t"] = tbl["_t"].astype(str)
    tbl = tbl.rename(columns={"_t": "time"})
    columns = [{"name": c, "id": c} for c in tbl.columns]
    return summary, fig, tbl.astype(str).to_dict("records"), columns


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)