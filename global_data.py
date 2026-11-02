"""Global GitHub activity explorer with user-defined activity groups.

Install:
    pip install pandas pyarrow plotly dash scipy

Run from the repository root:
    python global_trends_dash.py

Then open:
    http://127.0.0.1:8051

The input can be:
- The assignment Parquet database directory (containing Users.parquet, PullRequests.parquet, etc.)
- A Parquet/CSV/JSON/JSONL file
- A glob matching one or more supported files

For generic event files, configure USER_COL, TIME_COL, and ACTION_COL.
"""

# ---------------------------------------------------------------- CONFIG
DATA_PATH = "assignment2/data/db"  # repository Parquet database, file, or glob
USER_COL = None       # e.g. "user_id" (None = auto-detect)
TIME_COL = None       # e.g. "timestamp" (None = auto-detect)
ACTION_COL = None     # e.g. "action" (None = auto-detect)
ITEM_COL = None       # reserved for parity with the per-user explorer
TIME_UNIT = None      # "s" or "ms" if timestamps are numeric (None = guess)
PORT = 8051

# Transition rows with fewer previous-action events are hidden to avoid
# displaying very unstable conditional probabilities.
MIN_SUPPORT = 30

# ---------------------------------------------------------------- IMPORTS
import glob
import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dcc, html, dash_table

try:
    from scipy.stats import kruskal
except Exception:  # scipy is optional; significance tests will be skipped
    kruskal = None

# ---------------------------------------------------------------- INPUT DATA
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
    cohort_users = users.loc[users["is_cohort"].fillna(False)].set_index("user_id")

    events = []
    for table, actor_col, time_col, action in GITHUB_ACTIVITY_TABLES:
        columns = [actor_col, "repo_id", time_col]
        if table == "Issues":
            columns.append("is_pull_request")

        raw = pd.read_parquet(db_dir / f"{table}.parquet", columns=columns)
        if table == "Issues":
            # Issues.parquet also contains pull requests; keep ordinary issues.
            raw = raw.loc[~raw["is_pull_request"].fillna(False)]

        rows = raw.loc[
            raw[actor_col].isin(cohort_users.index) & raw[time_col].notna()
        ]
        events.append(
            pd.DataFrame(
                {
                    "user_id": rows[actor_col].map(cohort_users["github_user_id"]),
                    "timestamp": rows[time_col],
                    "action": action,
                    "item_id": rows["repo_id"],
                }
            )
        )

    result = pd.concat(events, ignore_index=True)
    return result.dropna(subset=["user_id", "timestamp"])


def read_any(path):
    """Read the configured database directory, file, or glob."""
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
    for filename in files:
        suffix = Path(filename).suffix.lower()
        if suffix == ".parquet":
            dfs.append(pd.read_parquet(filename))
        elif suffix in (".json", ".jsonl"):
            dfs.append(pd.read_json(filename, lines=suffix == ".jsonl"))
        else:
            dfs.append(pd.read_csv(filename, sep=None, engine="python"))
    return pd.concat(dfs, ignore_index=True)


def guess(cols, keys):
    """Guess a column name from common substrings."""
    low = {str(c).lower(): c for c in cols}
    for key in keys:
        for lower_name, original_name in low.items():
            if key in lower_name:
                return original_name
    return None


# Load data and determine columns.
df = read_any(DATA_PATH)
if df.empty:
    raise ValueError(f"No activity rows were loaded from DATA_PATH={DATA_PATH!r}.")

print("Columns:", list(df.columns), "| rows:", len(df))

USER_COL = USER_COL or guess(df.columns, ["user", "uid", "member", "customer"])
TIME_COL = TIME_COL or guess(df.columns, ["time", "date", "ts", "datetime"])
ACTION_COL = ACTION_COL or guess(
    df.columns, ["action", "event", "behavior", "type", "act"]
)
print(f"user={USER_COL}  time={TIME_COL}  action={ACTION_COL}")

if not USER_COL or not TIME_COL:
    raise ValueError("Could not identify user and time columns. Set them in CONFIG.")

missing_columns = [c for c in (USER_COL, TIME_COL) if c not in df.columns]
if missing_columns:
    raise ValueError(f"Configured column(s) not found in the input: {missing_columns}")

# Convert timestamps.
timestamp_series = df[TIME_COL]
if pd.api.types.is_numeric_dtype(timestamp_series):
    unit = TIME_UNIT or ("ms" if timestamp_series.median() > 1e11 else "s")
    df["_t"] = pd.to_datetime(timestamp_series, unit=unit, errors="coerce")
else:
    df["_t"] = pd.to_datetime(timestamp_series, errors="coerce")

df = df.dropna(subset=["_t", USER_COL]).copy()

if ACTION_COL is None:
    ACTION_COL = "_action"
    df[ACTION_COL] = "event"
else:
    if ACTION_COL not in df.columns:
        raise ValueError(f"Configured action column {ACTION_COL!r} was not found.")
    df = df[df[ACTION_COL].notna()].copy()

df[ACTION_COL] = df[ACTION_COL].astype(str)
df["_u"] = df[USER_COL].astype(str)

# Sort so adjacent rows form each user's chronological event sequence.
srt = df[["_u", "_t", ACTION_COL]].sort_values(["_u", "_t"]).reset_index(drop=True)
srt["_uc"] = pd.factorize(srt["_u"])[0]

# Build action codes and the user x action matrix.
actions_all = sorted(srt[ACTION_COL].dropna().unique())
K = len(actions_all)
if K == 0:
    raise ValueError("No action categories were found after cleaning the input.")

CODE = {action: index for index, action in enumerate(actions_all)}
srt["_a"] = srt[ACTION_COL].map(CODE).astype(np.int32)
ua = pd.crosstab(srt["_u"], srt[ACTION_COL]).reindex(
    columns=actions_all, fill_value=0
)

print(
    f"{len(srt):,} events | {srt['_uc'].nunique():,} users | "
    f"{K} actions: {actions_all}"
)

# ---------------------------------------------------------------- CONSTANTS
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B",
]
CMAP = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(actions_all)}
BINS = ["1h", "6h", "1D", "1W", "1MS"]
GAPS = {
    "any": np.inf,
    "1h": 3600,
    "6h": 6 * 3600,
    "1D": 86400,
    "7D": 7 * 86400,
    "30D": 30 * 86400,
}
GAP_LABELS = {
    "any": "No limit",
    "1h": "within 1 hour",
    "6h": "within 6 hours",
    "1D": "within 1 day",
    "7D": "within 7 days",
    "30D": "within 30 days",
}
METRICS = {
    "first": "Actions in first N days",
    "peak": "Actions in busiest N-day block",
    "total": "Total actions (whole period)",
}
GROUP_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
NORMS = {
    "count": "Counts",
    "prob": "P(next | previous), row %",
    "lift": "Lift = P(next | previous) / P(next)",
}
MAX_GROUPS = 5  # -> up to MAX_GROUPS - 1 threshold inputs

# ---------------------------------------------------------------- HELPERS


def empty_fig(msg):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        annotations=[
            dict(
                text=msg,
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=18),
            )
        ],
    )
    return fig


def fmt_secs(seconds):
    if not np.isfinite(seconds):
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def bucket_labels(edges):
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = hi - 1
        labels.append(f"{int(lo)}" if hi <= lo else f"{int(lo)}-{int(hi)}")
    return labels


def ordered(acts):
    selected = set(acts or [])
    return [action for action in actions_all if action in selected]


def parse_thresholds(q, raw_values):
    """Sorted, unique, valid upper-bound thresholds for the first q-1 inputs."""
    cleaned = []
    for value in list(raw_values)[: max(int(q) - 1, 0)]:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            cleaned.append(number)
    return tuple(sorted(set(cleaned)))


# ---------------------------------------------------------------- DASH APP
app = Dash(__name__)
app.title = "Global action trends"

TBL_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={"textAlign": "left", "fontSize": 13, "padding": "6px"},
    style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"},
)

TH_BOX_STYLE = {"width": "150px"}

app.layout = html.Div(
    style={
        "fontFamily": "sans-serif",
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "16px",
    },
    children=[
        html.H2("Global action trends"),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"width": "420px"},
                    children=[
                        html.Label("Actions"),
                        dcc.Dropdown(
                            id="actions",
                            options=[{"label": a, "value": a} for a in actions_all],
                            value=actions_all,
                            multi=True,
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "140px"},
                    children=[
                        html.Label("Time bin (trend)"),
                        dcc.Dropdown(
                            id="bin",
                            options=BINS,
                            value="1D",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "340px"},
                    children=[
                        html.Label("Transition matrix shows"),
                        dcc.Dropdown(
                            id="norm",
                            options=[
                                {"label": value, "value": key}
                                for key, value in NORMS.items()
                            ],
                            value="prob",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "200px"},
                    children=[
                        html.Label("Next action happens"),
                        dcc.Dropdown(
                            id="gap",
                            options=[
                                {"label": value, "value": key}
                                for key, value in GAP_LABELS.items()
                            ],
                            value="any",
                            clearable=False,
                        ),
                    ],
                ),
            ],
        ),
        html.Pre(
            id="summary",
            style={"whiteSpace": "pre-wrap", "marginTop": "12px"},
        ),
        dcc.Graph(id="overview"),
        html.H3("Which action leads to which?"),
        html.Div(
            id="note",
            style={"color": "#555", "marginBottom": "6px"},
        ),
        dcc.Graph(id="trans"),
        html.H4("Top transitions (previous -> next)"),
        dash_table.DataTable(
            id="bigram",
            page_size=15,
            sort_action="native",
            columns=[
                {"name": c, "id": c}
                for c in [
                    "from", "to", "count", "P(to|from) %", "lift",
                    "median time to next",
                ]
            ],
            **TBL_STYLE,
        ),
        html.H4("Top 3-step sequences"),
        dash_table.DataTable(
            id="trigram",
            page_size=15,
            sort_action="native",
            columns=[
                {"name": c, "id": c}
                for c in [
                    "step 1", "step 2", "step 3", "count",
                    "% of 3-step sequences",
                ]
            ],
            **TBL_STYLE,
        ),

        # ---------------------------------------------------------- USER GROUPS
        html.Hr(),
        html.H3("Do light and heavy users behave differently?"),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"width": "300px"},
                    children=[
                        html.Label("Split users by"),
                        dcc.Dropdown(
                            id="g-metric",
                            options=[
                                {"label": v, "value": k}
                                for k, v in METRICS.items()
                            ],
                            value="first",
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "140px"},
                    children=[
                        html.Label("N (days)"),
                        dcc.Dropdown(
                            id="g-window",
                            options=[
                                {
                                    "label": f"{w} day{'s' if w > 1 else ''}",
                                    "value": w,
                                }
                                for w in (1, 7, 30)
                            ],
                            value=7,
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "160px"},
                    children=[
                        html.Label("Number of groups"),
                        dcc.Dropdown(
                            id="g-q",
                            options=list(range(2, MAX_GROUPS + 1)),
                            value=3,
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "300px"},
                    children=[
                        html.Label("Transition heatmaps show"),
                        dcc.Dropdown(
                            id="g-view",
                            options=[
                                {
                                    "label": "P(next | previous) per group",
                                    "value": "prob",
                                },
                                {
                                    "label": "Difference vs all users (pp)",
                                    "value": "diff",
                                },
                            ],
                            value="diff",
                            clearable=False,
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            "Group thresholds (inclusive upper bound of each group; the last "
            "group is everything above the final threshold). Defaults are "
            "quantile-based suggestions; edit them and press Enter or click away.",
            style={"color": "#555", "margin": "12px 0 4px"},
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    id=f"g-th-box-{i}",
                    style=TH_BOX_STYLE,
                    children=[
                        html.Label(f"G{i + 1} upper bound (≤)"),
                        dcc.Input(
                            id=f"g-th-{i}",
                            type="number",
                            min=0,
                            step=1,
                            debounce=True,
                            style={"width": "100%"},
                        ),
                    ],
                )
                for i in range(MAX_GROUPS - 1)
            ],
        ),
        html.Div(
            id="g-note",
            style={
                "color": "#555",
                "margin": "8px 0",
                "whiteSpace": "pre-wrap",
            },
        ),
        html.Div(
            style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            children=[
                dcc.Graph(id="g-mix", style={"flex": "1 1 560px"}),
                dcc.Graph(id="g-pool", style={"flex": "1 1 560px"}),
            ],
        ),
        html.H4("Action mix by group (per-user share of actions)"),
        dash_table.DataTable(
            id="g-stats",
            page_size=15,
            sort_action="native",
            **TBL_STYLE,
        ),
        html.H4("Transitions by group"),
        dcc.Graph(id="g-trans"),
        html.Div(
            id="g-tnote",
            style={"color": "#555", "marginBottom": "6px"},
        ),
        html.H4("Transitions that differ most between groups"),
        dash_table.DataTable(
            id="g-tdiff",
            page_size=15,
            sort_action="native",
            columns=[
                {"name": c, "id": c}
                for c in [
                    "from", "to", "highest group", "lowest group",
                    "spread (pp)", "P highest %", "P lowest %",
                ]
            ],
            **TBL_STYLE,
        ),
    ],
)

# ---------------------------------------------------------------- GLOBAL OVERVIEW


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

    summary = (
        f"{len(d):,} events | {d['_uc'].nunique():,} users | "
        f"{d['_t'].min()} -> {d['_t'].max()}\n"
        f"Actions per user: median {totals.median():.0f}, "
        f"mean {totals.mean():.1f}, max {totals.max():.0f}\n"
        "Most common: "
        + ", ".join(f"{action} ({count / len(d):.1%})" for action, count in top.head(3).items())
    )

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{}, {}],
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=(
            "Events per action",
            "Distinct users per action",
            "Events over time",
            "Actions per user (histogram)",
            "Hour-of-day x day-of-week (all users)",
            "Per-user action mix (share of each user's actions)",
        ),
        row_heights=[0.2, 0.22, 0.25, 0.25],
        vertical_spacing=0.08,
    )

    # 1. Events per action.
    fig.add_trace(
        go.Bar(
            x=vc.index,
            y=vc.values,
            marker_color=[CMAP[a] for a in vc.index],
            text=[f"{v / vc.sum():.1%}" for v in vc.values],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Distinct users per action.
    nu = d.groupby(ACTION_COL)["_uc"].nunique().reindex(acts).fillna(0)
    nall = d["_uc"].nunique()
    fig.add_trace(
        go.Bar(
            x=nu.index,
            y=nu.values,
            marker_color=[CMAP[a] for a in nu.index],
            text=[f"{v / nall:.0%}" for v in nu.values],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Trend.
    grouped = (
        d.groupby([ACTION_COL, pd.Grouper(key="_t", freq=bin_)])
        .size()
        .unstack(0)
        .fillna(0)
    )
    for action in acts:
        if action in grouped.columns:
            fig.add_trace(
                go.Bar(
                    x=grouped.index,
                    y=grouped[action],
                    name=action,
                    marker_color=CMAP[action],
                ),
                row=2,
                col=1,
            )

    # 4. Actions-per-user histogram with log-spaced buckets.
    maximum = int(totals.max())
    edges = np.unique(np.round(np.logspace(0, np.log10(maximum + 1), 20)).astype(int))
    if len(edges) < 2:
        edges = np.array([1, maximum + 1])
    elif edges[-1] <= maximum:
        edges = np.append(edges, maximum + 1)
    hist, _ = np.histogram(totals, bins=edges)
    fig.add_trace(
        go.Bar(
            x=bucket_labels(edges),
            y=hist,
            marker_color="#4C78A8",
            showlegend=False,
            hovertemplate="%{x} actions: %{y} users<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # 5. Hour/day heatmap.
    heatmap = (
        d.assign(hour=d["_t"].dt.hour, weekday=d["_t"].dt.dayofweek)
        .groupby(["weekday", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    fig.add_trace(
        go.Heatmap(
            z=heatmap.values,
            x=list(range(24)),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale="Viridis",
            showscale=False,
        ),
        row=3,
        col=2,
    )

    # 6. Per-user action mix.
    selected_counts = ua.loc[totals.index, acts]
    shares = selected_counts.div(selected_counts.sum(axis=1), axis=0)
    for action in acts:
        fig.add_trace(
            go.Box(
                y=shares[action],
                name=action,
                marker_color=CMAP[action],
                boxpoints=False,
                boxmean=True,
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    fig.update_yaxes(title_text="events", row=1, col=1)
    fig.update_yaxes(title_text="users", row=1, col=2)
    fig.update_yaxes(title_text="share", tickformat=".0%", row=4, col=1)
    fig.update_xaxes(title_text="actions per user", row=3, col=1)
    fig.update_layout(
        barmode="stack",
        height=1450,
        template="plotly_white",
        legend=dict(orientation="h", y=1.03),
    )
    return summary, fig

# ---------------------------------------------------------------- GLOBAL TRANSITIONS


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

    action_codes = d["_a"].to_numpy()
    users = d["_uc"].to_numpy()
    timestamps = d["_t"].to_numpy()
    same_user = users[1:] == users[:-1]
    gap_seconds = (timestamps[1:] - timestamps[:-1]) / np.timedelta64(1, "s")
    valid = same_user & (gap_seconds >= 0) & (gap_seconds <= GAPS[gap_key])

    if valid.sum() == 0:
        return empty_fig("No transitions with this filter"), "", [], []

    from_codes = action_codes[:-1][valid]
    to_codes = action_codes[1:][valid]
    valid_gaps = gap_seconds[valid]
    pair_codes = from_codes.astype(np.int64) * K + to_codes.astype(np.int64)

    matrix = np.bincount(pair_codes, minlength=K * K).reshape(K, K).astype(float)
    medians = pd.Series(valid_gaps).groupby(pair_codes).median()
    median_gap_matrix = np.full(K * K, np.nan)
    median_gap_matrix[medians.index.to_numpy()] = medians.to_numpy()
    median_gap_matrix = median_gap_matrix.reshape(K, K)

    indices = [CODE[action] for action in acts]
    counts = matrix[np.ix_(indices, indices)]
    median_gaps = median_gap_matrix[np.ix_(indices, indices)]
    row_sums = counts.sum(axis=1, keepdims=True)
    overall_next_share = counts.sum(axis=0, keepdims=True) / counts.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.divide(
            counts, row_sums,
            out=np.full_like(counts, np.nan),
            where=row_sums > 0,
        )
        lift = np.divide(
            probabilities, overall_next_share,
            out=np.full_like(probabilities, np.nan),
            where=overall_next_share > 0,
        )

    if norm == "count":
        z = counts
        text_values = [[f"{v:,.0f}" for v in row] for row in counts]
        color_kwargs = dict(colorscale="Blues", colorbar=dict(title="count"))
    elif norm == "prob":
        z = probabilities * 100
        text_values = [
            ["" if np.isnan(v) else f"{v * 100:.1f}%" for v in row]
            for row in probabilities
        ]
        color_kwargs = dict(colorscale="Blues", zmin=0, zmax=100, colorbar=dict(title="%"))
    else:
        z = lift
        text_values = [
            ["" if np.isnan(v) else f"{v:.2f}x" for v in row]
            for row in lift
        ]
        color_kwargs = dict(
            colorscale="RdBu", zmin=0, zmax=2,
            colorbar=dict(title="lift (clipped at 2)"),
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.18,
        subplot_titles=(NORMS[norm], "Median time from previous to next action"),
    )
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=acts,
            y=acts,
            text=text_values,
            texttemplate="%{text}",
            hovertemplate="%{y} -> %{x}<br>%{text}<extra></extra>",
            **color_kwargs,
            colorbar_x=0.43,
        ),
        row=1,
        col=1,
    )
    log_gap = np.log10(np.maximum(median_gaps, 1))
    gap_text = [[fmt_secs(v) for v in row] for row in median_gaps]
    fig.add_trace(
        go.Heatmap(
            z=log_gap,
            x=acts,
            y=acts,
            text=gap_text,
            texttemplate="%{text}",
            colorscale="YlOrRd",
            showscale=False,
            hovertemplate="%{y} -> %{x}<br>median %{text}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(autorange="reversed", title_text="previous action")
    fig.update_xaxes(title_text="next action")
    fig.update_layout(template="plotly_white", height=520 + 25 * len(acts))

    note = (
        f"{int(valid.sum()):,} of {int(same_user.sum()):,} consecutive same-user "
        f"event pairs used ({GAP_LABELS[gap_key]}). Rows = previous action; "
        "columns = next action."
    )

    bigram_rows = []
    for i, from_action in enumerate(acts):
        for j, to_action in enumerate(acts):
            if counts[i, j] > 0:
                bigram_rows.append(
                    {
                        "from": from_action,
                        "to": to_action,
                        "count": int(counts[i, j]),
                        "P(to|from) %": round(float(probabilities[i, j] * 100), 2),
                        "lift": None if np.isnan(lift[i, j]) else round(float(lift[i, j]), 2),
                        "median time to next": fmt_secs(median_gaps[i, j]),
                    }
                )
    bigram_rows.sort(key=lambda row: -row["count"])

    # Trigrams: two consecutive valid event pairs from the same user.
    trigram_rows = []
    if len(action_codes) >= 3:
        valid_triples = valid[:-1] & valid[1:]
        if valid_triples.any():
            triple_codes = (
                (action_codes[:-2].astype(np.int64) * K + action_codes[1:-1]) * K
                + action_codes[2:]
            )
            triple_counts = pd.Series(triple_codes[valid_triples]).value_counts()
            total_triples = int(valid_triples.sum())
            for code, count in triple_counts.head(25).items():
                first, remainder = divmod(int(code), K * K)
                second, third = divmod(remainder, K)
                trigram_rows.append(
                    {
                        "step 1": actions_all[first],
                        "step 2": actions_all[second],
                        "step 3": actions_all[third],
                        "count": int(count),
                        "% of 3-step sequences": round(100 * count / total_triples, 2),
                    }
                )

    return fig, note, bigram_rows, trigram_rows

# ---------------------------------------------------------------- USER GROUPS


def metric_label(metric, window):
    return {
        "first": f"actions in first {window} days",
        "peak": f"actions in busiest {window}-day block",
        "total": "total actions",
    }[metric]


@lru_cache(maxsize=16)
def user_values(acts_t, metric, window):
    """Per-user activity metric (indexed by user code)."""
    d = srt[srt[ACTION_COL].isin(acts_t)]
    if d.empty:
        return pd.Series(dtype=float)

    if metric == "total":
        values = d.groupby("_uc").size()
    else:
        first_time = d.groupby("_uc")["_t"].transform("min")
        block = np.floor(
            (d["_t"] - first_time) / np.timedelta64(window, "D")
        ).astype(np.int64)
        if metric == "first":
            values = (block == 0).groupby(d["_uc"]).sum()
        else:
            values = (
                block.groupby([d["_uc"], block])
                .size()
                .groupby(level=0)
                .max()
            )
    return values.astype(float)


@lru_cache(maxsize=32)
def user_groups(acts_t, metric, window, thresholds):
    """Assign users to groups using user-defined inclusive upper bounds.

    Group 0 holds values <= thresholds[0], group 1 holds values in
    (thresholds[0], thresholds[1]], ..., and the last group holds values above
    the final threshold. Empty groups are dropped.
    """
    values = user_values(acts_t, metric, window)
    if values.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int), []

    n_raw = len(thresholds) + 1
    raw = pd.Series(
        np.searchsorted(np.asarray(thresholds, dtype=float), values.to_numpy(), side="left"),
        index=values.index,
    )
    present = sorted(raw.unique())
    remap = {group_id: index for index, group_id in enumerate(present)}
    groups = raw.map(remap).astype(int)

    def raw_label(group_id):
        if n_raw == 1:
            return "G1 (all users)"
        if group_id == 0:
            return f"G1 (≤{thresholds[0]:g})"
        if group_id == n_raw - 1:
            return f"G{group_id + 1} (>{thresholds[-1]:g})"
        return (
            f"G{group_id + 1} (>{thresholds[group_id - 1]:g} "
            f"to ≤{thresholds[group_id]:g})"
        )

    labels = [raw_label(group_id) for group_id in present]
    return values, groups, labels


@app.callback(
    *[Output(f"g-th-{i}", "value") for i in range(MAX_GROUPS - 1)],
    *[Output(f"g-th-box-{i}", "style") for i in range(MAX_GROUPS - 1)],
    Input("g-metric", "value"),
    Input("g-window", "value"),
    Input("g-q", "value"),
    State("actions", "value"),
)
def suggest_thresholds(metric, window, q, acts):
    """Show q-1 threshold inputs and pre-fill them with quantile cut points."""
    n_inputs = MAX_GROUPS - 1
    n_active = int(q) - 1
    styles = [
        TH_BOX_STYLE if i < n_active else {"display": "none"}
        for i in range(n_inputs)
    ]
    suggestions = [None] * n_inputs

    acts = ordered(acts)
    if acts:
        values = user_values(tuple(acts), metric, window)
        if len(values):
            quantiles = np.linspace(0, 1, int(q) + 1)[1:-1]
            cuts = np.unique(np.floor(values.quantile(quantiles).to_numpy()))
            for i, cut in enumerate(cuts[:n_active]):
                suggestions[i] = int(cut)

    return (*suggestions, *styles)


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
    *[Input(f"g-th-{i}", "value") for i in range(MAX_GROUPS - 1)],
)
def group_mix(acts, metric, window, q, *raw_thresholds):
    acts = ordered(acts)
    if not acts:
        empty = empty_fig("Select at least one action")
        return "Select at least one action.", empty, empty, [], []

    thresholds = parse_thresholds(q, raw_thresholds)
    values, groups, labels = user_groups(tuple(acts), metric, window, thresholds)
    if len(groups) == 0:
        empty = empty_fig("No users")
        return "No users for the selected actions.", empty, empty, [], []

    group_count = len(labels)
    short = [label.split(" ")[0] for label in labels]
    d = srt[srt[ACTION_COL].isin(acts)]
    counts = pd.crosstab(d["_uc"], d[ACTION_COL]).reindex(
        columns=acts, fill_value=0
    )
    groups = groups.reindex(counts.index)
    share = counts.div(counts.sum(axis=1), axis=0)
    sizes = [int((groups == group_id).sum()) for group_id in range(group_count)]

    note = (
        f"Users split into {group_count} group(s) by {metric_label(metric, window)} "
        f"using your thresholds (G1 = lightest). Metric range: "
        f"min {values.min():g}, median {values.median():g}, max {values.max():g}. "
        "Group sizes: "
        + "; ".join(
            f"{labels[group_id]} n={sizes[group_id]:,}"
            for group_id in range(group_count)
        )
    )
    if group_count < len(thresholds) + 1:
        note += "\n(empty groups omitted: no users fall in some threshold ranges)"
    if len(thresholds) < int(q) - 1:
        note += "\n(some threshold inputs are empty or duplicated, so fewer groups are used)"
    if kruskal is None:
        note += "\nInstall scipy (pip install scipy) to get significance tests."
    else:
        note += (
            "\nKruskal-Wallis tests whether per-user action shares differ across "
            "groups. With large samples, p-values can be small; also inspect "
            "effect size (epsilon-squared: ~0.01 small, 0.06 medium, 0.14 large)."
        )

    # Left: mean per-user share and approximate 95% CI.
    fig1 = go.Figure()
    for group_id in range(group_count):
        sub = share[groups == group_id]
        fig1.add_trace(
            go.Bar(
                x=acts,
                y=sub.mean().values,
                name=f"{labels[group_id]}, n={sizes[group_id]:,}",
                marker_color=GROUP_COLORS[group_id % len(GROUP_COLORS)],
                error_y=dict(
                    type="data",
                    array=(1.96 * sub.sem().fillna(0)).values,
                ),
            )
        )
    fig1.update_layout(
        barmode="group",
        template="plotly_white",
        height=460,
        title="Mean share of each user's actions (±95% CI)",
        yaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
    )

    # Right: pooled events per group; heavy users naturally contribute more.
    pooled = counts.groupby(groups).sum()
    pooled_share = pooled.div(pooled.sum(axis=1), axis=0)
    x_labels = [labels[int(group_id)] for group_id in pooled_share.index]
    fig2 = go.Figure()
    for action in acts:
        fig2.add_trace(
            go.Bar(
                x=x_labels,
                y=pooled_share[action].values,
                name=action,
                marker_color=CMAP[action],
            )
        )
    fig2.update_layout(
        barmode="stack",
        template="plotly_white",
        height=460,
        title="All events pooled per group (heavy users dominate)",
        yaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
    )

    rows = []
    n_users = len(share)
    for action in acts:
        row = {"action": action}
        means = []
        for group_id in range(group_count):
            mean_share = float(share.loc[groups == group_id, action].mean())
            means.append(mean_share)
            row[f"{short[group_id]} mean %"] = round(100 * mean_share, 2)
        row["spread (pp)"] = round(100 * (max(means) - min(means)), 2)

        p_value = effect = None
        if kruskal is not None and group_count >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    statistic, p_value = kruskal(
                        *[
                            share.loc[groups == group_id, action].to_numpy()
                            for group_id in range(group_count)
                        ]
                    )
                effect = statistic / (n_users - 1) if n_users > 1 else None
                if not (np.isfinite(statistic) and np.isfinite(p_value)):
                    p_value = effect = None
            except ValueError:
                p_value = effect = None

        row["Kruskal p"] = None if p_value is None else float(f"{p_value:.3g}")
        row["effect size (eps^2)"] = (
            None if effect is None else round(float(effect), 3)
        )
        row["signif. (Bonferroni)"] = (
            "" if p_value is None
            else ("yes" if p_value < 0.05 / len(acts) else "no")
        )
        rows.append(row)

    columns = [{"name": column, "id": column} for column in rows[0].keys()]
    return note, fig1, fig2, rows, columns


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
    *[Input(f"g-th-{i}", "value") for i in range(MAX_GROUPS - 1)],
)
def group_trans(acts, gap_key, metric, window, q, view, *raw_thresholds):
    acts = ordered(acts)
    if not acts:
        return empty_fig("Select at least one action"), "", []

    thresholds = parse_thresholds(q, raw_thresholds)
    values, groups, labels = user_groups(tuple(acts), metric, window, thresholds)
    group_count = len(labels)
    if group_count < 2:
        return empty_fig("Only one group: adjust the thresholds or metric"), "", []

    d = srt[srt[ACTION_COL].isin(acts)]
    if len(d) < 2:
        return empty_fig("Not enough events"), "", []

    action_codes = d["_a"].to_numpy()
    users = d["_uc"].to_numpy()
    timestamps = d["_t"].to_numpy()
    same_user = users[1:] == users[:-1]
    gap_seconds = (timestamps[1:] - timestamps[:-1]) / np.timedelta64(1, "s")
    valid = same_user & (gap_seconds >= 0) & (gap_seconds <= GAPS[gap_key])
    if valid.sum() == 0:
        return empty_fig("No transitions with this filter"), "", []

    group_map = np.full(int(srt["_uc"].max()) + 1, -1, dtype=np.int64)
    group_map[groups.index.to_numpy()] = groups.to_numpy()
    from_groups = group_map[users[:-1]][valid]
    pair_codes = (
        action_codes[:-1][valid].astype(np.int64) * K
        + action_codes[1:][valid]
    )
    all_group_pair_codes = from_groups * K * K + pair_codes
    all_counts = np.bincount(
        all_group_pair_codes,
        minlength=group_count * K * K,
    ).reshape(group_count, K, K).astype(float)

    indices = [CODE[action] for action in acts]
    group_counts = all_counts[:, indices][:, :, indices]
    row_support = group_counts.sum(axis=2)
    users_per_group = [int((groups == group_id).sum()) for group_id in range(group_count)]

    with np.errstate(divide="ignore", invalid="ignore"):
        group_probabilities = np.divide(
            group_counts,
            row_support[:, :, None],
            out=np.full_like(group_counts, np.nan),
            where=row_support[:, :, None] > 0,
        )
        all_counts_selected = group_counts.sum(axis=0)
        all_row_support = all_counts_selected.sum(axis=1, keepdims=True)
        all_probabilities = np.divide(
            all_counts_selected,
            all_row_support,
            out=np.full_like(all_counts_selected, np.nan),
            where=all_row_support > 0,
        )

    group_probabilities = np.where(
        (row_support >= MIN_SUPPORT)[:, :, None],
        group_probabilities,
        np.nan,
    )

    if view == "prob":
        values_to_plot = group_probabilities * 100
        formatter = lambda value: "" if np.isnan(value) else f"{value:.1f}%"
        color_kwargs = dict(colorscale="Blues", zmin=0, zmax=100, colorbar=dict(title="%"))
    else:
        values_to_plot = (group_probabilities - all_probabilities[None]) * 100
        finite = np.abs(values_to_plot[np.isfinite(values_to_plot)])
        color_limit = float(max(10, min(50, np.percentile(finite, 99)))) if finite.size else 10.0
        formatter = lambda value: "" if np.isnan(value) else f"{value:+.1f}"
        color_kwargs = dict(
            colorscale="RdBu", zmin=-color_limit, zmax=color_limit,
            colorbar=dict(title="pp vs all"),
        )

    ncols = min(group_count, 3)
    nrows = math.ceil(group_count / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[
            f"{labels[group_id]}, {users_per_group[group_id]:,} users"
            for group_id in range(group_count)
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    for group_id in range(group_count):
        fig.add_trace(
            go.Heatmap(
                z=values_to_plot[group_id],
                x=acts,
                y=acts,
                text=[[formatter(value) for value in row] for row in values_to_plot[group_id]],
                texttemplate="%{text}",
                showscale=(group_id == 0),
                **color_kwargs,
                hovertemplate="%{y} -> %{x}<br>%{text}<extra></extra>",
            ),
            row=group_id // ncols + 1,
            col=group_id % ncols + 1,
        )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=250 + 330 * nrows)

    note = (
        f"Rows = previous action, columns = next action ({GAP_LABELS[gap_key]}). "
        f"Cells where a group has fewer than {MIN_SUPPORT} previous-action events "
        "are blank. "
        + (
            "Values are the group's P(next | previous) in %."
            if view == "prob"
            else "Values are percentage points above (+) or below (-) all users."
        )
    )

    difference_rows = []
    for i, from_action in enumerate(acts):
        for j, to_action in enumerate(acts):
            available = [
                (group_id, group_probabilities[group_id, i, j] * 100)
                for group_id in range(group_count)
                if np.isfinite(group_probabilities[group_id, i, j])
            ]
            if len(available) < 2:
                continue
            highest = max(available, key=lambda item: item[1])
            lowest = min(available, key=lambda item: item[1])
            difference_rows.append(
                {
                    "from": from_action,
                    "to": to_action,
                    "highest group": labels[highest[0]],
                    "lowest group": labels[lowest[0]],
                    "spread (pp)": round(float(highest[1] - lowest[1]), 2),
                    "P highest %": round(float(highest[1]), 2),
                    "P lowest %": round(float(lowest[1]), 2),
                }
            )
    difference_rows.sort(key=lambda row: -row["spread (pp)"])
    return fig, note, difference_rows[:30]


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)