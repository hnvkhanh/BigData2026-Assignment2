"""Global GitHub activity explorer with bot-vs-human analysis.

Install:
    pip install pandas pyarrow plotly dash scipy

Run from the repository root:
    python global_trends_bot_human_dash.py

Then open:
    http://127.0.0.1:8051

The input can be:
- The assignment Parquet database directory (containing Users.parquet, PullRequests.parquet, etc.)
- A Parquet/CSV/JSON/JSONL file
- A glob matching one or more supported files

For generic event files, configure USER_COL, TIME_COL, ACTION_COL, and make
sure an is_bot column is present to enable the bot-vs-human section.
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
BOT_MIN_SUPPORT = 30

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
from dash import Dash, Input, Output, dcc, html, dash_table

try:
    from scipy.stats import kruskal, mannwhitneyu
except Exception:  # scipy is optional; significance tests will be skipped
    kruskal = None
    mannwhitneyu = None

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
        columns=["user_id", "github_user_id", "is_cohort", "is_bot"],
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
                    "is_bot": rows[actor_col].map(cohort_users["is_bot"]),
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


def parse_bot_flag(series):
    """Convert common bool/numeric/string bot labels to nullable booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True, "1": True, "1.0": True,
        "yes": True, "y": True, "t": True, "bot": True,
        "false": False, "0": False, "0.0": False,
        "no": False, "n": False, "f": False, "human": False,
    }
    return normalized.map(mapping).astype("boolean")


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
srt_cols = ["_u", "_t", ACTION_COL]
if "is_bot" in df.columns:
    srt_cols.append("is_bot")
srt = df[srt_cols].sort_values(["_u", "_t"]).reset_index(drop=True)
srt["_uc"] = pd.factorize(srt["_u"])[0]

# Normalize bot/human labels at the event level, then enforce a single
# user-level label. Conflicts are counted and resolved by majority vote.
BOT_COLUMN_PRESENT = "is_bot" in srt.columns
BOT_FLAG_CONFLICTS = 0
BOT_UNKNOWN_USERS = 0

if BOT_COLUMN_PRESENT:
    srt["_bot_event"] = parse_bot_flag(srt["is_bot"])
    flag_counts = srt.groupby("_u")["_bot_event"].nunique(dropna=True)
    BOT_FLAG_CONFLICTS = int((flag_counts > 1).sum())

    def majority_bot_flag(values):
        values = values.dropna()
        if values.empty:
            return pd.NA
        modes = values.mode()
        return modes.iloc[0]

    user_bot_flags = srt.groupby("_u")["_bot_event"].agg(majority_bot_flag)
    BOT_UNKNOWN_USERS = int(user_bot_flags.isna().sum())
    srt["_bot"] = srt["_u"].map(user_bot_flags).astype("boolean")
else:
    srt["_bot"] = pd.Series(pd.NA, index=srt.index, dtype="boolean")

BOT_AVAILABLE = BOT_COLUMN_PRESENT and srt["_bot"].notna().any()

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
if BOT_AVAILABLE:
    valid_user_flags = srt.drop_duplicates("_u")["_bot"]
    print(
        f"Bot/human labels: {(valid_user_flags == True).sum():,} bots | "
        f"{(valid_user_flags == False).sum():,} humans | "
        f"{BOT_UNKNOWN_USERS:,} unknown users | "
        f"{BOT_FLAG_CONFLICTS:,} users with conflicting labels"
    )
else:
    print("Bot/human analysis disabled: no usable is_bot labels were found.")

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
BOT_NAMES = {True: "Bot", False: "Human"}
BOT_COLORS = {True: "#F58518", False: "#4C78A8"}
BOT_GAP_LABELS = ["<1m", "1–10m", "10m–1h", "1–6h", "6–24h", "1–7d", ">7d"]
BOT_GAP_EDGES = [0, 60, 600, 3600, 21600, 86400, 604800, np.inf]

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


def bh_adjust(p_values):
    """Benjamini-Hochberg adjusted p-values. NaNs remain NaN."""
    p_values = np.asarray(p_values, dtype=float)
    result = np.full(len(p_values), np.nan)
    valid = np.isfinite(p_values)
    if not valid.any():
        return result

    p = p_values[valid]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    restored = np.empty(n)
    restored[order] = adjusted
    result[valid] = restored
    return result


def compare_bot_human(bot_values, human_values):
    """Mann-Whitney U and Cliff's delta (positive means bots tend higher)."""
    bot_values = np.asarray(bot_values, dtype=float)
    human_values = np.asarray(human_values, dtype=float)
    bot_values = bot_values[np.isfinite(bot_values)]
    human_values = human_values[np.isfinite(human_values)]

    if len(bot_values) == 0 or len(human_values) == 0 or mannwhitneyu is None:
        return None, None, None

    try:
        result = mannwhitneyu(
            bot_values,
            human_values,
            alternative="two-sided",
            method="auto",
        )
        u_stat = float(result.statistic)
        p_value = float(result.pvalue)
        cliff_delta = 2 * u_stat / (len(bot_values) * len(human_values)) - 1
        return u_stat, p_value, float(cliff_delta)
    except Exception:
        return None, None, None


def safe_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


# ---------------------------------------------------------------- DASH APP
app = Dash(__name__)
app.title = "Global action trends: Bot vs Human"

TBL_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={"textAlign": "left", "fontSize": 13, "padding": "6px"},
    style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"},
)

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
                            options=[2, 3, 4, 5],
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

        # ---------------------------------------------------------- BOT VS HUMAN
        html.Hr(),
        html.H2("Bot vs Human Behavior Analysis"),
        html.Div(
            "This section compares bot and human accounts using user-level "
            "activity statistics, event timing, inter-event gaps, and action "
            "transitions. The action filter and transition time filter above "
            "also apply to this section.",
            style={"color": "#555", "marginBottom": "12px"},
        ),
        html.Pre(
            id="bot-summary",
            style={
                "whiteSpace": "pre-wrap",
                "backgroundColor": "#f5f5f5",
                "padding": "12px",
                "borderRadius": "6px",
            },
        ),
        html.H3("1. Do bots and humans perform different actions?"),
        html.Div(
            "Action mix is calculated per user, so each account has equal "
            "weight in the mean action-share comparison. Activity counts are "
            "shown on a log scale because user activity is often skewed.",
            style={"color": "#555", "marginBottom": "6px"},
        ),
        dcc.Graph(id="bot-action"),
        html.H4("Statistical comparison by action and activity measure"),
        dash_table.DataTable(
            id="bot-stats",
            page_size=20,
            sort_action="native",
            columns=[
                {"name": "Measure", "id": "measure"},
                {"name": "Bot users", "id": "Bot n"},
                {"name": "Human users", "id": "Human n"},
                {"name": "Bot median", "id": "Bot median"},
                {"name": "Human median", "id": "Human median"},
                {"name": "Median difference", "id": "median difference"},
                {"name": "Cliff's delta", "id": "Cliff's delta"},
                {"name": "p-value", "id": "p-value"},
                {"name": "BH-adjusted p", "id": "BH-adjusted p"},
            ],
            **TBL_STYLE,
        ),
        html.H3("2. Do bots and humans act at different times?"),
        html.Div(
            "The heatmaps show the percentage of each group's events in each "
            "hour/day cell. The difference heatmap is Bot minus Human in "
            "percentage points. These are event-weighted profiles.",
            style={"color": "#555", "marginBottom": "6px"},
        ),
        dcc.Graph(id="bot-time"),
        html.H3("3. How much time passes between consecutive actions?"),
        html.Div(
            "Gaps are measured between consecutive selected actions from the "
            "same user. If an action is deselected above, it is skipped when "
            "forming these selected-action sequences.",
            style={"color": "#555", "marginBottom": "6px"},
        ),
        dcc.Graph(id="bot-gaps"),
        html.H3("4. Do bots and humans follow different action sequences?"),
        html.Div(
            id="bot-transition-note",
            style={
                "color": "#555",
                "marginBottom": "6px",
                "whiteSpace": "pre-wrap",
            },
        ),
        dcc.Graph(id="bot-transitions"),
        html.H4("Transitions with the largest bot-human differences"),
        dash_table.DataTable(
            id="bot-transition-table",
            page_size=20,
            sort_action="native",
            columns=[
                {"name": "Previous action", "id": "from"},
                {"name": "Next action", "id": "to"},
                {"name": "Bot previous events", "id": "Bot prev n"},
                {"name": "Human previous events", "id": "Human prev n"},
                {"name": "Bot probability (%)", "id": "P Bot %"},
                {"name": "Human probability (%)", "id": "P Human %"},
                {"name": "Bot - Human (pp)", "id": "Difference (pp)"},
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
def user_groups(acts_t, metric, window, q):
    """Per-user activity metric and quantile group (0 = lightest)."""
    d = srt[srt[ACTION_COL].isin(acts_t)]
    if d.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int), []

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

    values = values.astype(float)
    try:
        groups = pd.qcut(values, q, labels=False, duplicates="drop")
    except ValueError:
        groups = pd.Series(0, index=values.index)
    groups = groups.astype(int)

    labels = []
    for group_id in sorted(groups.unique()):
        group_values = values[groups == group_id]
        low, high = group_values.min(), group_values.max()
        labels.append(
            f"G{group_id + 1} ({low:g})"
            if low == high
            else f"G{group_id + 1} ({low:g}-{high:g})"
        )
    return values, groups, labels


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
        empty = empty_fig("Select at least one action")
        return "Select at least one action.", empty, empty, [], []

    values, groups, labels = user_groups(tuple(acts), metric, window, q)
    if len(groups) == 0:
        empty = empty_fig("No users")
        return "No users for the selected actions.", empty, empty, [], []

    group_count = len(labels)
    d = srt[srt[ACTION_COL].isin(acts)]
    counts = pd.crosstab(d["_uc"], d[ACTION_COL]).reindex(
        columns=acts, fill_value=0
    )
    groups = groups.reindex(counts.index)
    share = counts.div(counts.sum(axis=1), axis=0)
    sizes = [int((groups == group_id).sum()) for group_id in range(group_count)]

    note = (
        f"Users split into {group_count} group(s) by {metric_label(metric, window)} "
        f"(quantiles; G1 = lightest). Group sizes: "
        + "; ".join(
            f"{labels[group_id]} n={sizes[group_id]:,}"
            for group_id in range(group_count)
        )
    )
    if group_count < q:
        note += f"\n(only {group_count} groups possible: many users share the same value)"
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
            row[f"G{group_id + 1} mean %"] = round(100 * mean_share, 2)
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
)
def group_trans(acts, gap_key, metric, window, q, view):
    acts = ordered(acts)
    if not acts:
        return empty_fig("Select at least one action"), "", []

    values, groups, labels = user_groups(tuple(acts), metric, window, q)
    group_count = len(labels)
    if group_count < 2:
        return empty_fig("Only one group: pick a different metric / fewer groups"), "", []

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

# ---------------------------------------------------------------- BOT VS HUMAN: ACTIONS + USER-LEVEL STATISTICS


@app.callback(
    Output("bot-summary", "children"),
    Output("bot-action", "figure"),
    Output("bot-stats", "data"),
    Input("actions", "value"),
)
def bot_action_analysis(acts):
    if not BOT_AVAILABLE:
        message = (
            "Bot/human analysis is unavailable because the input has no usable "
            "is_bot column. For the assignment database, check Users.parquet."
        )
        return message, empty_fig("is_bot labels unavailable"), []

    acts = ordered(acts)
    if not acts:
        return "Select at least one action.", empty_fig("No actions selected"), []

    d = srt[
        srt[ACTION_COL].isin(acts) & srt["_bot"].notna()
    ].copy()
    if d.empty:
        return "No labeled activity events.", empty_fig("No labeled events"), []

    # One row per user, with one column per selected action.
    counts = pd.crosstab(d["_u"], d[ACTION_COL]).reindex(
        columns=acts, fill_value=0
    )
    total_per_user = counts.sum(axis=1)
    shares = counts.div(total_per_user, axis=0)

    meta = d.groupby("_u").agg(
        _bot=("_bot", "first"),
        first_time=("_t", "min"),
        last_time=("_t", "max"),
        events=(ACTION_COL, "size"),
        active_days=("_t", lambda values: values.dt.normalize().nunique()),
    )
    meta["events_per_active_day"] = meta["events"] / meta["active_days"].clip(lower=1)
    meta["observation_span_days"] = (
        (meta["last_time"] - meta["first_time"]).dt.total_seconds() / 86400
    )

    # Consecutive selected-action time gaps per user.
    d["_gap_s"] = d.groupby("_u")["_t"].diff().dt.total_seconds()
    user_gap_median = d.groupby("_u")["_gap_s"].median()
    meta["median_gap_seconds"] = user_gap_median.reindex(meta.index)

    bot_meta = meta[meta["_bot"] == True]
    human_meta = meta[meta["_bot"] == False]
    n_bot = len(bot_meta)
    n_human = len(human_meta)
    bot_events = int(bot_meta["events"].sum())
    human_events = int(human_meta["events"].sum())
    total_events = bot_events + human_events

    summary = (
        f"Users: {n_bot:,} bots | {n_human:,} humans\n"
        f"Events: {bot_events:,} bot events | {human_events:,} human events\n"
        f"Bot share of selected events: {bot_events / max(total_events, 1):.1%}\n\n"
        f"Median actions per user:\n"
        f"  Bot: {bot_meta['events'].median():.1f}\n"
        f"  Human: {human_meta['events'].median():.1f}\n\n"
        f"Median active days per user:\n"
        f"  Bot: {bot_meta['active_days'].median():.1f}\n"
        f"  Human: {human_meta['active_days'].median():.1f}\n\n"
        f"Median actions per active day:\n"
        f"  Bot: {bot_meta['events_per_active_day'].median():.2f}\n"
        f"  Human: {human_meta['events_per_active_day'].median():.2f}\n\n"
        f"Median time between consecutive selected actions:\n"
        f"  Bot: {fmt_secs(bot_meta['median_gap_seconds'].median())}\n"
        f"  Human: {fmt_secs(human_meta['median_gap_seconds'].median())}\n\n"
        f"Data quality:\n"
        f"  Users with conflicting is_bot labels: {BOT_FLAG_CONFLICTS:,}\n"
        f"  Users with unknown is_bot labels: {BOT_UNKNOWN_USERS:,}"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Mean per-user action share (approx. 95% CI)",
            "Actions per user (log10 scale)",
        ),
        horizontal_spacing=0.15,
    )

    for flag in (True, False):
        name = BOT_NAMES[flag]
        user_ids = meta.index[meta["_bot"] == flag]
        if len(user_ids) == 0:
            continue

        per_user_shares = shares.reindex(user_ids)
        mean_shares = per_user_shares.mean()
        errors = 1.96 * per_user_shares.sem().fillna(0)
        fig.add_trace(
            go.Bar(
                x=acts,
                y=mean_shares.values,
                name=name,
                marker_color=BOT_COLORS[flag],
                error_y=dict(type="data", array=errors.values),
            ),
            row=1,
            col=1,
        )
        activity = meta.loc[user_ids, "events"]
        fig.add_trace(
            go.Box(
                y=np.log10(activity + 1),
                name=name,
                marker_color=BOT_COLORS[flag],
                boxmean=True,
                boxpoints="outliers",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(title_text="Mean share", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="log10(1 + actions per user)", row=1, col=2)
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", y=-0.2),
    )

    # User-level statistical comparisons. This avoids treating every event
    # from a highly active user as an independent sample.
    test_data = [
        ("Total actions per user", meta["events"]),
        ("Active days per user", meta["active_days"]),
        ("Actions per active day", meta["events_per_active_day"]),
        ("Observation span (days)", meta["observation_span_days"]),
        ("Median inter-event gap (seconds)", meta["median_gap_seconds"]),
    ]
    for action in acts:
        test_data.append((f"Share of {action} (%)", shares[action] * 100))

    rows = []
    raw_p_values = []
    for measure, values in test_data:
        bot_values = values.reindex(bot_meta.index).dropna().to_numpy()
        human_values = values.reindex(human_meta.index).dropna().to_numpy()
        _, p_value, delta = compare_bot_human(bot_values, human_values)
        raw_p_values.append(np.nan if p_value is None else p_value)

        bot_median = safe_median(bot_values)
        human_median = safe_median(human_values)
        rows.append(
            {
                "measure": measure,
                "Bot n": len(bot_values),
                "Human n": len(human_values),
                "Bot median": None if np.isnan(bot_median) else round(bot_median, 3),
                "Human median": None if np.isnan(human_median) else round(human_median, 3),
                "median difference": (
                    None
                    if np.isnan(bot_median) or np.isnan(human_median)
                    else round(bot_median - human_median, 3)
                ),
                "Cliff's delta": None if delta is None else round(delta, 3),
                "p-value": None if p_value is None else float(f"{p_value:.4g}"),
                "BH-adjusted p": None,
            }
        )

    adjusted_p_values = bh_adjust(raw_p_values)
    for row, adjusted_p in zip(rows, adjusted_p_values):
        row["BH-adjusted p"] = (
            None if not np.isfinite(adjusted_p) else float(f"{adjusted_p:.4g}")
        )

    if mannwhitneyu is None:
        summary += "\n\nInstall scipy (`pip install scipy`) to calculate Mann-Whitney tests."

    return summary, fig, rows

# ---------------------------------------------------------------- BOT VS HUMAN: TEMPORAL PATTERNS


@app.callback(
    Output("bot-time", "figure"),
    Output("bot-gaps", "figure"),
    Input("actions", "value"),
)
def bot_temporal_analysis(acts):
    if not BOT_AVAILABLE:
        unavailable = empty_fig("is_bot labels unavailable")
        return unavailable, unavailable

    acts = ordered(acts)
    if not acts:
        empty = empty_fig("Select actions")
        return empty, empty

    d = srt[
        srt[ACTION_COL].isin(acts) & srt["_bot"].notna()
    ].copy()
    if d.empty:
        empty = empty_fig("No labeled events")
        return empty, empty

    # Hour x weekday percentages within each group (event-weighted).
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    profiles = {}
    for flag in (True, False):
        sub = d[d["_bot"] == flag]
        profile = np.zeros((7, 24), dtype=float)
        if len(sub):
            grouped = sub.groupby([
                sub["_t"].dt.dayofweek,
                sub["_t"].dt.hour,
            ]).size()
            for (weekday, hour), count in grouped.items():
                profile[int(weekday), int(hour)] = 100 * count / len(sub)
        profiles[flag] = profile

    difference = profiles[True] - profiles[False]
    diff_limit = max(0.1, float(np.nanmax(np.abs(difference))))

    fig_time = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Bot event distribution (%)",
            "Human event distribution (%)",
            "Bot - Human (percentage points)",
        ),
        horizontal_spacing=0.12,
    )
    fig_time.add_trace(
        go.Heatmap(
            z=profiles[True],
            x=list(range(24)),
            y=day_names,
            colorscale="Blues",
            zmin=0,
            showscale=False,
            hovertemplate="%{y}, %{x}:00<br>%{z:.3f}% of bot events<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_time.add_trace(
        go.Heatmap(
            z=profiles[False],
            x=list(range(24)),
            y=day_names,
            colorscale="Blues",
            zmin=0,
            showscale=False,
            hovertemplate="%{y}, %{x}:00<br>%{z:.3f}% of human events<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_time.add_trace(
        go.Heatmap(
            z=difference,
            x=list(range(24)),
            y=day_names,
            colorscale="RdBu",
            zmin=-diff_limit,
            zmax=diff_limit,
            colorbar=dict(title="pp"),
            hovertemplate="%{y}, %{x}:00<br>Bot - Human: %{z:+.3f} pp<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig_time.update_layout(template="plotly_white", height=430)
    fig_time.update_xaxes(title_text="Hour of day (0–23)")
    fig_time.update_yaxes(autorange="reversed")

    # Inter-event gaps between consecutive selected actions by the same user.
    d["_gap_s"] = d.groupby("_u")["_t"].diff().dt.total_seconds()
    gaps = d[
        d["_gap_s"].notna()
        & (d["_gap_s"] >= 0)
        & d["_bot"].notna()
    ].copy()
    if gaps.empty:
        return fig_time, empty_fig("No consecutive same-user event pairs")

    gaps["_gap_bucket"] = pd.cut(
        gaps["_gap_s"],
        bins=BOT_GAP_EDGES,
        labels=BOT_GAP_LABELS,
        right=False,
        include_lowest=True,
    )

    fig_gap = go.Figure()
    for flag in (True, False):
        sub = gaps[gaps["_bot"] == flag]
        bucket_counts = sub["_gap_bucket"].value_counts().reindex(
            BOT_GAP_LABELS, fill_value=0
        )
        percentages = 100 * bucket_counts / max(len(sub), 1)
        fig_gap.add_trace(
            go.Bar(
                x=BOT_GAP_LABELS,
                y=percentages.values,
                name=BOT_NAMES[flag],
                marker_color=BOT_COLORS[flag],
                customdata=bucket_counts.values,
                hovertemplate=(
                    "%{x}<br>%{y:.2f}% of gaps<br>%{customdata:,} gaps<extra></extra>"
                ),
            )
        )
    fig_gap.update_layout(
        barmode="group",
        template="plotly_white",
        height=460,
        title="Distribution of time between consecutive actions",
        xaxis_title="Time between actions",
        yaxis_title="Percentage of same-user event gaps (%)",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig_time, fig_gap

# ---------------------------------------------------------------- BOT VS HUMAN: ACTION TRANSITIONS


@app.callback(
    Output("bot-transitions", "figure"),
    Output("bot-transition-note", "children"),
    Output("bot-transition-table", "data"),
    Input("actions", "value"),
    Input("gap", "value"),
)
def bot_transition_analysis(acts, gap_key):
    if not BOT_AVAILABLE:
        return empty_fig("is_bot labels unavailable"), "No usable is_bot labels.", []

    acts = ordered(acts)
    if not acts:
        return empty_fig("Select actions"), "", []

    d = srt[
        srt[ACTION_COL].isin(acts) & srt["_bot"].notna()
    ].copy()
    if len(d) < 2:
        return empty_fig("Not enough events"), "", []

    action_codes = d["_a"].to_numpy()
    users = d["_u"].to_numpy()
    timestamps = d["_t"].to_numpy()
    bot_flags = d["_bot"].astype(bool).to_numpy()

    same_user = users[1:] == users[:-1]
    gap_seconds = (timestamps[1:] - timestamps[:-1]) / np.timedelta64(1, "s")
    valid = (
        same_user
        & (gap_seconds >= 0)
        & (gap_seconds <= GAPS[gap_key])
    )
    if not valid.any():
        return empty_fig("No transitions with this filter"), "", []

    previous_codes = action_codes[:-1]
    next_codes = action_codes[1:]
    # Bot label belongs to the user; using the previous event's label is safe
    # because both events in a valid pair belong to the same account.
    previous_is_bot = bot_flags[:-1]

    indices = [CODE[action] for action in acts]
    matrices = {}
    probabilities = {}
    row_support = {}

    for flag in (True, False):
        group_valid = valid & (previous_is_bot == flag)
        pair_codes = (
            previous_codes[group_valid].astype(np.int64) * K
            + next_codes[group_valid].astype(np.int64)
        )
        full_matrix = np.bincount(
            pair_codes,
            minlength=K * K,
        ).reshape(K, K).astype(float)
        selected_matrix = full_matrix[np.ix_(indices, indices)]
        support = selected_matrix.sum(axis=1)
        probability = np.divide(
            selected_matrix,
            support[:, None],
            out=np.full_like(selected_matrix, np.nan),
            where=support[:, None] >= BOT_MIN_SUPPORT,
        )
        matrices[flag] = selected_matrix
        probabilities[flag] = probability
        row_support[flag] = support

    bot_probability = probabilities[True]
    human_probability = probabilities[False]
    difference = (bot_probability - human_probability) * 100

    finite_difference = np.abs(difference[np.isfinite(difference)])
    if finite_difference.size == 0:
        bot_n = int((valid & previous_is_bot).sum())
        human_n = int((valid & ~previous_is_bot).sum())
        message = (
            f"Used {bot_n:,} bot transitions and {human_n:,} human transitions "
            f"({GAP_LABELS[gap_key]}), but there are not enough previous-action "
            f"events in both groups. Each group needs at least {BOT_MIN_SUPPORT} "
            "events for a previous-action row."
        )
        return empty_fig("Insufficient transition support"), message, []

    diff_limit = min(100.0, max(5.0, float(np.percentile(finite_difference, 95))))
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Bot: P(next | previous)",
            "Human: P(next | previous)",
            "Bot - Human (percentage points)",
        ),
        horizontal_spacing=0.13,
    )

    def probability_text(probability_matrix):
        return [
            ["" if not np.isfinite(value) else f"{value * 100:.1f}%" for value in row]
            for row in probability_matrix
        ]

    difference_text = [
        ["" if not np.isfinite(value) else f"{value:+.1f}" for value in row]
        for row in difference
    ]

    fig.add_trace(
        go.Heatmap(
            z=bot_probability * 100,
            x=acts,
            y=acts,
            text=probability_text(bot_probability),
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=100,
            showscale=False,
            hovertemplate="%{y} -> %{x}<br>Bot: %{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=human_probability * 100,
            x=acts,
            y=acts,
            text=probability_text(human_probability),
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=100,
            showscale=False,
            hovertemplate="%{y} -> %{x}<br>Human: %{text}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=difference,
            x=acts,
            y=acts,
            text=difference_text,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-diff_limit,
            zmax=diff_limit,
            colorbar=dict(title="pp"),
            hovertemplate="%{y} -> %{x}<br>Bot - Human: %{text} pp<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template="plotly_white",
        height=max(480, 300 + 25 * len(acts)),
    )

    bot_pairs = int((valid & previous_is_bot).sum())
    human_pairs = int((valid & ~previous_is_bot).sum())
    note = (
        f"Used {bot_pairs:,} bot transitions and {human_pairs:,} human transitions "
        f"({GAP_LABELS[gap_key]}). Rows = previous action; columns = next action.\n"
        "Difference = Bot probability minus Human probability. Positive values "
        "mean the transition is more common for bots; negative values mean it is "
        "more common for humans.\n"
        f"Rows with fewer than {BOT_MIN_SUPPORT} previous-action events in either "
        "group are hidden. These transition differences are descriptive, not "
        "individual significance tests."
    )

    rows = []
    for i, from_action in enumerate(acts):
        for j, to_action in enumerate(acts):
            if not (
                np.isfinite(bot_probability[i, j])
                and np.isfinite(human_probability[i, j])
            ):
                continue
            bot_probability_pct = float(bot_probability[i, j] * 100)
            human_probability_pct = float(human_probability[i, j] * 100)
            rows.append(
                {
                    "from": from_action,
                    "to": to_action,
                    "Bot prev n": int(row_support[True][i]),
                    "Human prev n": int(row_support[False][i]),
                    "P Bot %": round(bot_probability_pct, 2),
                    "P Human %": round(human_probability_pct, 2),
                    "Difference (pp)": round(
                        bot_probability_pct - human_probability_pct, 2
                    ),
                }
            )
    rows.sort(key=lambda row: abs(row["Difference (pp)"]), reverse=True)
    return fig, note, rows[:50]


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)
