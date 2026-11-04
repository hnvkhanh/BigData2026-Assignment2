"""Interactive analysis of the weekly activity-prediction task (Dash).

Install:
    pip install pandas pyarrow plotly dash scikit-learn

Run from the repository root:
    python data_analyze.py            # then open http://127.0.0.1:8052
    python data_analyze.py --port 8060
    python data_analyze.py --results-tag full    # only that one tag, instead of every tag

Inputs:
- the Parquet database in ``assignment2/data/db`` (daily volumes, eligibility, test list),
- the weekly examples cached by ``experiments/drift_lightgbm`` (labels, flags, features),
- every ``backtest_metrics.json`` under ``experiments/results/*/runs/*/`` (score, per-label F1,
  holdout probabilities where available) -- every tag and every run folder found there, not only
  experiments currently registered in ``run_all_experiments.py``.
Build the first with ``python -m experiments.drift_lightgbm.run backtest`` (examples cache) and
the second with ``python run_all_experiments.py`` (one or more times; each tag is picked up).

Tabs: Findings (written analysis computed from the data), Data quality, Weekly labels,
Population & test list, Habits, Feature explorer, Model results, Thresholds & errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from experiments.drift_lightgbm.data_quality import (
    BASELINE_WEEKS,
    anomaly_days,
    daily_counts,
)
from experiments.lightgbm_baseline.run import (
    DEFAULT_THRESHOLDS,
    ELIGIBILITY_WINDOW_DAYS,
    LABELS,
    chronological_splits,
    load_histories,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_all_experiments as rae  # noqa: E402  (registry + results of every experiment)

DB_DIR = ROOT / "assignment2" / "data" / "db"
TEST_INPUT = ROOT / "assignment2" / "evaluation" / "test_input.csv"
DRIFT_DIR = ROOT / "experiments" / "drift_lightgbm"
GRAPH_ELIGIBILITY_DAYS = 14
MIN_EVENTS = 5
L = list(LABELS)
PERIODS = ("train", "validation", "holdout")
PERIOD_COLORS = {"train": "#4C78A8", "validation": "#F58518", "holdout": "#54A24B"}
LABEL_COLORS = dict(zip(L, ("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2")))
NAIVE = "naive_last_week"
TEMPLATE = "plotly_white"


# ---------------------------------------------------------------- scoring
def _f1(tp: float, fp: float, fn: float) -> float:
    """Same edge case as assignment2/evaluation/evaluate.py: nothing to find and nothing found = 1."""
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def official(truth: np.ndarray, guess: np.ndarray) -> tuple[float, list[float]]:
    """(macro-F1 + micro-F1) / 2 of one week, plus per-label F1."""
    truth, guess = truth.astype(bool), guess.astype(bool)
    tp = (truth & guess).sum(0)
    fp = (~truth & guess).sum(0)
    fn = (truth & ~guess).sum(0)
    per_label = [_f1(*counts) for counts in zip(tp, fp, fn)]
    return (float(np.mean(per_label)) + _f1(tp.sum(), fp.sum(), fn.sum())) / 2, per_label


def fixed_thresholds() -> np.ndarray:
    return np.array([DEFAULT_THRESHOLDS[label] for label in L])


# ---------------------------------------------------------------- loading
@dataclass
class Data:
    daily: pd.DataFrame          # cohort rows per UTC day and label
    actors: pd.DataFrame         # distinct cohort users per UTC day and label
    ratio: pd.DataFrame          # daily rows / same-weekday 4-week median
    outage: pd.DataFrame
    burst: pd.DataFrame
    table_end: dict[str, pd.Timestamp]
    ex: pd.DataFrame             # weekly examples (90-day eligibility) with period and window counts
    weekly: pd.DataFrame         # one row per cutoff
    split: tuple[pd.Timestamp, pd.Timestamp]
    overlap: pd.DataFrame        # daily overlap between the test list and the eligible pool
    groups: pd.DataFrame         # test users at the list's snapshot + weekly pools: activity summaries
    results: pd.DataFrame        # one row per backtest variant
    feature_sets: dict[str, list[str]]
    first_seen_max: pd.Timestamp  # latest first_seen_at of any cohort user


def _latest(pattern: str) -> Path:
    paths = sorted(DRIFT_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise SystemExit(
            f"No {pattern} under {DRIFT_DIR}. Run: python -m experiments.drift_lightgbm.run backtest"
        )
    return paths[-1]


def _window_counts(events: pd.DataFrame, at: pd.Timestamp, days: int) -> pd.Series:
    """Events per user in [at - days, at), counting only rows already available at ``at``."""
    start = at - pd.Timedelta(days=days)
    times = events["action_at"].to_numpy()
    keep = (times >= np.datetime64(start)) & (times < np.datetime64(at))
    keep &= events["available_at"].to_numpy() < np.datetime64(at)
    return events.loc[keep, "user_id"].value_counts()


def _baseline_ratio(counts: pd.DataFrame) -> pd.DataFrame:
    previous = [counts.shift(7 * week) for week in range(1, BASELINE_WEEKS + 1)]
    baseline = pd.concat(previous).groupby(level=0).median().reindex(counts.index)
    return counts / baseline.where(baseline >= 5)


def _naive_row(ex: pd.DataFrame) -> dict:
    """'Positive if the user did it in the last 7 days' scored like the backtest."""
    def per_week(frame: pd.DataFrame) -> list[dict]:
        weeks = []
        for cutoff, week in frame.groupby("cutoff"):
            guess = np.column_stack([week[f"{label}_count_7d"].to_numpy() > 0 for label in L])
            score, per_label = official(week[L].to_numpy(), guess)
            weeks.append({"cutoff": str(cutoff), "score": score, "per_action_f1": dict(zip(L, per_label))})
        return weeks

    holdout = per_week(ex[ex["period"] == "holdout"])
    validation = per_week(ex[ex["period"] == "validation"])
    return {
        "variant": NAIVE,
        "description": "positive iff the user did the action in the last 7 days (no model)",
        "holdout": np.mean([week["score"] for week in holdout]),
        "validation": np.mean([week["score"] for week in validation]),
        "holdout_map": np.nan,
        "validation_map": np.nan,
        "per_week": holdout,
        **{f"f1_{label}": np.mean([week["per_action_f1"][label] for week in holdout]) for label in L},
        "thresholds": None,
        "csv": None,
    }


def _discover_metrics(results_tag: str | None) -> list[tuple[str, str, Path]]:
    """Every (experiment run-folder name, tag, backtest_metrics.json) under experiments/results/
    -- every tag subfolder found there, or just one with ``results_tag``. Nothing is deduplicated:
    the same experiment name under two different tags is kept as two separate rows."""
    if results_tag:
        tag_dirs = [rae.RESULTS_ROOT / results_tag]
    elif rae.RESULTS_ROOT.is_dir():
        tag_dirs = sorted(p for p in rae.RESULTS_ROOT.iterdir() if p.is_dir())
    else:
        tag_dirs = []
    return [
        (path.parent.name, tag_dir.name, path)
        for tag_dir in tag_dirs
        for path in sorted(tag_dir.glob("runs/*/backtest_metrics.json"))
    ]


def _branch_csv(run_dir: Path, branch: str) -> Path | None:
    """Holdout-probabilities CSV for one experiment/branch: drift_lightgbm keeps one per
    branch under variants/, every other experiment keeps every branch's columns in one file."""
    candidates = [run_dir / "variants" / f"{branch}_holdout.csv"] if branch else []
    candidates.append(run_dir / "holdout_probabilities.csv")
    return next((path for path in candidates if path.is_file()), None)


def load_results(ex: pd.DataFrame, results_tag: str | None) -> pd.DataFrame:
    """Every experiment with a finished backtest anywhere under experiments/results/ -- not only
    the ones currently registered in run_all_experiments.py's EXPERIMENTS list, and not deduplicated
    across tags: the experiment is named after its results/<tag>/runs/<name> run folder, qualified
    with its tag so the same name under two tags stays two distinct rows. Reuses
    run_all_experiments.py's own headline/branch flattening (``_metric_blocks``), so this stays in
    sync with that script."""
    found = _discover_metrics(results_tag)
    if not found:
        where = rae.RESULTS_ROOT / results_tag if results_tag else rae.RESULTS_ROOT
        raise SystemExit(f"No backtest_metrics.json under {where}. Run: python run_all_experiments.py")
    registry = {experiment.name: experiment for experiment in rae.EXPERIMENTS}
    rows = []
    for name, tag, path in found:
        metrics = json.loads(path.read_text())
        run_dir = path.parent
        registered = registry.get(name)
        group = registered.group if registered else "other"
        qualified = f"{name} ({tag})"
        for branch, block in rae._metric_blocks(metrics):
            if "score" not in block:
                continue
            description = registered.description if registered else ""
            if branch:
                description = f"{description} — branch: {branch}" if description else f"branch: {branch}"
            # The headline branch ("") has no name of its own in the shared schema, but
            # drift_lightgbm still keys its per-variant CSV by the underlying variant name.
            file_branch = branch or metrics.get("headline_variant", "")
            csv = _branch_csv(run_dir, file_branch)
            rows.append({
                "variant": f"{qualified}/{branch}" if branch else qualified,
                "experiment": name, "tag": tag, "branch": branch, "group": group,
                "description": description, "holdout": block["score"],
                "per_week": block.get("per_week", []),
                **{f"f1_{label}": block.get("per_action_f1", {}).get(label) for label in L},
                "thresholds": block.get("thresholds"),
                "csv": str(csv) if csv else None,
            })
    rows.append({**_naive_row(ex), "experiment": NAIVE, "branch": "", "group": "naive"})
    return pd.DataFrame(rows).sort_values("holdout", ascending=False).reset_index(drop=True)


def load_data(results_tag: str | None = None) -> Data:
    print("Loading histories ...", flush=True)
    users = pd.read_parquet(DB_DIR / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort", "first_seen_at"])
    cohort = users.loc[users["is_cohort"]]
    histories = load_histories(DB_DIR, cohort["user_id"].astype(int).tolist())
    daily = daily_counts(histories)
    outage, burst = anomaly_days(daily)
    # A few PRs/reviews carry creation times years before the collection window; show the window only.
    window_start = histories["push"]["action_at"].min().floor("D")
    daily, outage, burst = (frame.loc[window_start:] for frame in (daily, outage, burst))
    actors = pd.DataFrame(
        {label: histories[label].groupby(histories[label]["action_at"].dt.floor("D"))["user_id"].nunique()
         for label in L}
    ).reindex(daily.index).fillna(0)
    table_end = {label: histories[label]["action_at"].max() for label in L}
    events = pd.concat([histories[label][["user_id", "action_at", "available_at"]] for label in L])

    print("Loading weekly examples ...", flush=True)
    examples_path = _latest("cache/examples_lag0_*.parquet")
    feature_sets = json.loads(examples_path.with_suffix(".json").read_text())
    ex = pd.read_parquet(examples_path)
    floats = ex.select_dtypes("float64").columns
    ex[floats] = ex[floats].astype(np.float32)
    cutoffs = sorted(ex["cutoff"].unique())
    validation_start, holdout_start = chronological_splits([pd.Timestamp(c) for c in cutoffs])
    ex["period"] = np.select(
        [ex["cutoff"] < validation_start, ex["cutoff"] < holdout_start], ["train", "validation"], "holdout"
    )
    for days in (GRAPH_ELIGIBILITY_DAYS, ELIGIBILITY_WINDOW_DAYS):
        column = np.zeros(len(ex), dtype=np.int32)
        for cutoff in cutoffs:
            rows = (ex["cutoff"] == cutoff).to_numpy()
            counts = _window_counts(events, pd.Timestamp(cutoff), days)
            column[rows] = ex.loc[rows, "user_id"].map(counts).fillna(0).to_numpy()
        ex[f"events_{days}d"] = column

    weekly = ex.groupby("cutoff").agg(
        users=("user_id", "size"),
        period=("period", "first"),
        **{f"rate_{label}": (label, "mean") for label in L},
        **{f"flag_{label}": (f"flag_{label}", "max") for label in L},
    )
    graph_pool = ex[ex[f"events_{GRAPH_ELIGIBILITY_DAYS}d"] >= MIN_EVENTS].groupby("cutoff")
    weekly["users_14d"] = graph_pool.size()
    for label in L:
        weekly[f"rate14_{label}"] = graph_pool[label].mean()

    print("Comparing the test list with the eligible pool ...", flush=True)
    test_ids = pd.read_csv(TEST_INPUT)["UserID"]
    lookup = cohort.set_index("github_user_id")["user_id"]
    test_users = set(lookup.reindex(test_ids).dropna().astype(int))
    data_end = max(table_end[label] for label in L if label != "push")
    overlap_rows = []
    for day in pd.date_range("2026-08-01", data_end.floor("D"), freq="D"):
        at = day + pd.Timedelta(hours=16)
        eligible = set(_window_counts(events, at, ELIGIBILITY_WINDOW_DAYS).loc[lambda s: s >= MIN_EVENTS].index)
        both = len(eligible & test_users)
        overlap_rows.append({
            "at": at, "eligible": len(eligible),
            "test_eligible_share": both / len(test_users),
            "eligible_in_test_share": both / max(len(eligible), 1),
        })
    overlap = pd.DataFrame(overlap_rows)

    # The test users as of the day the list matches best, next to the weekly training pools.
    snapshot = overlap.loc[overlap["test_eligible_share"].idxmax(), "at"]
    test = pd.DataFrame({"user_id": sorted(test_users)})
    test["group"] = f"test list ({snapshot:%b %d})"
    test["events_90d"] = test["user_id"].map(_window_counts(events, snapshot, ELIGIBILITY_WINDOW_DAYS)).fillna(0)
    test["any_active_weeks_4w"] = sum(
        test["user_id"].isin(_window_counts(events, snapshot - pd.Timedelta(days=7 * week), 7).index).astype(int)
        for week in range(4)
    )
    for label in L:
        test[f"active_30d_{label}"] = test["user_id"].isin(_window_counts(histories[label], snapshot, 30).index)
    pools = ex[["user_id", "period", f"events_{ELIGIBILITY_WINDOW_DAYS}d", "any_active_weeks_4w"]].rename(
        columns={f"events_{ELIGIBILITY_WINDOW_DAYS}d": "events_90d"})
    pools["group"] = pools.pop("period") + " pool (user-weeks)"
    for label in L:
        pools[f"active_30d_{label}"] = ex[f"{label}_count_30d"].to_numpy() > 0
    groups = pd.concat([test, pools], ignore_index=True)

    where = rae.RESULTS_ROOT / results_tag if results_tag else f"{rae.RESULTS_ROOT} (every tag)"
    print(f"Loading backtest results from {where} ...", flush=True)
    results = load_results(ex, results_tag)
    return Data(daily, actors, _baseline_ratio(daily), outage, burst, table_end, ex, weekly,
                (pd.Timestamp(validation_start), pd.Timestamp(holdout_start)), overlap, groups,
                results, feature_sets, cohort["first_seen_at"].max())


@lru_cache(maxsize=16)
def holdout_frame(csv: str, branch: str = "") -> pd.DataFrame:
    """Holdout predictions with the active branch's probability columns renamed to p_<label>."""
    frame = pd.read_csv(csv, parse_dates=["cutoff"])
    for label in L:
        actual = f"actual_{label}"
        prefixes = sorted({c[: -len(f"_{label}")] for c in frame.columns
                           if c.endswith(f"_{label}") and c != actual})
        prefix = branch if branch in prefixes else next(iter(prefixes), None)
        if prefix:
            frame = frame.rename(columns={f"{prefix}_{label}": f"p_{label}"})
    return frame


# ---------------------------------------------------------------- figure helpers
def day_ranges(flags: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of True days as (first day, last day)."""
    runs, start, previous = [], None, None
    for day in flags.index[flags.to_numpy()]:
        if start is None or day - previous > pd.Timedelta(days=1):
            if start is not None:
                runs.append((start, previous))
            start = day
        previous = day
    if start is not None:
        runs.append((start, previous))
    return runs


def fmt_ranges(runs: list[tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    return ", ".join(
        a.strftime("%b %d") if a == b else f"{a.strftime('%b %d')}-{b.strftime('%b %d')}" for a, b in runs
    ) or "none"


def shade_periods(fig: go.Figure, data: Data, row: int | None = None) -> None:
    start, end = data.weekly.index.min(), data.weekly.index.max() + pd.Timedelta(days=7)
    validation_start, holdout_start = data.split
    holdout_end = data.ex.loc[data.ex["period"] == "holdout", "cutoff"].max() + pd.Timedelta(days=7)
    spans = (("train", start, validation_start), ("validation", validation_start, holdout_start),
             ("holdout", holdout_start, min(end, holdout_end)))
    for period, x0, x1 in spans:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=PERIOD_COLORS[period], opacity=0.07, line_width=0,
                      annotation_text=period, annotation_position="top left",
                      row=row, col=1 if row else None)


def bin_series(values: pd.Series, max_bins: int = 8) -> pd.Series:
    """Ordered categorical bins: exact values when few, otherwise 0 plus quantiles of the rest."""
    values = values.astype(float)
    unique = np.sort(values.dropna().unique())
    if len(unique) <= 10:
        labels = [f"{v:g}" for v in unique]
        return pd.Categorical(values.map(lambda v: f"{v:g}" if pd.notna(v) else None), categories=labels,
                              ordered=True)
    positive = values[values > 0]
    edges = np.unique(np.quantile(positive, np.linspace(0, 1, max_bins)))
    edges = np.concatenate([[-np.inf, 0], edges[1:-1], [np.inf]]) if (values <= 0).any() else \
        np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
    names = []
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 0:
            names.append("≤0")
        elif np.isinf(high):
            names.append(f">{low:.3g}")
        elif low in (-np.inf, 0):
            names.append(f"(0, {high:.3g}]" if low == 0 else f"≤{high:.3g}")
        else:
            names.append(f"({low:.3g}, {high:.3g}]")
    return pd.cut(values, edges, labels=names, ordered=True)


def empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(template=TEMPLATE, xaxis_visible=False, yaxis_visible=False, height=300)
    return fig


def card(title: str, *children, note: str | None = None) -> html.Div:
    parts = [html.H4(title, style={"margin": "0 0 6px"})]
    if note:
        parts.append(html.P(note, className="note"))
    return html.Div(parts + list(children), className="card")


# ---------------------------------------------------------------- findings
def threshold_sweep(frame: pd.DataFrame, label: str, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for cutoff, week in frame.groupby("cutoff"):
        truth = week[f"actual_{label}"].to_numpy() == 1
        probs = week[f"p_{label}"].to_numpy()
        for threshold in grid:
            guess = probs >= threshold
            rows.append({"cutoff": cutoff, "threshold": threshold,
                         "f1": _f1((truth & guess).sum(), (~truth & guess).sum(), (truth & ~guess).sum())})
    return pd.DataFrame(rows)


def findings_markdown(data: Data) -> str:
    ex, weekly, results = data.ex, data.weekly, data.results.set_index("variant")
    month = weekly.index.to_series().dt.strftime("%b")

    def mean_rate(label: str, months: list[str]) -> float:
        return 100 * weekly.loc[month.isin(months), f"rate_{label}"].mean()

    holdout = ex[ex["period"] == "holdout"]
    prev = {label: holdout[f"{label}_count_7d"] > 0 for label in L}
    persistence = "\n".join(
        f"| {label} | {100 * holdout[label].mean():.1f}% | {100 * holdout.loc[prev[label], label].mean():.0f}% "
        f"| {100 * holdout.loc[~prev[label], label].mean():.1f}% | {100 * (1 - prev[label][holdout[label] == 1].mean()):.0f}% |"
        for label in L
    )
    best = results.drop(index=NAIVE)["holdout"].idxmax()
    reference = next((n for n in results.index if n != NAIVE and "relational" in n.lower()), best)
    model_lines = [f"| `{name}` | {results.loc[name, 'group']} | {100 * results.loc[name, 'holdout']:.2f} |"
                   for name in dict.fromkeys((reference, best, NAIVE)) if name in results.index]

    threshold_lines, volume_lines, segment_text = [], [], ""
    if reference in results.index and results.loc[reference, "csv"]:
        frame = holdout_frame(results.loc[reference, "csv"], results.loc[reference, "branch"])
        grid = np.round(np.arange(0.05, 0.86, 0.025), 3)
        for label in L:
            sweep = threshold_sweep(frame, label, grid)
            best_per_week = sweep.loc[sweep.groupby("cutoff")["f1"].idxmax()]
            fixed = sweep[np.isclose(sweep["threshold"], DEFAULT_THRESHOLDS[label])]
            threshold_lines.append(
                f"| {label} | {DEFAULT_THRESHOLDS[label]:.2f} | {best_per_week['threshold'].min():.2f}-"
                f"{best_per_week['threshold'].max():.2f} | {100 * (best_per_week['f1'].mean() - fixed['f1'].mean()):+.1f} |"
            )
        thresholds = fixed_thresholds()
        for cutoff, week in frame.groupby("cutoff"):
            predicted = (week[[f"p_{label}" for label in L]].to_numpy() >= thresholds).sum(0)
            actual = week[[f"actual_{label}" for label in L]].to_numpy().sum(0)
            volume_lines.append(f"| {cutoff:%b %d} | " + " | ".join(f"{p / a:.2f}" if a else "–"
                                                                       for p, a in zip(predicted, actual)) + " |")
        merged = frame.merge(ex[["user_id", "cutoff", "any_active_weeks_4w"]], on=["user_id", "cutoff"])
        seg = []
        for weeks_active, part in merged.groupby("any_active_weeks_4w"):
            guess = part[[f"p_{label}" for label in L]].to_numpy() >= thresholds
            score, _ = official(part[[f"actual_{label}" for label in L]].to_numpy(), guess)
            seg.append(f"{int(weeks_active)}/4 weeks: {100 * score:.1f} ({len(part):,} rows)")
        segment_text = "; ".join(seg)

    overlap = data.overlap
    best_overlap = overlap.loc[overlap["test_eligible_share"].idxmax()]
    groups = data.groups
    test = groups[groups["group"].str.startswith("test")]
    holdout_pool = groups[groups["group"] == "holdout pool (user-weeks)"]
    train_pool = groups[groups["group"] == "train pool (user-weeks)"]
    push_end, other_end = data.table_end["push"], max(v for k, v in data.table_end.items() if k != "push")
    systemic_burst = data.burst["open_pr"]
    systemic_outage = data.outage["open_pr"]
    first_seen_max = data.first_seen_max
    columns = {"Mar-Apr": ["Mar", "Apr"], "May": ["May"], "Jul-Aug": ["Jul", "Aug"], "Sep": ["Sep"], "Oct": ["Oct"]}

    return f"""
### 1. The cohort's activity is not stationary; the test week will look like October, not like training

Mean weekly positive rate of the eligible users (≥{MIN_EVENTS} events in {ELIGIBILITY_WINDOW_DAYS} days):

| Label | {' | '.join(columns)} |
|---|{'---|' * len(columns)}
""" + "\n".join(
        f"| {label} | " + " | ".join(f"{mean_rate(label, months):.1f}%" for months in columns.values()) + " |"
        for label in L
    ) + f"""

* Every cohort user was first seen by {first_seen_max:%Y-%m-%d}, i.e. the cohort was **selected from users active
  in Dec-May**. Activity decays after the selection window (regression to the mean): from May to July-August
  open_pr falls {mean_rate('open_pr', ['May']) / mean_rate('open_pr', ['Jul', 'Aug']):.0f}x and push
  {mean_rate('push', ['May']) / mean_rate('push', ['Jul', 'Aug']):.1f}x. The weekly eligible pool shrinks with it
  ({weekly['users'].max():,} → {weekly['users'].iloc[-1]:,} users).
* In September the five non-push labels jump most of the way back to spring levels at once, while push
  does not move (see *Weekly labels*). The jump sits on days where all five non-push tables exceed 4x
  their usual volume together ({fmt_ranges(day_ranges(systemic_burst))}), which looks like a collection
  backfill rather than user behaviour.
* The default 70/15/15 split therefore trains on spring + summer, validates on the
  summer trough + September jump, and holds out the October decay. Validation scores (~32-35) and
  holdout scores (~54) are not comparable, and anything tuned on validation (thresholds) transfers badly.

### 2. Data-collection gaps hit the labels directly

* Push outage days (push rows < 20% of the usual same weekday): {fmt_ranges(day_ranges(data.outage['push']))}.
  A user who pushed on those days still gets push = 0, so push labels in those weeks are partly wrong.
* Systemic non-push outage days: {fmt_ranges(day_ranges(systemic_outage))}.
* Pushes end on **{push_end:%Y-%m-%d %H:%M}**, the other tables on **{other_end:%Y-%m-%d %H:%M}**. If the final
  release has the same lag, the push features of the test week are computed on ~{(other_end - push_end).days} missing
  days; the backtest shows this costs ~2 points unless training uses the same lag.

### 3. The test list is the {ELIGIBILITY_WINDOW_DAYS}-day eligible pool of about {best_overlap['at']:%b %d}

* {len(test):,} test users. On {best_overlap['at']:%b %d} {100 * best_overlap['test_eligible_share']:.1f}% of them have
  ≥{MIN_EVENTS} events in the previous {ELIGIBILITY_WINDOW_DAYS} days, and on {overlap['at'].iloc[-1]:%b %d}
  {100 * overlap['eligible_in_test_share'].iloc[-1]:.1f}% of the eligible users are in the list: the list is that
  pool, frozen about {(pd.Timestamp('2026-11-11 16:00') - best_overlap['at']).days} days before the test week.
* Compared with the training pools the test users are the survivors of the decay: median 90-day events
  {test['events_90d'].median():.0f} (holdout pool {holdout_pool['events_90d'].median():.0f}, train pool
  {train_pool['events_90d'].median():.0f}); {100 * (test['any_active_weeks_4w'] == 0).mean():.0f}% had no activity in
  the 4 weeks before the snapshot (holdout pool {100 * (holdout_pool['any_active_weeks_4w'] == 0).mean():.0f}%,
  train pool {100 * (train_pool['any_active_weeks_4w'] == 0).mean():.0f}%). The holdout pool is the closest match.
* The graph models' ≥{MIN_EVENTS}-in-{GRAPH_ELIGIBILITY_DAYS}-day rule keeps only ~{100 * (weekly['users_14d'] / weekly['users']).mean():.0f}% of
  the eligible users, the most active ones (holdout push rate
  {100 * weekly.loc[weekly['period'] == 'holdout', 'rate14_push'].mean():.0f}% vs
  {100 * weekly.loc[weekly['period'] == 'holdout', 'rate_push'].mean():.0f}%), so their old leaderboard scores were
  measured on an easier population than the test list.

### 4. Most of the signal is "did they do it recently"

Holdout weeks, eligible users:

| Label | Base rate | P(positive \\| did it last week) | P(positive \\| did not) | Positives that are new this week |
|---|---|---|---|---|
{persistence}

* Over half of the positives for every non-push label come from users who did *not* do that action last
  week; these are the hard cases and the main source of false negatives.
* `pr_review` and `pr_review_comment` co-occur strongly (see *Habits*), the other labels much less.

| Model | Group | Holdout |
|---|---|---|
{chr(10).join(model_lines)}

The no-model rule "positive iff they did it in the last 7 days" is
{100 * (results.loc[reference, 'holdout'] - results.loc[NAIVE, 'holdout']):.1f} points behind `{reference}` on the
holdout. Its gain comes from ranking the users in between, not from a new signal — see *Model results* for every
registered experiment (`python run_all_experiments.py --list`), ranked by the same holdout score.

### 5. Thresholds: the best threshold moves every week

`{reference}`, holdout weeks:

| Label | Fixed threshold | Best threshold per week | F1 gain if each week's best were known |
|---|---|---|---|
{chr(10).join(threshold_lines)}

Predicted positives / actual positives at the fixed thresholds (1.00 = right volume):

| Week | {' | '.join(L)} |
|---|{'---|' * len(L)}
{chr(10).join(volume_lines)}

* Knowing each week's best threshold in advance would add only 1-3 F1 points per label, and it cannot be
  known: the positive rate moves 3-10x between weeks. The fixed 0.30 / 0.45 thresholds are the safe choice.
* The model under-predicts the rare labels (open_issue, pr_review_comment) in most weeks; raising their
  probabilities is where a better model would gain.
* Push volume jumps when the label week contains a push outage (e.g. the week starting Oct 14 includes
  Oct 15-18): the model predicts real pushes that the data never recorded. Those weeks penalise good
  predictions, which is why masked scores are reported in the backtest.

### 6. Where the model fails

Official score by how many of the last 4 weeks the user was active at all ({reference}, holdout):
{segment_text}.
Users with no activity in the last 4 weeks score 0: none of the few who come back is caught. Quality rises
steadily with recent activity, so the score of a week depends heavily on its mix of quiet and regular users.
Quiet users are {100 * (test['any_active_weeks_4w'] == 0).mean():.0f}% of the test list at its snapshot vs
{100 * (holdout_pool['any_active_weeks_4w'] == 0).mean():.0f}% of the holdout rows, so on this count the test week should
not score below the holdout, provided the October decay does not continue.

### What this means for the submission

1. Train on all completed weeks (most recent behaviour matters most) with the relational features
   and the fixed thresholds: `python -m experiments.drift_lightgbm.run predict`.
2. Check the push lag in the final release and use a lag-matched variant if pushes trail the other tables.
3. Do not tune thresholds or pick models on the validation weeks; they are dominated by the September jump.
"""


# ---------------------------------------------------------------- app
CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f6f7f9; color: #1f2328; }
.page { max-width: 1400px; margin: 0 auto; padding: 16px; }
.card { background: white; border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; margin-bottom: 14px; }
.note { color: #57606a; font-size: 13px; margin: 0 0 8px; }
.controls { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; margin-bottom: 8px; }
.controls label { font-size: 13px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 14px; }
.markdown table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
.markdown td, .markdown th { border: 1px solid #d8dee4; padding: 3px 8px; }
.markdown h3 { margin-top: 22px; }
"""


def build_app(data: Data) -> Dash:
    app = Dash(__name__, title="Activity data analysis")
    app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
    results = data.results
    scored = results.loc[results["csv"].notna()].sort_values("holdout", ascending=False)
    with_csv = scored["variant"].tolist()
    default_variant = with_csv[0] if with_csv else None
    feature_options = [
        {"label": f"{column}  [{name}]", "value": column}
        for name, columns in data.feature_sets.items() for column in columns
    ] + [{"label": f"events_{d}d  [eligibility]", "value": f"events_{d}d"}
         for d in (GRAPH_ELIGIBILITY_DAYS, ELIGIBILITY_WINDOW_DAYS)]
    label_dropdown = lambda id_, value="open_pr": dcc.Dropdown(  # noqa: E731
        L, value, id=id_, clearable=False, style={"width": 220})
    variant_dropdown = lambda id_: dcc.Dropdown(  # noqa: E731
        with_csv, default_variant, id=id_, clearable=False, style={"width": 280})
    segment_options = ["any_active_weeks_4w", "any_active_days_7d", "total_actions_count_7d",
                       f"events_{ELIGIBILITY_WINDOW_DAYS}d", "days_since_first_action", "<label>_active_weeks_4w"]

    tabs = [
        dcc.Tab(label="Findings", children=[card(
            "Result analysis", dcc.Markdown(findings_markdown(data), className="markdown"),
            note="Computed from the current data release and backtest outputs when the app starts.")]),
        dcc.Tab(label="Data quality", children=[
            card("Daily volume per table (cohort users)",
                 html.Div([
                     dcc.RadioItems([{"label": "rows", "value": "rows"},
                                     {"label": "distinct users", "value": "actors"}], "rows", id="dq-measure", inline=True),
                     dcc.RadioItems(["daily", "7-day mean"], "daily", id="dq-smooth", inline=True),
                     dcc.RadioItems(["linear", "log"], "log", id="dq-scale", inline=True),
                     dcc.Checklist(L, L, id="dq-labels", inline=True),
                 ], className="controls"),
                 dcc.Graph(id="dq-volume"),
                 note="Shaded: red = push outage day, purple = all five non-push tables > 4x usual, "
                      "grey = all five non-push tables < 0.2x usual (data_quality.py)."),
            card("Volume relative to the usual same weekday", dcc.Graph(id="dq-ratio", figure=ratio_figure(data)),
                 note="log2(rows / median of the same weekday in the previous 4 weeks). "
                      "Vertical stripes across all tables are collection events, not user behaviour."),
        ]),
        dcc.Tab(label="Weekly labels", children=[
            card("Weekly positive rate per label",
                 html.Div([dcc.Checklist(L, L, id="wk-labels", inline=True),
                           dcc.RadioItems([{"label": f"≥{MIN_EVENTS} events in {ELIGIBILITY_WINDOW_DAYS}d (test rule)", "value": "90"},
                                           {"label": f"≥{MIN_EVENTS} events in {GRAPH_ELIGIBILITY_DAYS}d (old graph rule)", "value": "14"}],
                                          "90", id="wk-rule", inline=True)], className="controls"),
                 dcc.Graph(id="wk-rates"),
                 note="One point per Wednesday 16:00 UTC cutoff; the label covers the following 7 days. "
                      "× = label week touches a flagged collection anomaly."),
            card("Eligible users per week", dcc.Graph(figure=eligible_figure(data))),
        ]),
        dcc.Tab(label="Population & test list", children=[
            card("How the test list overlaps the eligible pool over time", dcc.Graph(figure=overlap_figure(data)),
                 note=f"Eligible = ≥{MIN_EVENTS} events in the previous {ELIGIBILITY_WINDOW_DAYS} days at 16:00 UTC. "
                      "Pushes stop after the push data end, so eligibility drops artificially at the end."),
            html.Div([
                card("90-day activity: test list vs training pools", dcc.Graph(figure=groups_hist_figure(data)),
                     note="Test users as of the day the list matches the eligible pool best; pools = every "
                          "eligible user-week of that period."),
                card("Share active in the last 30 days, by group", dcc.Graph(figure=groups_active_figure(data))),
            ], className="grid2"),
        ]),
        dcc.Tab(label="Habits", children=[
            card("Positive rate by recent habit",
                 html.Div([label_dropdown("hb-label"),
                           dcc.RadioItems(["4w", "12w", "26w"], "4w", id="hb-window", inline=True)], className="controls"),
                 dcc.Graph(id="hb-habit"),
                 note="x = number of the last N weeks in which the user did this action. Lines per period "
                      "show whether the relation is stable over time."),
            card("Persistence over time", dcc.Graph(id="hb-persist"),
                 note="P(positive next week) for users who did / did not do the action in the 7 days before the cutoff."),
            card("Label co-occurrence in the same week",
                 dcc.RadioItems([{"label": "P(column | row)", "value": "cond"}, {"label": "correlation", "value": "corr"}],
                                "cond", id="hb-cooc-mode", inline=True),
                 dcc.Graph(id="hb-cooc")),
        ]),
        dcc.Tab(label="Feature explorer", children=[
            card("One feature against one label",
                 html.Div([dcc.Dropdown(feature_options, "open_pr_count_7d", id="fx-feature", clearable=False,
                                        style={"width": 460}),
                           label_dropdown("fx-label")], className="controls"),
                 html.Div(id="fx-auc", className="note"),
                 html.Div([dcc.Graph(id="fx-rate"), dcc.Graph(id="fx-dist")], className="grid2"),
                 note="Bins: 0 on its own, then quantiles of the positive values. Diverging period lines = "
                      "the feature means something different over time."),
            card("Univariate AUC of every feature for the label", dcc.Graph(id="fx-top"),
                 note="ROC AUC of the raw feature (|AUC - 0.5| ranked), on a 40k-row sample per period."),
        ]),
        dcc.Tab(label="Model results", children=[
            card("Holdout score of every finished experiment", dcc.Graph(figure=variant_scatter(data)),
                 note="Every backtest_metrics.json found under experiments/results/*/runs/*/, scored the same "
                      "way: mean over holdout weeks of evaluate.py's Score. Multi-model experiments (chunks, "
                      "drift, relgt fusion) contribute one bar per branch. "
                      f"`{NAIVE}` is the no-model rule 'did it in the last 7 days'."),
            card("Per-week holdout score",
                 dcc.Dropdown(results["variant"].tolist(),
                              [v for v in results.sort_values("holdout", ascending=False)["variant"].head(3)] + [NAIVE],
                              id="mr-variants", multi=True),
                 dcc.Graph(id="mr-weeks")),
            card("Per-label holdout F1", dcc.Graph(figure=variant_heatmap(data))),
            card("All variants", dash_table.DataTable(
                data=results.assign(holdout=(100 * results["holdout"]).round(2))[
                    ["variant", "group", "holdout", "description"]
                ].to_dict("records"),
                sort_action="native", style_cell={"textAlign": "left", "fontSize": 13, "whiteSpace": "normal"},
                style_table={"overflowX": "auto"})),
        ]),
        dcc.Tab(label="Thresholds & errors", children=[
            card("Holdout predictions of one variant",
                 html.Div([variant_dropdown("th-variant"), label_dropdown("th-label")], className="controls"),
                 html.Div([dcc.Graph(id="th-sweep"), dcc.Graph(id="th-calib")], className="grid2"),
                 note="Left: F1 for every threshold, one line per holdout week (dashed = fixed threshold, "
                      "◆ = best threshold that week). Right: predicted probability vs observed rate per week."),
            card("Predicted vs actual number of positives (fixed thresholds)", dcc.Graph(id="th-volume")),
            card("Error segments",
                 dcc.Dropdown(segment_options, "any_active_weeks_4w", id="er-segment", clearable=False,
                              style={"width": 360}),
                 html.Div([dcc.Graph(id="er-f1"), dcc.Graph(id="er-counts")], className="grid2"),
                 note="<label>_active_weeks_4w uses each label's own habit feature."),
        ]),
    ]
    app.layout = html.Div([
        html.H2("GitHub user activity: data and result analysis", style={"margin": "4px 0 10px"}),
        dcc.Tabs(tabs),
    ], className="page")
    register_callbacks(app, data)
    return app


# ---------------------------------------------------------------- static figures
def ratio_figure(data: Data) -> go.Figure:
    ratio = np.log2(data.ratio.clip(lower=1 / 64, upper=64))
    fig = go.Figure(go.Heatmap(z=ratio.T.to_numpy(), x=ratio.index, y=L, colorscale="RdBu", zmid=0,
                               zmin=-4, zmax=4, colorbar={"title": "log2"}))
    fig.update_layout(template=TEMPLATE, height=300, margin={"t": 20, "b": 30})
    return fig


def eligible_figure(data: Data) -> go.Figure:
    weekly = data.weekly
    fig = go.Figure([
        go.Bar(x=weekly.index, y=weekly["users"], name=f"≥{MIN_EVENTS} in {ELIGIBILITY_WINDOW_DAYS}d", marker_color="#4C78A8"),
        go.Bar(x=weekly.index, y=weekly["users_14d"], name=f"≥{MIN_EVENTS} in {GRAPH_ELIGIBILITY_DAYS}d", marker_color="#F58518"),
    ])
    shade_periods(fig, data)
    fig.update_layout(template=TEMPLATE, barmode="overlay", height=300, yaxis_title="users", margin={"t": 30})
    return fig


def overlap_figure(data: Data) -> go.Figure:
    overlap = data.overlap
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=overlap["at"], y=100 * overlap["test_eligible_share"], name="% of test users eligible")
    fig.add_scatter(x=overlap["at"], y=100 * overlap["eligible_in_test_share"], name="% of eligible users in test list")
    fig.add_scatter(x=overlap["at"], y=overlap["eligible"], name="eligible users", line={"dash": "dot", "color": "grey"},
                    secondary_y=True)
    fig.add_vline(x=data.table_end["push"], line_dash="dash", line_color="red")
    fig.add_annotation(x=data.table_end["push"], y=1, yref="paper", text="push data ends", showarrow=False,
                       font={"color": "red"})
    fig.update_layout(template=TEMPLATE, height=340, margin={"t": 30})
    fig.update_yaxes(title_text="%", secondary_y=False)
    fig.update_yaxes(title_text="users", secondary_y=True)
    return fig


def _group_colors(data: Data) -> dict[str, str]:
    return {group: "#E45756" if group.startswith("test") else PERIOD_COLORS[group.split()[0]]
            for group in data.groups["group"].unique()}


def groups_hist_figure(data: Data) -> go.Figure:
    fig = go.Figure()
    for group, color in _group_colors(data).items():
        values = np.log10(1 + data.groups.loc[data.groups["group"] == group, "events_90d"])
        fig.add_histogram(x=values, name=f"{group} ({len(values):,})", histnorm="percent", opacity=0.5,
                          xbins={"size": 0.1}, marker_color=color)
    fig.update_layout(template=TEMPLATE, barmode="overlay", height=340, margin={"t": 20},
                      xaxis_title="log10(1 + events in the last 90 days)", yaxis_title="% of group")
    fig.add_vline(x=np.log10(1 + MIN_EVENTS), line_dash="dash")
    return fig


def groups_active_figure(data: Data) -> go.Figure:
    columns = [f"active_30d_{label}" for label in L]
    shares = data.groups.groupby("group")[columns].mean() * 100
    shares["inactive 4 weeks"] = data.groups.groupby("group")["any_active_weeks_4w"].apply(lambda s: (s == 0).mean()) * 100
    fig = go.Figure([go.Bar(x=L + ["no activity in last 4 weeks"], y=shares.loc[group].to_numpy(), name=group,
                            marker_color=color) for group, color in _group_colors(data).items()])
    fig.update_layout(template=TEMPLATE, barmode="group", height=340, yaxis_title="% of users", margin={"t": 20})
    return fig


def variant_scatter(data: Data) -> go.Figure:
    results = data.results.sort_values("holdout")
    groups = sorted(results["group"].unique())
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC"]
    group_color = dict(zip(groups, palette * (len(groups) // len(palette) + 1)))
    colors = [("#E45756" if v == NAIVE else group_color[g]) for v, g in zip(results["variant"], results["group"])]
    fig = go.Figure(go.Bar(
        x=100 * results["holdout"], y=results["variant"], orientation="h", marker_color=colors,
        customdata=np.column_stack([results["group"], results["description"]]),
        hovertemplate="%{y}<br>group: %{customdata[0]}<br>%{customdata[1]}<br>holdout %{x:.2f}<extra></extra>"))
    fig.update_layout(template=TEMPLATE, height=max(420, 22 * len(results)), xaxis_title="holdout score (%)",
                      margin={"t": 20, "l": 160})
    return fig


def variant_heatmap(data: Data) -> go.Figure:
    results = data.results
    z = 100 * results[[f"f1_{label}" for label in L]].to_numpy()
    fig = go.Figure(go.Heatmap(z=z, x=L, y=results["variant"], colorscale="Viridis", text=np.round(z, 1),
                               texttemplate="%{text}", colorbar={"title": "F1 %"}))
    fig.update_layout(template=TEMPLATE, height=max(320, 22 * len(results)), margin={"t": 20},
                      yaxis={"autorange": "reversed"})
    return fig


# ---------------------------------------------------------------- callbacks
def register_callbacks(app: Dash, data: Data) -> None:
    ex, results = data.ex, data.results.set_index("variant")

    @app.callback(Output("dq-volume", "figure"), Input("dq-measure", "value"), Input("dq-smooth", "value"),
                  Input("dq-scale", "value"), Input("dq-labels", "value"))
    def dq_volume(measure, smooth, scale, labels):
        frame = data.daily if measure == "rows" else data.actors
        if smooth == "7-day mean":
            frame = frame.rolling(7, min_periods=1).mean()
        fig = go.Figure([go.Scatter(x=frame.index, y=frame[label], name=label, line={"color": LABEL_COLORS[label]})
                         for label in labels])
        for flags, color in ((data.outage["push"], "red"), (data.burst["open_pr"], "purple"),
                             (data.outage["open_pr"], "grey")):
            for start, end in day_ranges(flags):
                fig.add_vrect(x0=start, x1=end + pd.Timedelta(days=1), fillcolor=color, opacity=0.15, line_width=0)
        fig.update_layout(template=TEMPLATE, height=420, yaxis_type=scale, margin={"t": 20},
                          yaxis_title="rows per day" if measure == "rows" else "users per day")
        return fig

    @app.callback(Output("wk-rates", "figure"), Input("wk-labels", "value"), Input("wk-rule", "value"))
    def wk_rates(labels, rule):
        weekly = data.weekly
        prefix = "rate_" if rule == "90" else "rate14_"
        fig = go.Figure()
        for label in labels:
            fig.add_scatter(x=weekly.index, y=100 * weekly[prefix + label], name=label, mode="lines+markers",
                            line={"color": LABEL_COLORS[label]})
            flagged = weekly[weekly[f"flag_{label}"] == 1]
            fig.add_scatter(x=flagged.index, y=100 * flagged[prefix + label], mode="markers", showlegend=False,
                            marker={"symbol": "x", "size": 11, "color": LABEL_COLORS[label]}, hoverinfo="skip")
        shade_periods(fig, data)
        fig.update_layout(template=TEMPLATE, height=460, yaxis_title="% of eligible users positive", margin={"t": 30})
        return fig

    @app.callback(Output("hb-habit", "figure"), Output("hb-persist", "figure"),
                  Input("hb-label", "value"), Input("hb-window", "value"))
    def habits(label, window):
        column = f"{label}_active_weeks_{window}"
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for period in PERIODS:
            part = ex[ex["period"] == period]
            stats = part.groupby(column)[label].agg(["mean", "size"])
            fig.add_scatter(x=stats.index, y=100 * stats["mean"], name=period, mode="lines+markers",
                            line={"color": PERIOD_COLORS[period]})
            fig.add_bar(x=stats.index, y=stats["size"], name=f"{period} rows", marker_color=PERIOD_COLORS[period],
                        opacity=0.2, showlegend=False, secondary_y=True)
        fig.update_layout(template=TEMPLATE, height=380, barmode="group", margin={"t": 20},
                          xaxis_title=f"weeks active in the last {window}")
        fig.update_yaxes(title_text="% positive next week", secondary_y=False)
        fig.update_yaxes(title_text="rows", secondary_y=True, showgrid=False)

        active = ex[f"{label}_count_7d"] > 0
        by_week = pd.DataFrame({
            "did it last week": ex[active].groupby("cutoff")[label].mean(),
            "did not": ex[~active].groupby("cutoff")[label].mean(),
            "all users": ex.groupby("cutoff")[label].mean(),
        })
        persist = go.Figure([go.Scatter(x=by_week.index, y=100 * by_week[name], name=name, mode="lines+markers")
                             for name in by_week.columns])
        shade_periods(persist, data)
        persist.update_layout(template=TEMPLATE, height=360, yaxis_title="% positive", margin={"t": 30})
        return fig, persist

    @app.callback(Output("hb-cooc", "figure"), Input("hb-cooc-mode", "value"))
    def cooccurrence(mode):
        y = ex[L].to_numpy().astype(float)
        if mode == "corr":
            z, title = np.corrcoef(y.T), "correlation"
        else:
            z, title = (y.T @ y) / y.sum(0)[:, None], "P(column | row)"
        fig = go.Figure(go.Heatmap(z=z, x=L, y=L, colorscale="Blues", text=np.round(z, 2), texttemplate="%{text}",
                                   colorbar={"title": title}))
        fig.update_layout(template=TEMPLATE, height=420, margin={"t": 20}, yaxis={"autorange": "reversed"})
        return fig

    @app.callback(Output("fx-rate", "figure"), Output("fx-dist", "figure"), Output("fx-auc", "children"),
                  Input("fx-feature", "value"), Input("fx-label", "value"))
    def feature_explorer(feature, label):
        bins = bin_series(ex[feature])
        frame = pd.DataFrame({"bin": bins, "period": ex["period"], "y": ex[label]})
        stats = frame.groupby(["period", "bin"], observed=False)["y"].agg(["mean", "size"]).reset_index()
        rate, dist = go.Figure(), go.Figure()
        aucs = []
        for period in PERIODS:
            part = stats[stats["period"] == period]
            rate.add_scatter(x=part["bin"].astype(str), y=100 * part["mean"], name=period, mode="lines+markers",
                             line={"color": PERIOD_COLORS[period]})
            dist.add_bar(x=part["bin"].astype(str), y=100 * part["size"] / part["size"].sum(), name=period,
                         marker_color=PERIOD_COLORS[period])
            rows = ex["period"] == period
            values = ex.loc[rows, feature].fillna(0)
            if ex.loc[rows, label].nunique() == 2:
                aucs.append(f"{period} AUC {roc_auc_score(ex.loc[rows, label], values):.3f}")
        rate.update_layout(template=TEMPLATE, height=380, yaxis_title=f"% {label} positive", margin={"t": 20},
                           xaxis_title=feature)
        dist.update_layout(template=TEMPLATE, height=380, yaxis_title="% of rows", barmode="group",
                           margin={"t": 20}, xaxis_title=feature)
        return rate, dist, " · ".join(aucs)

    @app.callback(Output("fx-top", "figure"), Input("fx-label", "value"))
    def feature_ranking(label):
        table = univariate_auc(label)
        top = table.head(25).iloc[::-1]
        fig = go.Figure([go.Bar(y=top.index, x=top[period], name=period, orientation="h",
                                marker_color=PERIOD_COLORS[period]) for period in PERIODS])
        fig.add_vline(x=0.5, line_dash="dash")
        fig.update_layout(template=TEMPLATE, height=720, barmode="group", xaxis_title="AUC", margin={"t": 20},
                          xaxis_range=[0, 1])
        return fig

    @lru_cache(maxsize=len(L))
    def univariate_auc(label: str) -> pd.DataFrame:
        features = [column for columns in data.feature_sets.values() for column in columns]
        table = {}
        for period in PERIODS:
            part = ex[ex["period"] == period]
            part = part.sample(min(len(part), 40_000), random_state=0)
            y = part[label].to_numpy()
            table[period] = {
                feature: roc_auc_score(y, part[feature].fillna(0).to_numpy()) if 0 < y.sum() < len(y) else np.nan
                for feature in features
            }
        table = pd.DataFrame(table)
        strength = (table - 0.5).abs().mean(axis=1)
        return table.loc[strength.sort_values(ascending=False).index]

    @app.callback(Output("mr-weeks", "figure"), Input("mr-variants", "value"))
    def per_week(variants):
        fig = go.Figure()
        for name in variants or []:
            weeks = results.loc[name, "per_week"]
            fig.add_scatter(x=[w["cutoff"] for w in weeks], y=[100 * w["score"] for w in weeks], name=name,
                            mode="lines+markers")
        fig.update_layout(template=TEMPLATE, height=380, yaxis_title="score (%)", margin={"t": 20},
                          xaxis_title="holdout week (cutoff)")
        return fig

    @app.callback(Output("th-sweep", "figure"), Output("th-calib", "figure"), Output("th-volume", "figure"),
                  Input("th-variant", "value"), Input("th-label", "value"))
    def thresholds(variant, label):
        if not variant or not results.loc[variant, "csv"]:
            return empty("no holdout predictions"), empty(""), empty("")
        frame = holdout_frame(results.loc[variant, "csv"], results.loc[variant, "branch"])
        used = results.loc[variant, "thresholds"] or DEFAULT_THRESHOLDS
        grid = np.round(np.arange(0.02, 0.91, 0.02), 3)
        sweep = threshold_sweep(frame, label, grid)
        fig = go.Figure()
        calib = go.Figure()
        for index, (cutoff, part) in enumerate(sweep.groupby("cutoff")):
            name = f"{cutoff:%b %d}"
            fig.add_scatter(x=part["threshold"], y=100 * part["f1"], name=name, mode="lines", legendgroup=name)
            best = part.loc[part["f1"].idxmax()]
            fig.add_scatter(x=[best["threshold"]], y=[100 * best["f1"]], mode="markers", legendgroup=name,
                            showlegend=False, marker={"symbol": "diamond", "size": 10})
            week = frame[frame["cutoff"] == cutoff]
            edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            buckets = pd.cut(week[f"p_{label}"], edges, include_lowest=True)
            stats = week.groupby(buckets, observed=True).agg(p=(f"p_{label}", "mean"), y=(f"actual_{label}", "mean"),
                                                             n=(f"actual_{label}", "size"))
            calib.add_scatter(x=stats["p"], y=stats["y"], name=name, mode="lines+markers", customdata=stats["n"],
                              hovertemplate="predicted %{x:.2f}, observed %{y:.2f}, n=%{customdata}")
        fig.add_vline(x=used[label], line_dash="dash")
        fig.update_layout(template=TEMPLATE, height=400, xaxis_title="threshold", yaxis_title=f"{label} F1 (%)",
                          margin={"t": 20})
        calib.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": "grey"}, showlegend=False)
        calib.update_layout(template=TEMPLATE, height=400, xaxis_title="mean predicted probability",
                            yaxis_title="observed positive rate", margin={"t": 20})

        threshold_vector = np.array([used[name] for name in L])
        rows = []
        for cutoff, week in frame.groupby("cutoff"):
            predicted = (week[[f"p_{name}" for name in L]].to_numpy() >= threshold_vector).sum(0)
            actual = week[[f"actual_{name}" for name in L]].to_numpy().sum(0)
            rows += [{"week": f"{cutoff:%b %d}", "label": name, "ratio": p / a if a else np.nan, "predicted": p,
                      "actual": a} for name, p, a in zip(L, predicted, actual)]
        volume = pd.DataFrame(rows)
        bars = go.Figure([go.Bar(x=part["week"], y=part["ratio"], name=name, marker_color=LABEL_COLORS[name],
                                 customdata=part[["predicted", "actual"]],
                                 hovertemplate="predicted %{customdata[0]} / actual %{customdata[1]}")
                          for name, part in volume.groupby("label", sort=False)])
        bars.add_hline(y=1, line_dash="dash")
        bars.update_layout(template=TEMPLATE, height=340, barmode="group", margin={"t": 20},
                           yaxis_title="predicted / actual positives")
        return fig, calib, bars

    @app.callback(Output("er-f1", "figure"), Output("er-counts", "figure"),
                  Input("th-variant", "value"), Input("er-segment", "value"))
    def error_segments(variant, segment):
        if not variant or not results.loc[variant, "csv"]:
            return empty("no holdout predictions"), empty("")
        frame = holdout_frame(results.loc[variant, "csv"], results.loc[variant, "branch"])
        used = results.loc[variant, "thresholds"] or DEFAULT_THRESHOLDS
        columns = sorted({c for c in ex.columns if c.endswith("_active_weeks_4w")} |
                         {"any_active_days_7d", "total_actions_count_7d", f"events_{ELIGIBILITY_WINDOW_DAYS}d",
                          "days_since_first_action"})
        merged = frame.merge(ex[["user_id", "cutoff", *columns]], on=["user_id", "cutoff"], how="left")
        z, counts, segment_names = [], [], None
        for label in L:
            source = f"{label}_active_weeks_4w" if segment == "<label>_active_weeks_4w" else segment
            bins = bin_series(merged[source], max_bins=6)
            truth = merged[f"actual_{label}"].to_numpy() == 1
            guess = merged[f"p_{label}"].to_numpy() >= used[label]
            frame_l = pd.DataFrame({"bin": bins, "tp": truth & guess, "fp": ~truth & guess, "fn": truth & ~guess})
            stats = frame_l.groupby("bin", observed=True)[["tp", "fp", "fn"]].sum()
            stats["n"] = frame_l.groupby("bin", observed=True).size()
            if segment_names is None or segment == "<label>_active_weeks_4w":
                segment_names = [str(b) for b in stats.index]
            z.append([_f1(*row) for row in stats[["tp", "fp", "fn"]].to_numpy()])
            counts.append(stats.assign(label=label))
        heat = go.Figure(go.Heatmap(z=100 * np.array(z), x=segment_names, y=L, colorscale="Viridis",
                                    text=np.round(100 * np.array(z), 1), texttemplate="%{text}",
                                    colorbar={"title": "F1 %"}))
        heat.update_layout(template=TEMPLATE, height=380, xaxis_title=segment, margin={"t": 20},
                           yaxis={"autorange": "reversed"})
        totals = pd.concat(counts).groupby(level=0, observed=True)[["tp", "fp", "fn", "n"]].sum()
        totals.index = totals.index.astype(str)
        bars = go.Figure([go.Bar(x=totals.index, y=totals[kind], name=name) for kind, name in
                          (("tp", "true positives"), ("fp", "false positives"), ("fn", "false negatives"))])
        bars.update_layout(template=TEMPLATE, height=380, barmode="stack", xaxis_title=segment,
                           yaxis_title="user-label pairs (all 6 labels)", margin={"t": 20})
        return heat, bars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--results-tag", default=None,
                        help="restrict to this one experiments/results/<tag> written by "
                             "run_all_experiments.py (default: every tag found there)")
    args = parser.parse_args()
    app = build_app(load_data(args.results_tag))
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
