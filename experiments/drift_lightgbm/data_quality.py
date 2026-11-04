"""Ingestion outages and bursts in the activity tables.

The release has whole days with no Pushes rows (e.g. 2026-09-02..04, 10-07,
10-15..17, everything after 10-26) and days where every other table jumps
together (2026-08-07, most of 09-03..09-17). A day is compared with the
median of the same weekday in the previous four weeks, so ordinary weekend
dips are not flagged.

``label_window_flags`` marks (cutoff, label) pairs whose 7-day target window
touches an anomalous day of that label's table; those labels are unreliable
for training and scoring. ``push_coverage_features`` counts push outage days
inside the feature windows; it only looks at days before the cutoff, and
Pushes rows are available at their creation time, so it is cutoff-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.lightgbm_baseline.run import HORIZON_DAYS, LABELS

OUTAGE_RATIO = 0.2
BURST_RATIO = 4.0
MIN_BASELINE = 5.0
BASELINE_WEEKS = 4


def daily_counts(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows per UTC calendar day for each label's source table (cohort rows)."""
    columns = {}
    for label in LABELS:
        days = histories[label]["action_at"].dt.floor("D")
        columns[label] = days.value_counts()
    counts = pd.DataFrame(columns).fillna(0)
    full_range = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    return counts.reindex(full_range, fill_value=0).astype(np.float64)


def anomaly_days(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return boolean (outage, burst) frames indexed like ``counts``.

    Push volume is stable, so a push outage is flagged on its own. Cohort
    activity in the other five tables swings 5-10x for weeks at a time (e.g.
    the September surge), so a non-push day is flagged only when all five
    tables are abnormal together, which points at ingestion rather than users.
    """
    previous = [counts.shift(7 * week) for week in range(1, BASELINE_WEEKS + 1)]
    baseline = pd.concat(previous).groupby(level=0).median().reindex(counts.index)
    reliable = baseline >= MIN_BASELINE
    low = (reliable & (counts < OUTAGE_RATIO * baseline)).fillna(False)
    high = (reliable & (counts > BURST_RATIO * baseline)).fillna(False)

    non_push = [label for label in LABELS if label != "push"]
    outage = pd.DataFrame(False, index=counts.index, columns=list(LABELS))
    burst = pd.DataFrame(False, index=counts.index, columns=list(LABELS))
    outage["push"] = low["push"]
    systemic_outage = low[non_push].all(axis=1)
    systemic_burst = high[non_push].all(axis=1)
    for label in non_push:
        outage[label] = systemic_outage
        burst[label] = systemic_burst
    return outage, burst


def _window_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Calendar days overlapping the half-open interval [start, end)."""
    return pd.date_range(start.floor("D"), (end - pd.Timedelta(microseconds=1)).floor("D"))


def label_window_flags(
    cutoffs: list[pd.Timestamp], outage: pd.DataFrame, burst: pd.DataFrame
) -> pd.DataFrame:
    """One row per cutoff: ``flag_<label>`` is 1 when the target window is anomalous."""
    anomalous = outage | burst
    rows = []
    for cutoff in cutoffs:
        days = _window_days(cutoff, cutoff + pd.Timedelta(days=HORIZON_DAYS))
        window = anomalous.reindex(days, fill_value=False)
        rows.append(
            {"cutoff": cutoff, **{f"flag_{label}": int(window[label].any()) for label in LABELS}}
        )
    return pd.DataFrame(rows)


def push_coverage_features(
    cutoff: pd.Timestamp, outage: pd.DataFrame, lag_days: int = 0
) -> dict[str, int]:
    """Push outage days among the full days before ``cutoff`` (7 and 30 day windows).

    With an ingestion lag, the last ``lag_days`` days hold no pushes yet and count as outages.
    """
    last_full_day = cutoff.floor("D") - pd.Timedelta(days=1)
    first_unseen_day = (cutoff - pd.Timedelta(days=lag_days)).floor("D")
    values = {}
    for days in (7, 30):
        window = pd.date_range(last_full_day - pd.Timedelta(days=days - 1), last_full_day)
        missing = outage["push"].reindex(window, fill_value=False).to_numpy()
        if lag_days:
            missing = missing | (window >= first_unseen_day)
        values[f"push_outage_days_{days}d"] = int(missing.sum())
    return values
