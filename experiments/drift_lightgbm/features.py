"""Longer-history and cohort-regime features, all computed from rows known before the cutoff."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.lightgbm_baseline.run import LABELS

HISTORY_WEEKS = 26
ACTIVE_WEEK_WINDOWS = (4, 12, 26)
LAGGED_WEEKS = (2, 3, 4)
REGIME_WEEKS = 4
DAY = pd.Timedelta(days=1)


def _known(history: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    return history.loc[
        (history["available_at"] < cutoff)
        & (history["action_at"] < cutoff)
        & (history["action_at"] >= start)
    ]


def _week_presence(known: pd.DataFrame, cutoff: pd.Timestamp, user_index: pd.Index) -> np.ndarray:
    """users x HISTORY_WEEKS boolean matrix; column k is the week [cutoff-7(k+1)d, cutoff-7kd)."""
    presence = np.zeros((len(user_index), HISTORY_WEEKS), dtype=bool)
    if known.empty:
        return presence
    weeks = ((cutoff - known["action_at"]) // pd.Timedelta(days=7)).to_numpy(dtype=np.int64)
    rows = user_index.get_indexer(known["user_id"])
    keep = (rows >= 0) & (weeks < HISTORY_WEEKS)
    presence[rows[keep], weeks[keep]] = True
    return presence


def history_features(
    cutoff: pd.Timestamp, histories: dict[str, pd.DataFrame], user_ids: list[int]
) -> pd.DataFrame:
    """Weekly habit, longer windows, trend, streak and account-age features."""
    user_index = pd.Index(user_ids)
    start = cutoff - pd.Timedelta(days=7 * HISTORY_WEEKS)
    values: dict[str, object] = {"user_id": user_ids}
    any_presence = np.zeros((len(user_ids), HISTORY_WEEKS), dtype=bool)
    day_frames = []
    first_by_label = []

    for label in LABELS:
        history = histories[label]
        known = _known(history, cutoff, start)
        presence = _week_presence(known, cutoff, user_index)
        any_presence |= presence
        for weeks in ACTIVE_WEEK_WINDOWS:
            values[f"{label}_active_weeks_{weeks}w"] = presence[:, :weeks].sum(axis=1).astype(np.int16)
        for week in LAGGED_WEEKS:
            values[f"{label}_active_week_minus{week}"] = presence[:, week - 1].astype(np.int8)

        age = (cutoff - known["action_at"]) / DAY
        count_7d = known.loc[age < 7].groupby("user_id").size().reindex(user_index, fill_value=0)
        count_30d = known.loc[age < 30].groupby("user_id").size().reindex(user_index, fill_value=0)
        count_90d = known.loc[age < 90].groupby("user_id").size().reindex(user_index, fill_value=0)
        values[f"{label}_count_30d"] = count_30d.to_numpy(dtype=np.int32)
        values[f"{label}_count_90d"] = count_90d.to_numpy(dtype=np.int32)
        expected_7d = count_30d.to_numpy(dtype=np.float32) * (7 / 30)
        values[f"{label}_trend_7d_vs_30d"] = (
            (count_7d.to_numpy(dtype=np.float32) + 1.0) / (expected_7d + 1.0)
        )

        day_frames.append(known.loc[age < 30, ["user_id", "action_at"]])
        all_known = history.loc[(history["available_at"] < cutoff) & (history["action_at"] < cutoff)]
        first_by_label.append(all_known.groupby("user_id")["action_at"].min().reindex(user_index))

    first_seen = pd.concat(first_by_label, axis=1).min(axis=1)
    recent = pd.concat(day_frames, ignore_index=True)
    recent_age = (cutoff - recent["action_at"]) / DAY
    recent = recent.assign(day=recent_age.astype(np.int64))
    for days in (7, 30):
        active_days = (
            recent.loc[recent["day"] < days]
            .drop_duplicates(["user_id", "day"])
            .groupby("user_id")
            .size()
            .reindex(user_index, fill_value=0)
        )
        values[f"any_active_days_{days}d"] = active_days.to_numpy(dtype=np.int16)

    # Consecutive active weeks ending at the cutoff (0 when the last week was idle).
    idle = ~any_presence
    first_idle = np.where(idle.any(axis=1), idle.argmax(axis=1), HISTORY_WEEKS)
    values["any_active_streak_weeks"] = first_idle.astype(np.int16)
    for weeks in ACTIVE_WEEK_WINDOWS:
        values[f"any_active_weeks_{weeks}w"] = any_presence[:, :weeks].sum(axis=1).astype(np.int16)
    values["days_since_first_action"] = (
        ((cutoff - first_seen) / DAY).fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
    )
    return pd.DataFrame(values)


def regime_features(
    cutoff: pd.Timestamp, histories: dict[str, pd.DataFrame], eligible_user_ids: list[int]
) -> dict[str, float]:
    """Share of eligible users doing each action in the last week, and vs the last 4 weeks.

    Every user at a cutoff gets the same values; they tell the model how active
    the whole cohort currently is, which drifts strongly between weeks.
    """
    eligible = pd.Index(eligible_user_ids)
    denominator = max(len(eligible), 1)
    values: dict[str, float] = {}
    for label in LABELS:
        known = _known(histories[label], cutoff, cutoff - pd.Timedelta(days=7 * REGIME_WEEKS))
        known = known.loc[known["user_id"].isin(eligible)]
        week = ((cutoff - known["action_at"]) // pd.Timedelta(days=7)).to_numpy(dtype=np.int64)
        rates = [
            known.loc[week == index, "user_id"].nunique() / denominator
            for index in range(REGIME_WEEKS)
        ]
        values[f"cohort_rate_{label}_1w"] = float(rates[0])
        values[f"cohort_rate_{label}_4w"] = float(np.mean(rates))
        values[f"cohort_rate_{label}_ratio"] = float((rates[0] + 1e-3) / (np.mean(rates) + 1e-3))
    return values
