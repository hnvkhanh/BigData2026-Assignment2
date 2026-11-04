"""Cutoff-safe daily activity sequences and calendar features.

This replaces the DM2026 daily weather table: each user-week example becomes a
``(window_days, channels)`` sequence of daily action counts ending at the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.relgt.data import HORIZON_DAYS, LABELS

DAY_NS = pd.Timedelta(days=1).value
COUNT_CHANNELS = (*(f"log_{label}" for label in LABELS), "log_total", "active")
CALENDAR_CHANNELS = ("dow_sin", "dow_cos")
SEQUENCE_CHANNELS = (*COUNT_CHANNELS, *CALENDAR_CHANNELS)
AUX_COLUMNS = (
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "weekofyear_sin",
    "weekofyear_cos",
)


def example_cutoff_ns(cutoff_days: np.ndarray, start_cutoff: pd.Timestamp) -> np.ndarray:
    """Recover exact cutoff times from the day-floored ``cutoff_days`` of the examples.

    Weekly cutoffs are ``start_cutoff + 7k days``, so they share its time of day.
    """
    offset = start_cutoff.value % DAY_NS
    cutoff_ns = cutoff_days.astype(np.int64) * DAY_NS + offset
    if ((cutoff_ns - start_cutoff.value) % (HORIZON_DAYS * DAY_NS)).any():
        raise ValueError("Example cutoffs are not weekly steps from --start-cutoff.")
    return cutoff_ns


def _day_of_week_channels(cutoff_ns: int, window_days: int) -> np.ndarray:
    """Weekday sin/cos for the midpoint of each 24-hour bin before the cutoff."""
    midpoints = cutoff_ns - (window_days - np.arange(window_days)) * DAY_NS + DAY_NS // 2
    weekday = pd.to_datetime(midpoints).dayofweek.to_numpy()
    angle = 2 * np.pi * weekday / 7.0
    return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


def build_daily_sequences(
    events: pd.DataFrame,
    user_ids: np.ndarray,
    cutoff_ns: np.ndarray,
    window_days: int,
) -> np.ndarray:
    """Return ``float32 (n, window_days, len(SEQUENCE_CHANNELS))`` ordered oldest first.

    Bin ``i`` holds actions in ``[cutoff - (window_days - i) days,
    cutoff - (window_days - i - 1) days)``. An action counts only if its action
    time, event row and target entity were all available before the cutoff.
    """
    if user_ids.shape != cutoff_ns.shape:
        raise ValueError("user_ids and cutoff_ns must have matching shapes.")
    if window_days < 1:
        raise ValueError("window_days must be positive.")
    sequences = np.zeros((len(user_ids), window_days, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    calendar = {
        int(cutoff): _day_of_week_channels(int(cutoff), window_days)
        for cutoff in np.unique(cutoff_ns)
    }
    for row, cutoff in enumerate(cutoff_ns):
        sequences[row, :, len(COUNT_CHANNELS) :] = calendar[int(cutoff)]
    if not len(user_ids) or events.empty:
        return sequences

    histories: dict[int, tuple[np.ndarray, ...]] = {}
    for user_id, group in events.groupby("user_id", sort=False):
        order = np.argsort(group["action_at"].astype("int64").to_numpy(), kind="stable")
        histories[int(user_id)] = tuple(
            group[column].astype("int64").to_numpy()[order]
            for column in ("action_at", "available_at", "entity_available_at")
        ) + (group["relation_id"].to_numpy(dtype=np.int64)[order] - 1,)

    window_ns = window_days * DAY_NS
    for row, (user_id, cutoff) in enumerate(zip(user_ids, cutoff_ns)):
        history = histories.get(int(user_id))
        if history is None:
            continue
        action_ns, available_ns, entity_ns, relations = history
        cutoff = int(cutoff)
        left = np.searchsorted(action_ns, cutoff - window_ns, side="left")
        right = np.searchsorted(action_ns, cutoff, side="left")
        known = np.arange(left, right)
        known = known[(available_ns[known] < cutoff) & (entity_ns[known] < cutoff)]
        if not known.size:
            continue
        day_index = window_days - 1 - (cutoff - action_ns[known] - 1) // DAY_NS
        counts = np.zeros((window_days, len(LABELS)), dtype=np.float32)
        np.add.at(counts, (day_index, relations[known]), 1.0)
        total = counts.sum(axis=1)
        sequences[row, :, : len(LABELS)] = np.log1p(counts)
        sequences[row, :, len(LABELS)] = np.log1p(total)
        sequences[row, :, len(LABELS) + 1] = total > 0
    return sequences


def calendar_features(cutoff_ns: np.ndarray) -> np.ndarray:
    """Cyclic month/day-of-year/week-of-year of each cutoff, ``float32 (n, 6)``."""
    timestamps = pd.to_datetime(cutoff_ns.astype(np.int64))
    columns = []
    for values, period in (
        (timestamps.month.to_numpy(), 12.0),
        (timestamps.dayofyear.to_numpy(), 366.0),
        (timestamps.isocalendar().week.to_numpy(dtype=np.int64), 53.0),
    ):
        angle = 2 * np.pi * values / period
        columns.extend([np.sin(angle), np.cos(angle)])
    return np.stack(columns, axis=1).astype(np.float32)


@dataclass(frozen=True)
class ChannelScaler:
    """Standardise the count channels with training-row statistics (as DM2026 does)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, sequences: np.ndarray) -> "ChannelScaler":
        if not len(sequences):
            raise ValueError("Cannot fit the channel scaler without training rows.")
        counts = sequences[..., : len(COUNT_CHANNELS)].reshape(-1, len(COUNT_CHANNELS))
        mean = counts.mean(axis=0, dtype=np.float64)
        std = counts.std(axis=0, dtype=np.float64)
        std[~np.isfinite(std) | (std <= 0)] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform_(self, sequences: np.ndarray) -> np.ndarray:
        """Scale ``sequences`` in place and return it."""
        counts = sequences[..., : len(COUNT_CHANNELS)]
        counts -= self.mean
        counts /= self.std
        return sequences

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            channel: {"mean": float(mean), "std": float(std)}
            for channel, mean, std in zip(COUNT_CHANNELS, self.mean, self.std)
        }
