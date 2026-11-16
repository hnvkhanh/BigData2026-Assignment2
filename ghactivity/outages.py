"""GH Archive collection outages: how much of a window they cover, and training-week weights."""
import pandas as pd

from . import config
from .config import DAY, GAP_FEATURE_DAYS, GAP_FEATURE_SHARE, RECENT_START, WEEK
from .evaluation import EVENT_COLS


def group_of(k):
    return "push" if k == "push" else "nonpush"


def gap_share(group, start, end, gaps=None):
    """Share of [start, end) that falls inside the collection outages of a source group.

    gaps: {group: [(start, end), ...]}, default config.GAPS."""
    covered = pd.Timedelta(0)
    for a, b in (config.GAPS if gaps is None else gaps)[group]:
        lo, hi = max(start, pd.Timestamp(a)), min(end, pd.Timestamp(b))
        if hi > lo:
            covered += hi - lo
    return covered / (end - start)


def training_weights(mode, weeks, lags, gap_weight, gaps=None):
    """Per label, a weight for each training week (0 drops it) under the chosen outage handling.

    gaps: the outages to use, default config.GAPS (experiments/missing_data passes re-measured ones)."""
    gaps = config.GAPS if gaps is None else gaps
    w = {k: {} for k in EVENT_COLS}
    for s in weeks:
        label_gap = {g: gap_share(g, s, s + WEEK, gaps) > 0 for g in gaps}
        feat_gap = {}
        for g, src in [("nonpush", "open_pr"), ("push", "push")]:
            c = s - lags[src]
            feat_gap[g] = gap_share(g, c - GAP_FEATURE_DAYS * DAY, c, gaps) > GAP_FEATURE_SHARE
        corrupted = any(label_gap.values()) or any(feat_gap.values())
        for k in EVENT_COLS:
            if mode == "none":
                w[k][s] = 1.0
            elif mode == "mask":
                w[k][s] = 0.0 if label_gap[group_of(k)] else 1.0
            elif mode == "drop":
                w[k][s] = 0.0 if corrupted else 1.0
            elif mode == "downweight":
                w[k][s] = gap_weight if corrupted else 1.0
            elif mode == "recent":
                w[k][s] = 1.0 if s >= RECENT_START else 0.0
    return w
