"""Paths, dates and constants shared by every experiment."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "db"  # written by assignment2/scripts/preprocess.py --out data
EVAL_DIR = ROOT / "assignment2" / "scripts"
TEST_INPUT = ROOT / "assignment2" / "submission" / "test_input.csv"

TEST_START = pd.Timestamp("2026-11-21 16:00")      # 2026-11-22 00:00 Taiwan time
# Test users: >= 3 records (the 6 activities, by event time) in the 90 days before this moment.
# Reproduces submission/test_input.csv exactly (4,465 users).
SELECTION_TIME = pd.Timestamp("2026-11-04 12:00")
DATA_START = pd.Timestamp("2025-12-01")
WEEK = pd.Timedelta(days=7)
DAY = pd.Timedelta(days=1)
WINDOWS = [7, 14, 30, 90]
MIN_RECORDS = 3

# Gaps in GH Archive, found by inspecting daily event counts of the cohort: non-push activity
# drops to ~5-20% of its usual level from June to early September (ending in a backfill burst on
# 9/4-9/6), and pushes are (nearly) missing for several days in August-October. [start, end) UTC.
GAPS = {
    "nonpush": [("2026-06-01", "2026-09-07")],
    "push": [("2026-08-18", "2026-09-13"), ("2026-10-07", "2026-10-08"),
             ("2026-10-15", "2026-10-19"), ("2026-10-23", "2026-10-24")],
}
# Since 10/27 GH Archive's PushEvents are ~1% of their usual volume (cohort: ~2,500 pushes a day
# before, 0-260 after). Push data is treated as ending there, so the push lag (and with it the
# staleness of push features at training time) matches what is really usable for the test week.
# Re-check this date when the GH Archive files are downloaded again.
FEED_END = {"push": pd.Timestamp("2026-10-27")}
GAP_FEATURE_DAYS = 30     # feature window checked for outages
GAP_FEATURE_SHARE = 0.25  # a feature window counts as corrupted above this share of outage days
RECENT_START = pd.Timestamp("2026-09-12 16:00")  # first week start after the summer outage

# LightGBM settings of the baseline (also used by experiments/missing_data and experiments/relations)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=0)
