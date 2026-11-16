"""Shared code of the experiments: data loading, per-week features and labels, outage handling,
data periods and scoring.

    config      paths (data/db, assignment2/), test week, time constants, GH Archive outages, LightGBM params
    data        read_tables, load: the tables as per-activity event lists of cohort users
    features    build_week (features + labels of one week), relative_features
    outages     gap_share, training_weights
    periods     add_period_args, split_weeks: the train / validation / final-fit weeks
    evaluation  EVENT_COLS, scores (the official scorer), best_threshold, to_output

Everything is re-exported here, so `import ghactivity as base` and `base.build_week(...)` work.
Entry scripts put the repo root on sys.path first (see the top of any experiments/*/run.py).
"""
from . import config, data, evaluation, features, outages, periods  # noqa: F401
from .config import (DATA_START, DAY, DB, EVAL_DIR, FEED_END, GAP_FEATURE_DAYS, GAP_FEATURE_SHARE, GAPS,
                     MIN_RECORDS, PARAMS, RECENT_START, ROOT, SELECTION_TIME, TEST_INPUT, TEST_START, WEEK,
                     WINDOWS)
from .data import TABLE_COLUMNS, load, read_tables
from .evaluation import EVENT_COLS, OUTPUT_COLUMNS, best_threshold, scores, to_output
from .features import COUNT_KEYS, build_week, open_count, relative_features, relative_views, source_features
from .outages import gap_share, group_of, training_weights
from .periods import add_period_args, split_weeks

__all__ = [
    "ROOT", "DB", "EVAL_DIR", "TEST_INPUT", "TEST_START", "SELECTION_TIME", "DATA_START", "WEEK", "DAY",
    "WINDOWS", "MIN_RECORDS", "GAPS", "FEED_END", "GAP_FEATURE_DAYS", "GAP_FEATURE_SHARE", "RECENT_START",
    "PARAMS", "TABLE_COLUMNS", "read_tables", "load", "EVENT_COLS", "OUTPUT_COLUMNS", "scores",
    "best_threshold", "to_output", "COUNT_KEYS", "source_features", "open_count", "build_week",
    "relative_views", "relative_features", "group_of", "gap_share", "training_weights", "add_period_args",
    "split_weeks",
]
