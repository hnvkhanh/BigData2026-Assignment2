"""The data-period options shared by all runners, and the train / validation / final-fit week split."""
import sys

import pandas as pd

from .config import TEST_START, WEEK
from .evaluation import EVENT_COLS


def add_period_args(ap):
    """Options choosing the data period, for all stages or for one stage (shared by all runners).

    A week (label window [s, s + 7d)) belongs to a period when it lies fully inside
    [start, end]. Dates are UTC; weeks start Saturday 16:00. A stage option overrides the
    global one for that stage only.
    """
    g = ap.add_argument_group("data period (UTC, e.g. 2026-03-01 or '2026-03-01 16:00')")
    g.add_argument("--start", default="2026-01-14 16:00", help="first moment of data, all stages")
    g.add_argument("--end", default=None, help="last moment of data, all stages (default: all labelled weeks)")
    g.add_argument("--train-start", default=None, help="override --start for training weeks")
    g.add_argument("--train-end", default=None, help="override --end for training weeks")
    g.add_argument("--val-start", default=None,
                   help="validation period start; with --val-end it replaces --val-weeks")
    g.add_argument("--val-end", default=None, help="validation period end")
    g.add_argument("--test-start", default=None,
                   help="override --start for the weeks the final (test) models are fitted on")
    g.add_argument("--test-end", default=None, help="override --end for the final (test) fit weeks")
    g.add_argument("--val-weeks", type=int, default=3,
                   help="without --val-start/--val-end: number of most recent weeks used for "
                        "validation / thresholds")


def split_weeks(args, data_end):
    """Return (weeks, train_weeks, val_weeks, final_weeks) for the chosen data periods.

    weeks = every labelled week inside the global period (the ones features are built for).
    Training labels must end before validation starts, so train weeks overlapping validation
    are dropped.
    """
    last = max(data_end[k] for k in EVENT_COLS) - WEEK
    ts = lambda x, d: pd.Timestamp(x) if x else d  # noqa: E731
    start, end = pd.Timestamp(args.start), ts(args.end, pd.Timestamp.max)
    weeks, s = [], TEST_START - WEEK
    while s >= start:
        if s <= last and s + WEEK <= end:
            weeks.append(s)
        s -= WEEK
    weeks = sorted(weeks)

    def inside(a, b):
        a, b = ts(a, start), ts(b, end)
        return [w for w in weeks if w >= a and w + WEEK <= b]

    if args.val_start or args.val_end:
        val_weeks = inside(args.val_start, args.val_end)
    else:
        val_weeks = weeks[-args.val_weeks:]
    if not val_weeks:
        sys.exit("no validation weeks in the selected period")
    train_weeks = [w for w in inside(args.train_start, args.train_end) if w + WEEK <= val_weeks[0]]
    final_weeks = inside(args.test_start, args.test_end)
    if not train_weeks or not final_weeks:
        sys.exit("no training / final-fit weeks in the selected period")
    return weeks, train_weeks, val_weeks, final_weeks
