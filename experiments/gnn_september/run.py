"""Backtest or predict with the relational GNN, trained on September weeks only.

Same examples, model, chronological split, calibration and official scoring
as ``experiments.gnn_no_summer`` / ``relgt.run --temporal-model gnn``. Only the
training pool changes: a training example is kept when its cutoff falls in
``--train-months`` (default 9). Calibration and holdout weeks are never filtered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gnn_no_summer.run import (
    TrainFilter,
    _parse_months,
    _week_strings,
    build_parser,
    create_run_output_dir,
    run_backtest,
    run_prediction,
    validate_args,
)

DEFAULT_TRAIN_MONTHS = (9,)
DEFAULT_HIDDEN_DIM = 32  # shared GNN default is 64
DEFAULT_DROPOUT = 0.2  # shared GNN default is 0.1
EXPERIMENT = "gnn_september"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def parse_train_months(value: str) -> tuple[int, ...]:
    return _parse_months(value, "--train-months")


def month_filter(months: tuple[int, ...]) -> TrainFilter:
    """Keep training weeks whose cutoff falls in ``months``; fail if none do."""

    def apply(train_indices: np.ndarray, cutoff_days: np.ndarray):
        cutoff_months = pd.to_datetime(
            cutoff_days[train_indices].astype(np.int64), unit="D"
        ).month
        keep = np.isin(cutoff_months, months)
        kept, dropped = train_indices[keep], train_indices[~keep]
        if not len(kept):
            raise ValueError(
                f"No training weeks have a cutoff in --train-months {list(months)}."
            )
        return kept, dropped, {
            "name": "train_months",
            "train_months": list(months),
            "rule": (
                "only training examples whose cutoff falls in train_months are kept; "
                "calibration/holdout are kept"
            ),
            "excluded_train_examples": int(len(dropped)),
            "excluded_train_weeks": _week_strings(cutoff_days[dropped]),
            "kept_train_weeks": _week_strings(cutoff_days[kept]),
        }

    return apply


def add_train_months_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-months",
        type=parse_train_months,
        default=DEFAULT_TRAIN_MONTHS,
        metavar="M,M,...",
        help="train only on weeks whose cutoff falls in these months (default 9)",
    )
    # September-only training data is small, so use a smaller model than the shared default.
    parser.set_defaults(hidden_dim=DEFAULT_HIDDEN_DIM, dropout=DEFAULT_DROPOUT)


def main() -> None:
    parser = build_parser(add_train_months_argument, __doc__)
    args = parser.parse_args()
    validate_args(parser, args)
    if args.calibration_months and set(args.calibration_months) & set(args.train_months):
        parser.error("--calibration-months and --train-months must not overlap.")
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    args.output_dir = create_run_output_dir(
        args.command, args.output_dir, DEFAULT_OUTPUT_ROOT
    )
    runner = run_backtest if args.command == "backtest" else run_prediction
    runner(args, month_filter(args.train_months), EXPERIMENT)


if __name__ == "__main__":
    main()
