"""Backtest or predict with the relational GNN, trained without summer weeks.

The examples, graph snapshots, chronological split, threshold selection and
scoring are the same as ``experiments.relgt.run --temporal-model gnn``. The
only change is that training examples whose 7-day label window touches a
summer month are dropped. Calibration and holdout weeks are never filtered.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.relgt.data import (
    HORIZON_DAYS,
    LABELS,
    load_entity_vocabulary,
    load_events,
    snapshot_tokens,
)
from experiments.relgt.run import (
    DEFAULT_DB_DIR,
    DEFAULT_PREDICTION_CUTOFF,
    DEFAULT_START_CUTOFF,
    DEFAULT_TEST_INPUT,
    calculate_metrics,
    choose_thresholds,
    chronological_splits,
    fit_model,
    load_or_build_examples,
    predict_probabilities,
    prediction_example_indices,
    select_device,
    subsample_train_indices,
)

DEFAULT_SUMMER_MONTHS = (6, 7, 8)
BRANCH = "gnn_no_summer"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

# (train pool indices, cutoff_days) -> (kept, dropped, record saved as "train_filter")
TrainFilter = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, dict]]


def create_run_output_dir(
    command: str, requested_dir: Path | None, root: Path = DEFAULT_OUTPUT_ROOT
) -> Path:
    if requested_dir is not None:
        requested_dir.mkdir(parents=True, exist_ok=True)
        return requested_dir
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = root / f"{command}_{timestamp}_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output_dir}")
    return output_dir


def _parse_months(value: str, flag: str) -> tuple[int, ...]:
    try:
        months = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{flag} must be comma-separated month numbers."
        ) from error
    if not months or any(month < 1 or month > 12 for month in months):
        raise argparse.ArgumentTypeError(f"{flag} must be between 1 and 12.")
    if len(months) == 12:
        raise argparse.ArgumentTypeError(f"{flag} cannot list every month.")
    return months


def parse_summer_months(value: str) -> tuple[int, ...]:
    """Parse ``6,7,8``; ``none`` disables the filter (the plain GNN control)."""
    if value.strip().lower() == "none":
        return ()
    return _parse_months(value, "--summer-months")


def parse_calibration_months(value: str) -> tuple[int, ...]:
    """Parse ``9`` or ``4,5``: calibrate on every usable week whose cutoff is in them."""
    return _parse_months(value, "--calibration-months")


def split_train_calibration(
    cutoff_days: np.ndarray,
    usable_indices: np.ndarray,
    train_end: int,
    calibration_months: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_pool, calibration)`` indices from the usable (pre-holdout) rows.

    Default: calibration is the usable weeks from ``train_end`` on (the week before
    the holdout). With ``calibration_months``, it is every usable week whose cutoff
    falls in those months, and all other usable weeks form the training pool.
    """
    days = cutoff_days[usable_indices]
    if calibration_months is None:
        is_calibration = days >= train_end
    else:
        is_calibration = np.isin(
            pd.to_datetime(days.astype(np.int64), unit="D").month, calibration_months
        )
    calibration = usable_indices[is_calibration]
    if not len(calibration):
        raise ValueError(
            f"No usable weeks before the holdout/prediction cutoff fall in "
            f"--calibration-months {list(calibration_months or ())}."
        )
    return usable_indices[~is_calibration], calibration


def summer_mask(cutoff_days: np.ndarray, months: tuple[int, ...]) -> np.ndarray:
    """True where any day of ``[cutoff, cutoff + 7 days)`` falls in ``months``."""
    if not months:
        return np.zeros(len(cutoff_days), dtype=bool)
    unique_days, inverse = np.unique(cutoff_days.astype(np.int64), return_inverse=True)
    in_summer = np.zeros(len(unique_days), dtype=bool)
    for offset in range(HORIZON_DAYS):
        window_months = pd.to_datetime(unique_days + offset, unit="D").month
        in_summer |= np.isin(window_months, months)
    return in_summer[inverse]


def drop_summer(
    train_indices: np.ndarray, cutoff_days: np.ndarray, months: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(kept, dropped)`` training indices; fail if nothing is left."""
    excluded = summer_mask(cutoff_days[train_indices], months)
    kept, dropped = train_indices[~excluded], train_indices[excluded]
    if not len(kept):
        raise ValueError(
            f"Excluding months {list(months)} removes every training example."
        )
    return kept, dropped


def _week_strings(cutoff_days: np.ndarray) -> list[str]:
    return [str(pd.Timestamp(int(day), unit="D").date()) for day in np.unique(cutoff_days)]


def _fit(examples, train_indices, device, args):
    return fit_model(
        examples,
        train_indices,
        examples.num_entities,
        device,
        args.epochs,
        args.batch_size,
        args.hidden_dim,
        1,  # num_heads is unused by the GNN
        args.num_layers,
        args.dropout,
        args.seed,
        "gnn",
        args.user_embedding,
        args.learning_rate,
    )


def _settings(args: argparse.Namespace) -> dict:
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "user_embedding": args.user_embedding,
        "train_fraction": args.train_fraction,
        "discard_unused_train": args.discard_unused_train,
        "eligibility_days": args.eligibility_days,
    }


def summer_filter(months: tuple[int, ...]) -> TrainFilter:
    """Drop training weeks whose label window touches ``months``."""

    def apply(train_indices: np.ndarray, cutoff_days: np.ndarray):
        kept, dropped = drop_summer(train_indices, cutoff_days, months)
        return kept, dropped, _summer_record(months, cutoff_days, kept, dropped)

    return apply


def _summer_record(months, cutoff_days, kept, dropped) -> dict:
    return {
        "name": "no_summer",
        "summer_months": list(months),
        "exclusion_rule": (
            "training examples whose label window [cutoff, cutoff + 7 days) "
            "touches a summer month are dropped; calibration/holdout are kept"
        ),
        "excluded_train_examples": int(len(dropped)),
        "excluded_train_weeks": _week_strings(cutoff_days[dropped]),
        "kept_train_weeks": _week_strings(cutoff_days[kept]),
    }


def _calibration_record(months, cutoff_days, calibration_base) -> dict:
    return {
        "calibration_months": None if months is None else list(months),
        "rule": (
            "the week before the holdout/prediction cutoff"
            if months is None
            else "every usable week whose cutoff falls in calibration_months"
        ),
        "weeks": _week_strings(cutoff_days[calibration_base]),
        "examples": int(len(calibration_base)),
    }


def run_backtest(
    args: argparse.Namespace,
    train_filter: TrainFilter | None = None,
    experiment: str = BRANCH,
) -> None:
    """Shared GNN backtest; ``train_filter`` defaults to ``--summer-months``."""
    train_filter = train_filter or summer_filter(args.summer_months)
    examples = load_or_build_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        eligibility_lookback_days=args.eligibility_days,
    )
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    train_indices, calibration_base = split_train_calibration(
        examples.cutoff_days,
        np.flatnonzero(examples.cutoff_days < calibration_end),
        train_end,
        args.calibration_months,
    )
    train_indices, dropped_indices, filter_record = train_filter(
        train_indices, examples.cutoff_days
    )
    available_train_examples = len(train_indices)
    train_indices, unused_train_indices = subsample_train_indices(train_indices, args)
    calibration_indices = np.union1d(calibration_base, unused_train_indices)
    holdout_indices = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not len(calibration_indices) or not len(holdout_indices):
        raise ValueError("The chronological split produced an empty calibration or holdout set.")

    device = select_device(args.device)
    print(
        f"Device: {device}; holdout >= {pd.Timestamp(calibration_end, unit='D')}; "
        f"calibration weeks {_week_strings(examples.cutoff_days[calibration_base])}; "
        f"train filter {filter_record['name']} kept {len(train_indices):,} and dropped "
        f"{len(dropped_indices):,} training examples"
    )
    model = _fit(examples, train_indices, device, args)
    calibration_probabilities = predict_probabilities(
        model, examples.tokens, calibration_indices, device, args.batch_size
    )
    holdout_probabilities = predict_probabilities(
        model, examples.tokens, holdout_indices, device, args.batch_size
    )
    thresholds = choose_thresholds(
        examples.labels[calibration_indices], calibration_probabilities
    )
    holdout_labels = examples.labels[holdout_indices]
    metrics = calculate_metrics(
        holdout_labels,
        holdout_probabilities,
        thresholds,
        examples.cutoff_days[holdout_indices],
    )

    torch.save(model.state_dict(), args.output_dir / f"{experiment}_model.pt")
    probability_columns = {
        "user_id": examples.tokens.user_ids[holdout_indices],
        "cutoff": pd.to_datetime(examples.cutoff_days[holdout_indices], unit="D"),
        **{
            f"actual_{label}": holdout_labels[:, column]
            for column, label in enumerate(LABELS)
        },
        **{
            f"{experiment}_{label}": holdout_probabilities[:, column]
            for column, label in enumerate(LABELS)
        },
    }
    probability_path = args.output_dir / "holdout_probabilities.csv"
    pd.DataFrame(probability_columns).to_csv(probability_path, index=False)

    results = {
        "experiment": experiment,
        "architecture": "Relation-aware message passing over sampled user ego graphs",
        "train_filter": filter_record,
        "calibration": _calibration_record(
            args.calibration_months, examples.cutoff_days, calibration_base
        ),
        "train_examples": int(len(train_indices)),
        "available_train_examples": int(available_train_examples),
        "calibration_examples": int(len(calibration_indices)),
        "holdout_examples": int(len(holdout_indices)),
        "train_end_exclusive": (
            str(pd.Timestamp(train_end, unit="D"))
            if args.calibration_months is None
            else None
        ),
        "holdout_start": str(pd.Timestamp(calibration_end, unit="D")),
        "device": str(device),
        "settings": _settings(args),
        "threshold_selection": "per label, on calibration weeks only",
        "thresholds": thresholds,
        "holdout_metrics": metrics,
    }
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        f"Holdout score {metrics['score']:.4f} "
        f"(macro-F1 {metrics['macro_f1']:.4f}, micro-F1 {metrics['micro_f1']:.4f})"
    )
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout probabilities to {probability_path}")


def run_prediction(
    args: argparse.Namespace,
    train_filter: TrainFilter | None = None,
    experiment: str = BRANCH,
) -> None:
    train_filter = train_filter or summer_filter(args.summer_months)
    examples = load_or_build_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        eligibility_lookback_days=args.eligibility_days,
    )
    usable_indices = prediction_example_indices(examples.cutoff_days, args.cutoff)
    usable_cutoffs = examples.cutoff_days[usable_indices]
    train_end, _ = chronological_splits(usable_cutoffs, holdout=False)
    train_indices, calibration_base = split_train_calibration(
        examples.cutoff_days, usable_indices, train_end, args.calibration_months
    )
    train_indices, dropped_indices, filter_record = train_filter(
        train_indices, examples.cutoff_days
    )
    train_indices, unused_train_indices = subsample_train_indices(train_indices, args)
    calibration_indices = np.union1d(calibration_base, unused_train_indices)

    device = select_device(args.device)
    model = _fit(examples, train_indices, device, args)
    thresholds = choose_thresholds(
        examples.labels[calibration_indices],
        predict_probabilities(
            model, examples.tokens, calibration_indices, device, args.batch_size
        ),
    )

    target_users, inference_tokens = load_test_tokens(args)
    probabilities = predict_probabilities(
        model,
        inference_tokens,
        np.arange(len(target_users), dtype=np.int64),
        device,
        args.batch_size,
    )
    save_predictions(args, target_users, probabilities, thresholds)
    torch.save(model.state_dict(), args.output_dir / f"{experiment}_model.pt")

    metadata = {
        "experiment": experiment,
        "prediction_cutoff_utc": str(args.cutoff),
        "training_start_cutoff_utc": str(args.start_cutoff),
        "train_filter": filter_record,
        "calibration": _calibration_record(
            args.calibration_months, examples.cutoff_days, calibration_base
        ),
        "training_examples": int(len(train_indices)),
        "training_weeks": int(np.unique(examples.cutoff_days[train_indices]).size),
        "calibration_weeks": int(np.unique(examples.cutoff_days[calibration_indices]).size),
        "labels": list(LABELS),
        "thresholds": thresholds,
        "device": str(device),
        "settings": _settings(args),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Device: {device}; calibrated thresholds: {json.dumps(thresholds)}")


def load_test_tokens(args: argparse.Namespace):
    """Return ``(target_users, graph tokens at --cutoff)`` in --test-input order."""
    target_users = pd.read_csv(args.test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{args.test_input} contains duplicate UserIDs.")
    users = pd.read_parquet(
        args.db_dir / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"]
    )
    cohort_users = users.loc[users["is_cohort"]]
    mapped_ids = target_users["UserID"].map(
        cohort_users.set_index("github_user_id")["user_id"]
    )
    if mapped_ids.isna().any():
        raise ValueError(
            f"{int(mapped_ids.isna().sum())} test UserID(s) are not in the cohort "
            "for this data release."
        )
    internal_ids = mapped_ids.to_numpy(dtype=np.int32)
    vocabulary = load_entity_vocabulary(args.db_dir)
    events = load_events(
        args.db_dir, cohort_users["user_id"].astype(int).tolist(), vocabulary
    )
    return target_users, snapshot_tokens(events, args.cutoff, internal_ids, vocabulary)


def save_predictions(
    args: argparse.Namespace,
    target_users: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    branch_probabilities: dict[str, np.ndarray] | None = None,
) -> None:
    """Write evaluator-format ``predictions.csv`` and unthresholded ``probabilities.csv``."""
    output = pd.DataFrame({"UserID": target_users["UserID"]})
    for column, label in enumerate(LABELS):
        output[label] = (probabilities[:, column] >= thresholds[label]).astype(np.uint8)
    prediction_path = args.output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)
    probability_output = pd.DataFrame({"UserID": target_users["UserID"]})
    for column, label in enumerate(LABELS):
        probability_output[f"{label}_probability"] = probabilities[:, column]
    for branch, values in (branch_probabilities or {}).items():
        for column, label in enumerate(LABELS):
            probability_output[f"{branch}_{label}_probability"] = values[:, column]
    probability_path = args.output_dir / "probabilities.csv"
    probability_output.to_csv(probability_path, index=False)
    print(f"Saved {len(output):,} predictions to {prediction_path}")
    print(f"Saved unthresholded probabilities to {probability_path}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output folder (default: a new timestamped folder per run)",
    )
    parser.add_argument(
        "--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF)
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--calibration-months",
        type=parse_calibration_months,
        default=None,
        metavar="M,M,...",
        help=(
            "calibrate thresholds on every week before the holdout (backtest) or with a "
            "complete label window before --cutoff (predict) whose cutoff falls in these "
            "months; the other weeks train (default: the week before the holdout)"
        ),
    )
    parser.add_argument(
        "--eligibility-days",
        type=int,
        default=None,
        help="users need 5 events in this many days before each cutoff (default: 14)",
    )
    parser.add_argument(
        "--user-embedding",
        action="store_true",
        help="add a learned per-user vector to the root node (unseen users get zero)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="fraction of the remaining training examples to use; the rest joins calibration",
    )
    parser.add_argument(
        "--discard-unused-train",
        action="store_true",
        help="with --train-fraction below 1, drop the unused examples instead",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)


def add_summer_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--summer-months",
        type=parse_summer_months,
        default=DEFAULT_SUMMER_MONTHS,
        metavar="M,M,...",
        help=(
            "months whose label weeks are dropped from training (default 6,7,8); "
            "'none' keeps every week, which is the plain GNN"
        ),
    )


def build_parser(
    add_filter_arguments: Callable[[argparse.ArgumentParser], None] = add_summer_argument,
    description: str | None = __doc__,
) -> argparse.ArgumentParser:
    """CLI shared by the GNN month-filter experiments; each adds its own filter flag."""
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest_parser = subparsers.add_parser("backtest")
    predict_parser = subparsers.add_parser("predict")
    for subparser in (backtest_parser, predict_parser):
        add_arguments(subparser)
        add_filter_arguments(subparser)
    predict_parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    predict_parser.add_argument(
        "--cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF),
        help="UTC time at which to make the prediction",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "hidden_dim", "num_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.eligibility_days is not None and args.eligibility_days < 14:
        parser.error("--eligibility-days must be at least the 14-day graph lookback.")
    if not 0.0 < args.train_fraction <= 1.0:
        parser.error("--train-fraction must be in (0, 1].")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1).")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    args.output_dir = create_run_output_dir(args.command, args.output_dir)
    if args.command == "backtest":
        run_backtest(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
