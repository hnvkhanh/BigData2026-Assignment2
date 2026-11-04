"""Ensemble two relational GNNs: one trained on every week, one on September only.

Both branches use the same examples, graph snapshots and calibration/holdout
weeks as ``experiments.gnn_no_summer`` / ``gnn_september``. They are trained
independently. Their probabilities are averaged as
``w * p_all + (1 - w) * p_september``. ``w`` (grid 0, 0.1, ..., 1) and the
per-label thresholds are chosen together on calibration only, unless
``--fusion-weight`` fixes ``w``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.gnn_no_summer.run import (
    _calibration_record,
    _fit,
    _settings,
    _week_strings,
    build_parser,
    create_run_output_dir,
    load_test_tokens,
    save_predictions,
    split_train_calibration,
    validate_args,
)
from experiments.gnn_september.run import add_train_months_argument, month_filter
from experiments.relgt.data import LABELS
from experiments.relgt.run import (
    calculate_metrics,
    choose_fusion_weight,
    choose_thresholds,
    chronological_splits,
    fuse_probabilities,
    load_or_build_examples,
    parse_unit_interval,
    predict_probabilities,
    prediction_example_indices,
    select_device,
)

EXPERIMENT = "gnn_september_ensemble"
BRANCHES = ("gnn_all", "gnn_september")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def _train_branches(args, examples, pool, device):
    """Fit both branches on the training pool; return models and their train rows."""
    september, dropped, filter_record = month_filter(args.train_months)(
        pool, examples.cutoff_days
    )
    train_sets = {"gnn_all": pool, "gnn_september": september}
    models = {}
    for branch, indices in train_sets.items():
        print(f"Training {branch} on {len(indices):,} examples")
        models[branch] = _fit(examples, indices, device, args)
    branch_record = {
        "gnn_all": {
            "train_examples": int(len(pool)),
            "train_weeks": _week_strings(examples.cutoff_days[pool]),
        },
        "gnn_september": {"train_examples": int(len(september)), **filter_record},
    }
    return models, branch_record


def _predict(models, tokens, indices, device, batch_size):
    return {
        branch: predict_probabilities(model, tokens, indices, device, batch_size)
        for branch, model in models.items()
    }


def _fusion(args, labels, calibration):
    weight, thresholds, calibration_metrics = choose_fusion_weight(
        labels, calibration["gnn_all"], calibration["gnn_september"], args.fusion_weight
    )
    record = {
        "method": (
            "manual probability averaging"
            if args.fusion_weight is not None
            else "calibration-selected probability averaging (grid step 0.1)"
        ),
        "gnn_all_weight": weight,
        "gnn_september_weight": 1.0 - weight,
        "calibration_score": calibration_metrics["score"],
    }
    return weight, thresholds, record


def run_backtest(args: argparse.Namespace) -> None:
    examples = load_or_build_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        eligibility_lookback_days=args.eligibility_days,
    )
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    pool, calibration_indices = split_train_calibration(
        examples.cutoff_days,
        np.flatnonzero(examples.cutoff_days < calibration_end),
        train_end,
        args.calibration_months,
    )
    holdout_indices = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not len(holdout_indices):
        raise ValueError("The chronological split produced an empty holdout set.")

    device = select_device(args.device)
    print(
        f"Device: {device}; holdout >= {pd.Timestamp(calibration_end, unit='D')}; "
        f"calibration weeks {_week_strings(examples.cutoff_days[calibration_indices])}"
    )
    models, branch_record = _train_branches(args, examples, pool, device)
    calibration_labels = examples.labels[calibration_indices]
    calibration = _predict(
        models, examples.tokens, calibration_indices, device, args.batch_size
    )
    holdout = _predict(models, examples.tokens, holdout_indices, device, args.batch_size)
    weight, thresholds, fusion_record = _fusion(args, calibration_labels, calibration)
    holdout["ensemble"] = fuse_probabilities(
        holdout["gnn_all"], holdout["gnn_september"], weight
    )

    holdout_labels = examples.labels[holdout_indices]
    holdout_cutoffs = examples.cutoff_days[holdout_indices]
    metrics = calculate_metrics(holdout_labels, holdout["ensemble"], thresholds, holdout_cutoffs)
    # Each branch alone, with its own calibration thresholds, for comparison.
    branch_metrics = {
        branch: calculate_metrics(
            holdout_labels,
            holdout[branch],
            choose_thresholds(calibration_labels, calibration[branch]),
            holdout_cutoffs,
        )
        for branch in BRANCHES
    }
    branch_metrics["ensemble"] = metrics

    for branch, model in models.items():
        torch.save(model.state_dict(), args.output_dir / f"{branch}_model.pt")
    probability_columns = {
        "user_id": examples.tokens.user_ids[holdout_indices],
        "cutoff": pd.to_datetime(holdout_cutoffs, unit="D"),
        **{
            f"actual_{label}": holdout_labels[:, column]
            for column, label in enumerate(LABELS)
        },
    }
    for branch, values in holdout.items():
        for column, label in enumerate(LABELS):
            probability_columns[f"{branch}_{label}"] = values[:, column]
    probability_path = args.output_dir / "holdout_probabilities.csv"
    pd.DataFrame(probability_columns).to_csv(probability_path, index=False)

    results = {
        "experiment": EXPERIMENT,
        "architecture": "Relation-aware message passing over sampled user ego graphs",
        "ensemble": "independently trained branches, weighted probability average",
        "branches": branch_record,
        "fusion": fusion_record,
        "calibration": _calibration_record(
            args.calibration_months, examples.cutoff_days, calibration_indices
        ),
        "calibration_examples": int(len(calibration_indices)),
        "holdout_examples": int(len(holdout_indices)),
        "holdout_start": str(pd.Timestamp(calibration_end, unit="D")),
        "device": str(device),
        "settings": _settings(args),
        "threshold_selection": "per label on calibration only, jointly with the fusion weight",
        "thresholds": thresholds,
        "holdout_metrics": metrics,
        "branch_holdout": branch_metrics,
    }
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for branch, values in branch_metrics.items():
        print(f"Holdout score {branch}: {values['score']:.4f}")
    print(f"gnn_all weight {weight:.1f}")
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout probabilities to {probability_path}")


def run_prediction(args: argparse.Namespace) -> None:
    examples = load_or_build_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        eligibility_lookback_days=args.eligibility_days,
    )
    usable_indices = prediction_example_indices(examples.cutoff_days, args.cutoff)
    train_end, _ = chronological_splits(examples.cutoff_days[usable_indices], holdout=False)
    pool, calibration_indices = split_train_calibration(
        examples.cutoff_days, usable_indices, train_end, args.calibration_months
    )

    device = select_device(args.device)
    models, branch_record = _train_branches(args, examples, pool, device)
    calibration = _predict(
        models, examples.tokens, calibration_indices, device, args.batch_size
    )
    weight, thresholds, fusion_record = _fusion(
        args, examples.labels[calibration_indices], calibration
    )

    target_users, tokens = load_test_tokens(args)
    branch_probabilities = _predict(
        models, tokens, np.arange(len(target_users), dtype=np.int64), device, args.batch_size
    )
    probabilities = fuse_probabilities(
        branch_probabilities["gnn_all"], branch_probabilities["gnn_september"], weight
    )
    save_predictions(args, target_users, probabilities, thresholds, branch_probabilities)
    for branch, model in models.items():
        torch.save(model.state_dict(), args.output_dir / f"{branch}_model.pt")

    metadata = {
        "experiment": EXPERIMENT,
        "prediction_cutoff_utc": str(args.cutoff),
        "training_start_cutoff_utc": str(args.start_cutoff),
        "branches": branch_record,
        "fusion": fusion_record,
        "calibration": _calibration_record(
            args.calibration_months, examples.cutoff_days, calibration_indices
        ),
        "labels": list(LABELS),
        "thresholds": thresholds,
        "device": str(device),
        "settings": _settings(args),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Device: {device}; gnn_all weight {weight:.1f}; thresholds {json.dumps(thresholds)}")


def add_ensemble_arguments(parser: argparse.ArgumentParser) -> None:
    add_train_months_argument(parser)
    parser.add_argument(
        "--fusion-weight",
        type=parse_unit_interval,
        default=None,
        metavar="FLOAT",
        help="fix the gnn_all weight w (default: chosen on calibration)",
    )


def main() -> None:
    parser = build_parser(add_ensemble_arguments, __doc__)
    args = parser.parse_args()
    validate_args(parser, args)
    if args.train_fraction != 1.0 or args.discard_unused_train:
        parser.error("--train-fraction/--discard-unused-train are not supported by the ensemble.")
    if args.calibration_months and set(args.calibration_months) & set(args.train_months):
        parser.error("--calibration-months and --train-months must not overlap.")
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    args.output_dir = create_run_output_dir(
        args.command, args.output_dir, DEFAULT_OUTPUT_ROOT
    )
    if args.command == "backtest":
        run_backtest(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
