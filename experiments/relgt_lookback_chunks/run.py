"""Backtest relational GNNs trained on activity chunks, each with its own lookback."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.relgt.data import (
    LABELS,
    TrainingExamples,
    load_entity_vocabulary,
    load_events,
)
from experiments.relgt.run import (
    _validate_aligned_examples,
    calculate_metrics,
    choose_thresholds,
    load_or_build_examples,
    select_device,
)
from experiments.relgt_action_specialists.run import (
    fit_specialist,
    predict_specialist,
)
from experiments.relgt_activity_chunks.run import (
    ALL_OUTPUTS,
    CHUNK_NAMES,
    _partition_indices,
    apply_train_fraction,
    assign_chunks,
    auto_thresholds,
    build_parser,
    metrics_from_predictions,
    partition_chunk_counts,
    prior_action_counts,
    require_populated_chunks,
    route_by_chunk,
    validate_args,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_LOOKBACKS = (180, 21, 7)
DEFAULT_GLOBAL_LOOKBACK = 14


def parse_lookbacks(value: str) -> tuple[int, int, int]:
    """Parse ``LOW,MID,HIGH`` lookback windows in days, e.g. ``180,21,7``."""
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "lookbacks must be three integers such as 180,21,7."
        ) from None
    if len(parts) != len(CHUNK_NAMES) or any(part < 1 for part in parts):
        raise argparse.ArgumentTypeError(
            "lookbacks need exactly three positive integers: LOW,MID,HIGH."
        )
    return parts


def _create_output_dir(requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        requested_dir.mkdir(parents=True, exist_ok=True)
        return requested_dir
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DEFAULT_OUTPUT_ROOT / f"backtest_{timestamp}_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output_dir}")
    return output_dir


def load_examples_by_lookback(
    args: argparse.Namespace, lookbacks: set[int]
) -> tuple[dict[int, TrainingExamples], int]:
    """Build one example set per lookback over identical (user, cutoff) rows.

    Every set uses the longest window for eligibility, so a user is eligible
    for all models or none and the sets stay row-aligned.
    """
    eligibility_lookback_days = max(lookbacks)
    examples_by_lookback = {
        days: load_or_build_examples(
            args.db_dir,
            args.output_dir,
            args.start_cutoff,
            lookback_days=days,
            eligibility_lookback_days=eligibility_lookback_days,
        )
        for days in sorted(lookbacks)
    }
    reference = next(iter(examples_by_lookback.values()))
    for examples in examples_by_lookback.values():
        _validate_aligned_examples(reference, examples)
    return examples_by_lookback, eligibility_lookback_days


def run_backtest(args: argparse.Namespace) -> None:
    args.output_dir = _create_output_dir(args.output_dir)
    chunk_lookbacks = dict(zip(CHUNK_NAMES, args.lookbacks))
    needed_lookbacks = set(chunk_lookbacks.values())
    if not args.no_global_baseline:
        needed_lookbacks.add(args.global_lookback)
    examples_by_lookback, eligibility_lookback_days = load_examples_by_lookback(
        args, needed_lookbacks
    )
    # Rows, users, cutoffs, and labels are identical across lookbacks.
    reference = examples_by_lookback[args.lookbacks[0]]

    partitions = _partition_indices(reference)
    train_end, calibration_end = (int(day) for day in partitions.pop("_bounds"))
    available_train_examples = len(partitions["train"])
    apply_train_fraction(partitions, args.train_fraction, args.seed)
    train_indices = partitions["train"]
    calibration_indices = partitions["calibration"]
    holdout_indices = partitions["holdout"]

    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    cohort_user_ids = users.loc[users["is_cohort"], "user_id"].astype(np.int32).tolist()
    vocabulary = load_entity_vocabulary(args.db_dir)
    events = load_events(args.db_dir, cohort_user_ids, vocabulary)

    counts = prior_action_counts(
        events,
        reference.tokens.user_ids,
        reference.cutoff_days,
        args.activity_window_days,
    )
    percentiles = np.percentile(counts[train_indices], [10, 25, 50, 75, 90, 99])
    print(
        "Training-partition prior-action percentiles "
        f"(10/25/50/75/90/99): {percentiles.tolist()}"
    )
    thresholds = args.chunk_thresholds or auto_thresholds(counts[train_indices])
    chunks = assign_chunks(counts, thresholds)
    chunk_sizes = partition_chunk_counts(chunks, partitions)
    print(f"Chunk thresholds {thresholds}; examples per chunk: {chunk_sizes}")
    if args.inspect_only:
        return
    require_populated_chunks(chunk_sizes, thresholds)

    device = select_device(args.device)
    holdout_labels = reference.labels[holdout_indices]

    holdout_probabilities: dict[str, np.ndarray] = {}
    chunk_thresholds: dict[str, dict[str, float]] = {}
    models = [
        (name, np.flatnonzero(chunks == chunk), args.seed + chunk, chunk_lookbacks[name])
        for chunk, name in enumerate(CHUNK_NAMES)
    ]
    if not args.no_global_baseline:
        models.append(
            (
                "global",
                np.arange(len(chunks)),
                args.seed + len(CHUNK_NAMES),
                args.global_lookback,
            )
        )
    balanced_size = None
    if args.balance_chunks:
        balanced_size = min(
            int(np.sum(chunks[train_indices] == chunk))
            for chunk in range(len(CHUNK_NAMES))
        )
        print(f"Balancing chunk models to {balanced_size} training examples each")
    for name, member_indices, seed, lookback_days in models:
        examples = examples_by_lookback[lookback_days]
        tokens = examples.tokens
        train_members = np.intersect1d(train_indices, member_indices)
        if balanced_size is not None and name in CHUNK_NAMES:
            train_members = np.sort(
                np.random.default_rng(seed).choice(
                    train_members, size=balanced_size, replace=False
                )
            )
        calibration_members = np.intersect1d(calibration_indices, member_indices)
        print(
            f"Training {name} model ({lookback_days}-day lookback) on "
            f"{len(train_members)} examples"
        )
        model = fit_specialist(
            examples,
            tokens,
            train_members,
            ALL_OUTPUTS,
            device,
            args,
            seed,
            user_embedding=args.user_embedding,
        )
        calibration = predict_specialist(
            model, tokens, calibration_members, device, args.batch_size
        )
        chunk_thresholds[name] = choose_thresholds(
            examples.labels[calibration_members], calibration
        )
        # Every model scores the whole holdout so cross-chunk behaviour can be
        # analysed; the chunked output keeps only each row's own model.
        holdout_probabilities[name] = predict_specialist(
            model, tokens, holdout_indices, device, args.batch_size
        )
        torch.save(model.state_dict(), args.output_dir / f"{name}_model.pt")

    holdout_chunks = chunks[holdout_indices]
    chunked_probabilities = route_by_chunk(holdout_probabilities, holdout_chunks)
    threshold_matrix = {
        name: np.tile(
            [chunk_thresholds[name][label] for label in LABELS],
            (len(holdout_indices), 1),
        )
        for name in CHUNK_NAMES
    }
    chunked_predictions = (
        chunked_probabilities >= route_by_chunk(threshold_matrix, holdout_chunks)
    ).astype(np.uint8)

    holdout_metrics: dict[str, Any] = {
        "chunked": metrics_from_predictions(holdout_labels, chunked_predictions)
        | {"thresholds": {name: chunk_thresholds[name] for name in CHUNK_NAMES}}
    }
    per_chunk: dict[str, Any] = {}
    for chunk, name in enumerate(CHUNK_NAMES):
        rows = holdout_chunks == chunk
        per_chunk[name] = {
            "lookback_days": chunk_lookbacks[name],
            "holdout_examples": int(rows.sum()),
            "chunk_model": metrics_from_predictions(
                holdout_labels[rows], chunked_predictions[rows]
            ),
        }
        if "global" in holdout_probabilities:
            per_chunk[name]["global_model"] = calculate_metrics(
                holdout_labels[rows],
                holdout_probabilities["global"][rows],
                chunk_thresholds["global"],
            )
    if "global" in holdout_probabilities:
        holdout_metrics["global"] = calculate_metrics(
            holdout_labels,
            holdout_probabilities["global"],
            chunk_thresholds["global"],
        )

    tokens = reference.tokens
    probability_columns: dict[str, np.ndarray] = {
        "user_id": tokens.user_ids[holdout_indices],
        "cutoff": pd.to_datetime(reference.cutoff_days[holdout_indices], unit="D"),
        "prior_actions": counts[holdout_indices],
        "chunk": [CHUNK_NAMES[chunk] for chunk in holdout_chunks],
        **{
            f"actual_{label}": holdout_labels[:, column]
            for column, label in enumerate(LABELS)
        },
    }
    for branch, probabilities in {
        "chunked": chunked_probabilities,
        **holdout_probabilities,
    }.items():
        probability_columns.update(
            {
                f"{branch}_{label}": probabilities[:, column]
                for column, label in enumerate(LABELS)
            }
        )
    probability_path = args.output_dir / "holdout_probabilities.csv"
    pd.DataFrame(probability_columns).to_csv(probability_path, index=False)

    results: dict[str, Any] = {
        "experiment": "Relational GNN activity-chunk models with per-chunk lookbacks",
        "architecture": "gnn",
        "lookback_days": {
            **chunk_lookbacks,
            **({} if args.no_global_baseline else {"global": args.global_lookback}),
        },
        "eligibility_lookback_days": eligibility_lookback_days,
        "chunking": {
            "measure": (
                "actions known before the cutoff"
                + (
                    f" within the last {args.activity_window_days} days"
                    if args.activity_window_days
                    else " (all history)"
                )
            ),
            "thresholds": list(thresholds),
            "threshold_source": "manual" if args.chunk_thresholds else "auto_train_tertiles",
            "rule": {
                "low": f"count < {thresholds[0]}",
                "mid": f"{thresholds[0]} <= count < {thresholds[1]}",
                "high": f"count >= {thresholds[1]}",
            },
            "examples_per_chunk": chunk_sizes,
        },
        "train_examples": int(len(train_indices)),
        "available_train_examples": available_train_examples,
        "train_fraction": args.train_fraction,
        "user_embedding": args.user_embedding,
        "balance_chunks": args.balance_chunks,
        "balanced_chunk_train_examples": balanced_size,
        "calibration_examples": int(len(calibration_indices)),
        "holdout_examples": int(len(holdout_indices)),
        "train_end_exclusive": str(pd.Timestamp(train_end, unit="D")),
        "holdout_start": str(pd.Timestamp(calibration_end, unit="D")),
        "device": str(device),
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "threshold_selection": "per model, on its calibration examples only",
        "thresholds": chunk_thresholds,
        "holdout_metrics": holdout_metrics["chunked"],
        "branch_holdout": holdout_metrics,
        "per_chunk_holdout": per_chunk,
    }
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout probabilities to {probability_path}")


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--lookbacks",
        type=parse_lookbacks,
        default=DEFAULT_LOOKBACKS,
        metavar="LOW,MID,HIGH",
        help=(
            "graph lookback window in days for the low, mid, and high chunk "
            "models (default 180,21,7)"
        ),
    )
    parser.add_argument(
        "--global-lookback",
        type=int,
        default=DEFAULT_GLOBAL_LOOKBACK,
        help="lookback window in days for the global baseline model (default 14)",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    if args.global_lookback < 1:
        parser.error("--global-lookback must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    run_backtest(args)


if __name__ == "__main__":
    main()
