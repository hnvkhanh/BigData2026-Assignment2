"""Backtest relational GNNs trained on user chunks split by prior activity."""

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
from sklearn.metrics import f1_score

from experiments.relgt.data import (
    LABELS,
    TrainingExamples,
    load_entity_vocabulary,
    load_events,
)
from experiments.relgt.run import (
    calculate_metrics,
    weekly_scores,
    choose_thresholds,
    chronological_splits,
    load_or_build_examples,
    select_device,
)
from experiments.relgt_action_specialists.run import (
    fit_specialist,
    predict_specialist,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_START_CUTOFF = "2026-03-11 16:00:00"
LOOKBACK_DAYS = 14
CHUNK_NAMES = ("low", "mid", "high")
ALL_OUTPUTS = tuple(range(len(LABELS)))
SECONDS_PER_DAY = 86_400
USER_KEY_SHIFT = 32


def prior_action_counts(
    events: pd.DataFrame,
    user_ids: np.ndarray,
    cutoff_days: np.ndarray,
    window_days: int = 0,
) -> np.ndarray:
    """Count each user's actions known before each example's cutoff.

    An action counts once its action, row, and target-entity times are all
    earlier than the cutoff, matching the snapshot availability rules. With
    ``window_days`` of zero, the whole cutoff-safe history is counted;
    otherwise only actions that became known in ``[cutoff - window, cutoff)``.
    """
    if window_days < 0:
        raise ValueError("window_days must not be negative.")
    known_at = (
        events[["action_at", "available_at", "entity_available_at"]]
        .max(axis=1)
        .to_numpy(dtype="datetime64[s]")
        .astype(np.int64)
    )
    keys = np.sort(
        (events["user_id"].to_numpy(dtype=np.int64) << USER_KEY_SHIFT) + known_at
    )
    user_base = user_ids.astype(np.int64) << USER_KEY_SHIFT
    cutoff_seconds = cutoff_days.astype(np.int64) * SECONDS_PER_DAY
    upper = np.searchsorted(keys, user_base + cutoff_seconds, side="left")
    lower_seconds = (
        np.zeros_like(cutoff_seconds)
        if window_days == 0
        else np.maximum(cutoff_seconds - window_days * SECONDS_PER_DAY, 0)
    )
    lower = np.searchsorted(keys, user_base + lower_seconds, side="left")
    return (upper - lower).astype(np.int64)


def assign_chunks(counts: np.ndarray, thresholds: tuple[int, int]) -> np.ndarray:
    """Map counts to chunk 0 (< t1), 1 (t1 <= count < t2), or 2 (>= t2)."""
    return np.searchsorted(np.asarray(thresholds), counts, side="right").astype(
        np.int8
    )


def parse_chunk_thresholds(value: str) -> tuple[int, int] | None:
    """Parse ``auto`` (None) or two increasing positive integers like ``10,100``."""
    if value.strip().lower() == "auto":
        return None
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "chunk thresholds must be 'auto' or two integers such as 10,100."
        ) from None
    if len(parts) != 2 or not 0 < parts[0] < parts[1]:
        raise argparse.ArgumentTypeError(
            "chunk thresholds need exactly two integers with 0 < low < high."
        )
    return parts


def auto_thresholds(train_counts: np.ndarray) -> tuple[int, int]:
    """Tertiles of the training partition's counts (no calibration/holdout use)."""
    low, high = (int(np.ceil(q)) for q in np.quantile(train_counts, [1 / 3, 2 / 3]))
    if not 0 < low < high:
        raise ValueError(
            f"Automatic tertile thresholds ({low}, {high}) are degenerate; "
            "pass --chunk-thresholds LOW,HIGH explicitly."
        )
    return low, high


def partition_chunk_counts(
    chunks: np.ndarray, partitions: dict[str, np.ndarray]
) -> dict[str, dict[str, int]]:
    return {
        name: {
            partition: int(np.sum(chunks[indices] == chunk))
            for partition, indices in partitions.items()
        }
        for chunk, name in enumerate(CHUNK_NAMES)
    }


def require_populated_chunks(
    counts: dict[str, dict[str, int]], thresholds: tuple[int, int]
) -> None:
    for name, sizes in counts.items():
        empty = [partition for partition, size in sizes.items() if size == 0]
        if empty:
            raise ValueError(
                f"Chunk '{name}' has no {', '.join(empty)} examples with "
                f"thresholds {thresholds}; adjust --chunk-thresholds "
                f"(sizes: {counts})."
            )


def metrics_from_predictions(
    labels: np.ndarray, predictions: np.ndarray, cutoffs: np.ndarray | None = None
) -> dict[str, Any]:
    if cutoffs is not None:
        return weekly_scores(labels, predictions, cutoffs)
    per_action = {
        label: float(
            f1_score(labels[:, column], predictions[:, column], zero_division=0)
        )
        for column, label in enumerate(LABELS)
    }
    macro = float(np.mean(list(per_action.values())))
    micro = float(f1_score(labels.ravel(), predictions.ravel(), zero_division=0))
    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "score": (macro + micro) / 2,
        "per_action_f1": per_action,
    }


def route_by_chunk(
    chunk_values: dict[str, np.ndarray],
    chunks: np.ndarray,
) -> np.ndarray:
    """Take each row's values from the model that owns the row's chunk."""
    routed = np.empty_like(chunk_values[CHUNK_NAMES[0]])
    for chunk, name in enumerate(CHUNK_NAMES):
        routed[chunks == chunk] = chunk_values[name][chunks == chunk]
    return routed


def _partition_indices(examples: TrainingExamples) -> dict[str, np.ndarray]:
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    partitions = {
        "train": np.flatnonzero(examples.cutoff_days < train_end),
        "calibration": np.flatnonzero(
            (examples.cutoff_days >= train_end)
            & (examples.cutoff_days < calibration_end)
        ),
        "holdout": np.flatnonzero(examples.cutoff_days >= calibration_end),
    }
    if not all(indices.size for indices in partitions.values()):
        raise ValueError(
            "The chronological split produced an empty train, calibration, "
            "or holdout set."
        )
    return partitions | {"_bounds": np.array([train_end, calibration_end])}


def apply_train_fraction(
    partitions: dict[str, np.ndarray], fraction: float, seed: int
) -> None:
    """Keep a random ``fraction`` of the train partition; the rest joins calibration."""
    if fraction >= 1.0:
        return
    available = len(partitions["train"])
    keep = max(1, round(available * fraction))
    chosen = np.sort(
        np.random.default_rng(seed).choice(
            partitions["train"], size=keep, replace=False
        )
    )
    partitions["calibration"] = np.union1d(
        partitions["calibration"], np.setdiff1d(partitions["train"], chosen)
    )
    partitions["train"] = chosen
    print(
        f"Using {keep} of {available} training examples "
        f"(--train-fraction {fraction}); the rest go to calibration"
    )


def _create_output_dir(requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        requested_dir.mkdir(parents=True, exist_ok=True)
        return requested_dir
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DEFAULT_OUTPUT_ROOT / f"backtest_{timestamp}_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output_dir}")
    return output_dir


def run_backtest(args: argparse.Namespace) -> None:
    args.output_dir = _create_output_dir(args.output_dir)
    examples = load_or_build_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        lookback_days=LOOKBACK_DAYS,
    )
    partitions = _partition_indices(examples)
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
        examples.tokens.user_ids,
        examples.cutoff_days,
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
    tokens = examples.tokens
    holdout_labels = examples.labels[holdout_indices]

    holdout_probabilities: dict[str, np.ndarray] = {}
    chunk_thresholds: dict[str, dict[str, float]] = {}
    models = [
        (name, np.flatnonzero(chunks == chunk), args.seed + chunk)
        for chunk, name in enumerate(CHUNK_NAMES)
    ]
    if not args.no_global_baseline:
        models.append(("global", np.arange(len(chunks)), args.seed + len(CHUNK_NAMES)))
    balanced_size = None
    if args.balance_chunks:
        balanced_size = min(
            int(np.sum(chunks[train_indices] == chunk))
            for chunk in range(len(CHUNK_NAMES))
        )
        print(f"Balancing chunk models to {balanced_size} training examples each")
    for name, member_indices, seed in models:
        train_members = np.intersect1d(train_indices, member_indices)
        if balanced_size is not None and name in CHUNK_NAMES:
            train_members = np.sort(
                np.random.default_rng(seed).choice(
                    train_members, size=balanced_size, replace=False
                )
            )
        calibration_members = np.intersect1d(calibration_indices, member_indices)
        print(f"Training {name} model on {len(train_members)} examples")
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

    holdout_cutoffs = examples.cutoff_days[holdout_indices]
    holdout_metrics: dict[str, Any] = {
        "chunked": metrics_from_predictions(
            holdout_labels, chunked_predictions, holdout_cutoffs
        )
        | {"thresholds": {name: chunk_thresholds[name] for name in CHUNK_NAMES}}
    }
    per_chunk: dict[str, Any] = {}
    for chunk, name in enumerate(CHUNK_NAMES):
        rows = holdout_chunks == chunk
        per_chunk[name] = {
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
            holdout_cutoffs,
        )

    probability_columns: dict[str, np.ndarray] = {
        "user_id": tokens.user_ids[holdout_indices],
        "cutoff": pd.to_datetime(examples.cutoff_days[holdout_indices], unit="D"),
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
        "experiment": "Relational GNN activity-chunk models",
        "lookback_days": LOOKBACK_DAYS,
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
        "user_embedding": args.user_embedding,
        "balance_chunks": args.balance_chunks,
        "balanced_chunk_train_examples": balanced_size,
        "train_fraction": args.train_fraction,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--start-cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(DEFAULT_START_CUTOFF),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--chunk-thresholds",
        type=parse_chunk_thresholds,
        default=None,
        metavar="LOW,HIGH",
        help=(
            "two action-count thresholds: low < LOW <= mid < HIGH <= high. "
            "Default 'auto' uses tertiles of the training partition."
        ),
    )
    parser.add_argument(
        "--activity-window-days",
        type=int,
        default=0,
        help="count actions in this many days before the cutoff; 0 = all history",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="print activity percentiles and chunk sizes, then exit without training",
    )
    parser.add_argument(
        "--no-global-baseline",
        action="store_true",
        help="skip the single model trained on all users for comparison",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help=(
            "fraction of the training partition to use, sampled at random "
            "(seeded by --seed); the unused remainder is added to calibration"
        ),
    )
    parser.add_argument(
        "--balance-chunks",
        action="store_true",
        help=(
            "randomly downsample larger chunks' training examples so every "
            "chunk model trains on as many as the smallest chunk has; the "
            "global baseline is unaffected"
        ),
    )
    parser.add_argument(
        "--user-embedding",
        action="store_true",
        help=(
            "give each model a learned per-user embedding added to the root node; "
            "users absent from a model's training examples get a zero embedding"
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "hidden_dim", "num_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.activity_window_days < 0:
        parser.error("--activity-window-days must not be negative.")
    if not 0.0 < args.train_fraction <= 1.0:
        parser.error("--train-fraction must be in (0, 1].")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1).")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def main() -> None:
    args = parse_args()
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    run_backtest(args)


if __name__ == "__main__":
    main()
