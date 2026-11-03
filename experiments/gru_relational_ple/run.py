"""Chronological backtest for GRU + relational attention + PLE."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.relgt.data import (
    HORIZON_DAYS,
    LABELS,
    TrainingExamples,
    load_entity_vocabulary,
    load_events,
)
from experiments.relgt.run import (
    calculate_metrics,
    choose_thresholds,
    chronological_splits,
    load_or_build_examples,
    select_device,
)
from experiments.gru_relational_ple.model import GRURelationalAttentionPLE


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_START_CUTOFF = "2026-03-11 16:00:00"
LOOKBACK_DAYS = 14
DAY_NS = pd.Timedelta(days=1).value


def build_daily_activity_sequences(
    events: pd.DataFrame,
    user_ids: np.ndarray,
    cutoff_days: np.ndarray,
) -> np.ndarray:
    """Build cutoff-safe 90-day daily action counts for each user-week example."""
    if user_ids.shape != cutoff_days.shape:
        raise ValueError("user_ids and cutoff_days must have matching shapes.")
    sequences = np.zeros(
        (len(user_ids), LOOKBACK_DAYS, len(LABELS)), dtype=np.float16
    )
    if not len(user_ids) or events.empty:
        return sequences

    histories: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for user_id, group in events.groupby("user_id", sort=False):
        action_ns = group["action_at"].astype("int64").to_numpy()
        available_ns = group["available_at"].astype("int64").to_numpy()
        relation_ids = group["relation_id"].to_numpy(dtype=np.int64) - 1
        order = np.argsort(action_ns, kind="stable")
        histories[int(user_id)] = (
            action_ns[order],
            available_ns[order],
            relation_ids[order],
        )

    example_order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[example_order]
    group_starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(example_order)]
    lookback_ns = LOOKBACK_DAYS * DAY_NS

    for start, end in zip(group_starts, group_ends):
        user_id = int(sorted_users[start])
        history = histories.get(user_id)
        if history is None:
            continue
        action_ns, available_ns, relation_ids = history
        for row_index in example_order[start:end]:
            cutoff_ns = int(cutoff_days[row_index]) * DAY_NS
            left = np.searchsorted(action_ns, cutoff_ns - lookback_ns, side="left")
            right = np.searchsorted(action_ns, cutoff_ns, side="left")
            event_indices = np.arange(left, right, dtype=np.int64)
            if not event_indices.size:
                continue
            event_indices = event_indices[available_ns[event_indices] < cutoff_ns]
            if not event_indices.size:
                continue

            age_days = (
                cutoff_ns - action_ns[event_indices] - 1
            ) // DAY_NS
            day_indices = LOOKBACK_DAYS - 1 - age_days.astype(np.int64)
            daily_counts = np.zeros(
                (LOOKBACK_DAYS, len(LABELS)), dtype=np.float32
            )
            np.add.at(
                daily_counts,
                (day_indices, relation_ids[event_indices]),
                1.0,
            )
            sequences[row_index] = np.log1p(daily_counts).astype(np.float16)
    return sequences


def _batch_inputs(
    examples: TrainingExamples,
    sequences: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    tokens = examples.tokens
    return (
        torch.as_tensor(tokens.entity_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(
            tokens.node_type_ids[indices], dtype=torch.long, device=device
        ),
        torch.as_tensor(
            tokens.relation_ids[indices], dtype=torch.long, device=device
        ),
        torch.as_tensor(tokens.hop_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(
            tokens.node_features[indices], dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            tokens.relative_days[indices], dtype=torch.float32, device=device
        ),
        torch.as_tensor(sequences[indices], dtype=torch.float32, device=device),
    )


def fit_model(
    examples: TrainingExamples,
    sequences: np.ndarray,
    train_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> GRURelationalAttentionPLE:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = GRURelationalAttentionPLE(
        num_entities=examples.num_entities,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_graph_layers=args.num_graph_layers,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        shuffled = np.random.permutation(train_indices)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            inputs = _batch_inputs(examples, sequences, batch_indices, device)
            labels = torch.as_tensor(
                examples.labels[batch_indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        print(f"Epoch {epoch + 1}/{args.epochs}: BCE={total_loss / batches:.5f}")
    return model


def predict_probabilities(
    model: GRURelationalAttentionPLE,
    examples: TrainingExamples,
    sequences: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            inputs = _batch_inputs(examples, sequences, batch_indices, device)
            predictions.append(torch.sigmoid(model(*inputs)).cpu().numpy())
    if not predictions:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


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
        args.db_dir, args.output_dir, args.start_cutoff
    )
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    train_indices = np.flatnonzero(examples.cutoff_days < train_end)
    calibration_indices = np.flatnonzero(
        (examples.cutoff_days >= train_end)
        & (examples.cutoff_days < calibration_end)
    )
    holdout_indices = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not len(train_indices) or not len(calibration_indices) or not len(holdout_indices):
        raise ValueError(
            "The chronological split produced an empty train, calibration, "
            "or holdout set."
        )

    vocabulary = load_entity_vocabulary(args.db_dir)
    users = pd.read_parquet(
        args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"]
    )
    cohort_user_ids = (
        users.loc[users["is_cohort"], "user_id"].astype(np.int32).tolist()
    )
    events = load_events(args.db_dir, cohort_user_ids, vocabulary)
    sequences = build_daily_activity_sequences(
        events, examples.tokens.user_ids, examples.cutoff_days
    )
    device = select_device(args.device)
    print(
        f"Device: {device}; chronological boundaries: train < "
        f"{pd.Timestamp(train_end, unit='D')}, holdout >= "
        f"{pd.Timestamp(calibration_end, unit='D')}"
    )
    model = fit_model(examples, sequences, train_indices, device, args)
    calibration_probabilities = predict_probabilities(
        model,
        examples,
        sequences,
        calibration_indices,
        device,
        args.batch_size,
    )
    holdout_probabilities = predict_probabilities(
        model,
        examples,
        sequences,
        holdout_indices,
        device,
        args.batch_size,
    )
    thresholds = choose_thresholds(
        examples.labels[calibration_indices], calibration_probabilities
    )
    metrics = calculate_metrics(
        examples.labels[holdout_indices],
        holdout_probabilities,
        thresholds,
        examples.cutoff_days[holdout_indices],
    )
    torch.save(model.state_dict(), args.output_dir / "gru_relational_ple_model.pt")
    holdout_cutoffs = pd.to_datetime(
        examples.cutoff_days[holdout_indices], unit="D"
    )
    pd.DataFrame(
        {
            "user_id": examples.tokens.user_ids[holdout_indices],
            "cutoff": holdout_cutoffs,
            **{
                label: holdout_probabilities[:, column]
                for column, label in enumerate(LABELS)
            },
        }
    ).to_csv(args.output_dir / "holdout_probabilities.csv", index=False)
    results: dict[str, Any] = {
        "experiment": "GRU + relational attention + PLE",
        "lookback_days": LOOKBACK_DAYS,
        "horizon_days": HORIZON_DAYS,
        "train_examples": int(len(train_indices)),
        "calibration_examples": int(len(calibration_indices)),
        "holdout_examples": int(len(holdout_indices)),
        "train_end_exclusive": str(pd.Timestamp(train_end, unit="D")),
        "holdout_start": str(pd.Timestamp(calibration_end, unit="D")),
        "device": str(device),
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_graph_layers": args.num_graph_layers,
            "num_gru_layers": args.num_gru_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "holdout_metrics": metrics,
    }
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved holdout metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--start-cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(DEFAULT_START_CUTOFF),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-graph-layers", type=int, default=2)
    parser.add_argument("--num-gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for argument in ("epochs", "batch_size", "hidden_dim", "num_heads",
                     "num_graph_layers", "num_gru_layers"):
        if getattr(args, argument) < 1:
            parser.error(f"--{argument.replace('_', '-')} must be positive.")
    if args.hidden_dim % args.num_heads:
        parser.error("--hidden-dim must be divisible by --num-heads.")
    if args.hidden_dim < 2:
        parser.error("--hidden-dim must be at least 2.")
    if args.dropout < 0 or args.dropout >= 1:
        parser.error("--dropout must be in [0, 1).")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    run_backtest(args)


if __name__ == "__main__":
    main()
