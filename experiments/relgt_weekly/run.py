"""Backtest RelGT on weekly activity vectors instead of event-level graph tokens."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.relgt.data import LABELS
from experiments.relgt.run import (
    calculate_metrics,
    choose_thresholds,
    chronological_splits,
    select_device,
    subsample_train_indices,
)
from experiments.relgt_weekly.data import (
    NUM_FEATURES,
    WeeklyExamples,
    load_or_build_weekly_examples,
)
from experiments.relgt_weekly.model import WeeklyGNN, WeeklyNN, WeeklyRelGT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_START_CUTOFF = "2026-03-11 16:00:00"
DEFAULT_WEEKS = 8
DEFAULT_ELIGIBILITY_DAYS = 14  # matches the other RelGT/GNN experiments
ARCHITECTURES = ("relgt", "nn", "gnn")
ARCHITECTURE_NAMES = {
    "relgt": "RelGT",
    "nn": "MLP on concatenated weekly vectors",
    "gnn": "Relational GNN",
}


def _create_output_dir(requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        requested_dir.mkdir(parents=True, exist_ok=True)
        return requested_dir
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DEFAULT_OUTPUT_ROOT / f"backtest_{timestamp}_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output_dir}")
    return output_dir


def _batch(
    examples: WeeklyExamples, indices: np.ndarray, device: torch.device, with_user: bool
) -> tuple[torch.Tensor, ...]:
    features = torch.as_tensor(
        examples.features[indices], dtype=torch.float32, device=device
    )
    if not with_user:
        return (features,)
    return features, torch.as_tensor(
        examples.user_ids[indices], dtype=torch.long, device=device
    )


def build_model(
    args: argparse.Namespace, known_user_ids: np.ndarray | None
) -> torch.nn.Module:
    shared = dict(
        num_weeks=args.weeks,
        num_features=NUM_FEATURES,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        known_user_ids=known_user_ids,
    )
    if args.architecture == "relgt":
        return WeeklyRelGT(
            num_heads=args.num_heads,
            num_local_layers=args.num_layers,
            num_centroids=args.num_centroids,
            **shared,
        )
    model_class = WeeklyNN if args.architecture == "nn" else WeeklyGNN
    return model_class(num_layers=args.num_layers, **shared)


def fit_model(
    examples: WeeklyExamples,
    train_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.nn.Module:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build_model(
        args,
        np.unique(examples.user_ids[train_indices]) if args.user_embedding else None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        shuffled = np.random.permutation(train_indices)
        total_loss, batches = 0.0, 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            targets = torch.as_tensor(
                examples.labels[batch_indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(*_batch(examples, batch_indices, device, args.user_embedding)),
                targets,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        print(f"Epoch {epoch + 1}/{args.epochs}: BCE={total_loss / max(batches, 1):.5f}")
    return model


def predict_probabilities(
    model: torch.nn.Module,
    examples: WeeklyExamples,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    with_user = model.user_embedding is not None
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            logits = model(*_batch(examples, batch_indices, device, with_user))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    if not outputs:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def run_backtest(args: argparse.Namespace) -> None:
    args.output_dir = _create_output_dir(args.output_dir)
    examples = load_or_build_weekly_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        args.weeks,
        args.eligibility_days,
    )
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    train_indices = np.flatnonzero(examples.cutoff_days < train_end)
    available_train_examples = len(train_indices)
    train_indices, unused_train_indices = subsample_train_indices(train_indices, args)
    calibration_indices = np.union1d(
        np.flatnonzero(
            (examples.cutoff_days >= train_end)
            & (examples.cutoff_days < calibration_end)
        ),
        unused_train_indices,
    )
    holdout_indices = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not len(calibration_indices) or not len(holdout_indices):
        raise ValueError(
            "The chronological split produced an empty calibration or holdout set."
        )

    device = select_device(args.device)
    print(
        f"Device: {device}; train < {pd.Timestamp(train_end, unit='D')}, "
        f"holdout >= {pd.Timestamp(calibration_end, unit='D')}; "
        f"{args.architecture} on {args.weeks} weekly vectors per sample"
    )
    model = fit_model(examples, train_indices, device, args)
    calibration_probabilities = predict_probabilities(
        model, examples, calibration_indices, device, args.batch_size
    )
    holdout_probabilities = predict_probabilities(
        model, examples, holdout_indices, device, args.batch_size
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

    torch.save(model.state_dict(), args.output_dir / f"weekly_{args.architecture}_model.pt")
    probability_columns = {
        "user_id": examples.user_ids[holdout_indices],
        "cutoff": pd.to_datetime(examples.cutoff_days[holdout_indices], unit="D"),
        **{
            f"actual_{label}": holdout_labels[:, column]
            for column, label in enumerate(LABELS)
        },
        **{
            f"weekly_{args.architecture}_{label}": holdout_probabilities[:, column]
            for column, label in enumerate(LABELS)
        },
    }
    probability_path = args.output_dir / "holdout_probabilities.csv"
    pd.DataFrame(probability_columns).to_csv(probability_path, index=False)

    results = {
        "experiment": f"{ARCHITECTURE_NAMES[args.architecture]} on weekly activity vectors",
        "architecture": args.architecture,
        "weeks": args.weeks,
        "week_features": "per relation: log1p(events), log1p(distinct entities), log1p(distinct repos)",
        "eligibility_days": args.eligibility_days,
        "user_embedding": args.user_embedding,
        "train_fraction": args.train_fraction,
        "train_examples": int(len(train_indices)),
        "available_train_examples": available_train_examples,
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
            "num_layers": args.num_layers,
            "num_centroids": args.num_centroids,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "threshold_selection": "per label, on calibration weeks only",
        "thresholds": thresholds,
        "holdout_metrics": metrics,
    }
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "holdout_metrics"}, indent=2))
    print(
        f"Holdout score {metrics['score']:.4f} "
        f"(macro-F1 {metrics['macro_f1']:.4f}, micro-F1 {metrics['micro_f1']:.4f})"
    )
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout probabilities to {probability_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF)
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="relgt",
        help="model over the weekly vectors (default %(default)s)",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=DEFAULT_WEEKS,
        help="number of past weekly vectors per sample (default %(default)s)",
    )
    parser.add_argument(
        "--eligibility-days",
        type=int,
        default=DEFAULT_ELIGIBILITY_DAYS,
        help="users need 5 actions in this many days to get a sample (default %(default)s)",
    )
    parser.add_argument(
        "--user-embedding",
        action="store_true",
        help="add a learned per-user vector to the root token (unseen users get zero)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="fraction of the training partition to use; the rest joins calibration",
    )
    parser.add_argument(
        "--discard-unused-train",
        action="store_true",
        help="with --train-fraction below 1, drop the unused examples instead",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-centroids", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "weeks", "eligibility_days", "epochs", "batch_size", "hidden_dim",
        "num_heads", "num_layers", "num_centroids",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.hidden_dim % args.num_heads:
        parser.error("--hidden-dim must be divisible by --num-heads.")
    if not 0.0 < args.train_fraction <= 1.0:
        parser.error("--train-fraction must be in (0, 1].")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1).")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    run_backtest(args)


if __name__ == "__main__":
    main()
