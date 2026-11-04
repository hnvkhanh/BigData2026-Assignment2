"""Backtest or predict with a Relational Transformer on the GitHub database.

Port of stanford-star/relational-transformer (Ranjan et al., 2025): every
known cell of the sampled rows is a token, and attention is restricted to the
same column, the row and its foreign-key parents, or the rows that reference
it. The target is a masked ``UserWeeks`` task row linked to the user; the
user's earlier, already-closed weeks are labelled in-context task rows.

Samples, labels, eligibility (>= 5 actions in 14 days), sampled rows
(``relgt_db``'s 1/2-hop sampler), the chronological split, epoch selection
(best calibration macro AUPRC) and threshold calibration match ``relgt_db``,
so holdout scores are directly comparable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from experiments.relgt.data import LABELS, load_entity_vocabulary, load_events  # noqa: E402
from experiments.relgt.database import (  # noqa: E402
    SECONDS_PER_DAY,
    load_database,
    timestamp_seconds,
)
from experiments.relgt.db_run import (  # noqa: E402
    DatabaseContext,
    DatabaseExamples,
    load_database_context,
    load_or_build_db_examples,
    macro_average_precision,
)
from experiments.relgt.db_sampler import TokenSets, sample_tokens  # noqa: E402
from experiments.relgt.run import (  # noqa: E402
    DEFAULT_DB_DIR,
    DEFAULT_PREDICTION_CUTOFF,
    DEFAULT_START_CUTOFF,
    DEFAULT_TEST_INPUT,
    calculate_metrics,
    choose_thresholds,
    chronological_splits,
    create_run_output_dir,
    parse_train_fraction,
    prediction_example_indices,
    select_device,
    subsample_train_indices,
)
from experiments.relational_transformer.cells import (  # noqa: E402
    SEM_NUMBER,
    WEEK_SECONDS,
    ActivityHistory,
    CellBuilder,
)
from experiments.relational_transformer.model import (
    ATTENTION_TYPES,
    RelationalTransformer,
)  # noqa: E402


BRANCH = "relational_transformer"
MODEL_FILE = "relational_transformer_model.pt"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
ARCHITECTURE = (
    "Relational Transformer (cell tokens; column / feature / neighbor masked attention) "
    "over the pkey/fkey database plus a UserWeeks task table"
)
UPSTREAM = "https://github.com/stanford-star/relational-transformer"
# RMSNorm keeps float32 weights under bfloat16 autocast; the unfused path is fine.
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")


def parse_attention(value: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    unknown = sorted(set(names) - set(ATTENTION_TYPES))
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"choose a comma-separated subset of {', '.join(ATTENTION_TYPES)}"
        )
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backtest", "predict"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
        sub.add_argument(
            "--output-dir", type=Path, default=None, help="default: a new timestamped folder"
        )
        sub.add_argument(
            "--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF)
        )
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument(
            "--train-fraction",
            type=parse_train_fraction,
            default=1.0,
            help="random fraction of training examples to train on; the rest is added to calibration",
        )
        sub.add_argument(
            "--discard-unused-train",
            action="store_true",
            help="with --train-fraction below 1, drop the unused examples instead",
        )
        context = sub.add_argument_group("context")
        context.add_argument(
            "--num-neighbors",
            type=int,
            default=64,
            help="database rows sampled per user-week, including the user row (default: 64)",
        )
        context.add_argument(
            "--context-mode",
            choices=["r0", "r1", "r2", "r3"],
            default="r2",
            help="relgt_db sampler context: r0 own rows, r1 + touching rows, r2 full 1/2-hop (default)",
        )
        context.add_argument(
            "--history-weeks",
            type=int,
            default=8,
            help="earlier closed UserWeeks rows of the same user with visible labels (default: 8)",
        )
        context.add_argument(
            "--max-cells", type=int, default=512, help="cell tokens per sequence (default: 512)"
        )
        context.add_argument("--text-buckets", type=int, default=2**16)
        model = sub.add_argument_group("model")
        model.add_argument("--d-model", type=int, default=128)
        model.add_argument("--num-heads", type=int, default=4)
        model.add_argument("--num-blocks", type=int, default=4)
        model.add_argument("--d-ff", type=int, default=256)
        model.add_argument(
            "--attention",
            type=parse_attention,
            default=("col", "feat", "nbr"),
            help="attention layers per block: col,feat,nbr (default; upstream) and optionally full (v1)",
        )
        training = sub.add_argument_group("training")
        training.add_argument("--epochs", type=int, default=5)
        training.add_argument("--batch-size", type=int, default=32)
        training.add_argument("--lr", type=float, default=3e-4)
        training.add_argument("--weight-decay", type=float, default=0.01)
        training.add_argument("--max-steps-per-epoch", type=int, default=None)
        training.add_argument(
            "--mtp-weight",
            type=float,
            default=0.0,
            help="weight of the auxiliary masked-number reconstruction loss (default: 0 = off)",
        )
        training.add_argument(
            "--mtp-prob", type=float, default=0.15, help="share of number cells masked for it"
        )
        training.add_argument(
            "--amp",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="bfloat16 autocast on CUDA (default: on)",
        )
        if command == "predict":
            sub.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
            sub.add_argument(
                "--cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF)
            )
    args = parser.parse_args(argv)
    for name in (
        "epochs",
        "batch_size",
        "num_blocks",
        "d_model",
        "d_ff",
        "num_heads",
        "text_buckets",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.d_model % args.num_heads:
        parser.error("--d-model must be divisible by --num-heads.")
    if not 2 <= args.num_neighbors <= 32_767:
        parser.error("--num-neighbors must be between 2 and 32767.")
    if args.history_weeks < 0:
        parser.error("--history-weeks must be zero or positive.")
    if args.max_cells < 7 * (1 + args.history_weeks) + 1:
        parser.error("--max-cells must fit the task row, every history row, and the user row.")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch < 1:
        parser.error("--max-steps-per-epoch must be positive.")
    if args.lr <= 0 or args.weight_decay < 0 or args.mtp_weight < 0:
        parser.error("--lr must be positive; --weight-decay and --mtp-weight non-negative.")
    if not 0 < args.mtp_prob < 1:
        parser.error("--mtp-prob must be in (0, 1).")
    # Read by relgt_db's example builder and cache signature.
    args.temporal_model = BRANCH
    if args.output_dir is None:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"{args.command}_{stamp}_{os.getpid()}"
    return args


def build_context(args: argparse.Namespace) -> tuple[DatabaseContext, CellBuilder]:
    database = load_database(args.db_dir)
    context = load_database_context(args.db_dir, args.text_buckets, database)
    builder = CellBuilder(database, context.graph, context.features)
    print(f"RT vocabulary: {builder.num_columns} columns, {builder.num_categories:,} categories")
    return context, builder


def model_cutoff(cutoff: pd.Timestamp) -> int:
    """Seconds at which inputs are read: midnight of the cutoff day.

    ``relgt_db`` stores example cutoffs as whole days, so training inputs are
    read at midnight; prediction uses the same convention.
    """
    return timestamp_seconds(cutoff) // SECONDS_PER_DAY * SECONDS_PER_DAY


def build_history(args: argparse.Namespace, last_cutoff: int) -> ActivityHistory:
    """Per-week first-known action times, starting ``--history-weeks`` before the first cutoff.

    Weeks start at midnight like the model cutoffs, so the week just before a
    cutoff is closed at that cutoff.
    """
    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    cohort_ids = users.loc[users["is_cohort"], "user_id"].astype(np.int32).to_numpy()
    events = load_events(args.db_dir, cohort_ids.tolist(), load_entity_vocabulary(args.db_dir))
    first_week = model_cutoff(args.start_cutoff) - max(args.history_weeks, 1) * WEEK_SECONDS
    return ActivityHistory.build(events, cohort_ids, first_week, last_cutoff)


def context_nodes(context: DatabaseContext, tokens: TokenSets) -> np.ndarray:
    return context.graph.table_offsets[tokens.node_types.astype(np.int64)] + tokens.node_rows


class Batcher:
    """Turn example indices into model-ready tensors."""

    def __init__(
        self,
        builder: CellBuilder,
        history: ActivityHistory,
        nodes: np.ndarray,
        user_ids: np.ndarray,
        cutoff_seconds: np.ndarray,
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
        self.builder = builder
        self.history = history
        self.nodes = nodes
        self.user_ids = user_ids
        self.cutoff_seconds = cutoff_seconds
        self.history_weeks = args.history_weeks
        self.max_cells = args.max_cells
        self.device = device

    def __call__(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        cutoffs = self.cutoff_seconds[indices]
        labels, starts, present = self.history.rows(
            self.user_ids[indices], cutoffs, self.history_weeks
        )
        arrays = self.builder.build(
            self.nodes[indices], cutoffs, labels, starts, present, self.max_cells
        )
        return {key: torch.as_tensor(value, device=self.device) for key, value in arrays.items()}


def build_model(args: argparse.Namespace, builder: CellBuilder) -> RelationalTransformer:
    return RelationalTransformer(
        num_columns=builder.num_columns,
        num_categories=builder.num_categories,
        text_buckets=args.text_buckets,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        d_ff=args.d_ff,
        attention_types=args.attention,
    )


def _autocast(args: argparse.Namespace, device: torch.device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"
    )


def predict_probabilities(
    model: RelationalTransformer,
    batcher: Batcher,
    indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode(), _autocast(args, device):
        for start in range(0, len(indices), args.batch_size):
            logits, _ = model(batcher(indices[start : start + args.batch_size]))
            predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
    if not predictions:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def fit_model(
    args: argparse.Namespace,
    builder: CellBuilder,
    batcher: Batcher,
    labels: np.ndarray,
    train_indices: np.ndarray,
    selection_indices: np.ndarray,
    device: torch.device,
) -> tuple[RelationalTransformer, list[dict[str, float]]]:
    """Train with BCE (plus optional MTP); keep the best calibration-AUPRC epoch."""
    if not len(train_indices) or not len(selection_indices):
        raise ValueError("Training and calibration examples are both required.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build_model(args, builder).to(device)
    print(f"Relational Transformer parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        shuffled = np.random.permutation(train_indices)
        totals = {"bce": 0.0, "mtp": 0.0, "cells": 0.0}
        steps = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            batch = batcher(batch_indices)
            target = torch.as_tensor(labels[batch_indices], dtype=torch.float32, device=device)
            mtp_mask = None
            if args.mtp_weight > 0:
                draw = torch.rand(batch["sem"].shape, generator=generator, device=device)
                mtp_mask = (
                    (draw < args.mtp_prob)
                    & (batch["sem"] == SEM_NUMBER)
                    & ~batch["is_target"]
                    & ~batch["padding"]
                )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(args, device):
                logits, mtp_loss = model(batch, mtp_mask)
            loss = criterion(logits.float(), target)
            totals["bce"] += float(loss.detach())
            if mtp_loss is not None:
                loss = loss + args.mtp_weight * mtp_loss
                totals["mtp"] += float(mtp_loss.detach())
            totals["cells"] += float((~batch["padding"]).sum(1).float().mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            steps += 1
            if steps % 500 == 0:
                print(f"  step {steps}: BCE={totals['bce'] / steps:.5f}")
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
        probabilities = predict_probabilities(model, batcher, selection_indices, args, device)
        score = macro_average_precision(labels[selection_indices], probabilities)
        row = {
            "epoch": epoch,
            "train_bce": totals["bce"] / max(steps, 1),
            "train_mtp": totals["mtp"] / max(steps, 1),
            "mean_cells": totals["cells"] / max(steps, 1),
            "selection_auprc": score,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}: BCE={row['train_bce']:.5f} "
            f"cells/seq={row['mean_cells']:.0f} calibration macro AUPRC={score:.4f} "
            f"({row['seconds']:.0f}s)"
        )
        if score >= best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _settings(
    args: argparse.Namespace, builder: CellBuilder, history: list[dict[str, float]]
) -> dict[str, Any]:
    best = max(history, key=lambda row: row["selection_auprc"]) if history else None
    return {
        "architecture": ARCHITECTURE,
        "upstream": UPSTREAM,
        "num_neighbors": args.num_neighbors,
        "context_mode": args.context_mode,
        "history_weeks": args.history_weeks,
        "max_cells": args.max_cells,
        "text_buckets": args.text_buckets,
        "num_columns": builder.num_columns,
        "num_categories": builder.num_categories,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_blocks": args.num_blocks,
        "d_ff": args.d_ff,
        "attention": list(args.attention),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "mtp_weight": args.mtp_weight,
        "mtp_prob": args.mtp_prob,
        "amp": args.amp,
        "seed": args.seed,
        "eligibility": ">= 5 cutoff-safe actions in the 14 days before the cutoff",
        "selected_epoch": best["epoch"] if best else None,
        "epoch_history": history,
        "train_fraction": args.train_fraction,
        "discard_unused_train": args.discard_unused_train,
        "threshold_selection": "per-label F1-maximizing threshold on calibration (relgt choose_thresholds)",
    }


def save_holdout_probabilities(
    examples: DatabaseExamples, indices: np.ndarray, probabilities: np.ndarray, path: Path
) -> None:
    output = pd.DataFrame(
        {
            "user_id": examples.user_ids[indices],
            "cutoff": pd.to_datetime(examples.cutoff_days[indices], unit="D"),
        }
    )
    for column, label in enumerate(LABELS):
        output[f"actual_{label}"] = examples.labels[indices, column]
    for column, label in enumerate(LABELS):
        output[f"{BRANCH}_{label}"] = probabilities[:, column]
    output.to_csv(path, index=False)


def run_backtest(args: argparse.Namespace) -> None:
    context, builder = build_context(args)
    examples = load_or_build_db_examples(args, context)
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    train_indices = np.flatnonzero(examples.cutoff_days < train_end)
    train_indices, unused_train_indices = subsample_train_indices(train_indices, args)
    calibration_indices = np.flatnonzero(
        (examples.cutoff_days >= train_end) & (examples.cutoff_days < calibration_end)
    )
    calibration_indices = np.union1d(calibration_indices, unused_train_indices)
    holdout_indices = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not len(calibration_indices) or not len(holdout_indices):
        raise ValueError("The chronological split produced an empty calibration or holdout set.")

    device = select_device(args.device)
    history = build_history(args, int(examples.cutoff_seconds.max()))
    batcher = Batcher(
        builder,
        history,
        context_nodes(context, examples.tokens),
        examples.user_ids,
        examples.cutoff_seconds,
        args,
        device,
    )
    print(
        f"Device: {device}; {len(train_indices):,} train / {len(calibration_indices):,} calibration / "
        f"{len(holdout_indices):,} holdout user-weeks; train < {pd.Timestamp(train_end, unit='D')}, "
        f"holdout >= {pd.Timestamp(calibration_end, unit='D')}"
    )
    model, epoch_history = fit_model(
        args, builder, batcher, examples.labels, train_indices, calibration_indices, device
    )
    thresholds = choose_thresholds(
        examples.labels[calibration_indices],
        predict_probabilities(model, batcher, calibration_indices, args, device),
    )
    holdout_probabilities = predict_probabilities(model, batcher, holdout_indices, args, device)
    results = calculate_metrics(
        examples.labels[holdout_indices],
        holdout_probabilities,
        thresholds,
        examples.cutoff_days[holdout_indices],
    )
    unique_cutoffs = np.unique(examples.cutoff_days)
    results["split"] = {
        "train_weeks": int(np.sum(unique_cutoffs < train_end)),
        "calibration_weeks": int(
            np.sum((unique_cutoffs >= train_end) & (unique_cutoffs < calibration_end))
        ),
        "holdout_weeks": int(np.sum(unique_cutoffs >= calibration_end)),
        "train_end": str(pd.Timestamp(train_end, unit="D")),
        "holdout_start": str(pd.Timestamp(calibration_end, unit="D")),
        "train_examples": int(len(train_indices)),
        "calibration_examples": int(len(calibration_indices)),
        "holdout_examples": int(len(holdout_indices)),
    }
    results["device"] = str(device)
    results["temporal_model"] = BRANCH
    results["experiment"] = args.output_dir.name
    results.update(_settings(args, builder, epoch_history))

    torch.save(model.state_dict(), args.output_dir / MODEL_FILE)
    holdout_path = args.output_dir / "holdout_probabilities.csv"
    save_holdout_probabilities(examples, holdout_indices, holdout_probabilities, holdout_path)
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("macro_f1", "micro_f1", "score")}, indent=2))
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout probabilities to {holdout_path}")


def run_prediction(args: argparse.Namespace) -> None:
    context, builder = build_context(args)
    examples = load_or_build_db_examples(args, context)
    usable_indices = prediction_example_indices(examples.cutoff_days, args.cutoff)
    usable_cutoffs = examples.cutoff_days[usable_indices]
    train_end, _ = chronological_splits(usable_cutoffs, holdout=False)
    train_indices = usable_indices[usable_cutoffs < train_end]
    train_indices, unused_train_indices = subsample_train_indices(train_indices, args)
    calibration_indices = np.union1d(
        usable_indices[usable_cutoffs >= train_end], unused_train_indices
    )

    target_users = pd.read_csv(args.test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{args.test_input} contains duplicate UserIDs.")
    users = pd.read_parquet(
        args.db_dir / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"]
    )
    user_lookup = users.loc[users["is_cohort"]].set_index("github_user_id")["user_id"]
    mapped_ids = target_users["UserID"].map(user_lookup)
    if mapped_ids.isna().any():
        raise ValueError(
            f"{int(mapped_ids.isna().sum())} test UserID(s) are not in the cohort for this data release."
        )
    internal_ids = mapped_ids.to_numpy(dtype=np.int32)
    cutoff = model_cutoff(args.cutoff)
    inference_tokens = sample_tokens(
        context.graph,
        context.graph.global_ids("Users", internal_ids),
        np.full(len(internal_ids), cutoff, dtype=np.int64),
        args.num_neighbors,
        args.seed,
        args.context_mode,
        context.seed_sampler,
    )

    # Training examples and prediction users share one batcher: prediction
    # rows are appended after the examples.
    device = select_device(args.device)
    history = build_history(args, max(cutoff, int(examples.cutoff_seconds.max())))
    batcher = Batcher(
        builder,
        history,
        np.concatenate(
            [context_nodes(context, examples.tokens), context_nodes(context, inference_tokens)]
        ),
        np.concatenate([examples.user_ids, internal_ids]),
        np.concatenate([examples.cutoff_seconds, np.full(len(internal_ids), cutoff, np.int64)]),
        args,
        device,
    )
    model, epoch_history = fit_model(
        args, builder, batcher, examples.labels, train_indices, calibration_indices, device
    )
    thresholds = choose_thresholds(
        examples.labels[calibration_indices],
        predict_probabilities(model, batcher, calibration_indices, args, device),
    )
    probabilities = predict_probabilities(
        model, batcher, len(examples.user_ids) + np.arange(len(internal_ids)), args, device
    )

    output = pd.DataFrame({"UserID": target_users["UserID"]})
    for column, label in enumerate(LABELS):
        output[label] = (probabilities[:, column] >= thresholds[label]).astype(np.uint8)
    prediction_path = args.output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)
    probability_output = pd.DataFrame(
        {f"{label}_probability": probabilities[:, column] for column, label in enumerate(LABELS)}
    )
    probability_output.insert(0, "UserID", target_users["UserID"].to_numpy())
    probability_path = args.output_dir / "probabilities.csv"
    probability_output.to_csv(probability_path, index=False)
    torch.save(model.state_dict(), args.output_dir / MODEL_FILE)
    metadata = {
        "prediction_cutoff_utc": str(args.cutoff),
        "model_input_cutoff_utc": str(pd.Timestamp(cutoff, unit="s")),
        "training_start_cutoff_utc": str(args.start_cutoff),
        "training_weeks": int(np.unique(examples.cutoff_days[train_indices]).size),
        "calibration_weeks": int(np.unique(examples.cutoff_days[calibration_indices]).size),
        "training_examples": int(len(train_indices)),
        "calibration_examples": int(len(calibration_indices)),
        "labels": list(LABELS),
        "thresholds": thresholds,
        "device": str(device),
        "temporal_model": BRANCH,
        "experiment": args.output_dir.name,
        **_settings(args, builder, epoch_history),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(output):,} predictions to {prediction_path}")
    print(f"Saved unthresholded probabilities to {probability_path}")
    print(f"Device: {device}; calibrated thresholds: {json.dumps(thresholds)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    args.output_dir = create_run_output_dir(args.command, args.output_dir)
    print(f"Experiment output directory: {args.output_dir}")
    if args.command == "backtest":
        run_backtest(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
