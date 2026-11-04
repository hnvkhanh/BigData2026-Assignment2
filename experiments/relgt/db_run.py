"""Backtest or predict with RelGT applied directly to the relational database.

Used by ``python -m experiments.relgt.run ... --temporal-model relgt_db``.
Samples, labels, eligibility, the chronological split, threshold
calibration, and output files match the ``relgt`` mode, so holdout scores are
directly comparable; only the model input differs.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from .data import (
    LABELS,
    LOOKBACK_DAYS,
    labels_for_cutoff,
    load_entity_vocabulary,
    load_events,
    weekly_cutoffs,
)
from .database import (
    SECONDS_PER_DAY,
    RelationalDatabase,
    TableFeatures,
    TemporalGraph,
    build_temporal_graph,
    load_database,
    materialize_features,
    timestamp_seconds,
)
from .db_model import RelGT
from .db_sampler import TokenSets, batch_edge_index, sample_seed, sample_tokens
from .run import (
    _data_signature,
    calculate_metrics,
    choose_thresholds,
    chronological_splits,
    prediction_example_indices,
    select_device,
    subsample_train_indices,
)


SAMPLER_VERSION = 1
BRANCH = "relgt_db"
MODEL_FILE = "relgt_db_model.pt"
ARCHITECTURE = (
    "Relational Graph Transformer over the pkey/fkey database graph "
    "(five-element row tokens, local + codebook global attention)"
)


@dataclass
class DatabaseExamples:
    """Weekly (cohort user, cutoff) samples with RelGT token sets."""

    user_ids: np.ndarray  # int32 Users row index of the seed
    cutoff_days: np.ndarray  # int32 days since the epoch
    labels: np.ndarray  # uint8 [samples, labels]
    tokens: TokenSets

    @property
    def cutoff_seconds(self) -> np.ndarray:
        return self.cutoff_days.astype(np.int64) * SECONDS_PER_DAY


@dataclass
class DatabaseContext:
    graph: TemporalGraph
    features: dict[str, TableFeatures]
    num_users: int
    # Replaces ``sample_seed`` (see ``sample_tokens``); ``signature`` must then
    # describe it so cached token sets are not reused across samplers.
    seed_sampler: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] = sample_seed
    signature: dict[str, Any] = field(default_factory=dict)


def eligible_users(
    events: pd.DataFrame,
    cutoff: pd.Timestamp,
    user_ids: np.ndarray,
    lookback_days: int = LOOKBACK_DAYS,
) -> np.ndarray:
    """Users with at least five cutoff-safe actions in the lookback window.

    Matches ``snapshot_tokens(..., eligible_only=True)`` so both modes score
    the same user-weeks.
    """
    start = cutoff - pd.Timedelta(days=lookback_days)
    recent = events.loc[
        (events["available_at"] < cutoff)
        & (events["entity_available_at"] < cutoff)
        & (events["action_at"] < cutoff)
        & (events["action_at"] >= start)
        & events["user_id"].isin(user_ids)
    ]
    counts = recent.groupby("user_id").size()
    eligible = set(counts.index[counts >= 5].astype(int))
    return np.asarray([uid for uid in user_ids if int(uid) in eligible], dtype=np.int32)


def load_database_context(
    db_dir: Path, text_buckets: int, database: RelationalDatabase | None = None
) -> DatabaseContext:
    database = database or load_database(db_dir)
    features = materialize_features(database, text_buckets)
    graph = build_temporal_graph(database)
    print(
        f"Loaded {len(database.tables)} tables as a pkey/fkey graph: "
        f"{graph.num_nodes:,} rows, {len(graph.indices) // 2:,} key edges"
    )
    return DatabaseContext(graph, features, len(database.tables["Users"]))


def _signature(args: argparse.Namespace, context: DatabaseContext) -> dict[str, Any]:
    signature = _data_signature(args.db_dir, args.start_cutoff, LOOKBACK_DAYS, LOOKBACK_DAYS)
    signature.update(
        sampler_version=SAMPLER_VERSION,
        num_neighbors=args.num_neighbors,
        seed=args.seed,
        context_mode=args.context_mode,
        **context.signature,
    )
    return signature


def load_or_build_db_examples(
    args: argparse.Namespace, context: DatabaseContext
) -> DatabaseExamples:
    cache_path = args.output_dir / "cache" / f"relgt_db_examples_k{args.num_neighbors}_{args.context_mode}.npz"
    metadata_path = cache_path.with_suffix(".json")
    signature = _signature(args, context)
    if cache_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == signature:
            with np.load(cache_path, allow_pickle=False) as stored:
                examples = DatabaseExamples(
                    user_ids=stored["user_ids"],
                    cutoff_days=stored["cutoff_days"],
                    labels=stored["labels"],
                    tokens=TokenSets(
                        node_types=stored["node_types"],
                        node_rows=stored["node_rows"],
                        hops=stored["hops"],
                        relative_days=stored["relative_days"],
                        edge_offsets=stored["edge_offsets"],
                        edges=stored["edges"],
                    ),
                )
            print(f"Loaded cached RelGT token sets from {cache_path}")
            return examples

    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    cohort_ids = users.loc[users["is_cohort"], "user_id"].astype(np.int32).to_numpy()
    events = load_events(args.db_dir, cohort_ids.tolist(), load_entity_vocabulary(args.db_dir))
    user_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    cutoff_batches: list[np.ndarray] = []
    for cutoff in weekly_cutoffs(events, args.start_cutoff):
        user_ids = eligible_users(events, cutoff, cohort_ids)
        user_batches.append(user_ids)
        label_batches.append(labels_for_cutoff(events, cutoff, user_ids))
        cutoff_batches.append(
            np.full(len(user_ids), cutoff.value // pd.Timedelta(days=1).value, np.int32)
        )
    user_ids = np.concatenate(user_batches)
    cutoff_days = np.concatenate(cutoff_batches)
    print(f"Sampling {args.num_neighbors} tokens for {len(user_ids):,} user-weeks")
    tokens = sample_tokens(
        context.graph,
        context.graph.global_ids("Users", user_ids),
        cutoff_days.astype(np.int64) * SECONDS_PER_DAY,
        args.num_neighbors,
        args.seed,
        args.context_mode,
        context.seed_sampler,
    )
    examples = DatabaseExamples(user_ids, cutoff_days, np.concatenate(label_batches), tokens)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    np.savez(
        temporary_path,
        user_ids=examples.user_ids,
        cutoff_days=examples.cutoff_days,
        labels=examples.labels,
        node_types=tokens.node_types,
        node_rows=tokens.node_rows,
        hops=tokens.hops,
        relative_days=tokens.relative_days,
        edge_offsets=tokens.edge_offsets,
        edges=tokens.edges,
    )
    temporary_path.replace(cache_path)
    metadata_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    print(f"Cached RelGT token sets at {cache_path}")
    return examples


def build_db_model(args: argparse.Namespace, context: DatabaseContext) -> RelGT:
    return RelGT(
        tables=[context.features[name] for name in context.graph.table_names],
        num_seed_nodes=context.num_users,
        channels=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_centroids=args.num_centroids,
        dropout=args.dropout,
        conv_type=args.conv_type,
        ablate=args.ablate,
        text_buckets=args.text_buckets,
    )


def batch_inputs(
    tokens: TokenSets,
    user_ids: np.ndarray,
    cutoff_seconds: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    def tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device)

    return (
        tensor(tokens.node_types[indices], torch.long),
        tensor(tokens.node_rows[indices], torch.long),
        tensor(tokens.hops[indices], torch.long),
        tensor(tokens.relative_days[indices], torch.float32),
        tensor(cutoff_seconds[indices], torch.long),
        tensor(user_ids[indices], torch.long),
        tensor(batch_edge_index(tokens, indices), torch.long),
    )


def predict_db_probabilities(
    model: RelGT,
    tokens: TokenSets,
    user_ids: np.ndarray,
    cutoff_seconds: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    # The positional encoding draws random node inputs; a fixed seed makes
    # predictions reproducible without resetting the training RNG.
    model.eval()
    predictions: list[np.ndarray] = []
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.manual_seed(seed)
        for start in range(0, len(indices), batch_size):
            inputs = batch_inputs(
                tokens, user_ids, cutoff_seconds, indices[start : start + batch_size], device
            )
            predictions.append(torch.sigmoid(model(*inputs)).float().cpu().numpy())
    if not predictions:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def macro_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    scores = [
        average_precision_score(labels[:, column], probabilities[:, column])
        for column in range(labels.shape[1])
        if labels[:, column].any()
    ]
    return float(np.mean(scores)) if scores else 0.0


def fit_db_model(
    args: argparse.Namespace,
    context: DatabaseContext,
    examples: DatabaseExamples,
    train_indices: np.ndarray,
    selection_indices: np.ndarray,
    device: torch.device,
) -> tuple[RelGT, list[dict[str, float]]]:
    """Train with BCE and keep the epoch with the best selection AUPRC.

    Upstream keeps the epoch with the best validation metric (macro AUPRC for
    multilabel tasks); the selection rows are the calibration weeks, never the
    holdout.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build_db_model(args, context).to(device)
    print(f"RelGT parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.db_lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    cutoff_seconds = examples.cutoff_seconds
    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = np.random.permutation(train_indices)
        total_loss = 0.0
        steps = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            if len(batch) < 2:  # batch norm needs two rows
                continue
            inputs = batch_inputs(
                examples.tokens, examples.user_ids, cutoff_seconds, batch, device
            )
            labels = torch.as_tensor(examples.labels[batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
        probabilities = predict_db_probabilities(
            model,
            examples.tokens,
            examples.user_ids,
            cutoff_seconds,
            selection_indices,
            device,
            args.batch_size,
            args.seed,
        )
        score = macro_average_precision(examples.labels[selection_indices], probabilities)
        history.append(
            {"epoch": epoch, "train_bce": total_loss / max(steps, 1), "selection_auprc": score}
        )
        print(
            f"Epoch {epoch}/{args.epochs}: BCE={total_loss / max(steps, 1):.5f} "
            f"calibration macro AUPRC={score:.4f}"
        )
        if score >= best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _settings(args: argparse.Namespace, history: list[dict[str, float]]) -> dict[str, Any]:
    best = max(history, key=lambda row: row["selection_auprc"]) if history else None
    return {
        "architecture": ARCHITECTURE,
        "official_relgt": "https://github.com/snap-stanford/relgt",
        "num_neighbors": args.num_neighbors,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "num_centroids": args.num_centroids,
        "conv_type": args.conv_type,
        "ablate": args.ablate,
        "context_mode": args.context_mode,
        "text_buckets": args.text_buckets,
        "learning_rate": args.db_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "selected_epoch": best["epoch"] if best else None,
        "epoch_history": history,
        "train_fraction": getattr(args, "train_fraction", 1.0),
        "discard_unused_train": getattr(args, "discard_unused_train", False),
    }


def save_db_holdout_probabilities(
    examples: DatabaseExamples,
    indices: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Write holdout labels and probabilities in the shared CSV schema."""
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
    output.to_csv(output_path, index=False)


def run_db_backtest(args: argparse.Namespace, context: DatabaseContext | None = None) -> None:
    context = context or load_database_context(args.db_dir, args.text_buckets)
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
    print(
        f"Device: {device}; chronological boundaries: train < "
        f"{pd.Timestamp(train_end, unit='D')}, holdout >= "
        f"{pd.Timestamp(calibration_end, unit='D')}"
    )
    model, history = fit_db_model(
        args, context, examples, train_indices, calibration_indices, device
    )

    def probabilities_for(indices: np.ndarray) -> np.ndarray:
        return predict_db_probabilities(
            model,
            examples.tokens,
            examples.user_ids,
            examples.cutoff_seconds,
            indices,
            device,
            args.batch_size,
            args.seed,
        )

    thresholds = choose_thresholds(
        examples.labels[calibration_indices], probabilities_for(calibration_indices)
    )
    holdout_probabilities = probabilities_for(holdout_indices)
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
        "holdout_start": str(pd.Timestamp(calibration_end, unit="D")),
    }
    results["device"] = str(device)
    results["temporal_model"] = args.temporal_model
    results["experiment"] = args.output_dir.name
    results.update(_settings(args, history))
    results.update(context.signature)

    torch.save(model.state_dict(), args.output_dir / MODEL_FILE)
    holdout_path = args.output_dir / "holdout_probabilities.csv"
    save_db_holdout_probabilities(examples, holdout_indices, holdout_probabilities, holdout_path)
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("macro_f1", "micro_f1", "score")}, indent=2))
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout probabilities to {holdout_path}")


def run_db_prediction(args: argparse.Namespace, context: DatabaseContext | None = None) -> None:
    context = context or load_database_context(args.db_dir, args.text_buckets)
    examples = load_or_build_db_examples(args, context)
    usable_indices = prediction_example_indices(examples.cutoff_days, args.cutoff)
    usable_cutoffs = examples.cutoff_days[usable_indices]
    train_end, _ = chronological_splits(usable_cutoffs, holdout=False)
    train_indices = usable_indices[usable_cutoffs < train_end]
    train_indices, unused_train_indices = subsample_train_indices(train_indices, args)
    calibration_indices = np.union1d(
        usable_indices[usable_cutoffs >= train_end], unused_train_indices
    )
    if not len(calibration_indices):
        raise ValueError("No chronological calibration examples are available.")

    device = select_device(args.device)
    model, history = fit_db_model(
        args, context, examples, train_indices, calibration_indices, device
    )
    calibration_probabilities = predict_db_probabilities(
        model,
        examples.tokens,
        examples.user_ids,
        examples.cutoff_seconds,
        calibration_indices,
        device,
        args.batch_size,
        args.seed,
    )
    thresholds = choose_thresholds(
        examples.labels[calibration_indices], calibration_probabilities
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
            f"{int(mapped_ids.isna().sum())} test UserID(s) are not in the cohort "
            "for this data release."
        )
    internal_ids = mapped_ids.to_numpy(dtype=np.int32)
    cutoff = timestamp_seconds(args.cutoff)
    cutoff_seconds = np.full(len(internal_ids), cutoff, dtype=np.int64)
    inference_tokens = sample_tokens(
        context.graph,
        context.graph.global_ids("Users", internal_ids),
        cutoff_seconds,
        args.num_neighbors,
        args.seed,
        args.context_mode,
        context.seed_sampler,
    )
    probabilities = predict_db_probabilities(
        model,
        inference_tokens,
        internal_ids,
        cutoff_seconds,
        np.arange(len(internal_ids)),
        device,
        args.batch_size,
        args.seed,
    )

    output = pd.DataFrame({"UserID": target_users["UserID"]})
    for column, label in enumerate(LABELS):
        output[label] = (probabilities[:, column] >= thresholds[label]).astype(np.uint8)
    prediction_path = args.output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)
    probability_output = pd.DataFrame(
        {
            f"{label}_probability": probabilities[:, column]
            for column, label in enumerate(LABELS)
        }
    )
    probability_output.insert(0, "UserID", target_users["UserID"].to_numpy())
    probability_path = args.output_dir / "probabilities.csv"
    probability_output.to_csv(probability_path, index=False)
    torch.save(model.state_dict(), args.output_dir / MODEL_FILE)

    metadata = {
        "prediction_cutoff_utc": str(args.cutoff),
        "training_start_cutoff_utc": str(args.start_cutoff),
        "training_weeks": int(np.unique(examples.cutoff_days[train_indices]).size),
        "calibration_weeks": int(np.unique(examples.cutoff_days[calibration_indices]).size),
        "labels": list(LABELS),
        "thresholds": thresholds,
        "device": str(device),
        "temporal_model": args.temporal_model,
        "experiment": args.output_dir.name,
        **_settings(args, history),
        **context.signature,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(output):,} predictions to {prediction_path}")
    print(f"Saved unthresholded probabilities to {probability_path}")
    print(f"Device: {device}; calibrated thresholds: {json.dumps(thresholds)}")
