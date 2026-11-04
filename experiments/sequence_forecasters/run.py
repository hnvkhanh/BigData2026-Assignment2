"""Backtest or predict with the DM2026-Final sequence forecasters and their blend.

Ports the LSTM, Informer and FEDformer of https://github.com/hnvkhanh/DM2026-Final
to per-user daily activity sequences. Each branch is trained independently on
the same examples, eligible users, weekly cutoffs and chronological split as
``experiments.relgt.run``. The ensemble is a weighted average of branch
probabilities (DM2026 ``ensemble.py``); weights on a simplex grid and per-label
thresholds are chosen together on the calibration week only, unless
``--fusion-weights`` fixes them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.gnn_no_summer.run import (
    _week_strings,
    create_run_output_dir,
    save_predictions,
)
from experiments.relgt.data import HORIZON_DAYS, LABELS, load_entity_vocabulary, load_events
from experiments.relgt.run import (
    DEFAULT_DB_DIR,
    DEFAULT_PREDICTION_CUTOFF,
    DEFAULT_START_CUTOFF,
    DEFAULT_TEST_INPUT,
    calculate_metrics,
    choose_thresholds,
    chronological_splits,
    load_or_build_examples,
    prediction_example_indices,
    select_device,
)
from experiments.sequence_forecasters.features import (
    AUX_COLUMNS,
    SEQUENCE_CHANNELS,
    ChannelScaler,
    build_daily_sequences,
    calendar_features,
    example_cutoff_ns,
)
from experiments.sequence_forecasters.model import (
    FEDformerForecaster,
    InformerForecaster,
    LSTMForecaster,
)

EXPERIMENT = "sequence_forecasters"
BRANCHES = ("lstm", "informer", "fedformer")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
# DM2026 per-model optimiser defaults (lstm_v2 / informer_updated / fedformer_updated).
BRANCH_DEFAULTS = {
    "lstm": {"epochs": 6, "learning_rate": 1e-3, "weight_decay": 1e-4},
    "informer": {"epochs": 8, "learning_rate": 1e-4, "weight_decay": 1e-3},
    "fedformer": {"epochs": 10, "learning_rate": 2e-4, "weight_decay": 1e-4},
}


def _build_model(branch: str, known_user_ids: np.ndarray | None, args) -> nn.Module:
    user_emb_dim = args.user_emb_dim
    if branch == "lstm":
        return LSTMForecaster(
            input_size=len(SEQUENCE_CHANNELS),
            aux_size=len(AUX_COLUMNS),
            hidden_size=args.hidden_dim,
            num_layers=args.lstm_layers,
            dropout=args.dropout,
            user_emb_dim=user_emb_dim,
            known_user_ids=known_user_ids,
        )
    if branch == "informer":
        return InformerForecaster(
            input_size=len(SEQUENCE_CHANNELS),
            window_days=args.window_days,
            label_days=args.label_days,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_ff=args.d_ff,
            factor=args.factor,
            attn=args.attn,
            distil=not args.no_distil,
            dropout=args.dropout,
            user_emb_dim=user_emb_dim,
            known_user_ids=known_user_ids,
        )
    if branch == "fedformer":
        return FEDformerForecaster(
            input_size=len(SEQUENCE_CHANNELS),
            window_days=args.window_days,
            d_model=args.d_model,
            d_ff=args.d_ff,
            e_layers=args.e_layers,
            modes=args.fed_modes,
            mode_select=args.mode_select,
            moving_avg=tuple(args.moving_avg),
            dropout=args.dropout,
            user_emb_dim=user_emb_dim,
            known_user_ids=known_user_ids,
        )
    raise ValueError(f"Unknown branch: {branch}")


def _branch_optimiser_settings(branch: str, args) -> dict[str, float]:
    settings = dict(BRANCH_DEFAULTS[branch])
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    override = getattr(args, f"{branch}_lr")
    if override is not None:
        settings["learning_rate"] = override
    return settings


class SequenceInputs:
    """Row-aligned model inputs: sequences, cutoff calendar features and user IDs."""

    def __init__(self, sequences: np.ndarray, aux: np.ndarray, user_ids: np.ndarray) -> None:
        if not len(sequences) == len(aux) == len(user_ids):
            raise ValueError("Sequences, aux features and user IDs must have equal rows.")
        if sequences.ndim != 3 or sequences.shape[2] != len(SEQUENCE_CHANNELS):
            raise ValueError(f"Sequences must have shape (n, days, {len(SEQUENCE_CHANNELS)}).")
        self.sequences, self.aux, self.user_ids = sequences, aux, user_ids

    def batch(self, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(self.sequences[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.aux[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.user_ids[indices], dtype=torch.long, device=device),
        )


def predict_probabilities(
    model: nn.Module,
    inputs: SequenceInputs,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = inputs.batch(indices[start : start + batch_size], device)
            predictions.append(torch.sigmoid(model(*batch)).float().cpu().numpy())
    if not predictions:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def _mean_bce(model, inputs, labels, indices, device, batch_size) -> float:
    probabilities = predict_probabilities(model, inputs, indices, device, batch_size)
    probabilities = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    targets = labels[indices]
    return float(
        -np.mean(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities))
    )


def fit_branch(
    branch: str,
    inputs: SequenceInputs,
    labels: np.ndarray,
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    device: torch.device,
    args,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one branch; keep the epoch with the lowest calibration BCE (as DM2026
    keeps its best validation epoch)."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    settings = _branch_optimiser_settings(branch, args)
    known = np.unique(inputs.user_ids[train_indices]) if args.user_emb_dim else None
    model = _build_model(branch, known, args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()
    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        shuffled = np.random.permutation(train_indices)
        total, batches = 0.0, 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            targets = torch.as_tensor(labels[batch_indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*inputs.batch(batch_indices, device)), targets)
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        calibration_bce = _mean_bce(
            model, inputs, labels, calibration_indices, device, args.batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "train_bce": total / max(batches, 1),
                "calibration_bce": calibration_bce,
            }
        )
        print(
            f"[{branch}] epoch {epoch}/{settings['epochs']}: train BCE "
            f"{total / max(batches, 1):.5f}, calibration BCE {calibration_bce:.5f}"
        )
        if best_state is None or calibration_bce < best_loss:
            best_loss, best_epoch = calibration_bce, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    record = {
        **settings,
        "selected_epoch": best_epoch,
        "epoch_selection": "lowest calibration BCE",
        "history": history,
        "known_users": 0 if known is None else int(len(known)),
    }
    return model, record


def simplex_grid(branches: tuple[str, ...], step: float):
    """Every weight assignment on a ``step`` grid that sums to 1."""
    units = int(round(1.0 / step))
    for parts in itertools.product(range(units + 1), repeat=len(branches) - 1):
        if sum(parts) <= units:
            yield dict(zip(branches, [p / units for p in (*parts, units - sum(parts))]))


def fuse(probabilities: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if set(weights) != set(probabilities):
        raise ValueError("Fusion weights must name exactly the trained branches.")
    if not np.isclose(sum(weights.values()), 1.0) or min(weights.values()) < 0:
        raise ValueError("Fusion weights must be non-negative and sum to 1.")
    return sum(weights[name] * values for name, values in probabilities.items())


def choose_weights(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    step: float,
    manual: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Return ``(weights, thresholds, calibration score)`` chosen on calibration only."""
    candidates = [manual] if manual is not None else simplex_grid(tuple(probabilities), step)
    best = None
    for weights in candidates:
        fused = fuse(probabilities, weights)
        thresholds = choose_thresholds(labels, fused)
        score = calculate_metrics(labels, fused, thresholds)["score"]
        if best is None or score > best[2]:
            best = (weights, thresholds, score)
    return best


def _prepare_training_inputs(args, examples, train_indices):
    """Build and scale cutoff-safe sequences for every example (scaler fit on train)."""
    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    cohort_ids = users.loc[users["is_cohort"], "user_id"].astype(int).tolist()
    events = load_events(args.db_dir, cohort_ids, load_entity_vocabulary(args.db_dir))
    cutoff_ns = example_cutoff_ns(examples.cutoff_days, args.start_cutoff)
    print(f"Building {args.window_days}-day sequences for {len(cutoff_ns):,} examples")
    sequences = build_daily_sequences(events, examples.tokens.user_ids, cutoff_ns, args.window_days)
    scaler = ChannelScaler.fit(sequences[train_indices])
    inputs = SequenceInputs(
        scaler.transform_(sequences), calendar_features(cutoff_ns), examples.tokens.user_ids
    )
    return inputs, scaler, events


def _train_all(args, inputs, labels, train_indices, calibration_indices, device):
    models, records, calibration = {}, {}, {}
    for branch in args.models:
        print(f"Training {branch} on {len(train_indices):,} examples")
        models[branch], records[branch] = fit_branch(
            branch, inputs, labels, train_indices, calibration_indices, device, args
        )
        calibration[branch] = predict_probabilities(
            models[branch], inputs, calibration_indices, device, args.batch_size
        )
    weights, thresholds, calibration_score = choose_weights(
        labels[calibration_indices], calibration, args.fusion_step, args.fusion_weights
    )
    fusion = {
        "method": (
            "manual probability averaging"
            if args.fusion_weights is not None
            else f"calibration-selected probability averaging (simplex grid step {args.fusion_step})"
        ),
        "weights": weights,
        "calibration_score": calibration_score,
    }
    return models, records, calibration, weights, thresholds, fusion


def _settings(args) -> dict[str, Any]:
    keys = (
        "models",
        "window_days",
        "batch_size",
        "dropout",
        "grad_clip",
        "user_emb_dim",
        "hidden_dim",
        "lstm_layers",
        "d_model",
        "d_ff",
        "n_heads",
        "e_layers",
        "d_layers",
        "label_days",
        "factor",
        "attn",
        "no_distil",
        "fed_modes",
        "mode_select",
        "moving_avg",
        "seed",
        "eligibility_days",
    )
    return {key: getattr(args, key) for key in keys}


def _feature_record(args, scaler: ChannelScaler) -> dict[str, Any]:
    return {
        "sequence_channels": list(SEQUENCE_CHANNELS),
        "aux_columns": list(AUX_COLUMNS),
        "window_days": args.window_days,
        "bins": "24-hour bins ending exactly at the cutoff, oldest first",
        "availability": "action_at, available_at and entity_available_at all < cutoff",
        "scaling": scaler.to_dict(),
    }


def _save_models(models, output_dir: Path) -> None:
    for branch, model in models.items():
        torch.save(model.state_dict(), output_dir / f"{branch}_model.pt")


def run_backtest(args: argparse.Namespace) -> None:
    examples = load_or_build_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        eligibility_lookback_days=args.eligibility_days,
    )
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    days = examples.cutoff_days
    train_indices = np.flatnonzero(days < train_end)
    calibration_indices = np.flatnonzero((days >= train_end) & (days < calibration_end))
    holdout_indices = np.flatnonzero(days >= calibration_end)
    if not (len(train_indices) and len(calibration_indices) and len(holdout_indices)):
        raise ValueError("The chronological split produced an empty partition.")

    inputs, scaler, _ = _prepare_training_inputs(args, examples, train_indices)
    device = select_device(args.device)
    print(f"Device: {device}; holdout >= {pd.Timestamp(calibration_end, unit='D')}")
    models, records, calibration, weights, thresholds, fusion = _train_all(
        args, inputs, examples.labels, train_indices, calibration_indices, device
    )

    holdout = {
        branch: predict_probabilities(model, inputs, holdout_indices, device, args.batch_size)
        for branch, model in models.items()
    }
    holdout["ensemble"] = fuse({b: holdout[b] for b in models}, weights)
    calibration_labels = examples.labels[calibration_indices]
    holdout_labels = examples.labels[holdout_indices]
    holdout_cutoffs = days[holdout_indices]
    metrics = calculate_metrics(holdout_labels, holdout["ensemble"], thresholds, holdout_cutoffs)
    # Each branch alone, with its own calibration thresholds, for comparison.
    branch_metrics = {
        branch: calculate_metrics(
            holdout_labels,
            holdout[branch],
            choose_thresholds(calibration_labels, calibration[branch]),
            holdout_cutoffs,
        )
        for branch in models
    }
    branch_metrics["ensemble"] = metrics

    _save_models(models, args.output_dir)
    columns = {
        "user_id": examples.tokens.user_ids[holdout_indices],
        "cutoff": pd.to_datetime(holdout_cutoffs, unit="D"),
        **{f"actual_{label}": holdout_labels[:, i] for i, label in enumerate(LABELS)},
    }
    for branch, values in holdout.items():
        for i, label in enumerate(LABELS):
            columns[f"{branch}_{label}"] = values[:, i]
    probability_path = args.output_dir / "holdout_probabilities.csv"
    pd.DataFrame(columns).to_csv(probability_path, index=False)

    results = {
        "experiment": EXPERIMENT,
        "architecture": "DM2026-Final LSTM / Informer / FEDformer over daily activity sequences",
        "ensemble": "independently trained branches, weighted probability average",
        "labels": list(LABELS),
        "horizon_days": HORIZON_DAYS,
        "features": _feature_record(args, scaler),
        "branches": records,
        "fusion": fusion,
        "train_examples": int(len(train_indices)),
        "train_weeks": _week_strings(days[train_indices]),
        "calibration_examples": int(len(calibration_indices)),
        "calibration_weeks": _week_strings(days[calibration_indices]),
        "holdout_examples": int(len(holdout_indices)),
        "holdout_start": str(pd.Timestamp(calibration_end, unit="D")),
        "device": str(device),
        "settings": _settings(args),
        "threshold_selection": "per label on calibration only, jointly with the fusion weights",
        "thresholds": thresholds,
        "holdout_metrics": metrics,
        "branch_holdout": branch_metrics,
    }
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for branch, values in branch_metrics.items():
        print(f"Holdout score {branch}: {values['score']:.4f}")
    print(f"Fusion weights {json.dumps(weights)}")
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout probabilities to {probability_path}")


def _target_users(args, cohort_users: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``(test_input rows, internal user IDs)`` in --test-input order."""
    target_users = pd.read_csv(args.test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{args.test_input} contains duplicate UserIDs.")
    mapped = target_users["UserID"].map(cohort_users.set_index("github_user_id")["user_id"])
    if mapped.isna().any():
        raise ValueError(
            f"{int(mapped.isna().sum())} test UserID(s) are not in the cohort for this data release."
        )
    return target_users, mapped.to_numpy(dtype=np.int64)


def run_prediction(args: argparse.Namespace) -> None:
    examples = load_or_build_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        eligibility_lookback_days=args.eligibility_days,
    )
    usable = prediction_example_indices(examples.cutoff_days, args.cutoff)
    train_end, _ = chronological_splits(examples.cutoff_days[usable], holdout=False)
    train_indices = usable[examples.cutoff_days[usable] < train_end]
    calibration_indices = usable[examples.cutoff_days[usable] >= train_end]

    inputs, scaler, events = _prepare_training_inputs(args, examples, train_indices)
    device = select_device(args.device)
    models, records, _, weights, thresholds, fusion = _train_all(
        args, inputs, examples.labels, train_indices, calibration_indices, device
    )

    users = pd.read_parquet(
        args.db_dir / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"]
    )
    target_users, internal_ids = _target_users(args, users.loc[users["is_cohort"]])
    cutoff_ns = np.full(len(internal_ids), args.cutoff.value, dtype=np.int64)
    test_inputs = SequenceInputs(
        scaler.transform_(build_daily_sequences(events, internal_ids, cutoff_ns, args.window_days)),
        calendar_features(cutoff_ns),
        internal_ids,
    )
    rows = np.arange(len(internal_ids))
    branch_probabilities = {
        branch: predict_probabilities(model, test_inputs, rows, device, args.batch_size)
        for branch, model in models.items()
    }
    probabilities = fuse(branch_probabilities, weights)
    save_predictions(args, target_users, probabilities, thresholds, branch_probabilities)
    _save_models(models, args.output_dir)

    metadata = {
        "experiment": EXPERIMENT,
        "prediction_cutoff_utc": str(args.cutoff),
        "training_start_cutoff_utc": str(args.start_cutoff),
        "labels": list(LABELS),
        "features": _feature_record(args, scaler),
        "branches": records,
        "fusion": fusion,
        "training_examples": int(len(train_indices)),
        "training_weeks": _week_strings(examples.cutoff_days[train_indices]),
        "calibration_weeks": _week_strings(examples.cutoff_days[calibration_indices]),
        "thresholds": thresholds,
        "device": str(device),
        "settings": _settings(args),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Device: {device}; fusion weights {json.dumps(weights)}")


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = set(models) - set(BRANCHES)
    if not models or unknown:
        raise argparse.ArgumentTypeError(
            f"--models must be a comma-separated subset of {','.join(BRANCHES)}."
        )
    return models


def parse_fusion_weights(value: str) -> dict[str, float]:
    """Parse ``lstm=0.4,informer=0.2,fedformer=0.4``."""
    try:
        weights = {
            name.strip(): float(weight)
            for name, weight in (item.split("=") for item in value.split(","))
        }
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--fusion-weights must look like lstm=0.4,informer=0.2,fedformer=0.4."
        ) from error
    if any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise argparse.ArgumentTypeError("--fusion-weights must be non-negative.")
    if not np.isclose(sum(weights.values()), 1.0):
        raise argparse.ArgumentTypeError("--fusion-weights must sum to 1.")
    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest_parser = subparsers.add_parser("backtest")
    predict_parser = subparsers.add_parser("predict")
    for sub in (backtest_parser, predict_parser):
        sub.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
        sub.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="output folder (default: a new timestamped folder per run)",
        )
        sub.add_argument(
            "--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF)
        )
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        sub.add_argument(
            "--eligibility-days",
            type=int,
            default=None,
            help="users need 5 events in this many days before each cutoff (default: 14)",
        )
        sub.add_argument(
            "--models",
            type=parse_models,
            default=BRANCHES,
            help="comma-separated branches to train (default: lstm,informer,fedformer)",
        )
        sub.add_argument(
            "--fusion-weights",
            type=parse_fusion_weights,
            default=None,
            help="fix blend weights, e.g. lstm=0.4,informer=0.2,fedformer=0.4 "
            "(default: chosen on calibration)",
        )
        sub.add_argument(
            "--fusion-step",
            type=float,
            default=0.1,
            help="grid step for calibration-selected weights",
        )
        sub.add_argument("--window-days", type=int, default=91)
        sub.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="epochs for every branch (default: lstm 6, informer 8, fedformer 10)",
        )
        for branch in BRANCHES:
            sub.add_argument(
                f"--{branch}-lr",
                type=float,
                default=None,
                help=f"default {BRANCH_DEFAULTS[branch]['learning_rate']}",
            )
        sub.add_argument("--batch-size", type=int, default=256)
        sub.add_argument("--dropout", type=float, default=0.2)
        sub.add_argument("--grad-clip", type=float, default=1.0)
        sub.add_argument(
            "--user-emb-dim", type=int, default=32, help="per-user embedding size; 0 disables it"
        )
        sub.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden size")
        sub.add_argument("--lstm-layers", type=int, default=2)
        sub.add_argument("--d-model", type=int, default=128)
        sub.add_argument("--d-ff", type=int, default=256)
        sub.add_argument("--n-heads", type=int, default=4)
        sub.add_argument("--e-layers", type=int, default=2)
        sub.add_argument("--d-layers", type=int, default=1)
        sub.add_argument(
            "--label-days", type=int, default=45, help="Informer decoder start-token length"
        )
        sub.add_argument("--factor", type=int, default=5, help="ProbSparse factor")
        sub.add_argument("--attn", choices=["prob", "full"], default="prob")
        sub.add_argument("--no-distil", action="store_true")
        sub.add_argument("--fed-modes", type=int, default=16)
        sub.add_argument("--mode-select", choices=["low", "random"], default="low")
        sub.add_argument("--moving-avg", type=int, nargs="+", default=[7, 14, 28])
        sub.add_argument("--seed", type=int, default=42)
    predict_parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    predict_parser.add_argument(
        "--cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF),
        help="UTC time at which to make the prediction",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "window_days",
        "hidden_dim",
        "lstm_layers",
        "d_model",
        "d_ff",
        "n_heads",
        "e_layers",
        "d_layers",
        "label_days",
        "factor",
        "fed_modes",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive.")
    for branch in BRANCHES:
        lr = getattr(args, f"{branch}_lr")
        if lr is not None and lr <= 0:
            parser.error(f"--{branch}-lr must be positive.")
    if args.user_emb_dim < 0:
        parser.error("--user-emb-dim must be non-negative.")
    if args.hidden_dim < 2:
        parser.error("--hidden-dim must be at least 2.")
    if args.d_model % args.n_heads:
        parser.error("--d-model must be divisible by --n-heads.")
    if args.label_days > args.window_days:
        parser.error("--label-days must not exceed --window-days.")
    if any(kernel < 1 for kernel in args.moving_avg):
        parser.error("--moving-avg kernels must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1).")
    if args.grad_clip < 0:
        parser.error("--grad-clip must be non-negative (0 disables clipping).")
    if args.eligibility_days is not None and args.eligibility_days < 14:
        parser.error("--eligibility-days must be at least the 14-day graph lookback.")
    units = 1.0 / args.fusion_step if args.fusion_step > 0 else 0.0
    if not 0.0 < args.fusion_step <= 1.0 or not np.isclose(units, round(units)):
        parser.error("--fusion-step must divide 1 evenly, e.g. 0.1, 0.25 or 0.5.")
    if args.fusion_weights is not None and set(args.fusion_weights) != set(args.models):
        parser.error("--fusion-weights must name exactly the branches in --models.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    args.output_dir = create_run_output_dir(args.command, args.output_dir, DEFAULT_OUTPUT_ROOT)
    if args.command == "backtest":
        run_backtest(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
