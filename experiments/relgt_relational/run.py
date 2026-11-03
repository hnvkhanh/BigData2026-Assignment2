"""RelGT / GNN with relational user features injected into the root token.

The graph samples, labels, eligibility, split, threshold calibration and
scoring are those of ``experiments.relgt.run`` (14-day snapshots). Each sample
additionally gets the cutoff-safe relational feature vector from
``experiments.relational_lightgbm.features`` (ownership, authorship, reviewer
history, assignments, push->PR sequences, mentions, threads, review cycle and
8-week habits). The vector is log1p-scaled, standardised with training-week
statistics, projected to the hidden size, and added to the user's root token,
so attention (RelGT) or message passing (GNN) is conditioned on it.
``--groups none`` trains the unchanged graph model as a control.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from experiments.lightgbm_baseline.run import load_histories
from experiments.relational_lightgbm.features import (
    FEATURE_GROUPS,
    feature_names,
    load_relational_tables,
    relational_features,
)
from experiments.relational_lightgbm.run import FEATURE_VERSION, parse_groups
from experiments.relgt.data import (
    LABELS,
    LOOKBACK_DAYS,
    GraphTokens,
    TrainingExamples,
    load_entity_vocabulary,
    load_events,
    snapshot_tokens,
)
from experiments.relgt.gnn_model import RelationalGNN
from experiments.relgt.model import RelGTInspired
from experiments.relgt.run import (
    _batch_inputs,
    _data_signature,
    calculate_metrics,
    choose_thresholds,
    chronological_splits,
    load_or_build_examples,
    prediction_example_indices,
    select_device,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_TEST_INPUT = ROOT / "assignment2" / "evaluation" / "test_input.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_START_CUTOFF = "2026-03-11 16:00:00"
DEFAULT_PREDICTION_CUTOFF = "2026-11-11 16:00:00"
ARCHITECTURES = ("relgt", "gnn")
DAY_NS = pd.Timedelta(days=1).value


@dataclass(frozen=True)
class RootFeatureScaler:
    """log1p followed by standardisation with statistics from training rows."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, raw: np.ndarray) -> "RootFeatureScaler":
        logged = np.log1p(np.clip(raw, 0, None))
        std = logged.std(axis=0)
        return cls(logged.mean(axis=0), np.where(std > 1e-6, std, 1.0))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.clip(raw, 0, None)) - self.mean) / self.std).astype(np.float32)


def cutoff_timestamps(cutoff_days: np.ndarray, start_cutoff: pd.Timestamp) -> dict[int, pd.Timestamp]:
    """Recover each sample's exact cutoff; ``cutoff_days`` keeps only the day."""
    time_of_day = start_cutoff - start_cutoff.floor("D")
    return {int(day): pd.Timestamp(int(day) * DAY_NS) + time_of_day for day in np.unique(cutoff_days)}


def compute_root_features(
    db_dir: Path,
    user_ids: np.ndarray,
    cutoff_days: np.ndarray,
    start_cutoff: pd.Timestamp,
    groups: list[str],
) -> np.ndarray:
    """Raw relational features, shape ``[samples, features]`` (float32), aligned to samples."""
    names = feature_names(groups)
    features = np.zeros((len(user_ids), len(names)), dtype=np.float32)
    if not names or not len(user_ids):
        return features
    users = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    cohort_ids = users.loc[users["is_cohort"], "user_id"].astype(int).tolist()
    histories = load_histories(db_dir, cohort_ids)
    tables = load_relational_tables(db_dir)
    cutoffs = cutoff_timestamps(cutoff_days, start_cutoff)
    for index, (day, cutoff) in enumerate(sorted(cutoffs.items()), start=1):
        rows = np.flatnonzero(cutoff_days == day)
        sample_users = user_ids[rows].astype(np.int64)
        frame = relational_features(cutoff, tables, histories, np.unique(sample_users).tolist(), groups)
        features[rows] = frame.set_index("user_id").loc[sample_users, names].to_numpy(dtype=np.float32)
        print(f"Built relational root features {index}/{len(cutoffs)}: {cutoff}")
    return features


def load_or_build_root_features(
    db_dir: Path,
    output_dir: Path,
    examples: TrainingExamples,
    start_cutoff: pd.Timestamp,
    groups: list[str],
) -> np.ndarray:
    """Root features for every training example, cached in the run folder."""
    cache_path = output_dir / "cache" / "root_features.npz"
    metadata_path = cache_path.with_suffix(".json")
    signature = {
        **_data_signature(db_dir, start_cutoff, LOOKBACK_DAYS, LOOKBACK_DAYS),
        "feature_version": FEATURE_VERSION,
        "groups": groups,
        "examples": int(len(examples.labels)),
    }
    if cache_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == signature:
            with np.load(cache_path, allow_pickle=False) as stored:
                print(f"Loaded cached root features from {cache_path}")
                return stored["features"]
        print(f"Ignoring stale root-feature cache {cache_path}")
    features = compute_root_features(
        db_dir, examples.tokens.user_ids, examples.cutoff_days, start_cutoff, groups
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features)
    metadata_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    return features


def build_model(
    args: argparse.Namespace, num_entities: int, num_root_features: int
) -> RelGTInspired | RelationalGNN:
    if args.architecture == "gnn":
        return RelationalGNN(
            num_entities=num_entities,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_root_features=num_root_features,
        )
    return RelGTInspired(
        num_entities=num_entities,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_local_layers=args.num_layers,
        dropout=args.dropout,
        num_root_features=num_root_features,
    )


def _logits(
    model: torch.nn.Module,
    tokens: GraphTokens,
    root_features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    inputs = _batch_inputs(tokens, indices, device)
    if root_features.shape[1] == 0:
        return model(*inputs)
    features = torch.as_tensor(root_features[indices], dtype=torch.float32, device=device)
    return model(*inputs, root_features=features)


def fit_model(
    examples: TrainingExamples,
    root_features: np.ndarray,
    train_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    calibration_indices: np.ndarray | None = None,
    model_factory: Callable[[], torch.nn.Module] | None = None,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    """Train; with ``--select-epoch`` keep the epoch with the best calibration-week score.

    ``model_factory`` builds the network after seeding (default: ``build_model``).
    Returns the model and the per-epoch history (training BCE and, when
    selecting, the calibration score with thresholds chosen on calibration).
    """
    select = args.select_epoch and calibration_indices is not None and len(calibration_indices)
    best_score, best_state = -np.inf, None
    history: list[dict[str, float]] = []
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = (
        model_factory() if model_factory is not None
        else build_model(args, examples.num_entities, root_features.shape[1])
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train()
        shuffled = np.random.permutation(train_indices)
        total_loss, batches = 0.0, 0
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            targets = torch.as_tensor(examples.labels[batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_logits(model, examples.tokens, root_features, batch, device), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        record = {"epoch": epoch + 1, "train_bce": total_loss / max(batches, 1)}
        if select:
            probabilities = predict_probabilities(
                model, examples.tokens, root_features, calibration_indices, device, args.batch_size
            )
            labels = examples.labels[calibration_indices]
            record["calibration_score"] = calculate_metrics(
                labels,
                probabilities,
                choose_thresholds(labels, probabilities),
                examples.cutoff_days[calibration_indices],
            )["score"]
            if record["calibration_score"] > best_score:
                best_score = record["calibration_score"]
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append(record)
        print(f"Epoch {epoch + 1}/{args.epochs}: " + ", ".join(
            f"{key}={value:.5f}" for key, value in record.items() if key != "epoch"
        ))
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Kept the epoch with the best calibration score ({best_score:.5f})")
    return model, history


def predict_probabilities(
    model: torch.nn.Module,
    tokens: GraphTokens,
    root_features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            outputs.append(torch.sigmoid(_logits(model, tokens, root_features, batch, device)).cpu().numpy())
    if not outputs:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _output_dir(command: str, requested: Path | None) -> Path:
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DEFAULT_OUTPUT_ROOT / f"{command}_{timestamp}_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output_dir}")
    return output_dir


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "architecture": args.architecture,
        "relational_groups": args.groups,
        "relational_features": feature_names(args.groups),
        "feature_version": FEATURE_VERSION,
        "root_feature_injection": "log1p + train standardisation -> Linear -> added to the root token",
        "graph_lookback_days": LOOKBACK_DAYS,
        "eligibility": f"5 actions in the {LOOKBACK_DAYS} days before the cutoff (as experiments.relgt)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "select_epoch": args.select_epoch,
    }


def _branch(args: argparse.Namespace) -> str:
    return f"{args.architecture}_relational" if args.groups else args.architecture


def run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _output_dir("backtest", args.output_dir)
    examples = load_or_build_examples(args.db_dir, output_dir, args.start_cutoff)
    raw = load_or_build_root_features(args.db_dir, output_dir, examples, args.start_cutoff, args.groups)
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    train = np.flatnonzero(examples.cutoff_days < train_end)
    calibration = np.flatnonzero((examples.cutoff_days >= train_end) & (examples.cutoff_days < calibration_end))
    holdout = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not len(train) or not len(calibration) or not len(holdout):
        raise ValueError("The chronological split produced an empty train, calibration or holdout set.")
    scaler = RootFeatureScaler.fit(raw[train])
    root_features = scaler.transform(raw)

    device = select_device(args.device)
    print(
        f"Device: {device}; {args.architecture} with {root_features.shape[1]} root features; "
        f"train < {pd.Timestamp(int(train_end) * DAY_NS)}, holdout >= {pd.Timestamp(int(calibration_end) * DAY_NS)}"
    )
    model, history = fit_model(examples, root_features, train, device, args, calibration)
    calibration_probabilities = predict_probabilities(
        model, examples.tokens, root_features, calibration, device, args.batch_size
    )
    thresholds = choose_thresholds(examples.labels[calibration], calibration_probabilities)
    holdout_probabilities = predict_probabilities(
        model, examples.tokens, root_features, holdout, device, args.batch_size
    )
    metrics = calculate_metrics(
        examples.labels[holdout], holdout_probabilities, thresholds, examples.cutoff_days[holdout]
    )
    branch = _branch(args)
    results = {
        **metrics,
        "experiment": "relgt_relational",
        "temporal_model": branch,
        **_settings(args),
        "device": str(device),
        "threshold_selection": "per label, on calibration weeks only",
        "epoch_history": history,
        "split": {
            "train_examples": int(len(train)),
            "calibration_examples": int(len(calibration)),
            "holdout_examples": int(len(holdout)),
            "train_weeks": int(len(np.unique(examples.cutoff_days[train]))),
            "calibration_weeks": int(len(np.unique(examples.cutoff_days[calibration]))),
            "holdout_weeks": int(len(np.unique(examples.cutoff_days[holdout]))),
            "holdout_start": str(pd.Timestamp(int(calibration_end) * DAY_NS)),
        },
        "root_feature_scaler": {
            name: {"mean": float(mean), "std": float(std)}
            for name, mean, std in zip(feature_names(args.groups), scaler.mean, scaler.std)
        },
    }

    torch.save(model.state_dict(), output_dir / f"{branch}_model.pt")
    labels = examples.labels[holdout]
    pd.DataFrame(
        {
            "user_id": examples.tokens.user_ids[holdout],
            "cutoff": pd.to_datetime(examples.cutoff_days[holdout].astype(np.int64) * DAY_NS),
            **{f"actual_{label}": labels[:, column] for column, label in enumerate(LABELS)},
            **{f"{branch}_{label}": holdout_probabilities[:, column] for column, label in enumerate(LABELS)},
        }
    ).to_csv(output_dir / "holdout_probabilities.csv", index=False)
    (output_dir / "backtest_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nHOLDOUT {branch}: score {metrics['score']:.4f} (macro {metrics['macro_f1']:.4f}, micro {metrics['micro_f1']:.4f})")
    for label in LABELS:
        print(f"  {label:<18} F1 {metrics['per_action_f1'][label]:.4f}")
    print(f"Saved metrics to {output_dir / 'backtest_metrics.json'}")
    return results


def run_prediction(args: argparse.Namespace) -> None:
    output_dir = _output_dir("predict", args.output_dir)
    examples = load_or_build_examples(args.db_dir, output_dir, args.start_cutoff)
    raw = load_or_build_root_features(args.db_dir, output_dir, examples, args.start_cutoff, args.groups)
    usable = prediction_example_indices(examples.cutoff_days, args.cutoff)
    train_end, _ = chronological_splits(examples.cutoff_days[usable])
    train = usable[examples.cutoff_days[usable] < train_end]
    calibration = usable[examples.cutoff_days[usable] >= train_end]
    if not len(train) or not len(calibration):
        raise ValueError("No chronological training or calibration examples precede the prediction cutoff.")
    scaler = RootFeatureScaler.fit(raw[train])
    root_features = scaler.transform(raw)
    device = select_device(args.device)
    model, _ = fit_model(examples, root_features, train, device, args, calibration)
    thresholds = choose_thresholds(
        examples.labels[calibration],
        predict_probabilities(model, examples.tokens, root_features, calibration, device, args.batch_size),
    )

    target_users = pd.read_csv(args.test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{args.test_input} contains duplicate UserIDs.")
    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"])
    cohort = users.loc[users["is_cohort"]]
    mapped = target_users["UserID"].map(cohort.set_index("github_user_id")["user_id"])
    if mapped.isna().any():
        raise ValueError(f"{int(mapped.isna().sum())} test UserID(s) are not in this release's cohort.")
    internal_ids = mapped.astype(np.int64).to_numpy()
    vocabulary = load_entity_vocabulary(args.db_dir)
    events = load_events(args.db_dir, cohort["user_id"].astype(int).tolist(), vocabulary)
    tokens = snapshot_tokens(events, args.cutoff, internal_ids, vocabulary)
    cutoff_day = np.full(len(internal_ids), args.cutoff.value // DAY_NS, dtype=np.int64)
    inference_raw = compute_root_features(
        args.db_dir, internal_ids, cutoff_day, args.cutoff, args.groups
    )
    probabilities = predict_probabilities(
        model, tokens, scaler.transform(inference_raw), np.arange(len(internal_ids)), device, args.batch_size
    )

    branch = _branch(args)
    predictions = pd.DataFrame({"UserID": target_users["UserID"]})
    probability_frame = predictions.copy()
    for column, label in enumerate(LABELS):
        predictions[label] = (probabilities[:, column] >= thresholds[label]).astype(np.int8)
        probability_frame[label] = probabilities[:, column]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    probability_frame.to_csv(output_dir / "probabilities.csv", index=False)
    torch.save(model.state_dict(), output_dir / f"{branch}_model.pt")
    metadata = {
        "experiment": "relgt_relational",
        "temporal_model": branch,
        **_settings(args),
        "prediction_cutoff_utc": str(args.cutoff),
        "training_examples": int(len(train)),
        "calibration_examples": int(len(calibration)),
        "thresholds": thresholds,
        "labels": list(LABELS),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions for {len(predictions):,} users to {output_dir / 'predictions.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backtest", "predict"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
        sub.add_argument("--output-dir", type=Path, default=None,
                         help="output folder (default: a new timestamped folder under outputs/)")
        sub.add_argument("--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF))
        sub.add_argument("--architecture", choices=ARCHITECTURES, default="relgt")
        sub.add_argument(
            "--groups",
            type=parse_groups,
            default=list(FEATURE_GROUPS),
            help=f"relational feature groups ({', '.join(FEATURE_GROUPS)}), comma-separated, "
                 "or 'none' for the plain graph model (default: all)",
        )
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        sub.add_argument("--epochs", type=int, default=5)
        sub.add_argument("--batch-size", type=int, default=256)
        sub.add_argument("--hidden-dim", type=int, default=64)
        sub.add_argument("--num-heads", type=int, default=4)
        sub.add_argument("--num-layers", type=int, default=2)
        sub.add_argument("--dropout", type=float, default=0.1)
        sub.add_argument("--learning-rate", type=float, default=1e-3)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument(
            "--select-epoch",
            action="store_true",
            help="keep the epoch with the best calibration-week score instead of the last one",
        )
    predict = subparsers.choices["predict"]
    predict.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    predict.add_argument("--cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF))
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "hidden_dim", "num_heads", "num_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.architecture == "relgt" and args.hidden_dim % args.num_heads:
        parser.error("--hidden-dim must be divisible by --num-heads.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1).")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "backtest":
        run_backtest(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
