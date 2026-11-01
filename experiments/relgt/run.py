"""Backtest or predict GitHub activity using a RelGT-inspired graph encoder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .data import (
    LABELS,
    GraphTokens,
    TrainingExamples,
    build_training_examples,
    load_entity_vocabulary,
    load_events,
    snapshot_tokens,
)
from .model import RelGTInspired


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_TEST_INPUT = ROOT / "assignment2" / "evaluation" / "test_input.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_START_CUTOFF = "2026-03-01 00:00:00"
DEFAULT_PREDICTION_CUTOFF = "2026-11-11 16:00:00"
CACHE_VERSION = 2


@dataclass(frozen=True)
class CudaDeviceInfo:
    index: int
    name: str
    utilization_percent: int | None
    memory_used_mib: int | None
    memory_total_mib: int | None


def _cuda_inventory() -> list[CudaDeviceInfo]:
    if not torch.cuda.is_available():
        print("CUDA GPUs available: 0")
        return []

    count = torch.cuda.device_count()
    gpu_stats: dict[int, tuple[int, int, int]] = {}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                if len(row) != 5:
                    continue
                index = int(row[0])
                gpu_stats[index] = (int(row[2]), int(row[3]), int(row[4]))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            print(f"Warning: could not read nvidia-smi utilization ({error}).")
    else:
        print("Warning: nvidia-smi not found; selecting CUDA GPU by free memory.")

    devices: list[CudaDeviceInfo] = []
    for index in range(count):
        name = torch.cuda.get_device_name(index)
        if index in gpu_stats:
            utilization, memory_used, memory_total = gpu_stats[index]
            info = CudaDeviceInfo(
                index, name, utilization, memory_used, memory_total
            )
            print(
                f"CUDA GPU {index}: {name}; utilization {utilization}%; "
                f"memory {memory_used}/{memory_total} MiB"
            )
        else:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except RuntimeError as error:
                print(
                    f"Warning: could not query free memory for CUDA GPU {index}: "
                    f"{error}"
                )
                free_bytes, total_bytes = 0, 0
            memory_total = total_bytes // (1024 * 1024)
            memory_used = memory_total - free_bytes // (1024 * 1024)
            info = CudaDeviceInfo(
                index,
                name,
                None,
                memory_used,
                memory_total,
            )
            print(
                f"CUDA GPU {index}: {name}; utilization unavailable; "
                f"memory {memory_used}/{memory_total} MiB"
            )
        devices.append(info)
    print(f"CUDA GPUs available: {count}")
    return devices


def _least_loaded_cuda_device(devices: list[CudaDeviceInfo]) -> torch.device:
    if not devices:
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")

    def load_key(info: CudaDeviceInfo) -> tuple[float, float, int]:
        utilization = (
            float(info.utilization_percent)
            if info.utilization_percent is not None
            else 100.0
        )
        if info.memory_used_mib is not None and info.memory_total_mib:
            memory_ratio = info.memory_used_mib / info.memory_total_mib
        else:
            memory_ratio = 1.0
        return (utilization, memory_ratio, info.index)

    selected = min(devices, key=load_key)
    if selected.utilization_percent is None:
        print(
            "Selecting by lowest GPU memory use because utilization readings "
            "are unavailable."
        )
    print(f"Selected CUDA GPU {selected.index}: {selected.name}")
    return torch.device("cuda", selected.index)


def select_device(requested: str) -> torch.device:
    cuda_devices = _cuda_inventory()
    if requested == "auto":
        if cuda_devices:
            return _least_loaded_cuda_device(cuda_devices)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        return _least_loaded_cuda_device(cuda_devices)
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but this PyTorch build has no MPS device.")
    if device.type != "cpu" and device.type != "mps":
        raise ValueError(f"Unsupported device: {requested}")
    return device


def create_run_output_dir(command: str, requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        requested_dir.mkdir(parents=True, exist_ok=True)
        return requested_dir

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DEFAULT_OUTPUT_ROOT / f"{command}_{timestamp}_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output_dir}")
    return output_dir


def _data_signature(db_dir: Path, start_cutoff: pd.Timestamp) -> dict[str, Any]:
    parquet_files = sorted(db_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet tables found in {db_dir}.")
    return {
        "version": CACHE_VERSION,
        "start_cutoff": str(start_cutoff),
        "tables": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in parquet_files
        ],
    }


def load_or_build_examples(
    db_dir: Path,
    output_dir: Path,
    start_cutoff: pd.Timestamp,
) -> TrainingExamples:
    cache_path = output_dir / "cache" / "weekly_graph_examples.npz"
    metadata_path = cache_path.with_suffix(".json")
    signature = _data_signature(db_dir, start_cutoff)

    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == signature:
            with np.load(cache_path, allow_pickle=False) as stored:
                tokens = GraphTokens(
                    user_ids=stored["user_ids"],
                    entity_ids=stored["entity_ids"],
                    node_type_ids=stored["node_type_ids"],
                    relation_ids=stored["relation_ids"],
                    hop_ids=stored["hop_ids"],
                    node_features=stored["node_features"],
                    relative_days=stored["relative_days"],
                )
                examples = TrainingExamples(
                    tokens=tokens,
                    labels=stored["labels"],
                    cutoff_days=stored["cutoff_days"],
                    num_entities=int(stored["num_entities"]),
                )
            print(f"Loaded cached training graphs from {cache_path}")
            return examples

    vocabulary = load_entity_vocabulary(db_dir)
    examples = build_training_examples(db_dir, start_cutoff, vocabulary)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    np.savez(
        temporary_path,
        user_ids=examples.tokens.user_ids,
        entity_ids=examples.tokens.entity_ids,
        node_type_ids=examples.tokens.node_type_ids,
        relation_ids=examples.tokens.relation_ids,
        hop_ids=examples.tokens.hop_ids,
        node_features=examples.tokens.node_features,
        relative_days=examples.tokens.relative_days,
        labels=examples.labels,
        cutoff_days=examples.cutoff_days,
        num_entities=np.asarray(examples.num_entities, dtype=np.int64),
    )
    temporary_path.replace(cache_path)
    metadata_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    print(f"Cached training graphs at {cache_path}")
    return examples


def chronological_splits(cutoff_days: np.ndarray) -> tuple[int, int]:
    unique_cutoffs = np.unique(cutoff_days)
    if len(unique_cutoffs) < 10:
        raise ValueError("At least 10 weekly cutoffs are needed for chronological splits.")
    train_end_index = max(1, int(len(unique_cutoffs) * 0.70))
    calibration_end_index = max(train_end_index + 1, int(len(unique_cutoffs) * 0.85))
    if calibration_end_index >= len(unique_cutoffs):
        calibration_end_index = len(unique_cutoffs) - 1
    return (
        int(unique_cutoffs[train_end_index]),
        int(unique_cutoffs[calibration_end_index]),
    )


def _model(
    num_entities: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
) -> RelGTInspired:
    if hidden_dim % num_heads:
        raise ValueError("--hidden-dim must be divisible by --num-heads.")
    return RelGTInspired(
        num_entities=num_entities,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_local_layers=num_layers,
        dropout=dropout,
    )


def _batch_inputs(
    tokens: GraphTokens, indices: np.ndarray, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        torch.as_tensor(tokens.entity_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(tokens.node_type_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(tokens.relation_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(tokens.hop_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(tokens.node_features[indices], dtype=torch.float32, device=device),
        torch.as_tensor(tokens.relative_days[indices], dtype=torch.float32, device=device),
    )


def fit_model(
    examples: TrainingExamples,
    train_indices: np.ndarray,
    num_entities: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    seed: int,
) -> RelGTInspired:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = _model(num_entities, hidden_dim, num_heads, num_layers, dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    targets = examples.labels

    for epoch in range(epochs):
        model.train()
        shuffled = np.random.permutation(train_indices)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            inputs = _batch_inputs(examples.tokens, batch_indices, device)
            labels = torch.as_tensor(
                targets[batch_indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        print(f"Epoch {epoch + 1}/{epochs}: BCE={total_loss / max(batches, 1):.5f}")
    return model


def predict_probabilities(
    model: RelGTInspired,
    tokens: GraphTokens,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            inputs = _batch_inputs(tokens, batch_indices, device)
            probabilities = torch.sigmoid(model(*inputs))
            predictions.append(probabilities.cpu().numpy())
    if not predictions:
        return np.empty((0, len(LABELS)), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def choose_thresholds(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    candidates = np.linspace(0.01, 0.99, 99)
    for column, label in enumerate(LABELS):
        scores = [
            f1_score(labels[:, column], probabilities[:, column] >= threshold, zero_division=0)
            for threshold in candidates
        ]
        thresholds[label] = float(candidates[int(np.argmax(scores))])
    return thresholds


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    threshold_array = np.array([thresholds[label] for label in LABELS])
    predictions = (probabilities >= threshold_array).astype(np.uint8)
    per_action = {
        label: float(
            f1_score(labels[:, index], predictions[:, index], zero_division=0)
        )
        for index, label in enumerate(LABELS)
    }
    macro = float(np.mean(list(per_action.values())))
    micro = float(
        f1_score(labels.ravel(), predictions.ravel(), zero_division=0)
    )
    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "score": (macro + micro) / 2,
        "per_action_f1": per_action,
        "thresholds": thresholds,
    }


def run_backtest(args: argparse.Namespace) -> None:
    examples = load_or_build_examples(args.db_dir, args.output_dir, args.start_cutoff)
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    train_indices = np.flatnonzero(examples.cutoff_days < train_end)
    calibration_indices = np.flatnonzero(
        (examples.cutoff_days >= train_end)
        & (examples.cutoff_days < calibration_end)
    )
    holdout_indices = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not len(calibration_indices) or not len(holdout_indices):
        raise ValueError("The chronological split produced an empty calibration or holdout set.")

    device = select_device(args.device)
    print(
        f"Device: {device}; chronological boundaries: train < "
        f"{pd.Timestamp(train_end, unit='D')}, holdout >= "
        f"{pd.Timestamp(calibration_end, unit='D')}"
    )
    model = fit_model(
        examples,
        train_indices,
        examples.num_entities,
        device,
        args.epochs,
        args.batch_size,
        args.hidden_dim,
        args.num_heads,
        args.num_layers,
        args.dropout,
        args.seed,
    )
    calibration_probabilities = predict_probabilities(
        model, examples.tokens, calibration_indices, device, args.batch_size
    )
    thresholds = choose_thresholds(
        examples.labels[calibration_indices], calibration_probabilities
    )
    holdout_probabilities = predict_probabilities(
        model, examples.tokens, holdout_indices, device, args.batch_size
    )
    results = calculate_metrics(
        examples.labels[holdout_indices], holdout_probabilities, thresholds
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
    results["epochs"] = args.epochs
    results["experiment"] = args.output_dir.name
    results["architecture"] = "RelGT-inspired sampled temporal neighborhood encoder"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved holdout metrics to {metrics_path}")


def run_prediction(args: argparse.Namespace) -> None:
    examples = load_or_build_examples(args.db_dir, args.output_dir, args.start_cutoff)
    train_end, _ = chronological_splits(examples.cutoff_days)
    train_indices = np.flatnonzero(examples.cutoff_days < train_end)
    calibration_indices = np.flatnonzero(examples.cutoff_days >= train_end)
    if not len(calibration_indices):
        raise ValueError("No chronological calibration examples are available.")

    device = select_device(args.device)
    model = fit_model(
        examples,
        train_indices,
        examples.num_entities,
        device,
        args.epochs,
        args.batch_size,
        args.hidden_dim,
        args.num_heads,
        args.num_layers,
        args.dropout,
        args.seed,
    )
    calibration_probabilities = predict_probabilities(
        model, examples.tokens, calibration_indices, device, args.batch_size
    )
    thresholds = choose_thresholds(
        examples.labels[calibration_indices], calibration_probabilities
    )

    target_users = pd.read_csv(args.test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{args.test_input} contains duplicate UserIDs.")
    users = pd.read_parquet(
        args.db_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"],
    )
    cohort_users = users.loc[users["is_cohort"]]
    user_lookup = cohort_users.set_index("github_user_id")["user_id"]
    mapped_ids = target_users["UserID"].map(user_lookup)
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
    inference_tokens = snapshot_tokens(events, args.cutoff, internal_ids, vocabulary)
    row_indices = np.arange(len(internal_ids), dtype=np.int64)
    probabilities = predict_probabilities(
        model, inference_tokens, row_indices, device, args.batch_size
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({"UserID": target_users["UserID"]})
    for column, label in enumerate(LABELS):
        output[label] = (probabilities[:, column] >= thresholds[label]).astype(np.uint8)
    prediction_path = args.output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)

    probability_output = pd.DataFrame(
        probabilities, columns=[f"{label}_probability" for label in LABELS]
    )
    probability_output.insert(0, "UserID", target_users["UserID"].to_numpy())
    probability_path = args.output_dir / "probabilities.csv"
    probability_output.to_csv(probability_path, index=False)

    torch.save(model.state_dict(), args.output_dir / "model.pt")
    metadata = {
        "prediction_cutoff_utc": str(args.cutoff),
        "training_start_cutoff_utc": str(args.start_cutoff),
        "training_weeks": int(np.unique(examples.cutoff_days[train_indices]).size),
        "calibration_weeks": int(np.unique(examples.cutoff_days[calibration_indices]).size),
        "labels": list(LABELS),
        "thresholds": thresholds,
        "device": str(device),
        "experiment": args.output_dir.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "architecture": "RelGT-inspired sampled temporal neighborhood encoder",
        "official_relgt": "https://github.com/snap-stanford/relgt",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(output):,} predictions to {prediction_path}")
    print(f"Saved unthresholded branch probabilities to {probability_path}")
    print(f"Device: {device}; calibrated thresholds: {json.dumps(thresholds)}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output folder (default: a new timestamped folder per run)",
    )
    parser.add_argument(
        "--start-cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(DEFAULT_START_CUTOFF),
        help="first weekly prediction cutoff used to construct training samples",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest_parser = subparsers.add_parser("backtest")
    add_common_arguments(backtest_parser)
    predict_parser = subparsers.add_parser("predict")
    add_common_arguments(predict_parser)
    predict_parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    predict_parser.add_argument(
        "--cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF),
        help="UTC time at which to make the prediction",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.num_layers < 1:
        parser.error("--num-layers must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    args.output_dir = create_run_output_dir(args.command, args.output_dir)
    if args.command == "backtest":
        run_backtest(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
