"""Backtest or predict GitHub activity using a RelGT-inspired graph encoder."""

from __future__ import annotations

import argparse
import csv
import importlib
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
    HORIZON_DAYS,
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


def prediction_example_indices(
    cutoff_days: np.ndarray, prediction_cutoff: pd.Timestamp
) -> np.ndarray:
    """Return samples whose complete target window ends by prediction time."""
    day_ns = pd.Timedelta(days=1).value
    horizon_ns = pd.Timedelta(days=HORIZON_DAYS).value
    cutoff_ns = cutoff_days.astype(np.int64) * day_ns
    return np.flatnonzero(cutoff_ns + horizon_ns <= prediction_cutoff.value)


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


def fuse_probabilities(
    graph_probabilities: np.ndarray,
    other_probabilities: np.ndarray,
    graph_weight: float,
) -> np.ndarray:
    if graph_probabilities.shape != other_probabilities.shape:
        raise ValueError("Probability arrays must have matching shapes.")
    if graph_probabilities.ndim != 2 or graph_probabilities.shape[1] != len(LABELS):
        raise ValueError(f"Probability arrays must have shape (n, {len(LABELS)}).")
    if not 0.0 <= graph_weight <= 1.0:
        raise ValueError("The RelGT fusion weight must be between 0 and 1.")
    if not (
        np.isfinite(graph_probabilities).all()
        and np.isfinite(other_probabilities).all()
    ):
        raise ValueError("Probability arrays must be finite.")
    return (
        graph_weight * graph_probabilities
        + (1.0 - graph_weight) * other_probabilities
    )


def choose_fusion_weight(
    labels: np.ndarray,
    graph_probabilities: np.ndarray,
    s2p2_probabilities: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    best_weight = 0.0
    best_thresholds: dict[str, float] = {}
    best_metrics: dict[str, Any] | None = None
    for graph_weight in np.linspace(0.0, 1.0, 11):
        fused = fuse_probabilities(
            graph_probabilities, s2p2_probabilities, float(graph_weight)
        )
        thresholds = choose_thresholds(labels, fused)
        metrics = calculate_metrics(labels, fused, thresholds)
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_weight = float(graph_weight)
            best_thresholds = thresholds
            best_metrics = metrics
    if best_metrics is None:
        raise ValueError("Cannot calibrate fusion without calibration examples.")
    return best_weight, best_thresholds, best_metrics


def fuse_three_probabilities(
    relgt_probabilities: np.ndarray,
    lightgbm_probabilities: np.ndarray,
    s2p2_probabilities: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    arrays = {
        "relgt": relgt_probabilities,
        "lightgbm": lightgbm_probabilities,
        "s2p2": s2p2_probabilities,
    }
    if set(weights) != set(arrays):
        raise ValueError("Fusion weights must be provided for RelGT, LightGBM, and S2P2.")
    reference = relgt_probabilities
    if reference.ndim != 2 or reference.shape[1] != len(LABELS):
        raise ValueError(f"Probability arrays must have shape (n, {len(LABELS)}).")
    if any(array.shape != reference.shape for array in arrays.values()):
        raise ValueError("Probability arrays must have matching shapes.")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Probability arrays must be finite.")
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Fusion weights must be finite and non-negative.")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Fusion weights must sum to 1.")
    return sum(weights[name] * array for name, array in arrays.items())


def choose_three_fusion_weights(
    labels: np.ndarray,
    relgt_probabilities: np.ndarray,
    lightgbm_probabilities: np.ndarray,
    s2p2_probabilities: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    expected_shape = (len(labels), len(LABELS))
    if labels.ndim != 2 or labels.shape != expected_shape or not len(labels):
        raise ValueError(
            f"Calibration labels must have non-empty shape (n, {len(LABELS)})."
        )
    best_weights: dict[str, float] | None = None
    best_thresholds: dict[str, float] = {}
    best_metrics: dict[str, Any] | None = None
    candidate_weights = np.linspace(0.0, 1.0, 5)
    for relgt_weight in candidate_weights:
        for lightgbm_weight in candidate_weights:
            s2p2_weight = 1.0 - relgt_weight - lightgbm_weight
            if s2p2_weight < -1e-9:
                continue
            weights = {
                "relgt": float(relgt_weight),
                "lightgbm": float(lightgbm_weight),
                "s2p2": max(0.0, float(s2p2_weight)),
            }
            fused = fuse_three_probabilities(
                relgt_probabilities,
                lightgbm_probabilities,
                s2p2_probabilities,
                weights,
            )
            thresholds = choose_thresholds(labels, fused)
            metrics = calculate_metrics(labels, fused, thresholds)
            if best_metrics is None or metrics["score"] > best_metrics["score"]:
                best_weights = weights
                best_thresholds = thresholds
                best_metrics = metrics
    if best_weights is None or best_metrics is None:
        raise ValueError("Cannot calibrate three-branch fusion without examples.")
    return best_weights, best_thresholds, best_metrics


def require_s2p2_dependency() -> None:
    try:
        importlib.import_module("easy_tpp.model.torch_model.torch_s2p2")
    except ImportError as error:
        raise ImportError(
            "RelGT+S2P2 modes require the optional EasyTPP dependency. Install it "
            "with: python -m pip install -r "
            "experiments/lightgbm_baseline/s2p2-requirements.txt"
        ) from error


def s2p2_examples_for_rows(
    examples: TrainingExamples, indices: np.ndarray
) -> pd.DataFrame:
    """Preserve each graph example's user/cutoff alignment for the S2P2 branch."""
    return pd.DataFrame(
        {
            "user_id": examples.tokens.user_ids[indices].astype(np.int64),
            "cutoff": pd.to_datetime(
                examples.cutoff_days[indices], unit="D"
            ),
        }
    )


def lightgbm_examples_for_graph(
    examples: TrainingExamples, data: pd.DataFrame
) -> pd.DataFrame:
    """Align tabular features to graph samples by user and weekly cutoff."""
    keys = s2p2_examples_for_rows(
        examples, np.arange(len(examples.cutoff_days), dtype=np.int64)
    )
    keyed_data = data.copy()
    keyed_data["cutoff"] = pd.to_datetime(keyed_data["cutoff"])
    keyed_data = keyed_data.set_index(["user_id", "cutoff"])
    if not keyed_data.index.is_unique:
        raise ValueError("LightGBM examples contain duplicate user/cutoff rows.")
    key_index = pd.MultiIndex.from_frame(keys[["user_id", "cutoff"]])
    missing = key_index.difference(keyed_data.index)
    if len(missing):
        raise ValueError(
            f"LightGBM features are missing {len(missing)} RelGT user/cutoff rows."
        )
    aligned = keyed_data.loc[key_index].reset_index()
    graph_labels = examples.labels
    if not np.array_equal(
        aligned[list(LABELS)].to_numpy(dtype=np.uint8), graph_labels
    ):
        raise ValueError("LightGBM and RelGT labels are not aligned.")
    return aligned


def lightgbm_model_params(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
    }


def fit_lightgbm_branch(
    args: argparse.Namespace,
    examples: TrainingExamples,
    train_indices: np.ndarray,
) -> tuple[Any, dict[str, pd.DataFrame], pd.DataFrame, list[str], dict[str, int | float]]:
    try:
        from experiments.lightgbm_baseline.run import (
            build_training_data,
            fit_models,
        )
    except ImportError as error:
        raise ImportError(
            "The RelGT+LightGBM mode requires the baseline dependencies. "
            "Install them with: python -m pip install -r requirements.txt"
        ) from error

    data, histories, _ = build_training_data(args.db_dir, args.start_cutoff)
    aligned_data = lightgbm_examples_for_graph(examples, data)
    feature_columns = [
        column
        for column in aligned_data.columns
        if column not in {"user_id", "cutoff", *LABELS}
    ]
    params = lightgbm_model_params(args)
    models = fit_models(aligned_data.iloc[train_indices], feature_columns, params)
    return models, histories, aligned_data, feature_columns, params


def lightgbm_probabilities(
    models: Any,
    data: pd.DataFrame,
    indices: np.ndarray,
    feature_columns: list[str],
) -> np.ndarray:
    from experiments.lightgbm_baseline.run import predict_probabilities as predict_lgbm

    by_label = predict_lgbm(models, data.iloc[indices], feature_columns)
    return np.column_stack([by_label[label] for label in LABELS])


def save_lightgbm_models(models: Any, output_dir: Path) -> None:
    for label, model in models.items():
        if model is not None:
            model.booster_.save_model(str(output_dir / f"lightgbm_{label}.txt"))


def fit_s2p2_branch(
    args: argparse.Namespace,
    examples: TrainingExamples,
    train_indices: np.ndarray,
    cohort_user_ids: list[int],
    histories: dict[str, pd.DataFrame] | None = None,
) -> tuple[Any, dict[str, pd.DataFrame], torch.device]:
    try:
        from experiments.lightgbm_baseline.run import load_histories
        from experiments.lightgbm_baseline.s2p2 import (
            S2P2TemporalModel,
            select_device as select_s2p2_device,
        )
    except ImportError as error:
        raise ImportError(
            "RelGT+S2P2 modes require the optional EasyTPP dependency. Install it "
            "with: python -m pip install -r "
            "experiments/lightgbm_baseline/s2p2-requirements.txt"
        ) from error

    s2p2_device_request = "cpu" if args.device == "mps" else args.device
    s2p2_device = select_s2p2_device(s2p2_device_request)
    if histories is None:
        histories = load_histories(args.db_dir, cohort_user_ids)
    model = S2P2TemporalModel(
        device=s2p2_device,
        epochs=args.s2p2_epochs,
        batch_size=args.s2p2_batch_size,
        max_sequence_length=args.s2p2_max_sequence_length,
        hidden_size=args.s2p2_hidden_size,
        state_size=args.s2p2_state_size,
        num_layers=args.s2p2_num_layers,
        integral_samples=args.s2p2_integral_samples,
        seed=args.seed,
    )
    model.fit(s2p2_examples_for_rows(examples, train_indices), histories)
    return model, histories, s2p2_device


def run_backtest(args: argparse.Namespace) -> None:
    if args.temporal_model in {"hybrid", "relgt_lightgbm_s2p2"}:
        require_s2p2_dependency()
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
    graph_calibration_probabilities = predict_probabilities(
        model, examples.tokens, calibration_indices, device, args.batch_size
    )
    graph_holdout_probabilities = predict_probabilities(
        model, examples.tokens, holdout_indices, device, args.batch_size
    )
    if args.temporal_model == "hybrid":
        cohort_user_ids = pd.read_parquet(
            args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"]
        ).loc[lambda users: users["is_cohort"], "user_id"].astype(int).tolist()
        s2p2_model, histories, s2p2_device = fit_s2p2_branch(
            args, examples, train_indices, cohort_user_ids
        )
        calibration_frame = s2p2_examples_for_rows(examples, calibration_indices)
        holdout_frame = s2p2_examples_for_rows(examples, holdout_indices)
        s2p2_calibration_by_label = s2p2_model.predict(calibration_frame, histories)
        s2p2_holdout_by_label = s2p2_model.predict(holdout_frame, histories)
        s2p2_calibration_probabilities = np.column_stack(
            [s2p2_calibration_by_label[label] for label in LABELS]
        )
        s2p2_holdout_probabilities = np.column_stack(
            [s2p2_holdout_by_label[label] for label in LABELS]
        )
        graph_weight, thresholds, calibration_metrics = choose_fusion_weight(
            examples.labels[calibration_indices],
            graph_calibration_probabilities,
            s2p2_calibration_probabilities,
        )
        graph_only_thresholds = choose_thresholds(
            examples.labels[calibration_indices], graph_calibration_probabilities
        )
        s2p2_only_thresholds = choose_thresholds(
            examples.labels[calibration_indices], s2p2_calibration_probabilities
        )
        graph_only_metrics = calculate_metrics(
            examples.labels[holdout_indices],
            graph_holdout_probabilities,
            graph_only_thresholds,
        )
        s2p2_only_metrics = calculate_metrics(
            examples.labels[holdout_indices],
            s2p2_holdout_probabilities,
            s2p2_only_thresholds,
        )
        holdout_probabilities = fuse_probabilities(
            graph_holdout_probabilities,
            s2p2_holdout_probabilities,
            graph_weight,
        )
        torch.save(model.state_dict(), args.output_dir / "relgt_model.pt")
        s2p2_model.save(args.output_dir / "s2p2_model.pt")
        pd.DataFrame(
            {
                "user_id": examples.tokens.user_ids[holdout_indices],
                "cutoff": s2p2_examples_for_rows(examples, holdout_indices)["cutoff"],
                **{
                    f"relgt_{label}": graph_holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
                **{
                    f"s2p2_{label}": s2p2_holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
                **{
                    f"fused_{label}": holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
            }
        ).to_csv(args.output_dir / "holdout_probabilities.csv", index=False)
    elif args.temporal_model == "relgt_lightgbm":
        (
            lightgbm_models,
            _,
            lightgbm_data,
            lightgbm_features,
            lightgbm_params,
        ) = fit_lightgbm_branch(args, examples, train_indices)
        lightgbm_calibration_probabilities = lightgbm_probabilities(
            lightgbm_models,
            lightgbm_data,
            calibration_indices,
            lightgbm_features,
        )
        lightgbm_holdout_probabilities = lightgbm_probabilities(
            lightgbm_models,
            lightgbm_data,
            holdout_indices,
            lightgbm_features,
        )
        graph_weight, thresholds, calibration_metrics = choose_fusion_weight(
            examples.labels[calibration_indices],
            graph_calibration_probabilities,
            lightgbm_calibration_probabilities,
        )
        graph_only_thresholds = choose_thresholds(
            examples.labels[calibration_indices], graph_calibration_probabilities
        )
        lightgbm_only_thresholds = choose_thresholds(
            examples.labels[calibration_indices],
            lightgbm_calibration_probabilities,
        )
        graph_only_metrics = calculate_metrics(
            examples.labels[holdout_indices],
            graph_holdout_probabilities,
            graph_only_thresholds,
        )
        lightgbm_only_metrics = calculate_metrics(
            examples.labels[holdout_indices],
            lightgbm_holdout_probabilities,
            lightgbm_only_thresholds,
        )
        holdout_probabilities = fuse_probabilities(
            graph_holdout_probabilities,
            lightgbm_holdout_probabilities,
            graph_weight,
        )
        torch.save(model.state_dict(), args.output_dir / "relgt_model.pt")
        save_lightgbm_models(lightgbm_models, args.output_dir)
        holdout_rows = s2p2_examples_for_rows(examples, holdout_indices)
        pd.DataFrame(
            {
                "user_id": examples.tokens.user_ids[holdout_indices],
                "cutoff": holdout_rows["cutoff"],
                **{
                    f"relgt_{label}": graph_holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
                **{
                    f"lightgbm_{label}": lightgbm_holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
                **{
                    f"fused_{label}": holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
            }
        ).to_csv(args.output_dir / "holdout_probabilities.csv", index=False)
    elif args.temporal_model == "relgt_lightgbm_s2p2":
        cohort_user_ids = pd.read_parquet(
            args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"]
        ).loc[lambda users: users["is_cohort"], "user_id"].astype(int).tolist()
        (
            lightgbm_models,
            histories,
            lightgbm_data,
            lightgbm_features,
            lightgbm_params,
        ) = fit_lightgbm_branch(args, examples, train_indices)
        s2p2_model, histories, s2p2_device = fit_s2p2_branch(
            args, examples, train_indices, cohort_user_ids, histories
        )
        calibration_frame = s2p2_examples_for_rows(examples, calibration_indices)
        holdout_frame = s2p2_examples_for_rows(examples, holdout_indices)
        s2p2_calibration_by_label = s2p2_model.predict(calibration_frame, histories)
        s2p2_holdout_by_label = s2p2_model.predict(holdout_frame, histories)
        s2p2_calibration_probabilities = np.column_stack(
            [s2p2_calibration_by_label[label] for label in LABELS]
        )
        s2p2_holdout_probabilities = np.column_stack(
            [s2p2_holdout_by_label[label] for label in LABELS]
        )
        lightgbm_calibration_probabilities = lightgbm_probabilities(
            lightgbm_models,
            lightgbm_data,
            calibration_indices,
            lightgbm_features,
        )
        lightgbm_holdout_probabilities = lightgbm_probabilities(
            lightgbm_models,
            lightgbm_data,
            holdout_indices,
            lightgbm_features,
        )
        fusion_weights, thresholds, calibration_metrics = choose_three_fusion_weights(
            examples.labels[calibration_indices],
            graph_calibration_probabilities,
            lightgbm_calibration_probabilities,
            s2p2_calibration_probabilities,
        )
        calibration_branch_probabilities = {
            "relgt": graph_calibration_probabilities,
            "lightgbm": lightgbm_calibration_probabilities,
            "s2p2": s2p2_calibration_probabilities,
        }
        branch_probabilities = {
            "relgt": graph_holdout_probabilities,
            "lightgbm": lightgbm_holdout_probabilities,
            "s2p2": s2p2_holdout_probabilities,
        }
        branch_thresholds = {
            name: choose_thresholds(
                examples.labels[calibration_indices],
                calibration_branch_probabilities[name],
            )
            for name in branch_probabilities
        }
        branch_metrics = {
            name: calculate_metrics(
                examples.labels[holdout_indices],
                branch_probabilities[name],
                branch_thresholds[name],
            )
            for name in branch_probabilities
        }
        holdout_probabilities = fuse_three_probabilities(
            graph_holdout_probabilities,
            lightgbm_holdout_probabilities,
            s2p2_holdout_probabilities,
            fusion_weights,
        )
        torch.save(model.state_dict(), args.output_dir / "relgt_model.pt")
        s2p2_model.save(args.output_dir / "s2p2_model.pt")
        save_lightgbm_models(lightgbm_models, args.output_dir)
        holdout_rows = s2p2_examples_for_rows(examples, holdout_indices)
        pd.DataFrame(
            {
                "user_id": examples.tokens.user_ids[holdout_indices],
                "cutoff": holdout_rows["cutoff"],
                **{
                    f"relgt_{label}": graph_holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
                **{
                    f"lightgbm_{label}": lightgbm_holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
                **{
                    f"s2p2_{label}": s2p2_holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
                **{
                    f"fused_{label}": holdout_probabilities[:, column]
                    for column, label in enumerate(LABELS)
                },
            }
        ).to_csv(args.output_dir / "holdout_probabilities.csv", index=False)
    else:
        s2p2_model = None
        s2p2_device = None
        graph_weight = 1.0
        calibration_metrics = None
        thresholds = choose_thresholds(
            examples.labels[calibration_indices], graph_calibration_probabilities
        )
        holdout_probabilities = graph_holdout_probabilities
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
    results["temporal_model"] = args.temporal_model
    results["epochs"] = args.epochs
    results["experiment"] = args.output_dir.name
    results["architecture"] = "RelGT-inspired sampled temporal neighborhood encoder"
    if args.temporal_model == "hybrid":
        results["fusion"] = {
            "method": "calibration-selected probability averaging",
            "graph_weight": graph_weight,
            "s2p2_weight": 1.0 - graph_weight,
            "calibration_score": calibration_metrics["score"],
            "s2p2_device": str(s2p2_device),
            "s2p2_epochs": args.s2p2_epochs,
        }
        results["branch_holdout"] = {
            "relgt": graph_only_metrics,
            "s2p2": s2p2_only_metrics,
        }
    elif args.temporal_model == "relgt_lightgbm":
        results["fusion"] = {
            "method": "calibration-selected probability averaging",
            "relgt_weight": graph_weight,
            "lightgbm_weight": 1.0 - graph_weight,
            "calibration_score": calibration_metrics["score"],
            "lightgbm_params": lightgbm_params,
        }
        results["branch_holdout"] = {
            "relgt": graph_only_metrics,
            "lightgbm": lightgbm_only_metrics,
        }
    elif args.temporal_model == "relgt_lightgbm_s2p2":
        results["fusion"] = {
            "method": "calibration-selected three-branch probability averaging",
            "weights": fusion_weights,
            "calibration_score": calibration_metrics["score"],
            "lightgbm_params": lightgbm_params,
            "s2p2_device": str(s2p2_device),
            "s2p2_epochs": args.s2p2_epochs,
        }
        results["branch_holdout"] = branch_metrics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved holdout metrics to {metrics_path}")


def run_prediction(args: argparse.Namespace) -> None:
    if args.temporal_model in {"hybrid", "relgt_lightgbm_s2p2"}:
        require_s2p2_dependency()
    examples = load_or_build_examples(args.db_dir, args.output_dir, args.start_cutoff)
    usable_indices = prediction_example_indices(examples.cutoff_days, args.cutoff)
    usable_cutoffs = examples.cutoff_days[usable_indices]
    train_end, _ = chronological_splits(usable_cutoffs)
    train_indices = usable_indices[usable_cutoffs < train_end]
    calibration_indices = usable_indices[usable_cutoffs >= train_end]
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
    graph_calibration_probabilities = predict_probabilities(
        model, examples.tokens, calibration_indices, device, args.batch_size
    )
    s2p2_model = None
    histories = None
    graph_weight = 1.0
    if args.temporal_model == "hybrid":
        cohort_user_ids = pd.read_parquet(
            args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"]
        ).loc[lambda users: users["is_cohort"], "user_id"].astype(int).tolist()
        s2p2_model, histories, s2p2_device = fit_s2p2_branch(
            args, examples, train_indices, cohort_user_ids
        )
        s2p2_calibration_by_label = s2p2_model.predict(
            s2p2_examples_for_rows(examples, calibration_indices), histories
        )
        s2p2_calibration_probabilities = np.column_stack(
            [s2p2_calibration_by_label[label] for label in LABELS]
        )
        graph_weight, thresholds, calibration_metrics = choose_fusion_weight(
            examples.labels[calibration_indices],
            graph_calibration_probabilities,
            s2p2_calibration_probabilities,
        )
    elif args.temporal_model == "relgt_lightgbm":
        (
            lightgbm_models,
            lightgbm_histories,
            lightgbm_data,
            lightgbm_features,
            lightgbm_params,
        ) = fit_lightgbm_branch(args, examples, train_indices)
        lightgbm_calibration_probabilities = lightgbm_probabilities(
            lightgbm_models,
            lightgbm_data,
            calibration_indices,
            lightgbm_features,
        )
        graph_weight, thresholds, calibration_metrics = choose_fusion_weight(
            examples.labels[calibration_indices],
            graph_calibration_probabilities,
            lightgbm_calibration_probabilities,
        )
    elif args.temporal_model == "relgt_lightgbm_s2p2":
        cohort_user_ids = pd.read_parquet(
            args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"]
        ).loc[lambda users: users["is_cohort"], "user_id"].astype(int).tolist()
        (
            lightgbm_models,
            lightgbm_histories,
            lightgbm_data,
            lightgbm_features,
            lightgbm_params,
        ) = fit_lightgbm_branch(args, examples, train_indices)
        s2p2_model, s2p2_histories, s2p2_device = fit_s2p2_branch(
            args,
            examples,
            train_indices,
            cohort_user_ids,
            lightgbm_histories,
        )
        s2p2_calibration_by_label = s2p2_model.predict(
            s2p2_examples_for_rows(examples, calibration_indices), s2p2_histories
        )
        s2p2_calibration_probabilities = np.column_stack(
            [s2p2_calibration_by_label[label] for label in LABELS]
        )
        lightgbm_calibration_probabilities = lightgbm_probabilities(
            lightgbm_models,
            lightgbm_data,
            calibration_indices,
            lightgbm_features,
        )
        fusion_weights, thresholds, calibration_metrics = choose_three_fusion_weights(
            examples.labels[calibration_indices],
            graph_calibration_probabilities,
            lightgbm_calibration_probabilities,
            s2p2_calibration_probabilities,
        )
    else:
        thresholds = choose_thresholds(
            examples.labels[calibration_indices], graph_calibration_probabilities
        )

    target_users = pd.read_csv(args.test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{args.test_input} contains duplicate UserIDs.")
    users = pd.read_parquet(
        args.db_dir / "Users.parquet",
        columns=[
            "user_id",
            "github_user_id",
            "is_cohort",
            "is_bot",
            "first_seen_at",
        ],
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
    graph_probabilities = predict_probabilities(
        model, inference_tokens, row_indices, device, args.batch_size
    )
    if args.temporal_model == "hybrid":
        inference_frame = pd.DataFrame(
            {
                "user_id": internal_ids.astype(np.int64),
                "cutoff": args.cutoff,
            }
        )
        s2p2_by_label = s2p2_model.predict(inference_frame, histories)
        s2p2_probabilities = np.column_stack(
            [s2p2_by_label[label] for label in LABELS]
        )
        probabilities = fuse_probabilities(
            graph_probabilities, s2p2_probabilities, graph_weight
        )
    elif args.temporal_model == "relgt_lightgbm":
        from experiments.lightgbm_baseline.run import (
            add_user_bot_feature,
            predict_probabilities as predict_lgbm,
            row_for_cutoff,
        )

        inference_features = row_for_cutoff(
            args.cutoff, lightgbm_histories, internal_ids.astype("int64").tolist()
        )
        inference_features = add_user_bot_feature(inference_features, cohort_users)
        lightgbm_by_label = predict_lgbm(
            lightgbm_models, inference_features, lightgbm_features
        )
        lightgbm_probabilities_for_users = np.column_stack(
            [lightgbm_by_label[label] for label in LABELS]
        )
        probabilities = fuse_probabilities(
            graph_probabilities, lightgbm_probabilities_for_users, graph_weight
        )
    elif args.temporal_model == "relgt_lightgbm_s2p2":
        inference_frame = pd.DataFrame(
            {
                "user_id": internal_ids.astype(np.int64),
                "cutoff": args.cutoff,
            }
        )
        s2p2_by_label = s2p2_model.predict(inference_frame, s2p2_histories)
        s2p2_probabilities_for_users = np.column_stack(
            [s2p2_by_label[label] for label in LABELS]
        )
        from experiments.lightgbm_baseline.run import (
            add_user_bot_feature,
            predict_probabilities as predict_lgbm,
            row_for_cutoff,
        )

        inference_features = row_for_cutoff(
            args.cutoff,
            lightgbm_histories,
            internal_ids.astype("int64").tolist(),
        )
        inference_features = add_user_bot_feature(inference_features, cohort_users)
        lightgbm_by_label = predict_lgbm(
            lightgbm_models, inference_features, lightgbm_features
        )
        lightgbm_probabilities_for_users = np.column_stack(
            [lightgbm_by_label[label] for label in LABELS]
        )
        probabilities = fuse_three_probabilities(
            graph_probabilities,
            lightgbm_probabilities_for_users,
            s2p2_probabilities_for_users,
            fusion_weights,
        )
    else:
        probabilities = graph_probabilities

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({"UserID": target_users["UserID"]})
    for column, label in enumerate(LABELS):
        output[label] = (probabilities[:, column] >= thresholds[label]).astype(np.uint8)
    prediction_path = args.output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)

    probability_values = {
        f"{label}_probability": probabilities[:, column]
        for column, label in enumerate(LABELS)
    }
    if args.temporal_model == "hybrid":
        probability_values.update(
            {
                f"relgt_{label}_probability": graph_probabilities[:, column]
                for column, label in enumerate(LABELS)
            }
        )
        probability_values.update(
            {
                f"s2p2_{label}_probability": s2p2_probabilities[:, column]
                for column, label in enumerate(LABELS)
            }
        )
    elif args.temporal_model == "relgt_lightgbm":
        probability_values.update(
            {
                f"relgt_{label}_probability": graph_probabilities[:, column]
                for column, label in enumerate(LABELS)
            }
        )
        probability_values.update(
            {
                f"lightgbm_{label}_probability": (
                    lightgbm_probabilities_for_users[:, column]
                )
                for column, label in enumerate(LABELS)
            }
        )
    elif args.temporal_model == "relgt_lightgbm_s2p2":
        probability_values.update(
            {
                f"relgt_{label}_probability": graph_probabilities[:, column]
                for column, label in enumerate(LABELS)
            }
        )
        probability_values.update(
            {
                f"lightgbm_{label}_probability": (
                    lightgbm_probabilities_for_users[:, column]
                )
                for column, label in enumerate(LABELS)
            }
        )
        probability_values.update(
            {
                f"s2p2_{label}_probability": s2p2_probabilities_for_users[:, column]
                for column, label in enumerate(LABELS)
            }
        )
    probability_output = pd.DataFrame(probability_values)
    probability_output.insert(0, "UserID", target_users["UserID"].to_numpy())
    probability_path = args.output_dir / "probabilities.csv"
    probability_output.to_csv(probability_path, index=False)

    relgt_model_path = (
        "model.pt" if args.temporal_model == "relgt" else "relgt_model.pt"
    )
    torch.save(model.state_dict(), args.output_dir / relgt_model_path)
    if args.temporal_model == "hybrid":
        s2p2_model.save(args.output_dir / "s2p2_model.pt")
    elif args.temporal_model == "relgt_lightgbm":
        save_lightgbm_models(lightgbm_models, args.output_dir)
    elif args.temporal_model == "relgt_lightgbm_s2p2":
        s2p2_model.save(args.output_dir / "s2p2_model.pt")
        save_lightgbm_models(lightgbm_models, args.output_dir)
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "architecture": "RelGT-inspired sampled temporal neighborhood encoder",
        "official_relgt": "https://github.com/snap-stanford/relgt",
    }
    if args.temporal_model == "hybrid":
        metadata["s2p2_device"] = str(s2p2_device)
        metadata["fusion"] = {
            "method": "calibration-selected probability averaging",
            "graph_weight": graph_weight,
            "s2p2_weight": 1.0 - graph_weight,
            "calibration_score": calibration_metrics["score"],
        }
    elif args.temporal_model == "relgt_lightgbm":
        metadata["lightgbm_params"] = lightgbm_params
        metadata["fusion"] = {
            "method": "calibration-selected probability averaging",
            "relgt_weight": graph_weight,
            "lightgbm_weight": 1.0 - graph_weight,
            "calibration_score": calibration_metrics["score"],
        }
    elif args.temporal_model == "relgt_lightgbm_s2p2":
        metadata["lightgbm_params"] = lightgbm_params
        metadata["s2p2_params"] = {
            "epochs": args.s2p2_epochs,
            "batch_size": args.s2p2_batch_size,
            "max_sequence_length": args.s2p2_max_sequence_length,
            "hidden_size": args.s2p2_hidden_size,
            "state_size": args.s2p2_state_size,
            "num_layers": args.s2p2_num_layers,
            "integral_samples": args.s2p2_integral_samples,
            "device": str(s2p2_device),
        }
        metadata["fusion"] = {
            "method": "calibration-selected three-branch probability averaging",
            "weights": fusion_weights,
            "calibration_score": calibration_metrics["score"],
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
    parser.add_argument(
        "--temporal-model",
        choices=["relgt", "hybrid", "relgt_lightgbm", "relgt_lightgbm_s2p2"],
        default="relgt",
        help="use RelGT alone or fuse with S2P2, LightGBM, or both",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--s2p2-epochs", type=int, default=3)
    parser.add_argument("--s2p2-batch-size", type=int, default=128)
    parser.add_argument("--s2p2-max-sequence-length", type=int, default=128)
    parser.add_argument("--s2p2-hidden-size", type=int, default=32)
    parser.add_argument("--s2p2-state-size", type=int, default=8)
    parser.add_argument("--s2p2-num-layers", type=int, default=2)
    parser.add_argument("--s2p2-integral-samples", type=int, default=8)


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
    if args.s2p2_epochs < 1 or args.s2p2_batch_size < 1:
        parser.error("--s2p2-epochs and --s2p2-batch-size must be positive.")
    if args.s2p2_max_sequence_length < 2:
        parser.error("--s2p2-max-sequence-length must be at least 2.")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be positive.")
    if args.num_leaves < 2:
        parser.error("--num-leaves must be at least 2.")
    if args.min_child_samples < 1:
        parser.error("--min-child-samples must be positive.")
    if not 0.0 < args.subsample <= 1.0:
        parser.error("--subsample must be in (0, 1].")
    if not 0.0 < args.colsample_bytree <= 1.0:
        parser.error("--colsample-bytree must be in (0, 1].")
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
