"""S2P2 point-process adapter for the six GitHub activity marks."""

from __future__ import annotations

import csv
import importlib
import shutil
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

DAY_NS = pd.Timedelta(days=1).value
LOOKBACK_DAYS = 90
PREDICTION_GRID_SIZE = 33
LABELS = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)


@dataclass(frozen=True)
class EventSequence:
    times: np.ndarray
    deltas: np.ndarray
    marks: np.ndarray
    forecast_offset_days: float


def build_event_sequences(
    examples: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    max_sequence_length: int,
    include_future: bool,
) -> list[EventSequence]:
    """Build per-user marked point-process sequences at each example cutoff."""
    events: list[pd.DataFrame] = []
    for mark, label in enumerate(LABELS):
        history = histories[label]
        selected = history[["user_id", "action_at", "available_at"]].copy()
        selected["mark"] = mark
        events.append(selected)

    all_events = pd.concat(events, ignore_index=True)
    all_events = all_events.dropna(
        subset=["user_id", "action_at", "available_at"]
    ).sort_values(["user_id", "action_at"], kind="stable")
    by_user: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for user_id, group in all_events.groupby("user_id", sort=False):
        by_user[int(user_id)] = (
            group["action_at"].astype("int64").to_numpy(),
            group["available_at"].astype("int64").to_numpy(),
            group["mark"].to_numpy(dtype=np.int64),
        )

    sequences: list[EventSequence] = []
    lookback_ns = LOOKBACK_DAYS * DAY_NS
    horizon_ns = 7 * DAY_NS
    for row in examples[["user_id", "cutoff"]].itertuples(index=False):
        user_id = int(row.user_id)
        cutoff_ns = pd.Timestamp(row.cutoff).value
        user_events = by_user.get(user_id)
        if user_events is None:
            action_ns = np.empty(0, dtype=np.int64)
            availability_ns = np.empty(0, dtype=np.int64)
            marks = np.empty(0, dtype=np.int64)
        else:
            action_ns, availability_ns, marks = user_events

        left = np.searchsorted(action_ns, cutoff_ns - lookback_ns, side="left")
        right = np.searchsorted(action_ns, cutoff_ns, side="left")
        context_indices = np.arange(left, right, dtype=np.int64)
        if context_indices.size:
            context_indices = context_indices[
                availability_ns[context_indices] < cutoff_ns
            ]

        selected_indices = context_indices
        if include_future:
            future_end = np.searchsorted(
                action_ns, cutoff_ns + horizon_ns, side="left"
            )
            future_indices = np.arange(right, future_end, dtype=np.int64)
            selected_indices = np.concatenate([context_indices, future_indices])

        selected_indices = selected_indices[-max_sequence_length:]
        selected_times = action_ns[selected_indices]
        selected_marks = marks[selected_indices]

        if selected_times.size == 0:
            sequences.append(
                EventSequence(
                    times=np.zeros(1, dtype=np.float32),
                    deltas=np.zeros(1, dtype=np.float32),
                    marks=np.zeros(1, dtype=np.int64),
                    forecast_offset_days=0.0,
                )
            )
            continue

        if context_indices.size:
            forecast_offset_days = float(
                (cutoff_ns - action_ns[context_indices[-1]]) / DAY_NS
            )
        else:
            forecast_offset_days = 0.0
        relative_times = (
            (selected_times - selected_times[0]).astype(np.float64) / DAY_NS
        ).astype(np.float32)
        deltas = np.diff(relative_times, prepend=relative_times[0]).astype(
            np.float32
        )
        sequences.append(
            EventSequence(
                times=relative_times,
                deltas=deltas,
                marks=selected_marks.astype(np.int64, copy=False),
                forecast_offset_days=forecast_offset_days,
            )
        )
    return sequences


def collate_sequences(
    sequences: list[EventSequence],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot create a batch from an empty event sequence list.")
    batch_size = len(sequences)
    max_length = max(len(sequence.marks) for sequence in sequences)
    times = np.zeros((batch_size, max_length), dtype=np.float32)
    deltas = np.zeros_like(times)
    marks = np.full((batch_size, max_length), pad_token_id, dtype=np.int64)
    mask = np.zeros((batch_size, max_length), dtype=np.float32)
    forecast_offsets = np.zeros(batch_size, dtype=np.float32)
    for index, sequence in enumerate(sequences):
        length = len(sequence.marks)
        times[index, :length] = sequence.times
        deltas[index, :length] = sequence.deltas
        marks[index, :length] = sequence.marks
        mask[index, :length] = 1.0
        forecast_offsets[index] = sequence.forecast_offset_days
        if length < max_length:
            times[index, length:] = sequence.times[-1]
    return (
        torch.as_tensor(times, dtype=torch.float32, device=device),
        torch.as_tensor(deltas, dtype=torch.float32, device=device),
        torch.as_tensor(marks, dtype=torch.long, device=device),
        torch.as_tensor(mask, dtype=torch.float32, device=device),
        torch.as_tensor(forecast_offsets, dtype=torch.float32, device=device),
    )


def select_device(requested: str = "auto") -> torch.device:
    """Choose CPU or the least-loaded CUDA device; S2P2 upstream cautions against MPS."""
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA GPUs available: {cuda_count}")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not cuda_count:
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported S2P2 device: {requested}")
    if not cuda_count:
        if torch.backends.mps.is_available():
            print("S2P2 uses CPU because upstream reports incorrect MPS results.")
        return torch.device("cpu")

    utilization: dict[int, tuple[int, int, int]] = {}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.run(
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
            for row in csv.reader(output.stdout.splitlines(), skipinitialspace=True):
                if len(row) == 5:
                    index = int(row[0])
                    utilization[index] = (int(row[2]), int(row[3]), int(row[4]))
                    print(
                        f"CUDA GPU {index}: {row[1]}; utilization {row[2]}%; "
                        f"memory {row[3]}/{row[4]} MiB"
                    )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            print(f"Warning: could not read nvidia-smi utilization ({error}).")
    else:
        print("Warning: nvidia-smi not found; selecting CUDA GPU by free memory.")

    def load_key(index: int) -> tuple[float, float, int]:
        if index in utilization:
            gpu_util, memory_used, memory_total = utilization[index]
            memory_ratio = memory_used / memory_total if memory_total else 1.0
            return (float(gpu_util), memory_ratio, index)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except RuntimeError as error:
            print(f"Warning: could not query free memory for CUDA GPU {index}: {error}")
            return (100.0, 1.0, index)
        memory_ratio = 1.0 - free_bytes / total_bytes if total_bytes else 1.0
        return (100.0, memory_ratio, index)

    selected_index = min(range(cuda_count), key=load_key)
    if selected_index not in utilization:
        print("GPU utilization unavailable; selected using available memory.")
    print(f"Selected CUDA GPU {selected_index}: {torch.cuda.get_device_name(selected_index)}")
    return torch.device("cuda", selected_index)


def make_model_config(
    device: torch.device,
    hidden_size: int = 32,
    state_size: int = 8,
    num_layers: int = 2,
    integral_samples: int = 8,
) -> SimpleNamespace:
    """Construct the small six-mark config required by upstream EasyTPP S2P2."""
    return SimpleNamespace(
        loss_integral_num_sample_per_step=integral_samples,
        hidden_size=hidden_size,
        num_event_types=len(LABELS),
        num_event_types_pad=len(LABELS) + 1,
        pad_token_id=len(LABELS),
        thinning=None,
        gpu=(device.index or 0) if device.type == "cuda" else -1,
        use_mc_samples=True,
        num_layers=num_layers,
        model_specs={
            "P": state_size,
            "dropout_rate": 0.1,
            "act_func": "gelu",
            "for_loop": False,
            "pre_norm": False,
            "post_norm": True,
            "int_backward_variant": True,
            "relative_time": True,
        },
    )


class S2P2TemporalModel:
    """Train official EasyTPP S2P2 and estimate six next-week mark risks."""

    def __init__(
        self,
        device: torch.device,
        epochs: int = 3,
        batch_size: int = 128,
        max_sequence_length: int = 128,
        hidden_size: int = 32,
        state_size: int = 8,
        num_layers: int = 2,
        integral_samples: int = 8,
        seed: int = 42,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.integral_samples = integral_samples
        self.seed = seed
        self.model: torch.nn.Module | None = None

    def _new_model(self) -> torch.nn.Module:
        try:
            module = importlib.import_module(
                "easy_tpp.model.torch_model.torch_s2p2"
            )
        except ImportError as error:
            raise ImportError(
                "S2P2 requires the optional upstream EasyTPP dependency. Install "
                "it with: python -m pip install -r "
                "experiments/lightgbm_baseline/s2p2-requirements.txt"
            ) from error
        model_class = getattr(module, "S2P2")
        model = model_class(
            make_model_config(
                self.device,
                self.hidden_size,
                self.state_size,
                self.num_layers,
                self.integral_samples,
            )
        )
        return model.to(self.device)

    def fit(
        self,
        examples: pd.DataFrame,
        histories: dict[str, pd.DataFrame],
    ) -> "S2P2TemporalModel":
        if examples.empty:
            raise ValueError("S2P2 training examples are empty.")
        if self.epochs < 1 or self.batch_size < 1 or self.max_sequence_length < 2:
            raise ValueError("S2P2 epochs/batch size must be positive and sequence length >= 2.")

        if self.device.type == "cpu":
            torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model = self._new_model()
        sequences = build_event_sequences(
            examples, histories, self.max_sequence_length, include_future=True
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        rng = np.random.default_rng(self.seed)
        sequences_by_length: dict[int, list[int]] = {}
        for index, sequence in enumerate(sequences):
            sequences_by_length.setdefault(len(sequence.marks), []).append(index)
        for epoch in range(self.epochs):
            self.model.train()
            batches: list[list[int]] = []
            for indices in sequences_by_length.values():
                shuffled = rng.permutation(indices)
                batches.extend(
                    shuffled[start : start + self.batch_size].tolist()
                    for start in range(0, len(shuffled), self.batch_size)
                )
            rng.shuffle(batches)
            epoch_loss = 0.0
            batch_count = 0
            for batch_indices in batches:
                selected = [sequences[index] for index in batch_indices]
                times, deltas, marks, mask, _ = collate_sequences(
                    selected, len(LABELS), self.device
                )
                optimizer.zero_grad(set_to_none=True)
                loss, num_events, *_ = self.model.loglike_loss(
                    (times, deltas, marks, mask, None)
                )
                event_count = int(num_events)
                if event_count == 0:
                    continue
                normalized_loss = loss / event_count
                if not torch.isfinite(normalized_loss):
                    raise FloatingPointError(
                        f"S2P2 produced a non-finite loss during epoch {epoch + 1}."
                    )
                normalized_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(normalized_loss.detach().cpu())
                batch_count += 1
            if not batch_count:
                raise ValueError("S2P2 training found no sequences with likelihood events.")
            print(
                f"S2P2 epoch {epoch + 1}/{self.epochs}: "
                f"mean normalized NLL={epoch_loss / batch_count:.5f}"
            )
        return self

    def predict(
        self,
        examples: pd.DataFrame,
        histories: dict[str, pd.DataFrame],
    ) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Fit the S2P2 model before requesting predictions.")
        sequences = build_event_sequences(
            examples, histories, self.max_sequence_length, include_future=False
        )
        grid = torch.linspace(
            0.0, 7.0, PREDICTION_GRID_SIZE, dtype=torch.float32, device=self.device
        )
        result = np.zeros((len(sequences), len(LABELS)), dtype=np.float32)
        sequences_by_length: dict[int, list[int]] = {}
        for index, sequence in enumerate(sequences):
            sequences_by_length.setdefault(len(sequence.marks), []).append(index)
        self.model.eval()
        with torch.inference_mode():
            for indices in sequences_by_length.values():
                for start in range(0, len(indices), self.batch_size):
                    batch_indices = indices[start : start + self.batch_size]
                    batch_sequences = [sequences[index] for index in batch_indices]
                    times, deltas, marks, _, forecast_offsets = collate_sequences(
                        batch_sequences, len(LABELS), self.device
                    )
                    sample_deltas = (
                        forecast_offsets[:, None, None] + grid.view(1, 1, -1)
                    )
                    intensities = self.model.compute_intensities_at_sample_times(
                        times,
                        deltas,
                        marks,
                        sample_deltas,
                        compute_last_step_only=True,
                    )
                    hazard = torch.trapezoid(intensities[:, 0], x=grid, dim=1)
                    probabilities = -torch.expm1(-hazard.clamp(min=0))
                    if not torch.isfinite(probabilities).all():
                        raise FloatingPointError(
                            "S2P2 produced non-finite weekly activity probabilities."
                        )
                    result[batch_indices] = probabilities.cpu().numpy()
        return {
            label: result[:, index] for index, label in enumerate(LABELS)
        }

    def save(self, path: Any) -> None:
        if self.model is None:
            raise RuntimeError("Fit the S2P2 model before saving.")
        torch.save(self.model.state_dict(), path)
