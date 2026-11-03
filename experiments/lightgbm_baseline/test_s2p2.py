import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.lightgbm_baseline import run as baseline_run

try:
    import torch

    from experiments.lightgbm_baseline import s2p2
    from experiments.lightgbm_baseline.s2p2 import (
        build_event_sequences,
        collate_sequences,
    )
except ImportError:
    S2P2_AVAILABLE = False
else:
    S2P2_AVAILABLE = True

LABELS = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)


def empty_histories() -> dict[str, pd.DataFrame]:
    return {
        label: pd.DataFrame(
            {
                "user_id": pd.Series(dtype="int64"),
                "action_at": pd.Series(dtype="datetime64[ns]"),
                "available_at": pd.Series(dtype="datetime64[ns]"),
            }
        )
        for label in LABELS
    }


@unittest.skipUnless(S2P2_AVAILABLE, "install the optional S2P2 requirements to run")
class S2P2SequenceTests(unittest.TestCase):
    def test_builds_time_ordered_leakage_safe_marked_sequence(self) -> None:
        histories = empty_histories()
        histories["push"] = pd.DataFrame(
            {
                "user_id": [7, 7, 7],
                "action_at": pd.to_datetime(
                    ["2024-01-06", "2024-01-07", "2024-01-10"]
                ),
                "available_at": pd.to_datetime(
                    ["2024-01-06", "2024-01-09", "2024-01-10"]
                ),
            }
        )
        histories["open_pr"] = pd.DataFrame(
            {
                "user_id": [7],
                "action_at": pd.to_datetime(["2024-01-05"]),
                "available_at": pd.to_datetime(["2024-01-05"]),
            }
        )
        examples = pd.DataFrame(
            {"user_id": [7], "cutoff": [pd.Timestamp("2024-01-08")]}
        )

        sequence = build_event_sequences(
            examples, histories, max_sequence_length=10, include_future=True
        )[0]

        self.assertEqual(sequence.marks.tolist(), [0, 5, 5])
        self.assertTrue(np.all(np.diff(sequence.times) >= 0))
        self.assertAlmostEqual(sequence.forecast_offset_days, 2.0)

    def test_caps_sequence_and_collates_masks_and_forecast_offset(self) -> None:
        histories = empty_histories()
        histories["push"] = pd.DataFrame(
            {
                "user_id": [7, 7, 7],
                "action_at": pd.to_datetime(
                    ["2024-01-01", "2024-01-03", "2024-01-05"]
                ),
                "available_at": pd.to_datetime(
                    ["2024-01-01", "2024-01-03", "2024-01-05"]
                ),
            }
        )
        examples = pd.DataFrame(
            {
                "user_id": [7, 8],
                "cutoff": [pd.Timestamp("2024-01-08")] * 2,
            }
        )

        sequences = build_event_sequences(
            examples, histories, max_sequence_length=2, include_future=False
        )
        times, deltas, marks, mask, offsets = collate_sequences(
            sequences, pad_token_id=len(LABELS), device=torch.device("cpu")
        )

        self.assertEqual(tuple(times.shape), (2, 2))
        self.assertEqual(marks[0].tolist(), [5, 5])
        self.assertEqual(marks[1].tolist(), [0, len(LABELS)])
        self.assertEqual(mask.tolist(), [[1.0, 1.0], [1.0, 0.0]])
        self.assertEqual(float(offsets[0]), 3.0)
        self.assertTrue(torch.isfinite(deltas).all())

    def test_selects_cuda_gpu_with_lowest_utilization(self) -> None:
        completed = SimpleNamespace(
            stdout="0, GPU 0, 60, 300, 1000\n1, GPU 1, 15, 700, 1000\n"
        )
        with (
            patch.object(s2p2.torch.cuda, "is_available", return_value=True),
            patch.object(s2p2.torch.cuda, "device_count", return_value=2),
            patch.object(
                s2p2.torch.cuda,
                "get_device_name",
                side_effect=lambda index: f"GPU {index}",
            ),
            patch.object(s2p2.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(s2p2.subprocess, "run", return_value=completed),
        ):
            device = s2p2.select_device("auto")
        self.assertEqual(device, torch.device("cuda", 1))


class CliTests(unittest.TestCase):
    def test_cli_accepts_s2p2_selection(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "experiments/lightgbm_baseline/run.py",
                "backtest",
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(
            "--temporal-model {lightgbm,frequency_nn,s2p2}", result.stdout
        )
        self.assertIn("--nn-output-mode {per_label,multi_label}", result.stdout)
        self.assertIn("--calibrated-threshold", result.stdout)
        self.assertIn("--device {auto,cpu,cuda}", result.stdout)

    def test_default_run_directories_are_timestamped_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            script = (
                "from pathlib import Path; "
                "from experiments.lightgbm_baseline import run; "
                "run.DEFAULT_OUTPUT_DIR = Path(__import__('sys').argv[1]); "
                "print(run.create_run_output_dir('backtest', 's2p2').name); "
                "print(run.create_run_output_dir('backtest', 's2p2').name)"
            )
            result = subprocess.run(
                [sys.executable, "-c", script, temporary_dir],
                check=True,
                capture_output=True,
                text=True,
            )
        first, second = result.stdout.splitlines()
        self.assertNotEqual(first, second)
        self.assertIn("backtest_s2p2_", first)
        self.assertIn("backtest_s2p2_", second)


if __name__ == "__main__":
    unittest.main()
