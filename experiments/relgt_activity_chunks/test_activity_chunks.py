from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.relgt.data import (
    LABELS,
    EntityVocabulary,
    GraphTokens,
    TrainingExamples,
)
from experiments.relgt_activity_chunks.run import (
    assign_chunks,
    auto_thresholds,
    parse_chunk_thresholds,
    prior_action_counts,
    require_populated_chunks,
    run_backtest,
)

EVENT_COLUMNS = [
    "user_id",
    "action_at",
    "available_at",
    "entity_available_at",
]


def _events(rows: list[tuple]) -> pd.DataFrame:
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    for column in EVENT_COLUMNS[1:]:
        events[column] = pd.to_datetime(events[column])
    return events


class ActivityCountTests(unittest.TestCase):
    def test_counts_only_actions_known_strictly_before_cutoff(self) -> None:
        cutoff = pd.Timestamp("2026-01-15")
        cutoff_day = np.int32(cutoff.value // pd.Timedelta(days=1).value)
        events = _events(
            [
                (1, "2026-01-01", "2026-01-01", "2026-01-01"),  # counted
                (1, "2026-01-14", "2026-01-14", "2026-01-14"),  # counted
                (1, "2026-01-15", "2026-01-15", "2026-01-15"),  # at cutoff
                (1, "2026-01-10", "2026-01-16", "2026-01-10"),  # row not yet known
                (1, "2026-01-10", "2026-01-10", "2026-01-20"),  # entity not yet known
                (2, "2026-01-02", "2026-01-02", "2026-01-02"),  # other user
            ]
        )
        users = np.array([1, 2, 3], dtype=np.int32)
        days = np.full(3, cutoff_day, dtype=np.int32)

        self.assertEqual(prior_action_counts(events, users, days).tolist(), [2, 1, 0])
        self.assertEqual(
            prior_action_counts(events, users, days, window_days=5).tolist(),
            [1, 0, 0],
        )

    def test_rejects_negative_window(self) -> None:
        with self.assertRaises(ValueError):
            prior_action_counts(
                _events([(1, "2026-01-01", "2026-01-01", "2026-01-01")]),
                np.array([1], dtype=np.int32),
                np.array([20000], dtype=np.int32),
                window_days=-1,
            )


class ChunkAssignmentTests(unittest.TestCase):
    def test_boundaries_are_half_open(self) -> None:
        counts = np.array([0, 9, 10, 99, 100, 5000])
        self.assertEqual(
            assign_chunks(counts, (10, 100)).tolist(), [0, 0, 1, 1, 2, 2]
        )

    def test_threshold_parsing(self) -> None:
        self.assertIsNone(parse_chunk_thresholds("auto"))
        self.assertEqual(parse_chunk_thresholds("10, 100"), (10, 100))
        for bad in ("10", "100,10", "0,5", "a,b", "1,2,3"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_chunk_thresholds(bad)

    def test_auto_thresholds_use_given_counts_and_reject_degenerate(self) -> None:
        self.assertEqual(auto_thresholds(np.arange(1, 100)), (34, 67))
        with self.assertRaises(ValueError):
            auto_thresholds(np.zeros(50))

    def test_empty_chunk_fails_clearly(self) -> None:
        sizes = {
            "low": {"train": 3, "calibration": 1, "holdout": 1},
            "mid": {"train": 3, "calibration": 0, "holdout": 1},
            "high": {"train": 3, "calibration": 1, "holdout": 1},
        }
        with self.assertRaisesRegex(ValueError, "mid"):
            require_populated_chunks(sizes, (10, 100))


class BacktestTests(unittest.TestCase):
    def test_backtest_saves_chunk_models_and_aligned_holdout(self) -> None:
        sample_count = 30
        user_ids = np.arange(1, sample_count + 1, dtype=np.int32)
        cutoff_dates = pd.date_range("2026-03-01", periods=10, freq="7D")
        cutoff_days = np.tile(
            (cutoff_dates.asi8 // pd.Timedelta(days=1).value).astype(np.int32), 3
        )
        cutoff_days.sort()
        tokens = GraphTokens(
            user_ids=user_ids,
            entity_ids=np.zeros((sample_count, 48), dtype=np.int32),
            node_type_ids=np.zeros((sample_count, 48), dtype=np.uint8),
            relation_ids=np.zeros((sample_count, 48), dtype=np.uint8),
            hop_ids=np.zeros((sample_count, 48), dtype=np.uint8),
            node_features=np.zeros((sample_count, 48, 4), dtype=np.float32),
            relative_days=np.zeros((sample_count, 48), dtype=np.float32),
        )
        labels = np.tile(
            np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.uint8),
            (sample_count // 2, 1),
        )
        examples = TrainingExamples(
            tokens=tokens, labels=labels, cutoff_days=cutoff_days, num_entities=8
        )
        # Every user gets 1, 3, or 6 earlier actions so each chunk is populated
        # in every partition.
        event_rows = []
        for index, (user_id, day) in enumerate(zip(user_ids, cutoff_days)):
            cutoff = pd.Timestamp(int(day), unit="D")
            for _ in range((1, 3, 6)[index % 3]):
                event_rows.append(
                    (
                        user_id,
                        cutoff - pd.Timedelta(days=1),
                        cutoff - pd.Timedelta(days=1),
                        cutoff - pd.Timedelta(days=1),
                    )
                )
        events = _events(event_rows)
        users = pd.DataFrame(
            {"user_id": user_ids, "is_cohort": np.ones(sample_count, dtype=bool)}
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = SimpleNamespace(
                db_dir=root,
                output_dir=root / "run",
                start_cutoff=cutoff_dates[0],
                device="cpu",
                chunk_thresholds=(2, 5),
                activity_window_days=0,
                inspect_only=False,
                no_global_baseline=False,
                epochs=1,
                batch_size=8,
                hidden_dim=12,
                num_layers=1,
                dropout=0.0,
                learning_rate=1e-3,
                seed=1,
            )
            module = "experiments.relgt_activity_chunks.run"
            with (
                patch(f"{module}.load_or_build_examples", return_value=examples),
                patch(
                    f"{module}.load_entity_vocabulary",
                    return_value=EntityVocabulary(2, 2, 2),
                ),
                patch(f"{module}.load_events", return_value=events),
                patch(f"{module}.select_device", return_value=torch.device("cpu")),
                patch(f"{module}.pd.read_parquet", return_value=users),
            ):
                run_backtest(args)

            for name in ("low", "mid", "high", "global"):
                self.assertTrue((args.output_dir / f"{name}_model.pt").is_file())
            holdout = pd.read_csv(args.output_dir / "holdout_probabilities.csv")
            metrics = json.loads(
                (args.output_dir / "backtest_metrics.json").read_text(encoding="utf-8")
            )

            self.assertEqual(set(holdout["chunk"]), {"low", "mid", "high"})
            for label in LABELS:
                self.assertIn(f"actual_{label}", holdout)
                for branch in ("chunked", "low", "mid", "high", "global"):
                    column = holdout[f"{branch}_{label}"]
                    self.assertTrue(column.between(0.0, 1.0).all())
            # Chunked probabilities come from the model that owns each row.
            for name in ("low", "mid", "high"):
                rows = holdout["chunk"] == name
                for label in LABELS:
                    np.testing.assert_allclose(
                        holdout.loc[rows, f"chunked_{label}"],
                        holdout.loc[rows, f"{name}_{label}"],
                    )
            expected_actual = labels[holdout["user_id"].to_numpy() - 1]
            np.testing.assert_array_equal(
                holdout[[f"actual_{label}" for label in LABELS]].to_numpy(),
                expected_actual,
            )
            self.assertEqual(metrics["chunking"]["thresholds"], [2, 5])
            self.assertEqual(metrics["chunking"]["threshold_source"], "manual")
            self.assertEqual(
                set(metrics["thresholds"]), {"low", "mid", "high", "global"}
            )
            self.assertEqual(set(metrics["per_chunk_holdout"]), {"low", "mid", "high"})
            self.assertIn("global", metrics["branch_holdout"])
            self.assertEqual(
                sum(
                    chunk["holdout_examples"]
                    for chunk in metrics["per_chunk_holdout"].values()
                ),
                metrics["holdout_examples"],
            )


if __name__ == "__main__":
    unittest.main()
