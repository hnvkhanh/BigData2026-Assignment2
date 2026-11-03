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
from experiments.relgt_lookback_chunks.run import parse_lookbacks, run_backtest

EVENT_COLUMNS = ["user_id", "action_at", "available_at", "entity_available_at"]


class ParseLookbacksTests(unittest.TestCase):
    def test_parses_three_positive_integers(self) -> None:
        self.assertEqual(parse_lookbacks("180,21,7"), (180, 21, 7))

    def test_rejects_wrong_count_or_non_positive(self) -> None:
        for value in ("180,21", "1,2,3,4", "180,0,7", "a,b,c"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                parse_lookbacks(value)


class BacktestTests(unittest.TestCase):
    def test_each_chunk_model_uses_its_own_lookback(self) -> None:
        sample_count = 30
        user_ids = np.arange(1, sample_count + 1, dtype=np.int32)
        cutoff_dates = pd.date_range("2026-03-01", periods=10, freq="7D")
        cutoff_days = np.tile(
            (cutoff_dates.asi8 // pd.Timedelta(days=1).value).astype(np.int32), 3
        )
        cutoff_days.sort()
        labels = np.tile(
            np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.uint8),
            (sample_count // 2, 1),
        )

        def examples_for(lookback_days: int) -> TrainingExamples:
            tokens = GraphTokens(
                user_ids=user_ids,
                entity_ids=np.zeros((sample_count, 48), dtype=np.int32),
                node_type_ids=np.zeros((sample_count, 48), dtype=np.uint8),
                relation_ids=np.zeros((sample_count, 48), dtype=np.uint8),
                hop_ids=np.zeros((sample_count, 48), dtype=np.uint8),
                node_features=np.zeros((sample_count, 48, 4), dtype=np.float32),
                relative_days=np.zeros((sample_count, 48), dtype=np.float32),
            )
            return TrainingExamples(
                tokens=tokens, labels=labels, cutoff_days=cutoff_days, num_entities=8
            )

        requested: list[tuple[int, int]] = []

        def fake_load(db_dir, output_dir, start_cutoff, lookback_days, **kwargs):
            requested.append((lookback_days, kwargs["eligibility_lookback_days"]))
            return examples_for(lookback_days)

        event_rows = []
        for index, (user_id, day) in enumerate(zip(user_ids, cutoff_days)):
            cutoff = pd.Timestamp(int(day), unit="D") - pd.Timedelta(days=1)
            event_rows.extend([(user_id, cutoff, cutoff, cutoff)] * (1, 3, 6)[index % 3])
        events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
        for column in EVENT_COLUMNS[1:]:
            events[column] = pd.to_datetime(events[column])
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
                lookbacks=(180, 21, 7),
                global_lookback=14,
                train_fraction=1.0,
                balance_chunks=False,
                user_embedding=False,
                epochs=1,
                batch_size=8,
                hidden_dim=12,
                num_layers=1,
                dropout=0.0,
                learning_rate=1e-3,
                seed=1,
            )
            module = "experiments.relgt_lookback_chunks.run"
            with (
                patch(f"{module}.load_or_build_examples", side_effect=fake_load),
                patch(
                    f"{module}.load_entity_vocabulary",
                    return_value=EntityVocabulary(2, 2, 2),
                ),
                patch(f"{module}.load_events", return_value=events),
                patch(f"{module}.select_device", return_value=torch.device("cpu")),
                patch(f"{module}.pd.read_parquet", return_value=users),
            ):
                run_backtest(args)

            # One example set per distinct lookback, all eligible on the longest.
            self.assertEqual(
                sorted(requested), [(7, 180), (14, 180), (21, 180), (180, 180)]
            )
            for name in ("low", "mid", "high", "global"):
                self.assertTrue((args.output_dir / f"{name}_model.pt").is_file())
            metrics = json.loads(
                (args.output_dir / "backtest_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metrics["lookback_days"],
                {"low": 180, "mid": 21, "high": 7, "global": 14},
            )
            self.assertEqual(metrics["eligibility_lookback_days"], 180)
            self.assertEqual(metrics["architecture"], "gnn")
            self.assertEqual(
                metrics["per_chunk_holdout"]["mid"]["lookback_days"], 21
            )
            holdout = pd.read_csv(args.output_dir / "holdout_probabilities.csv")
            for name in ("low", "mid", "high"):
                rows = holdout["chunk"] == name
                for label in LABELS:
                    np.testing.assert_allclose(
                        holdout.loc[rows, f"chunked_{label}"],
                        holdout.loc[rows, f"{name}_{label}"],
                    )


if __name__ == "__main__":
    unittest.main()
