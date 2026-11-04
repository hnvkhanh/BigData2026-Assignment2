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

from experiments.gnn_no_summer.run import run_backtest
from experiments.gnn_september.run import EXPERIMENT, month_filter, parse_train_months
from experiments.relgt.data import LABELS, GraphTokens, TrainingExamples


def _day(text: str) -> int:
    return int(pd.Timestamp(text).value // pd.Timedelta(days=1).value)


class MonthFilterTests(unittest.TestCase):
    def test_keeps_only_cutoffs_in_the_months(self) -> None:
        days = np.array(
            [_day(d) for d in ("2026-08-26", "2026-09-02", "2026-09-30", "2026-10-07")]
        )
        kept, dropped, record = month_filter((9,))(np.arange(4), days)
        np.testing.assert_array_equal(kept, [1, 2])
        np.testing.assert_array_equal(dropped, [0, 3])
        self.assertEqual(record["kept_train_weeks"], ["2026-09-02", "2026-09-30"])

    def test_fails_when_no_week_matches(self) -> None:
        with self.assertRaises(ValueError):
            month_filter((12,))(np.arange(1), np.array([_day("2026-09-02")]))

    def test_parse(self) -> None:
        self.assertEqual(parse_train_months("9"), (9,))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_train_months("13")


class BacktestTests(unittest.TestCase):
    def test_trains_only_on_september_and_scores_holdout(self) -> None:
        cutoffs = pd.date_range("2026-05-06", "2026-10-28", freq="7D")
        per_week = 4
        cutoff_days = np.repeat(
            (cutoffs.asi8 // pd.Timedelta(days=1).value).astype(np.int32), per_week
        )
        count = len(cutoff_days)
        rng = np.random.default_rng(0)
        tokens = GraphTokens(
            user_ids=np.tile(np.arange(1, per_week + 1, dtype=np.int32), len(cutoffs)),
            entity_ids=rng.integers(0, 8, (count, 48)).astype(np.int32),
            node_type_ids=np.zeros((count, 48), dtype=np.uint8),
            relation_ids=np.zeros((count, 48), dtype=np.uint8),
            hop_ids=np.zeros((count, 48), dtype=np.uint8),
            node_features=np.zeros((count, 48, 4), dtype=np.float32),
            relative_days=np.zeros((count, 48), dtype=np.float32),
        )
        examples = TrainingExamples(
            tokens=tokens,
            labels=rng.integers(0, 2, (count, len(LABELS))).astype(np.uint8),
            cutoff_days=cutoff_days,
            num_entities=8,
        )
        from experiments.relgt import run as relgt_run

        seen: list[np.ndarray] = []

        def recording_fit(examples, train_indices, *rest):
            seen.append(np.asarray(train_indices))
            return relgt_run.fit_model(examples, train_indices, *rest)

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(
                db_dir=Path(directory),
                output_dir=Path(directory) / "run",
                start_cutoff=cutoffs[0],
                device="cpu",
                calibration_months=None,
                eligibility_days=None,
                user_embedding=False,
                train_fraction=1.0,
                discard_unused_train=False,
                epochs=1,
                batch_size=8,
                hidden_dim=8,
                num_layers=1,
                dropout=0.0,
                seed=1,
            )
            args.output_dir.mkdir()
            module = "experiments.gnn_no_summer.run"
            with (
                patch(f"{module}.load_or_build_examples", return_value=examples),
                patch(f"{module}.select_device", return_value=torch.device("cpu")),
                patch(f"{module}.fit_model", side_effect=recording_fit),
            ):
                run_backtest(args, month_filter((9,)), EXPERIMENT)

            months = pd.to_datetime(cutoff_days[seen[0]].astype(np.int64), unit="D").month
            self.assertEqual(set(months), {9})
            metrics = json.loads(
                (args.output_dir / "backtest_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["experiment"], EXPERIMENT)
            self.assertEqual(metrics["train_filter"]["train_months"], [9])
            self.assertEqual(metrics["calibration"]["weeks"], ["2026-10-21"])
            self.assertIn("score", metrics["holdout_metrics"])
            holdout = pd.read_csv(args.output_dir / "holdout_probabilities.csv")
            self.assertIn(f"{EXPERIMENT}_push", holdout)
            self.assertTrue((args.output_dir / f"{EXPERIMENT}_model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
