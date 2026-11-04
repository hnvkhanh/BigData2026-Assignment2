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

from experiments.relgt.data import LABELS, GraphTokens, TrainingExamples
from experiments.gnn_no_summer.run import (
    build_parser,
    drop_summer,
    parse_calibration_months,
    parse_summer_months,
    run_backtest,
    split_train_calibration,
    summer_mask,
)


def _day(text: str) -> int:
    return int(pd.Timestamp(text).value // pd.Timedelta(days=1).value)


class SummerMaskTests(unittest.TestCase):
    def test_label_window_touching_summer_is_excluded(self) -> None:
        days = np.array(
            [
                _day("2026-05-20"),  # window ends May 26: kept
                _day("2026-05-27"),  # window reaches Jun 2: excluded
                _day("2026-07-15"),  # excluded
                _day("2026-08-26"),  # window Aug 26 - Sep 1: excluded
                _day("2026-09-01"),  # kept
            ]
        )
        np.testing.assert_array_equal(
            summer_mask(days, (6, 7, 8)), [False, True, True, True, False]
        )

    def test_no_months_keeps_everything(self) -> None:
        days = np.array([_day("2026-07-15")])
        self.assertFalse(summer_mask(days, ()).any())

    def test_drop_summer_fails_when_nothing_is_left(self) -> None:
        days = np.array([_day("2026-07-01"), _day("2026-07-08")])
        with self.assertRaises(ValueError):
            drop_summer(np.arange(2), days, (7,))

    def test_parse_summer_months(self) -> None:
        self.assertEqual(parse_summer_months("8,6,7,7"), (6, 7, 8))
        self.assertEqual(parse_summer_months("none"), ())
        for bad in ("13", "0", "a", ",", "1,2,3,4,5,6,7,8,9,10,11,12"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_summer_months(bad)

    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args(["backtest"])
        self.assertEqual(args.summer_months, (6, 7, 8))
        self.assertIsNone(args.calibration_months)
        args = build_parser().parse_args(["predict", "--calibration-months", "9,4"])
        self.assertEqual(args.calibration_months, (4, 9))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_calibration_months("none")


class CalibrationSplitTests(unittest.TestCase):
    days = np.array(
        [_day(d) for d in ("2026-04-29", "2026-05-06", "2026-09-02", "2026-09-30", "2026-10-07")]
    )

    def test_default_calibrates_on_the_last_usable_week(self) -> None:
        train, calibration = split_train_calibration(
            self.days, np.arange(5), int(self.days[4]), None
        )
        np.testing.assert_array_equal(train, [0, 1, 2, 3])
        np.testing.assert_array_equal(calibration, [4])

    def test_months_select_calibration_weeks_by_cutoff_month(self) -> None:
        train, calibration = split_train_calibration(
            self.days, np.arange(5), int(self.days[4]), (5, 9)
        )
        np.testing.assert_array_equal(train, [0, 4])
        np.testing.assert_array_equal(calibration, [1, 2, 3])

    def test_months_without_usable_weeks_fail(self) -> None:
        with self.assertRaises(ValueError):
            split_train_calibration(self.days, np.arange(5), int(self.days[4]), (1,))


class BacktestTests(unittest.TestCase):
    def test_summer_weeks_never_reach_training_and_holdout_is_saved(self) -> None:
        self._run(calibration_months=None)

    def test_calibration_months_choose_calibration_weeks(self) -> None:
        metrics, trained, cutoff_days = self._run(calibration_months=(9,))
        self.assertEqual(
            metrics["calibration"]["weeks"],
            ["2026-09-02", "2026-09-09", "2026-09-16", "2026-09-23", "2026-09-30"],
        )
        trained_months = pd.to_datetime(cutoff_days[trained].astype(np.int64), unit="D").month
        self.assertNotIn(9, set(trained_months))
        self.assertIn(10, set(trained_months))  # Oct week before holdout now trains

    def _run(self, calibration_months):
        cutoffs = pd.date_range("2026-05-06", "2026-10-28", freq="7D")
        users_per_week = 4
        cutoff_days = np.repeat(
            (cutoffs.asi8 // pd.Timedelta(days=1).value).astype(np.int32), users_per_week
        )
        count = len(cutoff_days)
        user_ids = np.tile(np.arange(1, users_per_week + 1, dtype=np.int32), len(cutoffs))
        rng = np.random.default_rng(0)
        tokens = GraphTokens(
            user_ids=user_ids,
            entity_ids=rng.integers(0, 8, (count, 48)).astype(np.int32),
            node_type_ids=np.zeros((count, 48), dtype=np.uint8),
            relation_ids=np.zeros((count, 48), dtype=np.uint8),
            hop_ids=np.zeros((count, 48), dtype=np.uint8),
            node_features=np.zeros((count, 48, 4), dtype=np.float32),
            relative_days=np.zeros((count, 48), dtype=np.float32),
        )
        labels = rng.integers(0, 2, (count, len(LABELS))).astype(np.uint8)
        examples = TrainingExamples(
            tokens=tokens, labels=labels, cutoff_days=cutoff_days, num_entities=8
        )

        seen_train: list[np.ndarray] = []
        from experiments.relgt import run as relgt_run

        def recording_fit(examples, train_indices, *rest):
            seen_train.append(np.asarray(train_indices))
            return relgt_run.fit_model(examples, train_indices, *rest)

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(
                db_dir=Path(directory),
                output_dir=Path(directory) / "run",
                start_cutoff=cutoffs[0],
                device="cpu",
                summer_months=(6, 7, 8),
                calibration_months=calibration_months,
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
                run_backtest(args)

            trained = seen_train[0]
            self.assertTrue(len(trained))
            self.assertFalse(summer_mask(cutoff_days[trained], (6, 7, 8)).any())
            metrics = json.loads(
                (args.output_dir / "backtest_metrics.json").read_text(encoding="utf-8")
            )
            self.assertGreater(metrics["train_filter"]["excluded_train_examples"], 0)
            self.assertIn("score", metrics["holdout_metrics"])
            self.assertEqual(metrics["holdout_examples"], users_per_week)
            holdout = pd.read_csv(args.output_dir / "holdout_probabilities.csv")
            self.assertEqual(len(holdout), users_per_week)
            for label in LABELS:
                self.assertIn(f"actual_{label}", holdout)
                self.assertTrue(holdout[f"gnn_no_summer_{label}"].between(0, 1).all())
            self.assertTrue((args.output_dir / "gnn_no_summer_model.pt").is_file())
            return metrics, trained, cutoff_days


if __name__ == "__main__":
    unittest.main()
