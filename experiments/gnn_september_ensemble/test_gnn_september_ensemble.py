from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.gnn_september_ensemble.run import BRANCHES, run_backtest
from experiments.relgt.data import LABELS, GraphTokens, TrainingExamples


def _examples(cutoffs: pd.DatetimeIndex, per_week: int) -> TrainingExamples:
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
    return TrainingExamples(
        tokens=tokens,
        labels=rng.integers(0, 2, (count, len(LABELS))).astype(np.uint8),
        cutoff_days=cutoff_days,
        num_entities=8,
    )


class EnsembleBacktestTests(unittest.TestCase):
    def _run(self, fusion_weight):
        cutoffs = pd.date_range("2026-05-06", "2026-10-28", freq="7D")
        examples = _examples(cutoffs, per_week=4)
        from experiments.relgt import run as relgt_run

        seen: list[np.ndarray] = []

        def recording_fit(examples, train_indices, *rest):
            seen.append(np.asarray(train_indices))
            return relgt_run.fit_model(examples, train_indices, *rest)

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        args = SimpleNamespace(
            db_dir=Path(directory.name),
            output_dir=Path(directory.name) / "run",
            start_cutoff=cutoffs[0],
            device="cpu",
            train_months=(9,),
            fusion_weight=fusion_weight,
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
        module = "experiments.gnn_september_ensemble.run"
        with (
            patch(f"{module}.load_or_build_examples", return_value=examples),
            patch(f"{module}.select_device", return_value=torch.device("cpu")),
            patch("experiments.gnn_no_summer.run.fit_model", side_effect=recording_fit),
        ):
            run_backtest(args)
        metrics = json.loads(
            (args.output_dir / "backtest_metrics.json").read_text(encoding="utf-8")
        )
        return args, examples, seen, metrics

    def test_branches_see_all_weeks_and_september_only(self) -> None:
        args, examples, seen, metrics = self._run(fusion_weight=None)
        all_rows, september_rows = seen
        days = examples.cutoff_days
        holdout_start, calibration_day = np.unique(days)[-1], np.unique(days)[-2]
        self.assertEqual(int(days[all_rows].max()), int(np.unique(days)[-3]))
        self.assertNotIn(calibration_day, days[all_rows])
        self.assertNotIn(holdout_start, days[all_rows])
        months = pd.to_datetime(days[september_rows].astype(np.int64), unit="D").month
        self.assertEqual(set(months), {9})
        self.assertEqual(len(all_rows), len(np.flatnonzero(days < calibration_day)))

        self.assertEqual(set(metrics["branch_holdout"]), {*BRANCHES, "ensemble"})
        weight = metrics["fusion"]["gnn_all_weight"]
        self.assertTrue(0.0 <= weight <= 1.0)
        holdout = pd.read_csv(args.output_dir / "holdout_probabilities.csv")
        for label in LABELS:
            np.testing.assert_allclose(
                holdout[f"ensemble_{label}"],
                weight * holdout[f"gnn_all_{label}"]
                + (1 - weight) * holdout[f"gnn_september_{label}"],
                rtol=1e-5,
            )
        for branch in BRANCHES:
            self.assertTrue((args.output_dir / f"{branch}_model.pt").is_file())

    def test_manual_fusion_weight_is_used(self) -> None:
        _, _, _, metrics = self._run(fusion_weight=0.3)
        self.assertEqual(metrics["fusion"]["gnn_all_weight"], 0.3)
        self.assertEqual(metrics["fusion"]["method"], "manual probability averaging")


if __name__ == "__main__":
    unittest.main()
