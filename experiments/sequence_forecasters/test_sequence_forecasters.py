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

from experiments.relgt.data import LABELS, GraphTokens, TrainingExamples
from experiments.sequence_forecasters.features import (
    AUX_COLUMNS,
    COUNT_CHANNELS,
    DAY_NS,
    SEQUENCE_CHANNELS,
    ChannelScaler,
    build_daily_sequences,
    calendar_features,
    example_cutoff_ns,
)
from experiments.sequence_forecasters.model import (
    FEDformerForecaster,
    InformerForecaster,
    LSTMForecaster,
)
from experiments.sequence_forecasters.run import (
    BRANCHES,
    build_parser,
    fuse,
    run_backtest,
    simplex_grid,
    validate_args,
)

CUTOFF = pd.Timestamp("2026-09-02 16:00:00")


def _events(rows):
    """rows: (user_id, relation_id, action_at, available_at, entity_available_at)."""
    frame = pd.DataFrame(
        rows,
        columns=["user_id", "relation_id", "action_at", "available_at", "entity_available_at"],
    )
    for column in ("action_at", "available_at", "entity_available_at"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


class SequenceFeatureTests(unittest.TestCase):
    def _build(self, rows, window_days=3):
        return build_daily_sequences(
            _events(rows),
            np.array([1]),
            np.array([CUTOFF.value], dtype=np.int64),
            window_days,
        )[0]

    def test_bins_end_exactly_at_cutoff(self) -> None:
        hour = pd.Timedelta(hours=1)
        day = pd.Timedelta(days=1)
        rows = [
            (1, 1, CUTOFF - hour, CUTOFF - hour, CUTOFF - hour),  # last bin
            (1, 1, CUTOFF - day - hour, CUTOFF - hour, CUTOFF - hour),  # middle bin
            (1, 6, CUTOFF - 3 * day, CUTOFF - hour, CUTOFF - hour),  # oldest bin start
            (1, 2, CUTOFF, CUTOFF, CUTOFF),  # at cutoff: label window, not history
            (1, 2, CUTOFF - 3 * day - hour, CUTOFF - hour, CUTOFF - hour),  # too old
        ]
        sequence = self._build(rows)
        counts = np.expm1(sequence[:, : len(LABELS)])
        np.testing.assert_allclose(counts[:, 0], [0, 1, 1])
        np.testing.assert_allclose(counts[:, 5], [1, 0, 0])
        np.testing.assert_allclose(counts[:, 1], [0, 0, 0])
        np.testing.assert_allclose(sequence[:, len(LABELS) + 1], [1, 1, 1])
        np.testing.assert_allclose(np.expm1(sequence[:, len(LABELS)]), [1, 1, 1])

    def test_unavailable_rows_and_entities_are_excluded(self) -> None:
        hour = pd.Timedelta(hours=1)
        rows = [
            (1, 1, CUTOFF - hour, CUTOFF, CUTOFF - hour),  # row published at cutoff
            (1, 1, CUTOFF - hour, CUTOFF - hour, CUTOFF + hour),  # entity later
            (2, 1, CUTOFF - hour, CUTOFF - hour, CUTOFF - hour),  # another user
        ]
        sequence = self._build(rows)
        self.assertEqual(float(sequence[:, : len(COUNT_CHANNELS)].sum()), 0.0)
        self.assertEqual(sequence.shape, (3, len(SEQUENCE_CHANNELS)))

    def test_exact_cutoffs_and_calendar(self) -> None:
        start = pd.Timestamp("2026-03-11 16:00:00")
        cutoffs = pd.date_range(start, periods=3, freq="7D")
        days = (cutoffs.asi8 // DAY_NS).astype(np.int32)
        np.testing.assert_array_equal(example_cutoff_ns(days, start), cutoffs.asi8)
        with self.assertRaises(ValueError):
            example_cutoff_ns(days + 1, start)
        aux = calendar_features(cutoffs.asi8)
        self.assertEqual(aux.shape, (3, len(AUX_COLUMNS)))
        np.testing.assert_allclose(
            aux[0, :2], [np.sin(2 * np.pi * 3 / 12), np.cos(2 * np.pi * 3 / 12)], rtol=1e-6
        )

    def test_scaler_uses_given_rows_and_leaves_calendar(self) -> None:
        rng = np.random.default_rng(0)
        sequences = rng.normal(3.0, 2.0, (20, 5, len(SEQUENCE_CHANNELS))).astype(np.float32)
        calendar = sequences[..., len(COUNT_CHANNELS) :].copy()
        scaler = ChannelScaler.fit(sequences[:10])
        scaled = scaler.transform_(sequences)
        flat = scaled[:10, :, : len(COUNT_CHANNELS)].reshape(-1, len(COUNT_CHANNELS))
        np.testing.assert_allclose(flat.mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(flat.std(0), 1, atol=1e-4)
        np.testing.assert_array_equal(scaled[..., len(COUNT_CHANNELS) :], calendar)


class ModelShapeTests(unittest.TestCase):
    def _inputs(self, batch=4, days=21):
        torch.manual_seed(0)
        return (
            torch.randn(batch, days, len(SEQUENCE_CHANNELS)),
            torch.randn(batch, len(AUX_COLUMNS)),
            torch.tensor([10, 11, 12, 99]),
        )

    def _models(self, known):
        channels = len(SEQUENCE_CHANNELS)
        return {
            "lstm": LSTMForecaster(
                channels, len(AUX_COLUMNS), hidden_size=8, user_emb_dim=4, known_user_ids=known
            ),
            "informer_prob": InformerForecaster(
                channels,
                21,
                label_days=7,
                d_model=8,
                n_heads=2,
                d_ff=16,
                user_emb_dim=4,
                known_user_ids=known,
            ),
            "informer_full": InformerForecaster(
                channels,
                21,
                label_days=7,
                d_model=8,
                n_heads=2,
                d_ff=16,
                attn="full",
                distil=False,
                user_emb_dim=4,
                known_user_ids=known,
            ),
            "fedformer": FEDformerForecaster(
                channels,
                21,
                d_model=8,
                d_ff=16,
                modes=4,
                moving_avg=(3, 7),
                user_emb_dim=4,
                known_user_ids=known,
            ),
        }

    def test_every_model_returns_one_logit_per_label_and_trains(self) -> None:
        for known in (None, np.array([10, 11, 12])):
            for name, model in self._models(known).items():
                with self.subTest(model=name, user_embedding=known is not None):
                    logits = model(*self._inputs())
                    self.assertEqual(tuple(logits.shape), (4, len(LABELS)))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, torch.ones_like(logits)
                    )
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))

    def test_unknown_users_share_the_zero_embedding(self) -> None:
        model = LSTMForecaster(
            len(SEQUENCE_CHANNELS), 1, hidden_size=8, user_emb_dim=4, known_user_ids=np.array([5])
        )
        vectors = model.user_emb(torch.tensor([5, 6, 7])).detach()
        self.assertGreater(float(vectors[0].abs().sum()), 0)
        self.assertEqual(float(vectors[1:].abs().sum()), 0)


class FusionTests(unittest.TestCase):
    def test_simplex_grid_sums_to_one(self) -> None:
        grid = list(simplex_grid(BRANCHES, 0.5))
        self.assertEqual(len(grid), 6)
        for weights in grid:
            self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(list(simplex_grid(("lstm",), 0.1)), [{"lstm": 1.0}])

    def test_fuse_validates_weights(self) -> None:
        probabilities = {"lstm": np.full((2, 6), 0.2), "fedformer": np.full((2, 6), 0.6)}
        np.testing.assert_allclose(fuse(probabilities, {"lstm": 0.5, "fedformer": 0.5}), 0.4)
        with self.assertRaises(ValueError):
            fuse(probabilities, {"lstm": 1.0})

    def test_cli_rejects_mismatched_weights(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["backtest", "--models", "lstm,fedformer", "--fusion-weights", "lstm=1"]
        )
        with self.assertRaises(SystemExit):
            validate_args(parser, args)
        with self.assertRaises(SystemExit):
            parser.parse_args(["backtest", "--models", "gru"])


class BacktestTests(unittest.TestCase):
    def test_backtest_writes_branch_and_ensemble_outputs(self) -> None:
        start = pd.Timestamp("2026-06-03 16:00:00")
        cutoffs = pd.date_range(start, periods=12, freq="7D")
        per_week = 6
        count = len(cutoffs) * per_week
        rng = np.random.default_rng(0)
        user_ids = np.tile(np.arange(1, per_week + 1, dtype=np.int32), len(cutoffs))
        tokens = GraphTokens(
            user_ids=user_ids,
            entity_ids=np.zeros((count, 4), dtype=np.int32),
            node_type_ids=np.zeros((count, 4), dtype=np.uint8),
            relation_ids=np.zeros((count, 4), dtype=np.uint8),
            hop_ids=np.zeros((count, 4), dtype=np.uint8),
            node_features=np.zeros((count, 4, 4), dtype=np.float32),
            relative_days=np.zeros((count, 4), dtype=np.float32),
        )
        examples = TrainingExamples(
            tokens=tokens,
            labels=rng.integers(0, 2, (count, len(LABELS))).astype(np.uint8),
            cutoff_days=np.repeat((cutoffs.asi8 // DAY_NS).astype(np.int32), per_week),
            num_entities=1,
        )
        times = start + pd.to_timedelta(rng.uniform(-20, 80, 300), unit="D")
        events = _events(
            [(int(rng.integers(1, per_week + 1)), int(rng.integers(1, 7)), t, t, t) for t in times]
        )

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_dir = Path(directory.name)
        pd.DataFrame({"user_id": np.arange(1, per_week + 1), "is_cohort": True}).to_parquet(
            db_dir / "Users.parquet"
        )
        parser = build_parser()
        args = parser.parse_args(
            [
                "backtest",
                "--db-dir",
                str(db_dir),
                "--output-dir",
                str(db_dir / "run"),
                "--start-cutoff",
                str(start),
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--batch-size",
                "16",
                "--window-days",
                "14",
                "--label-days",
                "7",
                "--hidden-dim",
                "8",
                "--d-model",
                "8",
                "--n-heads",
                "2",
                "--d-ff",
                "16",
                "--fed-modes",
                "4",
                "--moving-avg",
                "3",
                "7",
                "--user-emb-dim",
                "4",
                "--fusion-step",
                "0.5",
            ]
        )
        validate_args(parser, args)
        args.output_dir.mkdir()
        module = "experiments.sequence_forecasters.run"
        with (
            patch(f"{module}.load_or_build_examples", return_value=examples),
            patch(f"{module}.load_events", return_value=events),
            patch(f"{module}.load_entity_vocabulary", return_value=SimpleNamespace()),
            patch(f"{module}.select_device", return_value=torch.device("cpu")),
        ):
            run_backtest(args)

        metrics = json.loads(
            (args.output_dir / "backtest_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(metrics["branch_holdout"]), {*BRANCHES, "ensemble"})
        weights = metrics["fusion"]["weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(metrics["holdout_examples"], per_week)
        self.assertEqual(metrics["calibration_weeks"], [str(cutoffs[-2].date())])
        self.assertNotIn(str(cutoffs[-2].date()), metrics["train_weeks"])
        for branch in BRANCHES:
            self.assertIn(metrics["branches"][branch]["selected_epoch"], (1, 2))
            self.assertTrue((args.output_dir / f"{branch}_model.pt").is_file())

        holdout = pd.read_csv(args.output_dir / "holdout_probabilities.csv")
        self.assertEqual(len(holdout), per_week)
        expected_users = examples.tokens.user_ids[-per_week:]
        np.testing.assert_array_equal(holdout["user_id"], expected_users)
        np.testing.assert_array_equal(
            holdout[[f"actual_{label}" for label in LABELS]].to_numpy(),
            examples.labels[-per_week:],
        )
        for label in LABELS:
            expected = sum(weights[b] * holdout[f"{b}_{label}"] for b in BRANCHES)
            np.testing.assert_allclose(holdout[f"ensemble_{label}"], expected, rtol=1e-5)
            self.assertTrue(holdout[f"ensemble_{label}"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
