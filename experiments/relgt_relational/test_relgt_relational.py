from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from experiments.relational_lightgbm.features import FEATURE_GROUPS
from experiments.relgt.data import LABELS, MAX_NEIGHBORS, GraphTokens
from experiments.relgt_relational.run import (
    RootFeatureScaler,
    _logits,
    build_model,
    cutoff_timestamps,
    parse_args,
)


def _tokens(batch: int) -> GraphTokens:
    rng = np.random.default_rng(0)
    return GraphTokens(
        user_ids=np.arange(batch, dtype=np.int32),
        entity_ids=rng.integers(0, 20, (batch, MAX_NEIGHBORS)).astype(np.int32),
        node_type_ids=rng.integers(1, 4, (batch, MAX_NEIGHBORS)).astype(np.uint8),
        relation_ids=rng.integers(1, len(LABELS) + 2, (batch, MAX_NEIGHBORS)).astype(np.uint8),
        hop_ids=rng.integers(1, 3, (batch, MAX_NEIGHBORS)).astype(np.uint8),
        node_features=rng.random((batch, MAX_NEIGHBORS, 4)).astype(np.float32),
        relative_days=rng.random((batch, MAX_NEIGHBORS)).astype(np.float32) * 14,
    )


class RootFeatureTests(unittest.TestCase):
    def test_scaler_uses_training_statistics_and_handles_constant_columns(self) -> None:
        train = np.array([[0.0, 3.0], [np.e - 1, 3.0]], dtype=np.float32)
        scaler = RootFeatureScaler.fit(train)
        scaled = scaler.transform(train)
        np.testing.assert_allclose(scaled[:, 0], [-1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(scaled[:, 1], [0.0, 0.0], atol=1e-6)
        self.assertEqual(scaled.dtype, np.float32)

    def test_cutoff_timestamps_restore_the_time_of_day(self) -> None:
        start = pd.Timestamp("2026-03-11 16:00:00")
        days = np.array([start.value // pd.Timedelta(days=1).value] * 2 + [start.value // pd.Timedelta(days=1).value + 7])
        cutoffs = cutoff_timestamps(days, start)
        self.assertEqual(sorted(cutoffs.values()), [start, start + pd.Timedelta(days=7)])

    def test_models_accept_root_features_and_change_their_output(self) -> None:
        tokens = _tokens(4)
        features = np.random.default_rng(1).normal(size=(4, 5)).astype(np.float32)
        for architecture in ("relgt", "gnn"):
            args = parse_args(["backtest", "--architecture", architecture, "--hidden-dim", "16"])
            torch.manual_seed(0)
            model = build_model(args, num_entities=32, num_root_features=5).eval()
            logits = _logits(model, tokens, features, np.arange(4), torch.device("cpu"))
            self.assertEqual(tuple(logits.shape), (4, len(LABELS)))
            shifted = _logits(model, tokens, features + 3, np.arange(4), torch.device("cpu"))
            self.assertFalse(torch.allclose(logits, shifted))
            with self.assertRaises(ValueError):
                model(*[torch.as_tensor(getattr(tokens, name)).long() if name.endswith("ids") else
                        torch.as_tensor(getattr(tokens, name)) for name in
                        ("entity_ids", "node_type_ids", "relation_ids", "hop_ids",
                         "node_features", "relative_days")])

    def test_no_groups_builds_the_plain_graph_model(self) -> None:
        args = parse_args(["backtest", "--groups", "none", "--hidden-dim", "16"])
        self.assertEqual(args.groups, [])
        model = build_model(args, num_entities=32, num_root_features=0)
        self.assertIsNone(model.root_feature_projection)
        logits = _logits(model, _tokens(2), np.zeros((2, 0), np.float32), np.arange(2), torch.device("cpu"))
        self.assertEqual(tuple(logits.shape), (2, len(LABELS)))

    def test_cli_defaults_and_validation(self) -> None:
        args = parse_args(["backtest"])
        self.assertEqual(args.groups, list(FEATURE_GROUPS))
        self.assertEqual(args.architecture, "relgt")
        with self.assertRaises(SystemExit):
            parse_args(["backtest", "--hidden-dim", "10", "--num-heads", "4"])
        with self.assertRaises(SystemExit):
            parse_args(["backtest", "--groups", "unknown"])


if __name__ == "__main__":
    unittest.main()
