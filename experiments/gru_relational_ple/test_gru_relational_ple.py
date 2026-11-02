from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from experiments.gru_relational_ple.model import GRURelationalAttentionPLE
from experiments.gru_relational_ple.run import (
    DAY_NS,
    LOOKBACK_DAYS,
    build_daily_activity_sequences,
)
from experiments.relgt.data import LABELS


class ActivitySequenceTests(unittest.TestCase):
    def test_sequences_exclude_cutoff_and_not_yet_available_events(self) -> None:
        cutoff = pd.Timestamp("2026-01-15")
        events = pd.DataFrame(
            {
                "user_id": [7, 7, 7],
                "action_at": pd.to_datetime(
                    [
                        "2026-01-14 10:00",
                        "2026-01-13 00:00",
                        "2026-01-15 00:00",
                    ]
                ),
                "available_at": pd.to_datetime(
                    [
                        "2026-01-14 10:00",
                        "2026-01-15 00:00",
                        "2026-01-14 00:00",
                    ]
                ),
                "relation_id": [1, 2, 3],
            }
        )
        cutoff_days = np.array([cutoff.value // DAY_NS, cutoff.value // DAY_NS])

        sequences = build_daily_activity_sequences(
            events, np.array([7, 8]), cutoff_days
        )

        self.assertEqual(sequences.shape, (2, LOOKBACK_DAYS, len(LABELS)))
        self.assertAlmostEqual(float(sequences[0, -1, 0]), np.log1p(1), places=3)
        self.assertEqual(float(sequences[0, :, 1].sum()), 0.0)
        self.assertEqual(float(sequences[0, :, 2].sum()), 0.0)
        self.assertEqual(float(sequences[1].sum()), 0.0)

class GRURelationalAttentionPLETests(unittest.TestCase):
    def test_forward_and_backward_for_empty_and_populated_graphs(self) -> None:
        model = GRURelationalAttentionPLE(
            num_entities=32,
            hidden_dim=16,
            num_heads=4,
            num_graph_layers=1,
            num_gru_layers=1,
        )
        entity_ids = torch.tensor([[0, 0], [1, 8]], dtype=torch.long)
        node_type_ids = torch.tensor([[0, 0], [1, 3]], dtype=torch.long)
        relation_ids = torch.tensor([[0, 0], [1, 7]], dtype=torch.long)
        hop_ids = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        node_features = torch.zeros((2, 2, 4), dtype=torch.float32)
        relative_days = torch.zeros((2, 2), dtype=torch.float32)
        daily_activity = torch.zeros((2, 90, len(LABELS)), dtype=torch.float32)
        daily_activity[1, -3:, 0] = 1.0

        logits = model(
            entity_ids,
            node_type_ids,
            relation_ids,
            hop_ids,
            node_features,
            relative_days,
            daily_activity,
        )

        self.assertEqual(tuple(logits.shape), (2, len(LABELS)))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(model.sequence_gru.weight_ih_l0.grad)
        self.assertIsNotNone(model.graph_encoder.centroids.grad)
        self.assertIsNotNone(model.first_extraction.task_gates[0].weight.grad)

    def test_rejects_wrong_daily_activity_shape(self) -> None:
        model = GRURelationalAttentionPLE(
            num_entities=8, hidden_dim=8, num_heads=2, num_graph_layers=1
        )
        with self.assertRaisesRegex(ValueError, "daily_activity"):
            model(
                torch.zeros((1, 1), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
                torch.zeros((1, 1, 4)),
                torch.zeros((1, 1)),
                torch.zeros((1, 90)),
            )


if __name__ == "__main__":
    unittest.main()
