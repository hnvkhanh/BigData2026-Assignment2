from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.relgt.data import (
    LABELS,
    REPO_MEMBERSHIP_RELATION,
    EntityVocabulary,
    labels_for_cutoff,
    snapshot_tokens,
)
from experiments.relgt.model import RelGTInspired
from experiments.relgt.run import (
    CudaDeviceInfo,
    _least_loaded_cuda_device,
    create_run_output_dir,
)


class TemporalSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cutoff = pd.Timestamp("2026-01-15 00:00:00")
        self.vocabulary = EntityVocabulary(
            num_pull_requests=4,
            num_issues=4,
            num_repositories=2,
        )
        self.events = pd.DataFrame(
            [
                (10, 0, 0, 9, 3, 6, "2026-01-10", "2026-01-10", "2026-01-10"),
                (10, 0, 0, 9, 3, 6, "2026-01-11", "2026-01-11", "2026-01-10"),
                (10, 0, 0, 1, 1, 1, "2026-01-11", "2026-01-11", "2026-01-10"),
                (10, 0, 0, 1, 1, 4, "2026-01-12", "2026-01-12", "2026-01-10"),
                (10, 1, 0, 5, 2, 2, "2026-01-13", "2026-01-13", "2026-01-10"),
                (10, 1, 0, 5, 2, 3, "2026-01-14", "2026-01-14", "2026-01-10"),
                # Known only after the cutoff.
                (10, 1, 1, 10, 3, 6, "2026-01-14", "2026-01-16", "2026-01-10"),
                # An action at the cutoff belongs to the target, not the history.
                (10, 1, 1, 10, 3, 6, "2026-01-15", "2026-01-15", "2026-01-10"),
                # This cohort user is inactive in the lookback and must be omitted.
                (11, 1, 1, 6, 2, 2, "2026-01-14", "2026-01-14", "2026-01-10"),
            ],
            columns=[
                "user_id",
                "repo_id",
                "entity_id",
                "global_entity_id",
                "node_type",
                "relation_id",
                "action_at",
                "available_at",
                "entity_available_at",
            ],
        )
        for column in ("action_at", "available_at", "entity_available_at"):
            self.events[column] = pd.to_datetime(self.events[column])

    def test_snapshot_uses_only_available_history_and_adds_repo_hop(self) -> None:
        tokens = snapshot_tokens(
            self.events,
            self.cutoff,
            [10, 11],
            self.vocabulary,
            eligible_only=True,
        )

        self.assertEqual(tokens.user_ids.tolist(), [10])
        self.assertEqual(int(tokens.entity_ids[0, 0]), 1)
        self.assertEqual(int(tokens.node_type_ids[0, 0]), 1)
        self.assertEqual(int(tokens.relation_ids[0, 0]), 1)
        self.assertEqual(int(tokens.hop_ids[0, 0]), 1)
        self.assertAlmostEqual(float(tokens.node_features[0, 0, 0]), np.log1p(1))

        repo_slot = 24
        self.assertEqual(int(tokens.entity_ids[0, repo_slot]), 10)
        self.assertEqual(int(tokens.node_type_ids[0, repo_slot]), 3)
        self.assertEqual(
            int(tokens.relation_ids[0, repo_slot]), REPO_MEMBERSHIP_RELATION
        )
        self.assertEqual(int(tokens.hop_ids[0, repo_slot]), 2)
        self.assertFalse(np.any(tokens.entity_ids[0, 25:] == 10))
        self.assertEqual(int(tokens.entity_ids[0, 20]), 9)
        self.assertEqual(int(tokens.entity_ids[0, 21]), 0)

    def test_labels_use_the_seven_day_action_window(self) -> None:
        labels = labels_for_cutoff(self.events, self.cutoff, np.array([10]))

        self.assertEqual(labels.shape, (1, len(LABELS)))
        self.assertEqual(labels[0].tolist(), [0, 0, 0, 0, 0, 1])


class RelGTInspiredModelTests(unittest.TestCase):
    def test_forward_and_backward_for_empty_and_populated_neighborhoods(self) -> None:
        model = RelGTInspired(num_entities=32, hidden_dim=16, num_heads=4, num_local_layers=1)
        entity_ids = torch.tensor([[0, 0], [1, 8]], dtype=torch.long)
        node_type_ids = torch.tensor([[0, 0], [1, 3]], dtype=torch.long)
        relation_ids = torch.tensor([[0, 0], [1, 7]], dtype=torch.long)
        hop_ids = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        features = torch.zeros((2, 2, 4), dtype=torch.float32)
        relative_days = torch.zeros((2, 2), dtype=torch.float32)

        logits = model(
            entity_ids,
            node_type_ids,
            relation_ids,
            hop_ids,
            features,
            relative_days,
        )
        self.assertEqual(tuple(logits.shape), (2, len(LABELS)))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(model.centroids.grad)


class DeviceAndOutputTests(unittest.TestCase):
    def test_selects_cuda_gpu_with_lowest_utilization(self) -> None:
        selected = _least_loaded_cuda_device(
            [
                CudaDeviceInfo(0, "busy", 75, 12000, 16000),
                CudaDeviceInfo(1, "idle", 8, 2000, 16000),
                CudaDeviceInfo(2, "less-used-memory", 8, 7000, 16000),
            ]
        )

        self.assertEqual(selected, torch.device("cuda", 1))

    def test_uses_memory_as_tiebreaker_and_when_utilization_unavailable(self) -> None:
        selected = _least_loaded_cuda_device(
            [
                CudaDeviceInfo(0, "more-memory-used", None, 9000, 16000),
                CudaDeviceInfo(1, "less-memory-used", None, 3000, 16000),
            ]
        )

        self.assertEqual(selected, torch.device("cuda", 1))

    def test_default_run_output_paths_are_unique_and_timestamped(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with patch("experiments.relgt.run.DEFAULT_OUTPUT_ROOT", output_root):
                first = create_run_output_dir("backtest", None)
                second = create_run_output_dir("backtest", None)

        self.assertNotEqual(first, second)
        self.assertRegex(first.name, r"^backtest_\d{8}_\d{6}_\d{6}_\d+$")


if __name__ == "__main__":
    unittest.main()
