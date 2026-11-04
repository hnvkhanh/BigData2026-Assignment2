from __future__ import annotations

import argparse
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
    choose_fusion_weight,
    choose_three_fusion_weights,
    calculate_metrics,
    choose_thresholds,
    create_run_output_dir,
    fuse_probabilities,
    fuse_three_probabilities,
    lightgbm_examples_for_graph,
    parse_ensemble_lookbacks,
    parse_args,
    parse_three_branch_weights,
    parse_unit_interval,
    prediction_example_indices,
    require_s2p2_dependency,
    s2p2_examples_for_rows,
)
from experiments.relgt.data import GraphTokens, TrainingExamples


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

    def test_snapshot_structure_features_do_not_depend_on_requested_users(self) -> None:
        other_user_event = self.events.iloc[[2]].copy()
        other_user_event.loc[:, "user_id"] = 12
        events = pd.concat([self.events, other_user_event], ignore_index=True)

        one_user = snapshot_tokens(events, self.cutoff, [10], self.vocabulary)
        two_users = snapshot_tokens(events, self.cutoff, [10, 12], self.vocabulary)

        np.testing.assert_array_equal(one_user.entity_ids[0], two_users.entity_ids[0])
        np.testing.assert_array_equal(
            one_user.node_features[0], two_users.node_features[0]
        )
        self.assertAlmostEqual(
            float(one_user.node_features[0, 0, 2]), np.log1p(3)
        )

    def test_snapshot_respects_requested_lookback_window(self) -> None:
        short_window = snapshot_tokens(
            self.events,
            self.cutoff,
            [10],
            self.vocabulary,
            lookback_days=3,
        )
        long_window = snapshot_tokens(
            self.events,
            self.cutoff,
            [10],
            self.vocabulary,
            lookback_days=90,
        )

        self.assertNotIn(9, short_window.entity_ids[0, :24])
        self.assertIn(9, long_window.entity_ids[0, :24])

    def test_prediction_training_samples_have_labels_observed_before_cutoff(self) -> None:
        day_ns = pd.Timedelta(days=1).value
        cutoff_days = np.array(
            [
                pd.Timestamp("2026-01-01").value // day_ns,
                pd.Timestamp("2026-01-08").value // day_ns,
                pd.Timestamp("2026-01-15").value // day_ns,
            ],
            dtype=np.int32,
        )

        indices = prediction_example_indices(
            cutoff_days, pd.Timestamp("2026-01-15")
        )

        self.assertEqual(indices.tolist(), [0, 1])


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


class HybridFusionTests(unittest.TestCase):
    def test_parses_two_supported_lookback_windows(self) -> None:
        self.assertEqual(parse_ensemble_lookbacks("30,14"), (30, 14))
        self.assertEqual(parse_ensemble_lookbacks("90, 14"), (90, 14))

    def test_rejects_invalid_lookback_pairs(self) -> None:
        for value in ("30", "30,30", "30,7", "30,14,90"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                parse_ensemble_lookbacks(value)

    def test_parses_manual_fusion_weights(self) -> None:
        self.assertEqual(parse_unit_interval("0.35"), 0.35)
        self.assertEqual(
            parse_three_branch_weights(
                "relgt=0.2,lightgbm=0.5,s2p2=0.3"
            ),
            {"relgt": 0.2, "lightgbm": 0.5, "s2p2": 0.3},
        )
        for value in ("-0.1", "1.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                parse_unit_interval(value)
        for value in (
            "relgt=0.2,lightgbm=0.5",
            "relgt=0.2,lightgbm=0.5,s2p2=0.4",
            "relgt=-0.1,lightgbm=0.8,s2p2=0.3",
        ):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                parse_three_branch_weights(value)

    def test_manual_fusion_options_are_limited_to_matching_modes(self) -> None:
        with patch(
            "sys.argv",
            [
                "run.py",
                "backtest",
                "--temporal-model",
                "gnn_ensemble",
                "--fusion-weight",
                "0.7",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.fusion_weight, 0.7)

        with patch(
            "sys.argv",
            [
                "run.py",
                "backtest",
                "--temporal-model",
                "relgt",
                "--fusion-weight",
                "0.7",
            ],
        ), self.assertRaises(SystemExit):
            parse_args()

        with patch(
            "sys.argv",
            [
                "run.py",
                "predict",
                "--temporal-model",
                "relgt_lightgbm_s2p2",
                "--fusion-weights",
                "relgt=0.4,lightgbm=0.4,s2p2=0.2",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.fusion_weights["s2p2"], 0.2)

    def test_fuses_probabilities_and_selects_calibration_weight(self) -> None:
        labels = np.array(
            [
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
            ],
            dtype=np.uint8,
        )
        graph = labels.astype(np.float32) * 0.02 + 0.49
        s2p2 = 1.0 - labels.astype(np.float32) * 0.98 - 0.01
        fused = fuse_probabilities(graph, s2p2, graph_weight=1.0)
        weight, thresholds, metrics = choose_fusion_weight(labels, graph, s2p2)

        np.testing.assert_array_equal(fused, graph)
        self.assertEqual(weight, 1.0)
        self.assertEqual(set(thresholds), set(LABELS))
        self.assertAlmostEqual(metrics["score"], 1.0)

    def test_manual_two_branch_weight_controls_blend_and_calibration(self) -> None:
        labels = np.tile(
            np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=np.uint8),
            (2, 1),
        )
        graph = np.where(labels == 1, 0.95, 0.05).astype(np.float32)
        other = 1.0 - graph

        weight, thresholds, metrics = choose_fusion_weight(
            labels, graph, other, manual_weight=0.75
        )
        fused = fuse_probabilities(graph, other, weight)
        expected_thresholds = choose_thresholds(labels, fused)
        expected_metrics = calculate_metrics(labels, fused, expected_thresholds)

        self.assertEqual(weight, 0.75)
        self.assertEqual(thresholds, expected_thresholds)
        self.assertEqual(metrics, expected_metrics)
        self.assertAlmostEqual(metrics["score"], 1.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            choose_fusion_weight(labels, graph, other, manual_weight=1.5)

    def test_rejects_misaligned_probability_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            fuse_probabilities(np.zeros((2, 6)), np.zeros((3, 6)), 0.5)

    def test_selects_three_branch_weights_using_calibration_predictions(self) -> None:
        labels = np.tile(
            np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=np.uint8),
            (4, 1),
        )
        relgt = np.where(labels == 1, 0.2, 0.8).astype(np.float32)
        lightgbm = np.where(labels == 1, 0.95, 0.05).astype(np.float32)
        s2p2 = np.full(labels.shape, 0.5, dtype=np.float32)

        weights, thresholds, metrics = choose_three_fusion_weights(
            labels, relgt, lightgbm, s2p2
        )
        fused = fuse_three_probabilities(relgt, lightgbm, s2p2, weights)

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(set(weights), {"relgt", "lightgbm", "s2p2"})
        self.assertEqual(set(thresholds), set(LABELS))
        self.assertEqual(fused.shape, labels.shape)
        self.assertGreater(metrics["score"], 0.9)
        self.assertGreater(weights["lightgbm"], weights["relgt"])

    def test_manual_three_branch_weights_control_blend_and_calibration(self) -> None:
        labels = np.tile(
            np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=np.uint8),
            (2, 1),
        )
        relgt = np.where(labels == 1, 0.95, 0.05).astype(np.float32)
        lightgbm = np.full(labels.shape, 0.5, dtype=np.float32)
        s2p2 = 1.0 - relgt
        manual_weights = {"relgt": 0.6, "lightgbm": 0.2, "s2p2": 0.2}

        weights, thresholds, metrics = choose_three_fusion_weights(
            labels, relgt, lightgbm, s2p2, manual_weights
        )
        fused = fuse_three_probabilities(relgt, lightgbm, s2p2, manual_weights)
        expected_thresholds = choose_thresholds(labels, fused)
        expected_metrics = calculate_metrics(labels, fused, expected_thresholds)

        self.assertEqual(weights, manual_weights)
        self.assertEqual(thresholds, expected_thresholds)
        self.assertEqual(metrics, expected_metrics)
        self.assertAlmostEqual(metrics["score"], 1.0)

    def test_three_branch_fusion_rejects_invalid_weights(self) -> None:
        probabilities = np.zeros((1, len(LABELS)), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "sum to 1"):
            fuse_three_probabilities(
                probabilities,
                probabilities,
                probabilities,
                {"relgt": 0.5, "lightgbm": 0.4, "s2p2": 0.2},
            )

    def test_s2p2_dependency_preflight_reports_install_instructions(self) -> None:
        with self.assertRaisesRegex(ImportError, "s2p2-requirements.txt"):
            require_s2p2_dependency()

    def test_aligns_s2p2_rows_to_graph_user_and_cutoff(self) -> None:
        tokens = GraphTokens(
            user_ids=np.array([4, 8], dtype=np.int32),
            entity_ids=np.zeros((2, 1), dtype=np.int32),
            node_type_ids=np.zeros((2, 1), dtype=np.uint8),
            relation_ids=np.zeros((2, 1), dtype=np.uint8),
            hop_ids=np.zeros((2, 1), dtype=np.uint8),
            node_features=np.zeros((2, 1, 4), dtype=np.float32),
            relative_days=np.zeros((2, 1), dtype=np.float32),
        )
        examples = TrainingExamples(
            tokens=tokens,
            labels=np.zeros((2, len(LABELS)), dtype=np.uint8),
            cutoff_days=np.array(
                [
                    pd.Timestamp("2026-03-01").value // pd.Timedelta(days=1).value,
                    pd.Timestamp("2026-03-08").value // pd.Timedelta(days=1).value,
                ],
                dtype=np.int32,
            ),
            num_entities=10,
        )

        rows = s2p2_examples_for_rows(examples, np.array([1]))

        self.assertEqual(rows["user_id"].tolist(), [8])
        self.assertEqual(rows["cutoff"].tolist(), [pd.Timestamp("2026-03-08")])

    def test_aligns_lightgbm_features_to_graph_user_and_cutoff(self) -> None:
        tokens = GraphTokens(
            user_ids=np.array([4, 8], dtype=np.int32),
            entity_ids=np.zeros((2, 1), dtype=np.int32),
            node_type_ids=np.zeros((2, 1), dtype=np.uint8),
            relation_ids=np.zeros((2, 1), dtype=np.uint8),
            hop_ids=np.zeros((2, 1), dtype=np.uint8),
            node_features=np.zeros((2, 1, 4), dtype=np.float32),
            relative_days=np.zeros((2, 1), dtype=np.float32),
        )
        labels = np.zeros((2, len(LABELS)), dtype=np.uint8)
        examples = TrainingExamples(
            tokens=tokens,
            labels=labels,
            cutoff_days=np.array(
                [
                    pd.Timestamp("2026-03-01").value // pd.Timedelta(days=1).value,
                    pd.Timestamp("2026-03-08").value // pd.Timedelta(days=1).value,
                ],
                dtype=np.int32,
            ),
            num_entities=10,
        )
        data = pd.DataFrame(
            {
                "user_id": [8, 4],
                "cutoff": pd.to_datetime(["2026-03-08", "2026-03-01"]),
                "feature": [80, 40],
                **{label: [0, 0] for label in LABELS},
            }
        )

        aligned = lightgbm_examples_for_graph(examples, data)

        self.assertEqual(aligned["user_id"].tolist(), [4, 8])
        self.assertEqual(aligned["feature"].tolist(), [40, 80])


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
