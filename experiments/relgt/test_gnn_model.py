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

from experiments.relgt.data import (
    LABELS,
    EntityVocabulary,
    GraphTokens,
    TrainingExamples,
)
from experiments.relgt.gnn_model import RelationalGNN
from experiments.relgt.run import (
    fit_model,
    predict_probabilities,
    run_backtest,
    run_lookback_ensemble_prediction,
    save_holdout_probabilities,
)


class RelationalGNNTests(unittest.TestCase):
    def test_forward_and_backward_for_empty_and_populated_graphs(self) -> None:
        model = RelationalGNN(
            num_entities=32, hidden_dim=16, num_layers=2, dropout=0.0
        )
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
        self.assertIsNotNone(model.layers[0].relation_weights.grad)
        self.assertIsNotNone(model.entity_embedding.weight.grad)

    def test_training_and_prediction_use_gnn_architecture(self) -> None:
        tokens = GraphTokens(
            user_ids=np.array([1, 2, 3, 4], dtype=np.int32),
            entity_ids=np.array([[1, 0], [2, 3], [0, 0], [4, 5]], dtype=np.int32),
            node_type_ids=np.array(
                [[1, 0], [2, 3], [0, 0], [1, 3]], dtype=np.uint8
            ),
            relation_ids=np.array(
                [[1, 0], [2, 7], [0, 0], [4, 7]], dtype=np.uint8
            ),
            hop_ids=np.array([[1, 0], [1, 2], [0, 0], [1, 2]], dtype=np.uint8),
            node_features=np.zeros((4, 2, 4), dtype=np.float32),
            relative_days=np.zeros((4, 2), dtype=np.float32),
        )
        examples = TrainingExamples(
            tokens=tokens,
            labels=np.zeros((4, len(LABELS)), dtype=np.uint8),
            cutoff_days=np.arange(4, dtype=np.int32),
            num_entities=16,
        )
        model = fit_model(
            examples=examples,
            train_indices=np.arange(4),
            num_entities=examples.num_entities,
            device=torch.device("cpu"),
            epochs=1,
            batch_size=2,
            hidden_dim=16,
            num_heads=3,
            num_layers=1,
            dropout=0.0,
            seed=1,
            architecture="gnn",
        )
        probabilities = predict_probabilities(
            model,
            examples.tokens,
            np.arange(4),
            torch.device("cpu"),
            batch_size=2,
        )

        self.assertIsInstance(model, RelationalGNN)
        self.assertEqual(probabilities.shape, (4, len(LABELS)))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_saves_common_holdout_labels_and_probability_branch_columns(self) -> None:
        tokens = GraphTokens(
            user_ids=np.array([11, 12], dtype=np.int32),
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
            cutoff_days=np.array([20700, 20707], dtype=np.int32),
            num_entities=1,
        )
        relgt_probabilities = np.full(labels.shape, 0.25, dtype=np.float32)
        gnn_probabilities = np.full(labels.shape, 0.75, dtype=np.float32)

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "holdout_probabilities.csv"
            save_holdout_probabilities(
                examples,
                np.array([0, 1]),
                {
                    "relgt": relgt_probabilities,
                    "gnn": gnn_probabilities,
                },
                output_path,
            )
            result = pd.read_csv(output_path)

        expected_columns = [
            "user_id",
            "cutoff",
            *(f"actual_{label}" for label in LABELS),
            *(f"relgt_{label}" for label in LABELS),
            *(f"gnn_{label}" for label in LABELS),
        ]
        self.assertEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(result["user_id"].tolist(), [11, 12])
        np.testing.assert_array_equal(result["actual_open_pr"], [0, 0])
        np.testing.assert_allclose(result["relgt_open_pr"], [0.25, 0.25])
        np.testing.assert_allclose(result["gnn_open_pr"], [0.75, 0.75])

    def test_lookback_backtests_train_both_architectures_and_save_fusion(self) -> None:
        sample_count = 10
        tokens = GraphTokens(
            user_ids=np.arange(100, 100 + sample_count, dtype=np.int32),
            entity_ids=np.zeros((sample_count, 2), dtype=np.int32),
            node_type_ids=np.zeros((sample_count, 2), dtype=np.uint8),
            relation_ids=np.zeros((sample_count, 2), dtype=np.uint8),
            hop_ids=np.zeros((sample_count, 2), dtype=np.uint8),
            node_features=np.zeros((sample_count, 2, 4), dtype=np.float32),
            relative_days=np.zeros((sample_count, 2), dtype=np.float32),
        )
        examples = TrainingExamples(
            tokens=tokens,
            labels=np.tile(
                np.array([[0, 1, 0, 1, 0, 1]], dtype=np.uint8),
                (sample_count, 1),
            ),
            cutoff_days=np.arange(20700, 20700 + sample_count * 7, 7, dtype=np.int32),
            num_entities=1,
        )

        for temporal_model, architecture in (
            ("relgt_ensemble", "relgt"),
            ("gnn_ensemble", "gnn"),
        ):
            with self.subTest(temporal_model=temporal_model):
                with TemporaryDirectory() as temporary_directory:
                    output_dir = Path(temporary_directory)
                    args = SimpleNamespace(
                        temporal_model=temporal_model,
                        ensemble_lookbacks=(30, 14),
                        fusion_weight=None,
                        db_dir=output_dir,
                        output_dir=output_dir,
                        start_cutoff=pd.Timestamp("2026-09-01"),
                        device="cpu",
                        epochs=1,
                        batch_size=4,
                        hidden_dim=12,
                        num_heads=3,
                        num_layers=1,
                        dropout=0.0,
                        seed=1,
                    )
                    with (
                        patch(
                            "experiments.relgt.run.load_or_build_examples",
                            return_value=examples,
                        ),
                        patch(
                            "experiments.relgt.run.select_device",
                            return_value=torch.device("cpu"),
                        ),
                    ):
                        run_backtest(args)

                    holdout = pd.read_csv(output_dir / "holdout_probabilities.csv")
                    metrics = json.loads(
                        (output_dir / "backtest_metrics.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertIn("actual_open_pr", holdout.columns)
                    self.assertIn(f"{architecture}_30d_open_pr", holdout.columns)
                    self.assertIn(f"{architecture}_14d_open_pr", holdout.columns)
                    self.assertIn("fused_open_pr", holdout.columns)
                    self.assertTrue(
                        (output_dir / f"{architecture}_30d_model.pt").is_file()
                    )
                    self.assertTrue(
                        (output_dir / f"{architecture}_14d_model.pt").is_file()
                    )
                    self.assertEqual(
                        metrics["temporal_model"],
                        f"{architecture}_lookback_ensemble",
                    )

    def test_lookback_ensemble_prediction_writes_fused_and_branch_outputs(self) -> None:
        sample_count = 10
        tokens = GraphTokens(
            user_ids=np.arange(100, 100 + sample_count, dtype=np.int32),
            entity_ids=np.zeros((sample_count, 2), dtype=np.int32),
            node_type_ids=np.zeros((sample_count, 2), dtype=np.uint8),
            relation_ids=np.zeros((sample_count, 2), dtype=np.uint8),
            hop_ids=np.zeros((sample_count, 2), dtype=np.uint8),
            node_features=np.zeros((sample_count, 2, 4), dtype=np.float32),
            relative_days=np.zeros((sample_count, 2), dtype=np.float32),
        )
        examples = TrainingExamples(
            tokens=tokens,
            labels=np.tile(
                np.array([[0, 1, 0, 1, 0, 1]], dtype=np.uint8),
                (sample_count, 1),
            ),
            cutoff_days=np.arange(20700, 20700 + sample_count * 7, 7, dtype=np.int32),
            num_entities=1,
        )
        users = pd.DataFrame(
            {
                "user_id": [100],
                "github_user_id": [999],
                "is_cohort": [True],
            }
        )
        empty_events = pd.DataFrame(
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
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            args = SimpleNamespace(
                temporal_model="gnn_ensemble",
                ensemble_lookbacks=(30, 14),
                fusion_weight=None,
                db_dir=output_dir,
                output_dir=output_dir,
                start_cutoff=pd.Timestamp("2026-10-01"),
                cutoff=pd.Timestamp("2027-01-01"),
                test_input=output_dir / "test_input.csv",
                device="cpu",
                epochs=1,
                batch_size=4,
                hidden_dim=8,
                num_heads=3,
                num_layers=1,
                dropout=0.0,
                seed=1,
            )
            with (
                patch(
                    "experiments.relgt.run.load_or_build_examples",
                    return_value=examples,
                ),
                patch(
                    "experiments.relgt.run.select_device",
                    return_value=torch.device("cpu"),
                ),
                patch(
                    "experiments.relgt.run.pd.read_csv",
                    return_value=pd.DataFrame({"UserID": [999]}),
                ),
                patch(
                    "experiments.relgt.run.pd.read_parquet",
                    return_value=users,
                ),
                patch(
                    "experiments.relgt.run.load_entity_vocabulary",
                    return_value=EntityVocabulary(1, 1, 1),
                ),
                patch(
                    "experiments.relgt.run.load_events",
                    return_value=empty_events,
                ),
            ):
                run_lookback_ensemble_prediction(args, "gnn")

            predictions = pd.read_csv(output_dir / "predictions.csv")
            probabilities = pd.read_csv(output_dir / "probabilities.csv")
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(predictions["UserID"].tolist(), [999])
            self.assertIn("open_pr_probability", probabilities.columns)
            self.assertIn("gnn_30d_open_pr_probability", probabilities.columns)
            self.assertIn("gnn_14d_open_pr_probability", probabilities.columns)
            self.assertAlmostEqual(sum(metadata["fusion"]["weights"].values()), 1.0)
            self.assertTrue((output_dir / "gnn_30d_model.pt").is_file())
            self.assertTrue((output_dir / "gnn_14d_model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
