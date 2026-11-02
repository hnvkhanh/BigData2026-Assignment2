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
    REPO_MEMBERSHIP_RELATION,
    TrainingExamples,
)
from experiments.relgt.gnn_model import RelationalGNN
from experiments.relgt_action_specialists.run import (
    ISSUE_OUTPUTS,
    ISSUE_RELATIONS,
    NON_PUSH_OUTPUTS,
    PULL_REQUEST_OUTPUTS,
    PULL_REQUEST_RELATIONS,
    PUSH_OUTPUTS,
    build_non_push_tokens,
    build_tokens_for_relations,
    fit_specialist,
    predict_specialist,
    run_backtest,
)


class ActionSpecialistTests(unittest.TestCase):
    def test_specialist_gnns_have_only_their_assigned_outputs(self) -> None:
        push_model = RelationalGNN(
            num_entities=16, hidden_dim=12, num_layers=1, output_indices=PUSH_OUTPUTS
        )
        non_push_model = RelationalGNN(
            num_entities=16,
            hidden_dim=12,
            num_layers=1,
            output_indices=NON_PUSH_OUTPUTS,
        )
        entity_ids = torch.tensor([[1, 2]], dtype=torch.long)
        node_type_ids = torch.tensor([[1, 3]], dtype=torch.long)
        relation_ids = torch.tensor([[1, 7]], dtype=torch.long)
        hop_ids = torch.tensor([[1, 2]], dtype=torch.long)
        features = torch.zeros((1, 2, 4), dtype=torch.float32)
        relative_days = torch.zeros((1, 2), dtype=torch.float32)
        inputs = (
            entity_ids,
            node_type_ids,
            relation_ids,
            hop_ids,
            features,
            relative_days,
        )

        self.assertEqual(tuple(push_model(*inputs).shape), (1, 1))
        self.assertEqual(tuple(non_push_model(*inputs).shape), (1, 5))

    def test_trains_and_predicts_both_output_shapes(self) -> None:
        sample_count = 4
        tokens = GraphTokens(
            user_ids=np.arange(sample_count, dtype=np.int32),
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
            cutoff_days=np.arange(sample_count, dtype=np.int32),
            num_entities=1,
        )
        args = SimpleNamespace(
            hidden_dim=12,
            num_layers=1,
            dropout=0.0,
            learning_rate=1e-3,
            epochs=1,
            batch_size=2,
        )

        for output_indices, expected_outputs in (
            (PUSH_OUTPUTS, 1),
            (NON_PUSH_OUTPUTS, 5),
        ):
            model = fit_specialist(
                examples,
                tokens,
                np.arange(sample_count),
                output_indices,
                torch.device("cpu"),
                args,
                seed=1,
            )
            probabilities = predict_specialist(
                model,
                tokens,
                np.arange(sample_count),
                torch.device("cpu"),
                batch_size=2,
            )
            self.assertEqual(probabilities.shape, (sample_count, expected_outputs))
            self.assertTrue(np.isfinite(probabilities).all())

    def test_non_push_input_removes_push_nodes_and_push_only_repositories(self) -> None:
        day_ns = pd.Timedelta(days=1).value
        cutoff_days = np.array(
            [
                pd.Timestamp("2026-01-15").value // day_ns,
                pd.Timestamp("2026-01-22").value // day_ns,
            ],
            dtype=np.int32,
        )
        tokens = GraphTokens(
            user_ids=np.array([10, 10], dtype=np.int32),
            entity_ids=np.zeros((2, 48), dtype=np.int32),
            node_type_ids=np.zeros((2, 48), dtype=np.uint8),
            relation_ids=np.zeros((2, 48), dtype=np.uint8),
            hop_ids=np.zeros((2, 48), dtype=np.uint8),
            node_features=np.zeros((2, 48, 4), dtype=np.float32),
            relative_days=np.zeros((2, 48), dtype=np.float32),
        )
        examples = TrainingExamples(
            tokens=tokens,
            labels=np.zeros((2, len(LABELS)), dtype=np.uint8),
            cutoff_days=cutoff_days,
            num_entities=32,
        )
        events = pd.DataFrame(
            [
                (10, 5, 3, 10, 3, 6, "2026-01-10", "2026-01-10", "2026-01-10"),
                (10, 6, 4, 11, 1, 1, "2026-01-11", "2026-01-11", "2026-01-11"),
                (10, 7, 5, 12, 2, 2, "2026-01-12", "2026-01-12", "2026-01-12"),
                (10, 7, 5, 12, 2, 3, "2026-01-12", "2026-01-12", "2026-01-12"),
                (10, 6, 6, 13, 1, 4, "2026-01-13", "2026-01-13", "2026-01-13"),
                (10, 6, 7, 14, 1, 5, "2026-01-14", "2026-01-14", "2026-01-14"),
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
            events[column] = pd.to_datetime(events[column])
        vocabulary = EntityVocabulary(
            num_pull_requests=8, num_issues=8, num_repositories=8
        )

        result = build_non_push_tokens(examples, events, vocabulary)

        self.assertEqual(result.user_ids.tolist(), [10, 10])
        self.assertTrue(np.all(result.relation_ids != 6))
        self.assertIn(11, result.entity_ids[0])
        self.assertNotIn(10, result.entity_ids[0])
        self.assertIn(23, result.entity_ids[0])
        self.assertNotIn(22, result.entity_ids[0])
        self.assertTrue(np.array_equal(result.entity_ids[0], result.entity_ids[1]))
        pr_tokens = build_tokens_for_relations(
            examples, events, vocabulary, PULL_REQUEST_RELATIONS
        )
        issue_tokens = build_tokens_for_relations(
            examples, events, vocabulary, ISSUE_RELATIONS
        )
        self.assertEqual(
            set(np.unique(pr_tokens.relation_ids)) - {0},
            set(PULL_REQUEST_RELATIONS) | {REPO_MEMBERSHIP_RELATION},
        )
        self.assertEqual(
            set(np.unique(issue_tokens.relation_ids)) - {0},
            set(ISSUE_RELATIONS) | {REPO_MEMBERSHIP_RELATION},
        )
        self.assertEqual(len(PULL_REQUEST_OUTPUTS), 3)
        self.assertEqual(len(ISSUE_OUTPUTS), 2)

    def test_backtest_saves_both_specialists_and_complete_holdout(self) -> None:
        sample_count = 10
        user_ids = np.arange(1, sample_count + 1, dtype=np.int32)
        cutoff_dates = pd.date_range("2026-03-01", periods=sample_count, freq="7D")
        cutoff_days = (cutoff_dates.asi8 // pd.Timedelta(days=1).value).astype(
            np.int32
        )
        tokens = GraphTokens(
            user_ids=user_ids,
            entity_ids=np.zeros((sample_count, 48), dtype=np.int32),
            node_type_ids=np.zeros((sample_count, 48), dtype=np.uint8),
            relation_ids=np.zeros((sample_count, 48), dtype=np.uint8),
            hop_ids=np.zeros((sample_count, 48), dtype=np.uint8),
            node_features=np.zeros((sample_count, 48, 4), dtype=np.float32),
            relative_days=np.zeros((sample_count, 48), dtype=np.float32),
        )
        labels = np.tile(
            np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.uint8),
            (sample_count // 2, 1),
        )
        examples = TrainingExamples(
            tokens=tokens,
            labels=labels,
            cutoff_days=cutoff_days,
            num_entities=8,
        )
        event_rows = []
        for user_id, cutoff in zip(user_ids, cutoff_dates):
            event_rows.extend(
                [
                    (
                        user_id, 0, 0, 1, 1, 1,
                        cutoff - pd.Timedelta(days=2),
                        cutoff - pd.Timedelta(days=2),
                        cutoff - pd.Timedelta(days=2),
                    ),
                    (
                        user_id, 1, 1, 6, 3, 6,
                        cutoff - pd.Timedelta(days=1),
                        cutoff - pd.Timedelta(days=1),
                        cutoff - pd.Timedelta(days=1),
                    ),
                ]
            )
        events = pd.DataFrame(
            event_rows,
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
            events[column] = pd.to_datetime(events[column])

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = SimpleNamespace(
                db_dir=root,
                output_dir=root / "run",
                start_cutoff=cutoff_dates[0],
                device="cpu",
                epochs=1,
                batch_size=4,
                hidden_dim=12,
                num_layers=1,
                dropout=0.0,
                learning_rate=1e-3,
                seed=1,
                variant="both",
            )
            users = pd.DataFrame(
                {"user_id": user_ids, "is_cohort": np.ones(sample_count, dtype=bool)}
            )
            with (
                patch(
                    "experiments.relgt_action_specialists.run.load_or_build_examples",
                    return_value=examples,
                ),
                patch(
                    "experiments.relgt_action_specialists.run.load_entity_vocabulary",
                    return_value=EntityVocabulary(2, 2, 2),
                ),
                patch(
                    "experiments.relgt_action_specialists.run.load_events",
                    return_value=events,
                ),
                patch(
                    "experiments.relgt_action_specialists.run.select_device",
                    return_value=torch.device("cpu"),
                ),
                patch(
                    "experiments.relgt_action_specialists.run.pd.read_parquet",
                    return_value=users,
                ),
            ):
                run_backtest(args)

            holdout = pd.read_csv(args.output_dir / "holdout_probabilities.csv")
            metrics = json.loads(
                (args.output_dir / "backtest_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                (args.output_dir / "push_specialist_model.pt").is_file()
            )
            self.assertTrue(
                (args.output_dir / "non_push_specialist_model.pt").is_file()
            )
            self.assertTrue((args.output_dir / "pr_specialist_model.pt").is_file())
            self.assertTrue((args.output_dir / "issue_specialist_model.pt").is_file())
            self.assertTrue(
                all(f"combined_specialists_{label}" in holdout for label in LABELS)
            )
            self.assertTrue(
                all(f"combined_grouped_{label}" in holdout for label in LABELS)
            )
            self.assertEqual(
                set(metrics["thresholds"]),
                set(LABELS),
            )
            self.assertIn("combined_specialists", metrics["branch_holdout"])
            self.assertIn("combined_grouped", metrics["branch_holdout"])


if __name__ == "__main__":
    unittest.main()
