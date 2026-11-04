from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from experiments.relgt.data import LABELS  # noqa: E402
from experiments.relgt.database import NEVER, materialize_features, timestamp_seconds  # noqa: E402
from experiments.relgt.test_relgt_db import DatabaseTestCase  # noqa: E402
from experiments.relational_transformer.cells import (  # noqa: E402
    PAD_ROW,
    SEM_DATETIME,
    SEM_NUMBER,
    WEEK_SECONDS,
    ActivityHistory,
    CellBuilder,
)
from experiments.relational_transformer.model import (
    RelationalTransformer,
    attention_masks,
)  # noqa: E402
from experiments.relational_transformer.run import model_cutoff, parse_args  # noqa: E402


DAY = 86_400


def _no_history(batch_size: int, weeks: int = 0):
    return (
        np.zeros((batch_size, weeks, len(LABELS)), np.uint8),
        np.zeros((batch_size, weeks), np.int64),
        np.zeros((batch_size, weeks), bool),
    )


class CellTestCase(DatabaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.builder = CellBuilder(
            self.database, self.graph, materialize_features(self.database, text_buckets=97)
        )

    def column(self, name: str) -> int:
        return self.builder.column_names.index(name)

    def row_columns(self, table: str, row: int, cutoff: int) -> set[str]:
        table_index = self.graph.table_names.index(table)
        cells = self.builder.table_cells(
            table_index, np.array([row]), np.array([cutoff], np.int64), np.array([0])
        )
        return {self.builder.column_names[c] for c in cells.column}


class CellTests(CellTestCase):
    def test_late_values_are_hidden_until_known(self) -> None:
        before = self.row_columns("PullRequests", 0, self.cutoff)
        after = self.row_columns("PullRequests", 0, timestamp_seconds(pd.Timestamp("2026-01-26")))
        self.assertNotIn("state of PullRequests", before)
        self.assertNotIn("merged_at of PullRequests", before)
        self.assertIn("state of PullRequests", after)
        self.assertIn("merged_at of PullRequests", after)
        self.assertIn("title of PullRequests", before)

    def test_missing_values_and_keys_are_not_cells(self) -> None:
        later = timestamp_seconds(pd.Timestamp("2026-02-01"))
        self.assertNotIn("title of PullRequests", self.row_columns("PullRequests", 2, later))
        self.assertNotIn("labels of PullRequests", self.row_columns("PullRequests", 1, later))
        for name in self.builder.column_names:
            self.assertFalse(name.split(" of ")[0].endswith("_id"), name)

    def test_parents_are_linked_only_once_known_and_in_context(self) -> None:
        nodes = np.array(
            [
                [
                    self.node("Users", 0),
                    self.node("PullRequests", 0),
                    self.node("Users", 1),
                    self.node("Repositories", 0),
                    self.node("PullRequests", 1),
                ]
            ]
        )
        batch = self.builder.build(nodes, np.array([self.cutoff]), *_no_history(1), max_cells=512)
        parents = batch["row_parents"][0]
        alice, pr0, bob, repo0, pr1 = range(1, 6)
        # merged_by_id = alice is only known at 2026-01-25.
        self.assertEqual(set(parents[pr0]) - {PAD_ROW}, {repo0, bob})
        self.assertEqual(set(parents[pr1]) - {PAD_ROW}, {repo0, alice})
        self.assertEqual(set(parents[0]) - {PAD_ROW}, {alice})  # task row -> user
        self.assertEqual(set(parents[alice]) - {PAD_ROW}, set())

    def test_targets_lead_and_survive_truncation(self) -> None:
        nodes = np.array(
            [
                [
                    self.node("Users", 0),
                    self.node("Pushes", 0),
                    self.node("Pushes", 0),
                    self.node("Pushes", 1),
                ],
                [
                    self.node("Users", 1),
                    self.node("PullRequests", 0),
                    self.node("Repositories", 0),
                    self.node("Users", 1),
                ],
            ]
        )
        cutoffs = np.array([self.cutoff, self.cutoff])
        full = self.builder.build(nodes, cutoffs, *_no_history(2), max_cells=512)
        short = self.builder.build(nodes, cutoffs, *_no_history(2), max_cells=9)
        self.assertEqual(short["row"].shape[1], 9)
        for batch in (full, short):
            self.assertEqual(int(batch["is_target"].sum()), 2 * len(LABELS))
            for b in range(2):
                positions = batch["target_pos"][b]
                self.assertTrue(batch["is_target"][b, positions].all())
                self.assertTrue((batch["row"][b, positions] == 0).all())
                self.assertEqual(
                    batch["column"][b, positions].tolist(), self.builder.label_columns.tolist()
                )
        # The repeated push and repeated seed are each one row.
        self.assertEqual(int(full["row"][0].max()), 3)
        self.assertEqual(int(full["row"][1].max()), 3)
        self.assertTrue(full["padding"][full["row"] == PAD_ROW].all())

    def test_history_rows_carry_visible_labels(self) -> None:
        labels = np.zeros((1, 2, len(LABELS)), np.uint8)
        labels[0, 0, 5] = 1
        starts = np.array([[self.cutoff - WEEK_SECONDS, self.cutoff - 2 * WEEK_SECONDS]])
        present = np.array([[True, False]])
        batch = self.builder.build(
            np.array([[self.node("Users", 0)]]),
            np.array([self.cutoff]),
            labels,
            starts,
            present,
            512,
        )
        history = batch["row"][0] == 1
        self.assertFalse((batch["row"][0] == 2).any())  # absent week has no cells
        self.assertFalse(batch["is_target"][0, history].any())
        push = history & (batch["column"][0] == self.column("push of UserWeeks"))
        self.assertEqual(batch["number"][0, push].tolist(), [1.0])
        cutoff_cell = history & (batch["sem"][0] == SEM_DATETIME)
        self.assertAlmostEqual(
            float(batch["datetime"][0, cutoff_cell][0, 0]), np.log1p(7.0), places=5
        )
        self.assertTrue((batch["sem"][0, batch["target_pos"][0]] == SEM_NUMBER).all())


class HistoryTests(unittest.TestCase):
    def test_labels_respect_windows_and_ingestion_time(self) -> None:
        start = timestamp_seconds(pd.Timestamp("2026-01-01"))
        cutoff = start + 3 * WEEK_SECONDS

        def at(seconds: int) -> pd.Timestamp:
            return pd.Timestamp(seconds, unit="s")

        events = pd.DataFrame(
            {
                "user_id": [7, 7, 7, 9],
                "relation_id": [6, 1, 2, 6],
                # push in week 2 (ingested before the cutoff), PR in week 2 but
                # ingested at the cutoff, issue exactly at week 1's end (= week 2).
                "action_at": [
                    at(start + 2 * WEEK_SECONDS + DAY),
                    at(start + 2 * WEEK_SECONDS + 2 * DAY),
                    at(start + 2 * WEEK_SECONDS),
                    at(start + DAY),
                ],
                "available_at": [
                    at(cutoff - 1),
                    at(cutoff),
                    at(start + 2 * WEEK_SECONDS),
                    at(start + DAY),
                ],
                "entity_available_at": [at(start)] * 4,
            }
        )
        history = ActivityHistory.build(events, np.array([9, 7]), start, cutoff + WEEK_SECONDS)
        labels, starts, present = history.rows(np.array([7, 9]), np.array([cutoff, cutoff]), 4)
        self.assertEqual(present.tolist(), [[True, True, True, False]] * 2)
        self.assertEqual(
            starts[0, :3].tolist(), [start + 2 * WEEK_SECONDS, start + WEEK_SECONDS, start]
        )
        self.assertEqual(labels[0, 0].tolist(), [0, 1, 0, 0, 0, 1])  # PR ingested too late
        self.assertFalse(labels[0, 1:].any())
        self.assertEqual(labels[1, 2].tolist(), [0, 0, 0, 0, 0, 1])
        # One week later the late PR is known.
        later, _, _ = history.rows(np.array([7]), np.array([cutoff + WEEK_SECONDS]), 2)
        self.assertEqual(later[0, 1].tolist(), [1, 1, 0, 0, 0, 1])
        self.assertEqual(history.first_known[0, 2, 0], timestamp_seconds(at(cutoff)))
        self.assertEqual(history.first_known[0, 0, 0], NEVER)
        with self.assertRaisesRegex(ValueError, "every requested user"):
            history.rows(np.array([8]), np.array([cutoff]), 1)


class ModelTests(CellTestCase):
    def model(self, **kwargs) -> RelationalTransformer:
        torch.manual_seed(0)
        model = RelationalTransformer(
            num_columns=self.builder.num_columns,
            num_categories=self.builder.num_categories,
            text_buckets=97,
            d_model=16,
            num_heads=2,
            num_blocks=2,
            d_ff=32,
            **kwargs,
        )
        # Upstream zero-initializes residual outputs and decoders; perturb them
        # so the tests see attention.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        return model.eval()

    def batch(self, nodes: list[list[int]], max_cells: int = 512) -> dict[str, torch.Tensor]:
        labels = np.zeros((len(nodes), 1, len(LABELS)), np.uint8)
        labels[:, 0, 0] = 1
        starts = np.full((len(nodes), 1), self.cutoff - WEEK_SECONDS)
        present = np.ones((len(nodes), 1), bool)
        arrays = self.builder.build(
            np.array(nodes), np.full(len(nodes), self.cutoff), labels, starts, present, max_cells
        )
        return {key: torch.as_tensor(value) for key, value in arrays.items()}

    def test_masks_follow_foreign_keys(self) -> None:
        alice, pr1, repo0 = (
            self.node("Users", 0),
            self.node("PullRequests", 1),
            self.node("Repositories", 0),
        )
        batch = self.batch([[alice, pr1, repo0]])
        masks = attention_masks(batch, ("col", "feat", "nbr", "full"))
        row = batch["row"][0]
        user_rows, pr_rows, repo_rows = 2, 3, 4  # after the task row and one history row
        q = int(torch.nonzero(row == pr_rows)[0])
        feat_rows = set(row[masks["feat"][0, q]].tolist())
        self.assertEqual(feat_rows, {pr_rows, user_rows, repo_rows})
        q_user = int(torch.nonzero(row == user_rows)[0])
        self.assertEqual(set(row[masks["nbr"][0, q_user]].tolist()), {0, 1, pr_rows})
        self.assertEqual(set(row[masks["feat"][0, q_user]].tolist()), {user_rows})
        target = int(batch["target_pos"][0, 5])
        col_rows = set(row[masks["col"][0, target]].tolist())
        self.assertEqual(col_rows, {0, 1})  # the target sees last week's push label
        self.assertFalse(masks["full"][0, :, batch["padding"][0]].any())

    def test_padding_and_batch_neighbors_do_not_change_predictions(self) -> None:
        model = self.model(attention_types=("col", "feat", "nbr", "full"))
        alice, bob = self.node("Users", 0), self.node("Users", 1)
        short = [alice, self.node("Pushes", 0), alice]
        long = [bob, self.node("PullRequests", 0), self.node("Repositories", 0)]
        with torch.no_grad():
            alone, _ = model(self.batch([short]))
            together, _ = model(self.batch([short, long]))
        self.assertEqual(tuple(together.shape), (2, len(LABELS)))
        torch.testing.assert_close(alone[0], together[0], atol=1e-5, rtol=1e-5)

    def test_targets_are_hidden_from_the_model(self) -> None:
        model = self.model()
        batch = self.batch([[self.node("Users", 0), self.node("Pushes", 1)]])
        changed = {key: value.clone() for key, value in batch.items()}
        changed["number"][batch["is_target"]] = 1.0
        with torch.no_grad():
            torch.testing.assert_close(model(batch)[0], model(changed)[0])

    def test_training_step_and_mtp_loss(self) -> None:
        model = self.model().train()
        batch = self.batch(
            [
                [self.node("Users", 0), self.node("Pushes", 0)],
                [self.node("Users", 1), self.node("PullRequests", 0)],
            ]
        )
        mtp = (batch["sem"] == SEM_NUMBER) & ~batch["is_target"] & ~batch["padding"]
        logits, mtp_loss = model(batch, mtp)
        self.assertIsNotNone(mtp_loss)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
            + mtp_loss
        )
        loss.backward()
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )
        self.assertIsNone(model(batch)[1])

    def test_invalid_attention_types(self) -> None:
        with self.assertRaises(ValueError):
            self.model(attention_types=("rows",))


class ArgumentTests(unittest.TestCase):
    def test_model_cutoff_is_midnight_like_training_examples(self) -> None:
        cutoff = pd.Timestamp("2026-07-01 16:00:00")
        days = cutoff.value // pd.Timedelta(days=1).value  # relgt_db's cutoff_days
        self.assertEqual(model_cutoff(cutoff), days * DAY)

    def test_defaults(self) -> None:
        args = parse_args(["backtest"])
        self.assertEqual(args.attention, ("col", "feat", "nbr"))
        self.assertEqual((args.num_neighbors, args.history_weeks, args.max_cells), (64, 8, 512))
        self.assertIn("outputs", str(args.output_dir))
        args = parse_args(["predict", "--attention", "col,feat,nbr,full"])
        self.assertEqual(args.attention[-1], "full")

    def test_invalid_values_are_rejected(self) -> None:
        for argv in (
            ["backtest", "--attention", "rows"],
            ["backtest", "--d-model", "30", "--num-heads", "4"],
            ["backtest", "--max-cells", "10"],
            ["backtest", "--history-weeks", "-1"],
            ["backtest", "--mtp-prob", "1.5"],
            ["backtest", "--num-neighbors", "1"],
            ["backtest", "--epochs", "0"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
