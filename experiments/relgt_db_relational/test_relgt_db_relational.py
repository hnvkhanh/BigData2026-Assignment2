from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.relational_lightgbm.features import RelationalTables
from experiments.relational_lightgbm.test_relational_lightgbm import CUTOFF, DAY, OTHER, OWNER, _tables
from experiments.relgt.database import RelationalDatabase, build_temporal_graph, load_database, timestamp_seconds
from experiments.relgt.db_sampler import sample_seed, sample_tokens
from experiments.relgt.test_relgt_db import SCHEMA, _tables as _db_tables
from experiments.relgt_db_relational.run import parse_args, parse_tables
from experiments.relgt_db_relational.sampler import reserved_sampler, reserved_tokens
from experiments.relgt_db_relational.tables import (
    DERIVED_SCHEMA,
    DERIVED_TABLES,
    collaborations,
    notification_events,
    notifications,
)


def _relational_tables() -> RelationalTables:
    tables = _tables()
    mentions = tables.mentions.assign(node_type=2, entity_id=1)
    return RelationalTables(**{**tables.__dict__, "mentions": mentions})


class TableTests(unittest.TestCase):
    def setUp(self) -> None:
        issue_pr = pd.Series(pd.NA, index=pd.Index([0, 1, 2], name="issue_id"), dtype="Int64")
        self.events = notification_events(_relational_tables(), issue_pr)

    def test_notifications_follow_the_relationships(self) -> None:
        owner = self.events.loc[self.events["user_id"] == OWNER]
        self.assertEqual(
            sorted(owner["kind"].unique()),
            [
                "comment_on_authored",
                "issue_on_owned_repo",
                "mention",
                "pr_on_owned_repo",
                "review_comment_on_authored_pr",
                "review_on_authored_pr",
            ],
        )
        # Issue 1 is the owner's own, so only issues 0 and 2 notify them.
        owned = owner.loc[owner["kind"] == "issue_on_owned_repo", "issue_id"]
        self.assertEqual(sorted(owned.tolist()), [0, 2])
        # The owner merges OTHER's PR 1, known only at merged_at.
        other = self.events.loc[self.events["user_id"] == OTHER]
        self.assertEqual(other["kind"].tolist(), ["authored_pr_merged_by_other"])
        self.assertEqual(other["known_at"].iloc[0], CUTOFF + DAY)
        self.assertFalse((self.events["user_id"] == self.events["actor_id"]).any())
        self.assertTrue((self.events["known_at"] >= self.events["acted_at"]).all())

    def test_tables_are_sorted_and_keyed_by_row(self) -> None:
        for frame, name in (
            (notifications(self.events), "Notifications"),
            (collaborations(self.events), "Collaborations"),
        ):
            schema = DERIVED_SCHEMA[name]
            np.testing.assert_array_equal(frame[schema.primary_key], np.arange(len(frame)))
            self.assertTrue(frame[schema.time_column].is_monotonic_increasing)
        pairs = collaborations(self.events)
        self.assertEqual(pairs[["user_a_id", "user_b_id"]].values.tolist(), [[OWNER, OTHER]])
        self.assertEqual(pairs["known_at"].iloc[0], self.events["known_at"].min())


class SamplerTests(unittest.TestCase):
    """The ``relgt_db`` test database plus a small Notifications table."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_dir = Path(directory.name)
        for name, frame in _db_tables().items():
            frame.to_parquet(db_dir / f"{name}.parquet", index=False)
        base = load_database(db_dir, SCHEMA)
        notes = pd.DataFrame(
            {
                "notification_id": [0, 1, 2],
                "user_id": pd.array([0, 0, 0], dtype="Int64"),
                "pr_id": pd.array([0, 1, 2], dtype="Int64"),
                "issue_id": pd.array([pd.NA] * 3, dtype="Int64"),
                "kind": ["mention", "pr_on_owned_repo", "mention"],
                "known_at": pd.to_datetime(["2026-01-08", "2026-01-12", "2026-01-25"]),
            }
        ).drop(columns="issue_id")
        schema = DERIVED_SCHEMA["Notifications"]
        schema = type(schema)(
            schema.name,
            schema.primary_key,
            schema.time_column,
            foreign_keys={"user_id": "Users", "pr_id": "PullRequests"},
        )
        database = RelationalDatabase(SCHEMA + (schema,), {**base.tables, "Notifications": notes})
        self.graph = build_temporal_graph(database)
        self.cutoff = timestamp_seconds(pd.Timestamp("2026-01-20"))
        self.alice = int(self.graph.global_ids("Users", np.array([0]))[0])
        self.types = np.array([self.graph.node_type("Notifications")])

    def node(self, table: str, row: int) -> int:
        return int(self.graph.global_ids(table, np.array([row]))[0])

    def test_reserved_rows_are_known_recent_first_with_their_targets(self) -> None:
        nodes, hops = reserved_tokens(self.graph, self.alice, self.cutoff, 8, self.types)
        # Notification 2 is known after the cutoff; 1 is newer than 0.
        expected = [
            self.node("Notifications", 1),
            self.node("PullRequests", 1),
            self.node("Notifications", 0),
            self.node("PullRequests", 0),
        ]
        self.assertEqual(nodes.tolist(), expected)
        # Alice authored PR 1 (a direct neighbour); PR 0 links to her only after the cutoff.
        self.assertEqual(hops.tolist(), [1, 1, 1, 2])
        self.assertTrue(np.all(self.graph.node_time[nodes] < self.cutoff))
        capped, _ = reserved_tokens(self.graph, self.alice, self.cutoff, 3, self.types)
        self.assertEqual(capped.tolist(), expected[:3])

    def test_reserved_sampler_keeps_rows_and_known_edges(self) -> None:
        sampler = reserved_sampler(self.graph, ["Notifications"], 4)
        nodes, hops, edges = sampler(self.graph, self.alice, self.cutoff, 16, np.random.default_rng(0))
        self.assertEqual(nodes[0], self.alice)
        self.assertEqual(hops[0], 0)
        self.assertEqual(len(nodes), 16)
        self.assertIn(self.node("Notifications", 1), nodes[1:5])
        self.assertTrue(np.all(self.graph.node_time[nodes] < self.cutoff))
        self.assertGreater(edges.shape[1], 0)
        self.assertLess(edges.max(), 16)
        tokens = sample_tokens(
            self.graph, np.array([self.alice]), np.array([self.cutoff]), 16, seed=1, seed_sampler=sampler
        )
        self.assertEqual(tokens.node_types.shape, (1, 16))

    def test_no_reserve_is_upstream_sampling(self) -> None:
        self.assertIs(reserved_sampler(self.graph, ["Notifications"], 0), sample_seed)
        self.assertIs(reserved_sampler(self.graph, [], 8), sample_seed)


class CliTests(unittest.TestCase):
    def test_tables_and_reserve_options(self) -> None:
        self.assertEqual(parse_tables("none"), [])
        self.assertEqual(parse_tables("all"), list(DERIVED_TABLES))
        self.assertEqual(parse_tables("Collaborations,RepoOwners"), ["RepoOwners", "Collaborations"])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_tables("Nope")
        args = parse_args(["backtest", "--reserve", "32", "--output-dir", "unused"])
        self.assertEqual((args.temporal_model, args.reserve, args.tables), ("relgt_db", 32, list(DERIVED_TABLES)))
        with self.assertRaises(SystemExit):
            parse_args(["backtest", "--tables", "none", "--reserve", "8"])


if __name__ == "__main__":
    unittest.main()
