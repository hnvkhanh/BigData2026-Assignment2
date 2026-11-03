from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.relgt.data import EntityVocabulary, LABELS, snapshot_tokens
from experiments.relgt.database import (
    NEVER,
    TableSchema,
    build_temporal_graph,
    load_database,
    materialize_features,
    materialize_table,
    stable_hash,
    timestamp_seconds,
)
from experiments.relgt.db_model import RelGT, TableEncoder
from experiments.relgt.db_run import eligible_users, predict_db_probabilities
from experiments.relgt.db_sampler import (
    FALLBACK_HOP,
    batch_edge_index,
    sample_seed,
    sample_tokens,
)
from experiments.relgt.run import parse_args


# Tiny tables make cardinality-based typing arbitrary, so stypes are declared.
SCHEMA = (
    TableSchema(
        "Users",
        "user_id",
        "first_seen_at",
        drop_columns=("github_user_id",),
        stypes={"login": "text"},
    ),
    TableSchema("Repositories", "repo_id", "first_seen_at"),
    TableSchema(
        "PullRequests",
        "pr_id",
        "first_seen_at",
        foreign_keys={
            "repo_id": "Repositories",
            "author_id": "Users",
            "merged_by_id": "Users",
        },
        late_columns={"merged_at": "merged_at", "merged_by_id": "merged_at",
                      "state": "merged_at"},
        stypes={
            "title": "text",
            "labels": "multicategorical",
            "state": "categorical",
            "merged_at": "timestamp",
        },
    ),
    TableSchema(
        "Pushes",
        "push_id",
        "created_at",
        foreign_keys={"user_id": "Users", "repo_id": "Repositories"},
    ),
)


def _tables() -> dict[str, pd.DataFrame]:
    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "github_user_id": [100, 101, 102],
            "login": ["alice", "bob-dev", "renovate[bot]"],
            "is_bot": pd.array([False, False, True], dtype="boolean"),
            "first_seen_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )
    repositories = pd.DataFrame(
        {
            "repo_id": [0, 1],
            "name": ["octo/core", "octo/docs"],
            "first_seen_at": pd.to_datetime(["2026-01-01", "2026-01-04"]),
        }
    )
    pull_requests = pd.DataFrame(
        {
            "pr_id": [0, 1, 2],
            "repo_id": pd.array([0, 0, 1], dtype="Int64"),
            "author_id": pd.array([1, 0, pd.NA], dtype="Int64"),
            "merged_by_id": pd.array([0, pd.NA, 1], dtype="Int64"),
            "number": pd.array([1, 2, 3], dtype="Int64"),
            "title": ["Fix the parser", "Add docs", None],
            "labels": ["bug|core", "", "docs"],
            "state": ["merged", None, "closed"],
            "constant": ["x", "x", "x"],
            # PR 0 is merged after the 2026-01-20 cutoff; PR 2 never merges
            # in time for its key to be known.
            "merged_at": pd.to_datetime(["2026-01-25", None, None]),
            "first_seen_at": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        }
    )
    pushes = pd.DataFrame(
        {
            "push_id": [0, 1, 2],
            "user_id": pd.array([0, 0, 1], dtype="Int64"),
            "repo_id": pd.array([0, 1, 0], dtype="Int64"),
            "branch": ["main", "feature/x", "main"],
            "created_at": pd.to_datetime(["2026-01-08", "2026-01-10", "2026-01-30"]),
        }
    )
    return {
        "Users": users,
        "Repositories": repositories,
        "PullRequests": pull_requests,
        "Pushes": pushes,
    }


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.db_dir = Path(self.directory.name)
        for name, frame in _tables().items():
            frame.to_parquet(self.db_dir / f"{name}.parquet", index=False)
        self.database = load_database(self.db_dir, SCHEMA)
        self.graph = build_temporal_graph(self.database)
        self.cutoff = timestamp_seconds(pd.Timestamp("2026-01-20"))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def node(self, table: str, row: int) -> int:
        return int(self.graph.global_ids(table, np.array([row]))[0])

    def neighbors(self, node: int) -> dict[int, int]:
        start, end = self.graph.indptr[node], self.graph.indptr[node + 1]
        return dict(zip(self.graph.indices[start:end].tolist(),
                        self.graph.edge_time[start:end].tolist()))


class SchemaAndFeatureTests(DatabaseTestCase):
    def test_columns_are_typed_and_keys_are_not_features(self) -> None:
        features = materialize_features(self.database, text_buckets=97)
        pulls = features["PullRequests"]
        self.assertEqual(pulls.numerical_columns, ["number"])
        self.assertEqual(pulls.categorical_columns, ["state"])
        self.assertEqual(pulls.text_columns, ["title", "labels"])
        self.assertEqual(pulls.timestamp_columns, ["merged_at", "first_seen_at"])
        self.assertNotIn("constant", pulls.text_columns + pulls.categorical_columns)
        users = features["Users"]
        self.assertEqual(users.categorical_columns, ["is_bot"])
        self.assertEqual(users.text_columns, ["login"])

    def test_late_columns_point_at_their_known_time(self) -> None:
        pulls = materialize_features(self.database, text_buckets=97)["PullRequests"]
        self.assertEqual(pulls.known_at.shape, (3, 1))
        self.assertEqual(pulls.known_at[0, 0], timestamp_seconds(pd.Timestamp("2026-01-25")))
        self.assertEqual(pulls.known_at[1, 0], NEVER)
        self.assertEqual(pulls.categorical_known_at.tolist(), [0])
        self.assertEqual(pulls.timestamp_known_at.tolist(), [0, -1])
        self.assertEqual(pulls.numerical_known_at.tolist(), [-1])

    def test_text_and_multicategorical_hashing_is_stable(self) -> None:
        self.assertEqual(stable_hash("main", 97), stable_hash("main", 97))
        self.assertTrue(1 <= stable_hash("main", 97) < 97)
        pulls = materialize_features(self.database, text_buckets=97)["PullRequests"]
        title, labels = pulls.text
        self.assertEqual(int((title[0] > 0).sum()), 3)  # fix, the, parser
        self.assertFalse(title[2].any())
        self.assertEqual(int((labels[0] > 0).sum()), 2)
        self.assertFalse(labels[1].any())

    def test_primary_keys_must_be_row_indices(self) -> None:
        users = _tables()["Users"]
        users["user_id"] = [5, 6, 7]
        users.to_parquet(self.db_dir / "Users.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "row index"):
            load_database(self.db_dir, SCHEMA)


class TemporalGraphTests(DatabaseTestCase):
    def test_foreign_keys_become_undirected_edges(self) -> None:
        pr0 = self.node("PullRequests", 0)
        user1 = self.node("Users", 1)
        self.assertIn(user1, self.neighbors(pr0))
        self.assertIn(pr0, self.neighbors(user1))
        self.assertIn(self.node("Repositories", 0), self.neighbors(pr0))

    def test_late_key_edge_is_known_only_when_its_value_is(self) -> None:
        pr0 = self.node("PullRequests", 0)
        alice = self.node("Users", 0)
        self.assertEqual(
            self.neighbors(pr0)[alice], timestamp_seconds(pd.Timestamp("2026-01-25"))
        )
        # PR 2's merged_by_id has no merged_at, so its edge is never known.
        self.assertNotIn(self.node("Users", 1), self.neighbors(self.node("PullRequests", 2)))

    def test_edges_are_known_after_both_rows(self) -> None:
        push1 = self.node("Pushes", 1)
        docs = self.node("Repositories", 1)
        self.assertEqual(
            self.neighbors(push1)[docs], timestamp_seconds(pd.Timestamp("2026-01-10"))
        )
        for node in range(self.graph.num_nodes):
            times = self.graph.edge_time[self.graph.indptr[node] : self.graph.indptr[node + 1]]
            self.assertTrue(np.all(np.diff(times) >= 0))


class SamplerTests(DatabaseTestCase):
    def test_tokens_only_use_rows_and_edges_known_before_seed_time(self) -> None:
        alice = self.node("Users", 0)
        nodes, hops, edges = sample_seed(
            self.graph, alice, self.cutoff, 16, np.random.default_rng(0)
        )
        self.assertEqual(nodes[0], alice)
        self.assertEqual(hops[0], 0)
        self.assertTrue(np.all(self.graph.node_time[nodes] < self.cutoff))
        # PR 0 links to alice only through merged_by_id, merged after the cutoff.
        self.assertNotIn(self.node("PullRequests", 0), nodes[hops == 1])
        for source, target in edges.T:
            self.assertLess(self.neighbors(int(nodes[source]))[int(nodes[target])], self.cutoff)

    def test_hops_follow_the_key_graph(self) -> None:
        alice = self.node("Users", 0)
        nodes, hops, _ = sample_seed(
            self.graph, alice, self.cutoff, 64, np.random.default_rng(1)
        )
        found = dict(zip(nodes[1:].tolist(), hops[1:].tolist()))
        self.assertEqual(found[self.node("Pushes", 0)], 1)
        self.assertEqual(found[self.node("PullRequests", 1)], 1)
        self.assertEqual(found[self.node("Repositories", 0)], 2)
        self.assertNotIn(self.node("Pushes", 2), found)  # created after the cutoff

    def test_context_modes(self) -> None:
        alice = self.node("Users", 0)
        repo = self.node("Repositories", 0)

        def run(mode: str):
            return sample_seed(
                self.graph, alice, self.cutoff, 64, np.random.default_rng(1), mode
            )

        _, hops, _ = run("r0")
        self.assertTrue(set(hops[1:].tolist()) <= {1})
        nodes, hops, _ = run("r1")
        self.assertNotIn(repo, nodes)
        self.assertIn(1, hops)
        nodes, hops, _ = run("r2")
        self.assertIn(repo, nodes)
        nodes, hops, _ = run("r3")
        self.assertTrue(np.all(hops[1:] == FALLBACK_HOP))
        self.assertTrue(np.all(self.graph.node_time[nodes] < self.cutoff))
        with self.assertRaises(ValueError):
            run("r9")

    def test_merged_by_edge_appears_once_known(self) -> None:
        later = timestamp_seconds(pd.Timestamp("2026-01-26"))
        nodes, hops, _ = sample_seed(
            self.graph, self.node("Users", 0), later, 64, np.random.default_rng(2)
        )
        self.assertIn(self.node("PullRequests", 0), nodes[hops == 1])

    def test_isolated_seed_falls_back_to_known_random_rows(self) -> None:
        bot = self.node("Users", 2)
        nodes, hops, _ = sample_seed(
            self.graph, bot, self.cutoff, 8, np.random.default_rng(3)
        )
        self.assertTrue(np.all(hops[1:] == FALLBACK_HOP))
        self.assertTrue(np.all(self.graph.node_time[nodes] < self.cutoff))

    def test_samples_are_deterministic_and_batch_edges_are_offset(self) -> None:
        seeds = self.graph.global_ids("Users", np.array([0, 1]))
        times = np.full(2, self.cutoff)
        first = sample_tokens(self.graph, seeds, times, 6, seed=7)
        second = sample_tokens(self.graph, seeds, times, 6, seed=7)
        np.testing.assert_array_equal(first.node_rows, second.node_rows)
        np.testing.assert_array_equal(first.edges, second.edges)
        self.assertEqual(first.node_types[0, 0], self.graph.node_type("Users"))
        self.assertEqual(first.relative_days[0, 0], 0.0)
        edge_index = batch_edge_index(first, np.array([1]))
        np.testing.assert_array_equal(
            edge_index, first.edges[:, first.edge_offsets[1] : first.edge_offsets[2]]
        )
        both = batch_edge_index(first, np.array([0, 1]))
        self.assertTrue(np.all(both[:, first.edge_offsets[1] :] >= 6))


class ModelTests(DatabaseTestCase):
    def _encoder_output(self, merged_at: list[str | None], cutoff: str) -> torch.Tensor:
        frame = self.database.tables["PullRequests"].copy()
        frame["merged_at"] = pd.to_datetime(merged_at)
        features = materialize_table(frame, SCHEMA[2], text_buckets=97)
        torch.manual_seed(0)
        encoder = TableEncoder(features, channels=8).eval()
        embedding = torch.nn.Embedding(97, 8, padding_idx=0)
        cutoffs = torch.full((3,), timestamp_seconds(pd.Timestamp(cutoff)))
        with torch.no_grad():
            return encoder(torch.arange(3), cutoffs, embedding)

    def test_values_known_after_the_cutoff_do_not_change_features(self) -> None:
        merged = self._encoder_output(["2026-01-25", None, None], "2026-01-20")
        unmerged = self._encoder_output([None, None, None], "2026-01-20")
        torch.testing.assert_close(merged, unmerged)
        later_merged = self._encoder_output(["2026-01-25", None, None], "2026-01-26")
        later_unmerged = self._encoder_output([None, None, None], "2026-01-26")
        self.assertFalse(torch.allclose(later_merged[0], later_unmerged[0]))

    def test_relgt_trains_for_each_attention_variant(self) -> None:
        features = materialize_features(self.database, text_buckets=97)
        seeds = self.graph.global_ids("Users", np.array([0, 1, 0, 1]))
        times = np.array([self.cutoff, self.cutoff, self.cutoff + 86_400, self.cutoff])
        tokens = sample_tokens(self.graph, seeds, times, 6, seed=0)
        indices = np.arange(4)
        inputs = (
            torch.as_tensor(tokens.node_types, dtype=torch.long),
            torch.as_tensor(tokens.node_rows, dtype=torch.long),
            torch.as_tensor(tokens.hops, dtype=torch.long),
            torch.as_tensor(tokens.relative_days),
            torch.as_tensor(times),
            torch.tensor([0, 1, 0, 1]),
            torch.as_tensor(batch_edge_index(tokens, indices)),
        )
        for conv_type, ablate in (("full", "none"), ("local", "gnn"), ("global", "type")):
            with self.subTest(conv_type=conv_type, ablate=ablate):
                torch.manual_seed(0)
                model = RelGT(
                    [features[name] for name in self.graph.table_names],
                    num_seed_nodes=3,
                    channels=16,
                    num_heads=2,
                    num_layers=1,
                    num_centroids=4,
                    conv_type=conv_type,
                    ablate=ablate,
                    text_buckets=97,
                )
                logits = model(*inputs)
                self.assertEqual(logits.shape, (4, len(LABELS)))
                logits.sum().backward()
                self.assertTrue(torch.isfinite(logits).all())
                # Table arrays are not checkpointed and the text vocabulary
                # is stored once, not once per table.
                keys = set(model.state_dict())
                self.assertNotIn("tfs_encoder.encoders.0.text_0", keys)
                self.assertNotIn("tfs_encoder.encoders.3.timestamps", keys)
                self.assertEqual(
                    sum(key.endswith("text_embedding.weight") for key in keys), 1
                )


    def test_inference_is_reproducible_and_keeps_the_training_rng(self) -> None:
        features = materialize_features(self.database, text_buckets=97)
        user_ids = np.array([0, 1, 0], dtype=np.int32)
        cutoffs = np.full(3, self.cutoff)
        tokens = sample_tokens(
            self.graph, self.graph.global_ids("Users", user_ids), cutoffs, 6, seed=0
        )
        model = RelGT(
            [features[name] for name in self.graph.table_names],
            num_seed_nodes=3, channels=16, num_heads=2, num_layers=1,
            num_centroids=4, text_buckets=97,
        )
        state = torch.random.get_rng_state()
        first, second = (
            predict_db_probabilities(
                model, tokens, user_ids, cutoffs, np.arange(3), torch.device("cpu"), 2, 5
            )
            for _ in range(2)
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


class SampleAlignmentTests(unittest.TestCase):
    def test_eligibility_matches_relgt_snapshots(self) -> None:
        cutoff = pd.Timestamp("2026-01-15")
        rows = []
        for user, count, available in ((10, 6, "2026-01-10"), (11, 4, "2026-01-10"),
                                        (12, 5, "2026-01-16"), (13, 5, "2026-01-12")):
            for index in range(count):
                rows.append((user, 0, index, 1 + index, 1, 1,
                             f"2026-01-{10 + index % 4:02d}", available, "2026-01-01"))
        events = pd.DataFrame(
            rows,
            columns=["user_id", "repo_id", "entity_id", "global_entity_id", "node_type",
                     "relation_id", "action_at", "available_at", "entity_available_at"],
        )
        for column in ("action_at", "available_at", "entity_available_at"):
            events[column] = pd.to_datetime(events[column])
        user_ids = np.array([13, 12, 11, 10], dtype=np.int32)
        vocabulary = EntityVocabulary(num_pull_requests=8, num_issues=0, num_repositories=1)
        expected = snapshot_tokens(events, cutoff, user_ids, vocabulary, eligible_only=True)
        np.testing.assert_array_equal(eligible_users(events, cutoff, user_ids), expected.user_ids)
        np.testing.assert_array_equal(expected.user_ids, [13, 10])


class ArgumentTests(unittest.TestCase):
    def test_relgt_db_arguments(self) -> None:
        argv = ["run.py", "backtest", "--temporal-model", "relgt_db",
                "--num-neighbors", "64", "--conv-type", "local", "--ablate", "gnn"]
        with patch("sys.argv", argv):
            args = parse_args()
        self.assertEqual(args.num_neighbors, 64)
        self.assertEqual(args.conv_type, "local")
        self.assertEqual(args.ablate, "gnn")

    def test_relgt_db_rejects_user_embedding(self) -> None:
        argv = ["run.py", "backtest", "--temporal-model", "relgt_db", "--user-embedding"]
        with patch("sys.argv", argv), patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args()


if __name__ == "__main__":
    unittest.main()
