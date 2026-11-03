from __future__ import annotations

import argparse
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.relational_lightgbm.features import RelationalTables
from experiments.relational_lightgbm.test_relational_lightgbm import CUTOFF, DAY, OTHER, OWNER, _tables
from experiments.relgt.data import MAX_NEIGHBORS, EntityVocabulary, GraphTokens
from experiments.relgt_relational.run import _logits
from experiments.relgt_relgraph.graph import (
    ISSUE_NODE,
    PR_NODE,
    RELATION_IDS,
    RELATIONS,
    REPO_NODE,
    USER_NODE,
    GlobalIds,
    Snapshot,
    concat_tokens,
    num_relation_slots,
    relation_slots,
    relation_tokens,
)
from experiments.relgt_relgraph.run import model_factory, parse_args, parse_relations

IDS = GlobalIds(EntityVocabulary(num_pull_requests=3, num_issues=3, num_repositories=30), num_users=2)
COHORT = pd.Index([OWNER, OTHER])


def _graph_tables() -> RelationalTables:
    tables = _tables()
    mentions = tables.mentions.assign(node_type=ISSUE_NODE, entity_id=1)
    return RelationalTables(**{**tables.__dict__, "mentions": mentions})


def _edges(tokens: GraphTokens, row: int, relation: str) -> list[tuple[int, int]]:
    """(node type, global id) of one user's edges for one relation."""
    slots = relation_slots(list(RELATIONS))[relation]
    mask = tokens.relation_ids[row, slots] == RELATION_IDS[relation]
    return list(zip(tokens.node_type_ids[row, slots][mask].tolist(), tokens.entity_ids[row, slots][mask].tolist()))


class RelationTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        tables = _graph_tables()
        self.tokens = relation_tokens(
            Snapshot.at(tables, CUTOFF), np.array([OWNER, OTHER]), list(RELATIONS), IDS, tables.repo_owner, COHORT
        )

    def test_shapes_and_relation_ids(self) -> None:
        width = num_relation_slots(list(RELATIONS))
        self.assertEqual(self.tokens.entity_ids.shape, (2, width))
        self.assertEqual(self.tokens.node_features.shape, (2, width, 4))
        used = set(np.unique(self.tokens.relation_ids)) - {0}
        self.assertTrue(used <= set(RELATION_IDS.values()))
        self.assertTrue(np.all(self.tokens.relative_days >= 0))

    def test_ownership_review_and_author_edges(self) -> None:
        repo = lambda repo_id: IDS.of(REPO_NODE, [repo_id])[0]  # noqa: E731
        pr = lambda pr_id: IDS.of(PR_NODE, [pr_id])[0]  # noqa: E731
        self.assertEqual(_edges(self.tokens, 0, "owns"), [(REPO_NODE, repo(10))])
        self.assertEqual(_edges(self.tokens, 1, "owns"), [])
        # PR 0 (by the owner) is open; feat-a has no PR yet and lives on repo 10.
        self.assertEqual(_edges(self.tokens, 0, "open_authored_pr"), [(PR_NODE, pr(0))])
        self.assertEqual(_edges(self.tokens, 0, "unpaired_branch"), [(REPO_NODE, repo(10))])
        # OTHER reviewed on repo 20; PR 2 there is not yet available, PR 0 is older than 30 days.
        self.assertEqual(_edges(self.tokens, 1, "reviewed_repo"), [(REPO_NODE, repo(20))])
        self.assertEqual(_edges(self.tokens, 1, "review_candidate"), [])
        # The owner is a review candidate for PR 1 (by OTHER) on the owned repo 10.
        self.assertEqual(_edges(self.tokens, 0, "review_candidate"), [(PR_NODE, pr(1))])

    def test_issue_and_people_edges(self) -> None:
        issue = lambda issue_id: IDS.of(ISSUE_NODE, [issue_id])[0]  # noqa: E731
        user = lambda user_id: IDS.of(USER_NODE, [user_id])[0]  # noqa: E731
        self.assertEqual(_edges(self.tokens, 1, "assigned"), [(ISSUE_NODE, issue(0))])  # issue 2 is closed
        self.assertEqual(_edges(self.tokens, 0, "mentioned_in"), [(ISSUE_NODE, issue(1))])
        self.assertEqual(_edges(self.tokens, 0, "thread"), [(ISSUE_NODE, issue(1))])
        self.assertEqual(_edges(self.tokens, 0, "interacted"), [(USER_NODE, user(OTHER))])
        self.assertEqual(_edges(self.tokens, 1, "interacted"), [(USER_NODE, user(OWNER))])
        slots = relation_slots(list(RELATIONS))["interacted"]
        partner = self.tokens.node_features[0, slots][0]
        self.assertEqual(partner[2], 1.0)  # the partner is a cohort user

    def test_future_rows_create_no_edges(self) -> None:
        tables = _graph_tables()
        early = CUTOFF - pd.Timedelta(days=200)
        tokens = relation_tokens(
            Snapshot.at(tables, early), np.array([OWNER, OTHER]), list(RELATIONS), IDS, tables.repo_owner, COHORT
        )
        self.assertFalse(tokens.entity_ids.any())

    def test_slot_limit_keeps_the_highest_scoring_edges(self) -> None:
        tables = _graph_tables()
        many = pd.DataFrame(
            {
                "user_id": OWNER,
                "repo_id": np.arange(10, 16),
                "branch": [f"feat-{index}" for index in range(6)],
                "at": [CUTOFF - DAY * (index + 1) / 10 for index in range(6)],
            }
        )
        tables = RelationalTables(**{**tables.__dict__, "pushes": many})
        tokens = relation_tokens(
            Snapshot.at(tables, CUTOFF), np.array([OWNER]), ["unpaired_branch"], IDS, tables.repo_owner, COHORT
        )
        self.assertEqual(tokens.entity_ids.shape[1], RELATIONS["unpaired_branch"][1])
        # Most recent first: repos 10..13 of the six pushed ones.
        self.assertEqual(tokens.entity_ids[0].tolist(), IDS.of(REPO_NODE, [10, 11, 12, 13]).tolist())

    def test_concat_requires_aligned_users(self) -> None:
        base = relation_tokens(Snapshot.at(_graph_tables(), CUTOFF), np.array([OWNER]), [], IDS, pd.Series(dtype="int64"), COHORT)
        other = relation_tokens(Snapshot.at(_graph_tables(), CUTOFF), np.array([OTHER]), [], IDS, pd.Series(dtype="int64"), COHORT)
        with self.assertRaises(ValueError):
            concat_tokens(base, other)


class ModelAndCliTests(unittest.TestCase):
    def test_models_run_on_augmented_tokens(self) -> None:
        tables = _graph_tables()
        extra = relation_tokens(
            Snapshot.at(tables, CUTOFF), np.array([OWNER, OTHER]), list(RELATIONS), IDS, tables.repo_owner, COHORT
        )
        base = GraphTokens(
            user_ids=extra.user_ids,
            entity_ids=np.zeros((2, MAX_NEIGHBORS), np.int32),
            node_type_ids=np.zeros((2, MAX_NEIGHBORS), np.uint8),
            relation_ids=np.zeros((2, MAX_NEIGHBORS), np.uint8),
            hop_ids=np.zeros((2, MAX_NEIGHBORS), np.uint8),
            node_features=np.zeros((2, MAX_NEIGHBORS, 4), np.float32),
            relative_days=np.zeros((2, MAX_NEIGHBORS), np.float32),
        )
        tokens = concat_tokens(base, extra)
        for architecture in ("relgt", "gnn"):
            args = parse_args(["backtest", "--architecture", architecture, "--hidden-dim", "16"])
            model = model_factory(args, IDS.num_entities)().eval()
            logits = _logits(model, tokens, np.zeros((2, 0), np.float32), np.arange(2), torch.device("cpu"))
            self.assertEqual(tuple(logits.shape), (2, 6))
            self.assertTrue(torch.isfinite(logits).all())

    def test_relations_option(self) -> None:
        self.assertEqual(parse_relations("none"), [])
        self.assertEqual(parse_relations("all"), list(RELATIONS))
        self.assertEqual(parse_relations("interacted, owns"), ["owns", "interacted"])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_relations("owns,unknown")
        self.assertEqual(parse_args(["backtest"]).relations, list(RELATIONS))


if __name__ == "__main__":
    unittest.main()
