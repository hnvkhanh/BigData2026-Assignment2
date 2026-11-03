from __future__ import annotations

import argparse
import unittest

import pandas as pd

from experiments.relational_lightgbm.features import (
    FEATURE_GROUPS,
    RelationalTables,
    extract_mentions,
    feature_names,
    hypothesis_lift,
    relational_features,
)
from experiments.relational_lightgbm.run import parse_args, parse_groups

CUTOFF = pd.Timestamp("2026-06-10 16:00:00")
DAY = pd.Timedelta(days=1)
OWNER, OTHER = 0, 1


def _tables() -> RelationalTables:
    """User 0 owns repo 10; user 1 owns nothing. Repo 20 belongs to an outsider."""
    return RelationalTables(
        repo_owner=pd.Series([OWNER], index=pd.Index([10], name="repo_id")),
        pushes=pd.DataFrame(
            {
                "user_id": [OWNER, OWNER, OWNER, OTHER],
                "repo_id": [10, 10, 20, 20],
                "branch": ["main", "feat-a", "feat-b", "main"],
                "at": [CUTOFF - 2 * DAY, CUTOFF - DAY, CUTOFF - 3 * DAY, CUTOFF - DAY],
            }
        ),
        prs=pd.DataFrame(
            {
                "pr_id": [0, 1, 2],
                "author_id": pd.array([OWNER, OTHER, OTHER], dtype="Int64"),
                "repo_id": [20, 10, 20],
                "head_ref": ["feat-b", "fix", "later"],
                "opened_at": [CUTOFF - 40 * DAY, CUTOFF - 2 * DAY, CUTOFF - DAY],
                # PR 2 is opened before the cutoff but only becomes known after it.
                "available_at": [CUTOFF - 40 * DAY, CUTOFF - 2 * DAY, CUTOFF + DAY],
                "closed_at": [pd.NaT, pd.NaT, pd.NaT],
                # PR 0 is still open; PR 1 is merged by the owner after the cutoff.
                "merged_at": [pd.NaT, CUTOFF + DAY, pd.NaT],
                "merged_by_id": pd.array([pd.NA, OWNER, pd.NA], dtype="Int64"),
            }
        ),
        issues=pd.DataFrame(
            {
                "issue_id": [0, 1, 2],
                "author_id": pd.array([OTHER, OWNER, OTHER], dtype="Int64"),
                "assignee_id": pd.array([OTHER, pd.NA, OTHER], dtype="Int64"),
                "repo_id": [10, 10, 10],
                "is_pr": [False, False, False],
                "opened_at": [CUTOFF - 3 * DAY, CUTOFF - 3 * DAY, CUTOFF - 20 * DAY],
                "available_at": [CUTOFF - 3 * DAY, CUTOFF - 3 * DAY, CUTOFF - 20 * DAY],
                # Issue 2 closed before the cutoff, so it is no longer open.
                "closed_at": [pd.NaT, pd.NaT, CUTOFF - 5 * DAY],
            }
        ),
        issue_comments=pd.DataFrame(
            {
                "issue_id": [1, 1],
                "repo_id": [10, 10],
                "author_id": [OTHER, OWNER],
                "at": [CUTOFF - DAY, CUTOFF - DAY],
            }
        ),
        reviews=pd.DataFrame(
            {
                "pr_id": [0],
                "repo_id": [20],
                "reviewer_id": [OTHER],
                "state": ["approved"],
                "at": [CUTOFF - 10 * DAY],
                "available_at": [CUTOFF - 10 * DAY],
            }
        ),
        review_comments=pd.DataFrame(
            {"pr_id": [0], "repo_id": [20], "author_id": [OTHER], "at": [CUTOFF - DAY]}
        ),
        mentions=pd.DataFrame(
            {
                "user_id": [OWNER, OWNER, OWNER],
                "author_id": pd.array([OTHER, OTHER, OTHER], dtype="Int64"),
                # Recent, older than 7 days, and posted before but only known after the cutoff.
                "at": [CUTOFF - DAY, CUTOFF - 20 * DAY, CUTOFF - DAY],
                "available_at": [CUTOFF - DAY, CUTOFF - 20 * DAY, CUTOFF + DAY],
            }
        ),
    )


def _histories() -> dict[str, pd.DataFrame]:
    return {
        "push": pd.DataFrame(
            {
                "user_id": [OWNER, OTHER],
                "repo_id": [10, 20],
                "action_at": [CUTOFF - DAY, CUTOFF - DAY],
                "available_at": [CUTOFF - DAY, CUTOFF - DAY],
            }
        )
    }


class RelationalFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        frame = relational_features(CUTOFF, _tables(), _histories(), [OWNER, OTHER])
        self.rows = frame.set_index("user_id")

    def test_returns_every_feature_once_per_user(self) -> None:
        self.assertEqual(list(self.rows.columns), feature_names(list(FEATURE_GROUPS)))
        self.assertEqual(list(self.rows.index), [OWNER, OTHER])
        self.assertFalse(self.rows.isna().any().any())

    def test_ownership_counts_only_other_peoples_activity(self) -> None:
        owner = self.rows.loc[OWNER]
        self.assertAlmostEqual(owner["own_push_share_30d"], 2 / 3)
        self.assertEqual(owner["own_repos_pushed_30d"], 1)
        self.assertEqual(owner["own_repo_new_issues_by_others_7d"], 1)  # own issue 1 excluded
        self.assertEqual(owner["own_repo_open_issues_by_others"], 1)  # issue 2 closed
        self.assertEqual(owner["own_repo_issue_comments_by_others_7d"], 1)
        self.assertEqual(owner["own_repo_new_prs_by_others_7d"], 1)
        self.assertEqual(self.rows.loc[OTHER, "own_push_share_30d"], 0)

    def test_unpaired_branches_exclude_mainline_and_branches_with_a_pr(self) -> None:
        owner = self.rows.loc[OWNER]
        self.assertEqual(owner["feature_branch_pushes_7d"], 2)
        self.assertEqual(owner["unpaired_branches_7d"], 1)  # feat-a; feat-b already has PR 0

    def test_reviewer_and_author_relations(self) -> None:
        other = self.rows.loc[OTHER]
        self.assertEqual(other["reviewed_repos_90d"], 1)
        # PR 2 on the reviewed repo is not yet available at the cutoff.
        self.assertEqual(other["new_prs_on_reviewed_repos_7d"], 0)
        self.assertEqual(other["open_prs_reviewed_by_user"], 1)
        self.assertEqual(other["assigned_open_issues"], 1)
        owner = self.rows.loc[OWNER]
        self.assertEqual(owner["review_comments_received_7d"], 1)
        self.assertEqual(owner["issue_comments_received_7d"], 1)
        self.assertEqual(owner["open_authored_prs"], 1)

    def test_discovered_relations(self) -> None:
        owner, other = self.rows.loc[OWNER], self.rows.loc[OTHER]
        self.assertEqual(owner["mentions_received_7d"], 1)
        self.assertEqual(owner["mentions_received_30d"], 2)
        # Both joined issue 1 and each sees the other's comment there; own comments never count.
        self.assertEqual(owner["thread_new_comments_7d"], 1)
        self.assertEqual(other["thread_new_comments_7d"], 1)
        # PR 1 is merged only after the cutoff, so it is not visible yet.
        self.assertEqual(owner["merged_others_prs_7d"], 0)
        self.assertEqual(owner["approvals_received_30d"], 1)
        # Open PR 0 has head branch feat-b, which the owner pushed 3 days ago.
        self.assertEqual(owner["open_prs_with_recent_push"], 1)
        # One push in week 0 (1 day ago); the owner's pushes are 1-3 days ago.
        self.assertEqual(owner["push_active_weeks_8w"], 1)
        self.assertEqual(owner["any_active_weeks_8w"], 1)
        self.assertEqual(owner["open_pr_active_weeks_8w"], 0)

    def test_mentions_are_extracted_from_text(self) -> None:
        bodies = pd.DataFrame(
            {
                "body": ["thanks @Alice and @bob!", "mail a@b.com @alice", "@carol"],
                "author_id": pd.array([2, 2, 0], dtype="Int64"),
                "at": [CUTOFF] * 3,
                "available_at": [CUTOFF] * 3,
            }
        )
        logins = pd.Series({"alice": 0, "bob": 1, "carol": 0})
        found = extract_mentions(bodies, logins)
        # alice twice (once per text), bob once; e-mail and carol's self-mention are skipped.
        self.assertEqual(sorted(found["user_id"].tolist()), [0, 0, 1])

    def test_events_after_the_cutoff_are_ignored(self) -> None:
        tables = _tables()
        future = tables.pushes.assign(at=CUTOFF)
        shifted = RelationalTables(**{**tables.__dict__, "pushes": future})
        frame = relational_features(CUTOFF, shifted, _histories(), [OWNER, OTHER])
        self.assertEqual(frame["feature_branch_pushes_7d"].sum(), 0)
        self.assertEqual(frame["own_push_share_30d"].sum(), 0)

    def test_group_selection_and_unknown_groups(self) -> None:
        frame = relational_features(CUTOFF, _tables(), _histories(), [OWNER], ["assigned"])
        self.assertEqual(list(frame.columns), ["user_id", *FEATURE_GROUPS["assigned"]])
        with self.assertRaises(ValueError):
            feature_names(["nope"])

    def test_hypothesis_lift(self) -> None:
        frame = pd.DataFrame(
            {"unpaired_branches_7d": [0, 0, 1, 2], "open_pr": [0, 1, 1, 1]}
        )
        lift = hypothesis_lift(frame).set_index(["feature", "label"])
        row = lift.loc[("unpaired_branches_7d", "open_pr")]
        self.assertEqual(row["rows_with_feature"], 2)
        self.assertAlmostEqual(row["lift"], 2.0)


class CliTests(unittest.TestCase):
    def test_groups_option(self) -> None:
        self.assertEqual(parse_groups("none"), [])
        self.assertEqual(parse_groups("owner, reviewer"), ["owner", "reviewer"])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_groups("owner,unknown")
        self.assertEqual(parse_args(["backtest"]).groups, list(FEATURE_GROUPS))

    def test_rejects_invalid_model_settings(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["backtest", "--learning-rate", "0"])


if __name__ == "__main__":
    unittest.main()
