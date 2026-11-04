"""Relationship tables derived from the GitHub database for ``relgt_db``.

RelGT reads a database as a pkey/fkey graph and samples tokens from each
user's 1- and 2-hop neighbourhood. Most relationships behind next week's PRs,
reviews and comments are three or more hops away in the raw schema (a new PR
by someone else on a repo the user owns is user -> push -> repo -> PR), or
not keys at all (ownership is a login match, mentions are text).

The tables built here turn them into rows with foreign keys, the same way a
RelBench database would model them:

- ``RepoOwners``: one row per (user, repository the user owns).
- ``Notifications``: one row per event by someone else that concerns the user
  (an issue or PR on their repo, feedback on their PR, a mention, ...). It
  links the user to the PR or issue in one hop, so the PR or issue itself is a
  2-hop token.
- ``Collaborations``: one row per pair of users who interacted (review,
  merge, comment, mention). It is symmetric, so the row means the same from
  either side.

Every row's time column is the moment the row is known: the later of the
source event's availability and the referenced rows' availability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.relational_lightgbm.features import RelationalTables, load_relational_tables
from experiments.relgt.database import GITHUB_SCHEMA, RelationalDatabase, TableSchema, load_database

PR_MENTION = 1
ISSUE_MENTION = 2

DERIVED_SCHEMA: dict[str, TableSchema] = {
    "RepoOwners": TableSchema(
        "RepoOwners",
        "owner_link_id",
        "known_at",
        foreign_keys={"user_id": "Users", "repo_id": "Repositories"},
    ),
    "Notifications": TableSchema(
        "Notifications",
        "notification_id",
        "known_at",
        foreign_keys={"user_id": "Users", "pr_id": "PullRequests", "issue_id": "Issues"},
        stypes={"kind": "categorical", "acted_at": "timestamp", "known_at": "timestamp"},
    ),
    "Collaborations": TableSchema(
        "Collaborations",
        "collaboration_id",
        "known_at",
        foreign_keys={"user_a_id": "Users", "user_b_id": "Users"},
        stypes={"known_at": "timestamp"},
    ),
}
DERIVED_TABLES = tuple(DERIVED_SCHEMA)

NOTIFICATION_KINDS = (
    "issue_on_owned_repo",
    "pr_on_owned_repo",
    "pr_on_reviewed_repo",
    "comment_on_authored",
    "review_on_authored_pr",
    "review_comment_on_authored_pr",
    "authored_pr_merged_by_other",
    "mention",
    "thread_reply",
)


def _finalize(frame: pd.DataFrame, table: TableSchema) -> pd.DataFrame:
    """Sort by the time column and number rows 0..n-1, as ``load_database`` requires."""
    frame = frame.dropna(subset=[table.time_column])
    frame = frame.sort_values(table.time_column, kind="stable").reset_index(drop=True)
    frame.insert(0, table.primary_key, np.arange(len(frame), dtype=np.int64))
    for column in table.foreign_keys:
        frame[column] = frame[column].astype("Int64")
    return frame


def _known_at(*times: pd.Series) -> pd.Series:
    """Latest of several times; missing in any of them means never known."""
    stacked = pd.concat(times, axis=1)
    return stacked.max(axis=1).where(stacked.notna().all(axis=1))


def repo_owners(
    tables: RelationalTables, user_seen: pd.Series, repo_seen: pd.Series
) -> pd.DataFrame:
    owners = pd.DataFrame(
        {"user_id": tables.repo_owner.to_numpy(), "repo_id": tables.repo_owner.index.to_numpy()}
    )
    owners["known_at"] = _known_at(
        owners["user_id"].map(user_seen), owners["repo_id"].map(repo_seen)
    )
    return _finalize(owners, DERIVED_SCHEMA["RepoOwners"])


def _notify(
    kind: str,
    user_id: pd.Series,
    actor_id: pd.Series,
    acted_at: pd.Series,
    known_at: pd.Series,
    pr_id: pd.Series | None = None,
    issue_id: pd.Series | None = None,
) -> pd.DataFrame:
    index = user_id.index
    frame = pd.DataFrame(
        {
            "user_id": user_id.astype("Int64"),
            "actor_id": actor_id.astype("Int64"),
            "kind": kind,
            "pr_id": (pr_id if pr_id is not None else pd.Series(pd.NA, index=index)).astype("Int64"),
            "issue_id": (
                issue_id if issue_id is not None else pd.Series(pd.NA, index=index)
            ).astype("Int64"),
            "acted_at": acted_at,
            "known_at": known_at,
        },
        index=index,
    )
    # Self-notifications carry no relationship.
    other = frame["actor_id"].isna() | (frame["actor_id"] != frame["user_id"])
    return frame.loc[frame["user_id"].notna() & other.fillna(True)]


def notification_events(tables: RelationalTables, issue_pr: pd.Series) -> pd.DataFrame:
    """Every notification with its actor (``actor_id`` is kept for ``Collaborations``)."""
    owner = tables.repo_owner
    prs, issues = tables.prs, tables.issues
    pr_author = prs.set_index("pr_id")["author_id"]
    issue_author = issues.set_index("issue_id")["author_id"]
    pr_seen = prs.set_index("pr_id")["available_at"]
    issue_seen = issues.set_index("issue_id")["available_at"]
    plain_issues = issues.loc[~issues["is_pr"]]

    parts = [
        _notify(
            "issue_on_owned_repo",
            plain_issues["repo_id"].map(owner),
            plain_issues["author_id"],
            plain_issues["opened_at"],
            plain_issues["available_at"],
            issue_id=plain_issues["issue_id"],
        ),
        _notify(
            "pr_on_owned_repo",
            prs["repo_id"].map(owner),
            prs["author_id"],
            prs["opened_at"],
            prs["available_at"],
            pr_id=prs["pr_id"],
        ),
    ]

    # New PRs by others on a repo where the user's first review is already known.
    first_review = (
        tables.reviews.dropna(subset=["reviewer_id", "repo_id"])
        .groupby(["reviewer_id", "repo_id"])["available_at"]
        .min()
        .rename("first_review_at")
        .reset_index()
    )
    candidates = prs.dropna(subset=["repo_id"]).merge(first_review, on="repo_id")
    candidates = candidates.loc[candidates["available_at"] > candidates["first_review_at"]]
    parts.append(
        _notify(
            "pr_on_reviewed_repo",
            candidates["reviewer_id"],
            candidates["author_id"],
            candidates["opened_at"],
            candidates["available_at"],
            pr_id=candidates["pr_id"],
        )
    )

    comments = tables.issue_comments.dropna(subset=["issue_id"])
    parts.append(
        _notify(
            "comment_on_authored",
            comments["issue_id"].map(issue_author),
            comments["author_id"],
            comments["at"],
            _known_at(comments["at"], comments["issue_id"].map(issue_seen)),
            pr_id=comments["issue_id"].map(issue_pr),
            issue_id=comments["issue_id"],
        )
    )
    reviews = tables.reviews.dropna(subset=["pr_id"])
    parts.append(
        _notify(
            "review_on_authored_pr",
            reviews["pr_id"].map(pr_author),
            reviews["reviewer_id"],
            reviews["at"],
            _known_at(reviews["available_at"], reviews["pr_id"].map(pr_seen)),
            pr_id=reviews["pr_id"],
        )
    )
    review_comments = tables.review_comments.dropna(subset=["pr_id"])
    parts.append(
        _notify(
            "review_comment_on_authored_pr",
            review_comments["pr_id"].map(pr_author),
            review_comments["author_id"],
            review_comments["at"],
            _known_at(review_comments["at"], review_comments["pr_id"].map(pr_seen)),
            pr_id=review_comments["pr_id"],
        )
    )
    merged = prs.dropna(subset=["merged_at", "merged_by_id"])
    parts.append(
        _notify(
            "authored_pr_merged_by_other",
            merged["author_id"],
            merged["merged_by_id"],
            merged["merged_at"],
            _known_at(merged["merged_at"], merged["available_at"]),
            pr_id=merged["pr_id"],
        )
    )

    mentions = tables.mentions
    on_pr = mentions["node_type"] == PR_MENTION
    entity_seen = np.where(
        on_pr, mentions["entity_id"].map(pr_seen), mentions["entity_id"].map(issue_seen)
    )
    parts.append(
        _notify(
            "mention",
            mentions["user_id"],
            mentions["author_id"],
            mentions["at"],
            _known_at(mentions["available_at"], pd.Series(entity_seen, index=mentions.index)),
            pr_id=mentions["entity_id"].where(on_pr),
            issue_id=mentions["entity_id"].where(~on_pr),
        )
    )

    # Later comments by others in an issue thread the user joined (not as its author).
    first_comment = (
        comments.dropna(subset=["author_id"])
        .groupby(["author_id", "issue_id"])["at"]
        .min()
        .rename("joined_at")
        .reset_index()
        .rename(columns={"author_id": "participant_id"})
    )
    replies = comments.merge(first_comment, on="issue_id")
    replies = replies.loc[
        (replies["at"] > replies["joined_at"])
        & (replies["participant_id"] != replies["issue_id"].map(issue_author))
    ]
    parts.append(
        _notify(
            "thread_reply",
            replies["participant_id"],
            replies["author_id"],
            replies["at"],
            _known_at(replies["at"], replies["issue_id"].map(issue_seen)),
            pr_id=replies["issue_id"].map(issue_pr),
            issue_id=replies["issue_id"],
        )
    )
    events = pd.concat(parts, ignore_index=True)
    # An event that is several kinds for one user (e.g. a PR on a repo the user
    # owns and reviews) keeps its first kind.
    return events.drop_duplicates(
        subset=["user_id", "actor_id", "pr_id", "issue_id", "acted_at"], keep="first"
    )


def notifications(events: pd.DataFrame) -> pd.DataFrame:
    return _finalize(events.drop(columns="actor_id"), DERIVED_SCHEMA["Notifications"])


def collaborations(events: pd.DataFrame) -> pd.DataFrame:
    """One symmetric row per pair of users, known from their first interaction."""
    pairs = events.dropna(subset=["actor_id"])
    low = np.minimum(pairs["user_id"], pairs["actor_id"])
    high = np.maximum(pairs["user_id"], pairs["actor_id"])
    frame = (
        pd.DataFrame({"user_a_id": low, "user_b_id": high, "known_at": pairs["known_at"]})
        .groupby(["user_a_id", "user_b_id"], as_index=False)["known_at"]
        .min()
    )
    return _finalize(frame, DERIVED_SCHEMA["Collaborations"])


def derived_tables(db_dir: Path, names: tuple[str, ...] | list[str]) -> dict[str, pd.DataFrame]:
    unknown = sorted(set(names) - set(DERIVED_TABLES))
    if unknown:
        raise ValueError(f"Unknown relationship table(s): {', '.join(unknown)}")
    if not names:
        return {}
    tables = load_relational_tables(db_dir)
    users = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "first_seen_at"])
    repos = pd.read_parquet(db_dir / "Repositories.parquet", columns=["repo_id", "first_seen_at"])
    issue_pr = pd.read_parquet(db_dir / "Issues.parquet", columns=["issue_id", "pull_request_id"])
    issue_pr = issue_pr.set_index("issue_id")["pull_request_id"].astype("Int64")

    output: dict[str, pd.DataFrame] = {}
    if "RepoOwners" in names:
        output["RepoOwners"] = repo_owners(
            tables,
            users.set_index("user_id")["first_seen_at"],
            repos.set_index("repo_id")["first_seen_at"],
        )
    if {"Notifications", "Collaborations"} & set(names):
        events = notification_events(tables, issue_pr)
        if "Notifications" in names:
            output["Notifications"] = notifications(events)
        if "Collaborations" in names:
            output["Collaborations"] = collaborations(events)
    return output


def load_relational_database(
    db_dir: Path, names: tuple[str, ...] | list[str]
) -> RelationalDatabase:
    """The assignment database plus the requested relationship tables."""
    database = load_database(db_dir)
    extra = derived_tables(db_dir, names)
    schema = GITHUB_SCHEMA + tuple(DERIVED_SCHEMA[name] for name in DERIVED_TABLES if name in extra)
    return RelationalDatabase(schema=schema, tables={**database.tables, **extra})
