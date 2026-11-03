"""Relationship edges that extend the RelGT/GNN user neighbourhood.

``experiments.relgt.data.snapshot_tokens`` links a user only to entities they
acted on in the last 14 days (relations 1-6 plus repository membership 7).
This module adds typed edges for relationships that are not the user's own
recent actions: repositories they own, issues assigned to them, PRs/issues
where they were @mentioned, open PRs by others on repositories they review or
own, their open PRs and issues, threads they joined that got new comments,
repositories they reviewed in the last 90 days, repositories with pushed
branches that have no PR yet, and other users they interacted with. Each edge
becomes one token of the same form as the base tokens (entity, node type,
relation, hop, 4 node features, age), so attention and message passing see the
concrete PRs, issues, repositories and people behind each relationship.

All rows must be acted and available strictly before the cutoff. Lifecycle
values known only later (``closed_at``, ``merged_at``) are used only when
earlier than the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.relational_lightgbm.features import (
    CONTEXT_DAYS,
    MAINLINE_BRANCHES,
    OPEN_ITEM_DAYS,
    RECENT_DAYS,
    RelationalTables,
)
from experiments.relgt.data import LABELS, EntityVocabulary, GraphTokens

PR_NODE, ISSUE_NODE, REPO_NODE, USER_NODE = 1, 2, 3, 4
NUM_NODE_TYPES = 5  # 0 is the user root

# Base relations are 0 (root), 1-6 (actions, LABELS order) and 7 (repo membership).
FIRST_RELATION = len(LABELS) + 2
# Relation name -> (node type, slots per user).
RELATIONS: dict[str, tuple[int, int]] = {
    "owns": (REPO_NODE, 4),
    "assigned": (ISSUE_NODE, 4),
    "mentioned_in": (ISSUE_NODE, 4),  # PR or issue; the node type is set per token
    "review_candidate": (PR_NODE, 4),
    "open_authored_pr": (PR_NODE, 4),
    "open_authored_issue": (ISSUE_NODE, 4),
    "thread": (ISSUE_NODE, 4),
    "reviewed_repo": (REPO_NODE, 4),
    "unpaired_branch": (REPO_NODE, 4),
    "interacted": (USER_NODE, 8),
}
RELATION_IDS = {name: FIRST_RELATION + index for index, name in enumerate(RELATIONS)}
NUM_RELATIONS = FIRST_RELATION + len(RELATIONS)
HOP = 1  # every relationship edge starts at the user


def relation_slots(relations: list[str]) -> dict[str, slice]:
    """Fixed slot range for each selected relation."""
    slots, start = {}, 0
    for name in relations:
        width = RELATIONS[name][1]
        slots[name] = slice(start, start + width)
        start += width
    return slots


def num_relation_slots(relations: list[str]) -> int:
    return sum(RELATIONS[name][1] for name in relations)


@dataclass(frozen=True)
class GlobalIds:
    """Embedding ids: the base PR/issue/repo vocabulary followed by users."""

    vocabulary: EntityVocabulary
    num_users: int

    @property
    def num_entities(self) -> int:
        return self.vocabulary.num_entities + self.num_users

    def of(self, node_type: int, ids: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(ids, dtype=np.int64)
        if node_type == USER_NODE:
            return values + self.vocabulary.num_entities
        return values + self.vocabulary.offset_for_type(node_type)


def _log(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(values, dtype=np.float64), 0, None)).astype(np.float32)


def _days(cutoff: pd.Timestamp, times: pd.Series) -> np.ndarray:
    return ((cutoff - times).dt.total_seconds() / 86_400).to_numpy(dtype=np.float32)


def _count(frame: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    return frame.groupby(keys).size().rename(name).reset_index()


def _edges(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    node_type: int | pd.Series,
    entity: str,
    features: list[np.ndarray],
    last_at: pd.Series,
    score: np.ndarray,
) -> pd.DataFrame:
    """Normalise one relation's candidate edges to the common token columns."""
    padded = [*features, *[np.zeros(len(frame), np.float32)] * (4 - len(features))]
    return pd.DataFrame(
        {
            "user_id": frame["user_id"].to_numpy(dtype=np.int64),
            "node_type": np.broadcast_to(np.asarray(node_type), len(frame)).astype(np.int64),
            "entity_id": frame[entity].to_numpy(dtype=np.int64),
            **{f"f{index}": np.asarray(value, np.float32) for index, value in enumerate(padded)},
            "age_days": np.clip(_days(cutoff, last_at), 0, 3650),
            "score": np.asarray(score, dtype=np.float64),
        }
    )


@dataclass(frozen=True)
class Snapshot:
    """Rows of the relational tables known before one cutoff."""

    cutoff: pd.Timestamp
    prs: pd.DataFrame
    open_prs: pd.DataFrame
    issues: pd.DataFrame  # includes PR conversations
    open_issues: pd.DataFrame  # plain issues only
    comments: pd.DataFrame
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    pushes: pd.DataFrame
    mentions: pd.DataFrame

    @classmethod
    def at(cls, tables: RelationalTables, cutoff: pd.Timestamp) -> "Snapshot":
        open_window = cutoff - pd.Timedelta(days=OPEN_ITEM_DAYS)
        prs = tables.prs.loc[(tables.prs["available_at"] < cutoff) & (tables.prs["opened_at"] < cutoff)]
        issues = tables.issues.loc[
            (tables.issues["available_at"] < cutoff) & (tables.issues["opened_at"] < cutoff)
        ]
        is_open = lambda frame: frame["closed_at"].isna() | (frame["closed_at"] >= cutoff)  # noqa: E731
        plain = issues.loc[~issues["is_pr"]]
        return cls(
            cutoff=cutoff,
            prs=prs,
            open_prs=prs.loc[is_open(prs) & (prs["available_at"] >= open_window)],
            issues=issues,
            open_issues=plain.loc[is_open(plain) & (plain["available_at"] >= open_window)],
            comments=tables.issue_comments.loc[tables.issue_comments["at"] < cutoff],
            reviews=tables.reviews.loc[
                (tables.reviews["at"] < cutoff) & (tables.reviews["available_at"] < cutoff)
            ],
            review_comments=tables.review_comments.loc[tables.review_comments["at"] < cutoff],
            pushes=tables.pushes.loc[tables.pushes["at"] < cutoff],
            mentions=tables.mentions.loc[
                (tables.mentions["at"] < cutoff) & (tables.mentions["available_at"] < cutoff)
            ],
        )

    def since(self, frame: pd.DataFrame, days: int, column: str = "at") -> pd.DataFrame:
        return frame.loc[frame[column] >= self.cutoff - pd.Timedelta(days=days)]


def _repo_events(snap: Snapshot, days: int) -> pd.DataFrame:
    """(repo_id, actor, at) for every recent event on a repository."""
    parts = [
        snap.since(snap.prs, days, "opened_at")[["repo_id", "author_id", "opened_at"]],
        snap.since(snap.issues.loc[~snap.issues["is_pr"]], days, "opened_at")[["repo_id", "author_id", "opened_at"]],
        snap.since(snap.comments, days)[["repo_id", "author_id", "at"]],
        snap.since(snap.reviews, days)[["repo_id", "reviewer_id", "at"]],
        snap.since(snap.review_comments, days)[["repo_id", "author_id", "at"]],
        snap.since(snap.pushes, days)[["repo_id", "user_id", "at"]],
    ]
    return pd.concat([part.set_axis(["repo_id", "actor", "at"], axis=1) for part in parts], ignore_index=True)


def owns(snap: Snapshot, users: pd.Index, owner: pd.Series) -> pd.DataFrame:
    owned = owner.loc[owner.isin(users)].rename("user_id").rename_axis("repo_id").reset_index()
    events = _repo_events(snap, CONTEXT_DAYS).merge(owned, on="repo_id")
    events["by_owner"] = events["actor"].eq(events["user_id"]).fillna(False)
    stats = events.groupby(["user_id", "repo_id"]).agg(
        others=("by_owner", lambda flags: int((~flags).sum())),
        own=("by_owner", "sum"),
        last_at=("at", "max"),
    ).reset_index()
    for name, frame in (("open_issues", snap.open_issues), ("open_prs", snap.open_prs)):
        mine = frame.merge(owned, on="repo_id")
        mine = mine.loc[mine["author_id"].isna() | (mine["author_id"] != mine["user_id"])]
        stats = stats.merge(_count(mine, ["user_id", "repo_id"], name), on=["user_id", "repo_id"], how="left")
    stats = stats.fillna({"open_issues": 0, "open_prs": 0})
    return _edges(
        stats, snap.cutoff, REPO_NODE, "repo_id",
        [_log(stats["others"]), _log(stats["own"]), _log(stats["open_issues"]), _log(stats["open_prs"])],
        stats["last_at"], stats["others"] + stats["own"],
    )


def assigned(snap: Snapshot, users: pd.Index) -> pd.DataFrame:
    issues = snap.open_issues.dropna(subset=["assignee_id"]).astype({"assignee_id": "int64"})
    issues = issues.loc[issues["assignee_id"].isin(users)].rename(columns={"assignee_id": "user_id"})
    recent = snap.since(snap.comments, RECENT_DAYS)
    issues = issues.merge(_count(recent, ["issue_id"], "recent_comments"), on="issue_id", how="left")
    issues = issues.merge(_count(snap.comments, ["issue_id"], "comments"), on="issue_id", how="left")
    issues = issues.fillna({"recent_comments": 0, "comments": 0})
    return _edges(
        issues, snap.cutoff, ISSUE_NODE, "issue_id",
        [_log(issues["recent_comments"]), _log(_days(snap.cutoff, issues["opened_at"])), _log(issues["comments"])],
        issues["available_at"], -_days(snap.cutoff, issues["available_at"]),
    )


def mentioned_in(snap: Snapshot, users: pd.Index) -> pd.DataFrame:
    mentions = snap.since(snap.mentions, CONTEXT_DAYS)
    mentions = mentions.loc[mentions["user_id"].isin(users)]
    known = pd.concat(
        [
            pd.DataFrame({"node_type": PR_NODE, "entity_id": snap.prs["pr_id"]}),
            pd.DataFrame({"node_type": ISSUE_NODE, "entity_id": snap.issues["issue_id"]}),
        ]
    )
    mentions = mentions.merge(known, on=["node_type", "entity_id"])
    stats = mentions.groupby(["user_id", "node_type", "entity_id"]).agg(
        count=("at", "size"), last_at=("at", "max")
    ).reset_index()
    return _edges(
        stats, snap.cutoff, stats["node_type"], "entity_id", [_log(stats["count"])],
        stats["last_at"], -_days(snap.cutoff, stats["last_at"]),
    )


def _reviewed_repos(snap: Snapshot, users: pd.Index) -> pd.DataFrame:
    reviews = snap.since(snap.reviews, OPEN_ITEM_DAYS)
    reviews = reviews.loc[reviews["reviewer_id"].isin(users)].rename(columns={"reviewer_id": "user_id"})
    return reviews.groupby(["user_id", "repo_id"]).agg(reviews=("at", "size"), last_at=("at", "max")).reset_index()


def review_candidate(snap: Snapshot, users: pd.Index, owner: pd.Series) -> pd.DataFrame:
    reviewed = _reviewed_repos(snap, users)[["user_id", "repo_id", "reviews"]]
    owned = owner.loc[owner.isin(users)].rename("user_id").rename_axis("repo_id").reset_index()
    linked = reviewed.merge(owned.assign(is_owner=1.0), on=["user_id", "repo_id"], how="outer")
    linked = linked.fillna({"reviews": 0, "is_owner": 0})
    prs = snap.since(snap.open_prs, CONTEXT_DAYS, "available_at").merge(linked, on="repo_id")
    prs = prs.loc[prs["author_id"].isna() | (prs["author_id"] != prs["user_id"])]
    prs = prs.merge(_count(snap.reviews, ["pr_id"], "pr_reviews"), on="pr_id", how="left").fillna({"pr_reviews": 0})
    return _edges(
        prs, snap.cutoff, PR_NODE, "pr_id",
        [_log(prs["reviews"]), _log(_days(snap.cutoff, prs["opened_at"])), _log(prs["pr_reviews"]), prs["is_owner"]],
        prs["opened_at"], -_days(snap.cutoff, prs["opened_at"]),
    )


def open_authored_pr(snap: Snapshot, users: pd.Index) -> pd.DataFrame:
    prs = snap.open_prs.dropna(subset=["author_id"]).astype({"author_id": "int64"})
    prs = prs.loc[prs["author_id"].isin(users)].rename(columns={"author_id": "user_id"})
    received = snap.reviews.merge(prs[["pr_id", "user_id"]], on="pr_id")
    received = received.loc[received["reviewer_id"] != received["user_id"]]
    feedback = snap.review_comments.merge(prs[["pr_id", "user_id"]], on="pr_id")
    feedback = feedback.loc[feedback["author_id"] != feedback["user_id"]]
    pushed = snap.since(snap.pushes, RECENT_DAYS)[["user_id", "branch"]].drop_duplicates().assign(pushed=1.0)
    prs = (
        prs.merge(_count(received.loc[received["state"] == "approved"], ["pr_id"], "approvals"), on="pr_id", how="left")
        .merge(_count(pd.concat([received[["pr_id", "at"]], feedback[["pr_id", "at"]]]), ["pr_id"], "feedback"),
               on="pr_id", how="left")
        .merge(pd.concat([received[["pr_id", "at"]], feedback[["pr_id", "at"]]]).groupby("pr_id")["at"].max()
               .rename("last_feedback").reset_index(), on="pr_id", how="left")
        .merge(pushed, left_on=["user_id", "head_ref"], right_on=["user_id", "branch"], how="left")
        .fillna({"approvals": 0, "feedback": 0, "pushed": 0})
    )
    last_at = prs[["opened_at", "last_feedback"]].max(axis=1)
    return _edges(
        prs, snap.cutoff, PR_NODE, "pr_id",
        [_log(prs["approvals"]), _log(prs["feedback"]), prs["pushed"], _log(_days(snap.cutoff, prs["opened_at"]))],
        last_at, -_days(snap.cutoff, last_at),
    )


def open_authored_issue(snap: Snapshot, users: pd.Index) -> pd.DataFrame:
    issues = snap.open_issues.dropna(subset=["author_id"]).astype({"author_id": "int64"})
    issues = issues.loc[issues["author_id"].isin(users)].rename(columns={"author_id": "user_id"})
    others = snap.comments.merge(issues[["issue_id", "user_id"]], on="issue_id")
    others = others.loc[others["author_id"] != others["user_id"]]
    stats = others.groupby("issue_id").agg(comments=("at", "size"), last_comment=("at", "max")).reset_index()
    recent = _count(snap.since(others, RECENT_DAYS), ["issue_id"], "recent")
    issues = issues.merge(stats, on="issue_id", how="left").merge(recent, on="issue_id", how="left")
    issues = issues.fillna({"comments": 0, "recent": 0})
    last_at = issues[["opened_at", "last_comment"]].max(axis=1)
    return _edges(
        issues, snap.cutoff, ISSUE_NODE, "issue_id",
        [_log(issues["comments"]), _log(issues["recent"]), np.zeros(len(issues)), _log(_days(snap.cutoff, issues["opened_at"]))],
        last_at, -_days(snap.cutoff, last_at),
    )


def thread(snap: Snapshot, users: pd.Index) -> pd.DataFrame:
    own_comments = snap.since(snap.comments, CONTEXT_DAYS)
    own_comments = own_comments.loc[own_comments["author_id"].isin(users)]
    joined = pd.concat(
        [
            own_comments[["author_id", "issue_id"]],
            snap.since(snap.issues, OPEN_ITEM_DAYS, "available_at")[["author_id", "issue_id"]],
        ]
    ).dropna().drop_duplicates().astype({"author_id": "int64"}).rename(columns={"author_id": "user_id"})
    joined = joined.loc[joined["user_id"].isin(users)]
    new = snap.since(snap.comments, RECENT_DAYS)[["issue_id", "author_id", "at"]].merge(joined, on="issue_id")
    new = new.loc[new["author_id"] != new["user_id"]]
    stats = new.groupby(["user_id", "issue_id"]).agg(new=("at", "size"), last_at=("at", "max")).reset_index()
    stats = stats.merge(
        _count(own_comments.rename(columns={"author_id": "user_id"}), ["user_id", "issue_id"], "mine"),
        on=["user_id", "issue_id"], how="left",
    ).fillna({"mine": 0})
    return _edges(
        stats, snap.cutoff, ISSUE_NODE, "issue_id", [_log(stats["new"]), _log(stats["mine"])],
        stats["last_at"], -_days(snap.cutoff, stats["last_at"]),
    )


def reviewed_repo(snap: Snapshot, users: pd.Index) -> pd.DataFrame:
    stats = _reviewed_repos(snap, users)
    new_prs = snap.since(snap.prs, RECENT_DAYS, "opened_at")
    for name, frame in (("new_prs", new_prs), ("open_prs", snap.open_prs)):
        mine = frame[["repo_id", "author_id"]].merge(stats[["user_id", "repo_id"]], on="repo_id")
        mine = mine.loc[mine["author_id"].isna() | (mine["author_id"] != mine["user_id"])]
        stats = stats.merge(_count(mine, ["user_id", "repo_id"], name), on=["user_id", "repo_id"], how="left")
    stats = stats.fillna({"new_prs": 0, "open_prs": 0})
    return _edges(
        stats, snap.cutoff, REPO_NODE, "repo_id",
        [_log(stats["reviews"]), _log(stats["new_prs"]), _log(stats["open_prs"])],
        stats["last_at"], stats["new_prs"] * 100 + stats["reviews"],
    )


def unpaired_branch(snap: Snapshot, users: pd.Index) -> pd.DataFrame:
    pushes = snap.since(snap.pushes, RECENT_DAYS)
    pushes = pushes.loc[pushes["user_id"].isin(users) & ~pushes["branch"].isin(MAINLINE_BRANCHES)]
    paired = (
        snap.prs.dropna(subset=["author_id"])[["author_id", "head_ref"]].drop_duplicates()
        .rename(columns={"author_id": "user_id", "head_ref": "branch"}).astype({"user_id": "int64"})
    )
    pushes = pushes.merge(paired, on=["user_id", "branch"], how="left", indicator=True)
    pushes = pushes.loc[pushes["_merge"] == "left_only"]
    stats = pushes.groupby(["user_id", "repo_id"]).agg(
        pushes=("at", "size"), branches=("branch", "nunique"), last_at=("at", "max")
    ).reset_index()
    return _edges(
        stats, snap.cutoff, REPO_NODE, "repo_id", [_log(stats["pushes"]), _log(stats["branches"])],
        stats["last_at"], -_days(snap.cutoff, stats["last_at"]),
    )


def interacted(snap: Snapshot, users: pd.Index, cohort: pd.Index) -> pd.DataFrame:
    """Other users the user reviewed, was reviewed by, replied to, merged or @mentioned (90 days)."""
    pr_authors = snap.prs.dropna(subset=["author_id"])[["pr_id", "author_id"]].rename(columns={"author_id": "owner"})
    issue_authors = snap.issues.dropna(subset=["author_id"])[["issue_id", "author_id"]].rename(
        columns={"author_id": "owner"}
    )
    window = lambda frame: snap.since(frame, OPEN_ITEM_DAYS)  # noqa: E731
    merged = snap.prs.loc[snap.prs["merged_at"] < snap.cutoff]
    merged = merged.loc[merged["merged_at"] >= snap.cutoff - pd.Timedelta(days=OPEN_ITEM_DAYS)]
    # (actor, counterpart, time): reviewer -> PR author, commenter -> PR/issue author,
    # merger -> PR author, mention author -> mentioned user.
    parts = (
        window(snap.reviews).merge(pr_authors, on="pr_id")[["reviewer_id", "owner", "at"]],
        window(snap.review_comments).merge(pr_authors, on="pr_id")[["author_id", "owner", "at"]],
        window(snap.comments).merge(issue_authors, on="issue_id")[["author_id", "owner", "at"]],
        merged[["merged_by_id", "author_id", "merged_at"]],
        window(snap.mentions)[["author_id", "user_id", "at"]],
    )
    pairs = pd.concat(
        [part.set_axis(["u", "v", "at"], axis=1) for part in parts], ignore_index=True
    ).dropna()
    pairs = pairs.astype({"u": "int64", "v": "int64"})
    pairs = pairs.loc[pairs["u"] != pairs["v"]]
    both = pd.concat([pairs, pairs.rename(columns={"u": "v", "v": "u"})], ignore_index=True)
    both = both.loc[both["u"].isin(users)].rename(columns={"u": "user_id", "v": "partner_id"})
    stats = both.groupby(["user_id", "partner_id"]).agg(count=("at", "size"), last_at=("at", "max")).reset_index()
    recent_prs = snap.since(snap.prs, RECENT_DAYS, "opened_at").dropna(subset=["author_id"])
    recent_issues = snap.since(snap.issues.loc[~snap.issues["is_pr"]], RECENT_DAYS, "opened_at").dropna(
        subset=["author_id"]
    )
    for name, frame in (("partner_prs", recent_prs), ("partner_issues", recent_issues)):
        counts = frame.groupby("author_id").size().rename(name)
        counts.index = counts.index.astype("int64")
        stats = stats.merge(counts, left_on="partner_id", right_index=True, how="left")
    stats = stats.fillna({"partner_prs": 0, "partner_issues": 0})
    return _edges(
        stats, snap.cutoff, USER_NODE, "partner_id",
        [_log(stats["count"]), _log(stats["partner_prs"]), stats["partner_id"].isin(cohort).astype(np.float32),
         _log(stats["partner_issues"])],
        stats["last_at"], stats["count"] * 1e3 - _days(snap.cutoff, stats["last_at"]),
    )


def relation_tokens(
    snap: Snapshot,
    user_ids: np.ndarray,
    relations: list[str],
    ids: GlobalIds,
    repo_owner: pd.Series,
    cohort: pd.Index,
) -> GraphTokens:
    """Relationship tokens for ``user_ids`` (one row each, in that order) at ``snap.cutoff``."""
    count, width = len(user_ids), num_relation_slots(relations)
    tokens = GraphTokens(
        user_ids=np.asarray(user_ids, dtype=np.int32),
        entity_ids=np.zeros((count, width), dtype=np.int32),
        node_type_ids=np.zeros((count, width), dtype=np.uint8),
        relation_ids=np.zeros((count, width), dtype=np.uint8),
        hop_ids=np.zeros((count, width), dtype=np.uint8),
        node_features=np.zeros((count, width, 4), dtype=np.float32),
        relative_days=np.zeros((count, width), dtype=np.float32),
    )
    if not count or not width:
        return tokens
    users = pd.Index(np.unique(user_ids).astype(np.int64))
    builders = {
        "owns": lambda: owns(snap, users, repo_owner),
        "assigned": lambda: assigned(snap, users),
        "mentioned_in": lambda: mentioned_in(snap, users),
        "review_candidate": lambda: review_candidate(snap, users, repo_owner),
        "open_authored_pr": lambda: open_authored_pr(snap, users),
        "open_authored_issue": lambda: open_authored_issue(snap, users),
        "thread": lambda: thread(snap, users),
        "reviewed_repo": lambda: reviewed_repo(snap, users),
        "unpaired_branch": lambda: unpaired_branch(snap, users),
        "interacted": lambda: interacted(snap, users, cohort),
    }
    rows_of = pd.Series(np.arange(count), index=np.asarray(user_ids, dtype=np.int64))
    if rows_of.index.has_duplicates:
        raise ValueError("relation_tokens expects each user at most once per cutoff.")
    for name, slots in relation_slots(relations).items():
        edges = builders[name]()
        if edges.empty:
            continue
        edges = edges.sort_values(["user_id", "score", "age_days"], ascending=[True, False, True], kind="stable")
        edges = edges.groupby("user_id", sort=False).head(slots.stop - slots.start)
        rows = rows_of.loc[edges["user_id"]].to_numpy(dtype=np.intp)
        columns = slots.start + edges.groupby("user_id", sort=False).cumcount().to_numpy(dtype=np.intp)
        global_ids = np.zeros(len(edges), dtype=np.int64)
        for node_type in np.unique(edges["node_type"]):
            mask = edges["node_type"].to_numpy() == node_type
            global_ids[mask] = ids.of(int(node_type), edges["entity_id"].to_numpy()[mask])
        tokens.entity_ids[rows, columns] = global_ids
        tokens.node_type_ids[rows, columns] = edges["node_type"].to_numpy(dtype=np.uint8)
        tokens.relation_ids[rows, columns] = RELATION_IDS[name]
        tokens.hop_ids[rows, columns] = HOP
        tokens.node_features[rows, columns] = edges[["f0", "f1", "f2", "f3"]].to_numpy(dtype=np.float32)
        tokens.relative_days[rows, columns] = edges["age_days"].to_numpy(dtype=np.float32)
    return tokens


def concat_tokens(base: GraphTokens, extra: GraphTokens) -> GraphTokens:
    if not np.array_equal(base.user_ids, extra.user_ids):
        raise ValueError("Base and relationship tokens are not aligned by user.")
    return GraphTokens(
        user_ids=base.user_ids,
        **{
            name: np.concatenate([getattr(base, name), getattr(extra, name)], axis=1)
            for name in ("entity_ids", "node_type_ids", "relation_ids", "hop_ids", "node_features", "relative_days")
        },
    )
