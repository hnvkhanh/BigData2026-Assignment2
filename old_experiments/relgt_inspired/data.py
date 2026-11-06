"""Temporal heterogeneous graph snapshots for the GitHub activity task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet


LABELS = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)

HORIZON_DAYS = 7
LOOKBACK_DAYS = 14
MAX_OBJECTS_PER_RELATION = 4
MAX_OBJECT_NEIGHBORS = len(LABELS) * MAX_OBJECTS_PER_RELATION
MAX_REPOSITORY_NEIGHBORS = 24
MAX_NEIGHBORS = MAX_OBJECT_NEIGHBORS + MAX_REPOSITORY_NEIGHBORS
REPO_MEMBERSHIP_RELATION = len(LABELS) + 1

# Entity types: 1=PullRequest, 2=Issue, 3=Repository.
ACTION_SPECS = {
    "open_pr": ("PullRequests", "author_id", "created_at", "first_seen_at", "pr_id", 1),
    "open_issue": ("Issues", "author_id", "created_at", "first_seen_at", "issue_id", 2),
    "issue_comment": (
        "IssueComments",
        "author_id",
        "created_at",
        "created_at",
        "issue_id",
        2,
    ),
    "pr_review": (
        "PRReviews",
        "reviewer_id",
        "submitted_at",
        "first_seen_at",
        "pr_id",
        1,
    ),
    "pr_review_comment": (
        "PRReviewComments",
        "author_id",
        "created_at",
        "created_at",
        "pr_id",
        1,
    ),
    "push": ("Pushes", "user_id", "created_at", "created_at", "repo_id", 3),
}

ENTITY_TABLES = {
    1: ("PullRequests", "pr_id", "first_seen_at"),
    2: ("Issues", "issue_id", "first_seen_at"),
    3: ("Repositories", "repo_id", "first_seen_at"),
}


@dataclass(frozen=True)
class EntityVocabulary:
    """Disjoint embedding ranges for PR, issue, and repository nodes."""

    num_pull_requests: int
    num_issues: int
    num_repositories: int

    @property
    def num_entities(self) -> int:
        return 1 + self.num_pull_requests + self.num_issues + self.num_repositories

    def offset_for_type(self, node_type: int) -> int:
        if node_type == 1:
            return 1
        if node_type == 2:
            return 1 + self.num_pull_requests
        if node_type == 3:
            return 1 + self.num_pull_requests + self.num_issues
        raise ValueError(f"Unknown entity node type: {node_type}")


@dataclass
class GraphTokens:
    """Fixed-size typed neighborhoods for target users."""

    user_ids: np.ndarray
    entity_ids: np.ndarray
    node_type_ids: np.ndarray
    relation_ids: np.ndarray
    hop_ids: np.ndarray
    node_features: np.ndarray
    relative_days: np.ndarray


@dataclass
class TrainingExamples:
    """Graph neighborhoods and labels for chronological user-week samples."""

    tokens: GraphTokens
    labels: np.ndarray
    cutoff_days: np.ndarray
    num_entities: int


def load_entity_vocabulary(db_dir: Path) -> EntityVocabulary:
    """Read entity counts from Parquet metadata without loading table columns."""
    counts = {
        node_type: parquet.ParquetFile(db_dir / f"{table}.parquet").metadata.num_rows
        for node_type, (table, _, _) in ENTITY_TABLES.items()
    }
    return EntityVocabulary(
        num_pull_requests=counts[1],
        num_issues=counts[2],
        num_repositories=counts[3],
    )


def _entity_global_ids(
    node_type: int, entity_ids: np.ndarray, vocabulary: EntityVocabulary
) -> np.ndarray:
    return entity_ids.astype(np.int64) + vocabulary.offset_for_type(node_type)


def load_events(
    db_dir: Path,
    cohort_user_ids: list[int],
    vocabulary: EntityVocabulary,
) -> pd.DataFrame:
    """Load cohort actions with action, row, and target-node availability times."""
    cohort = set(cohort_user_ids)
    entity_availability: dict[int, pd.DataFrame] = {}
    for node_type, (table, key, time_col) in ENTITY_TABLES.items():
        frame = pd.read_parquet(db_dir / f"{table}.parquet", columns=[key, time_col])
        entity_availability[node_type] = frame.rename(
            columns={key: "entity_row_id", time_col: "entity_available_at"}
        )

    frames: list[pd.DataFrame] = []
    for relation_id, (table, actor_col, action_col, available_col, entity_col, node_type) in enumerate(
        ACTION_SPECS.values(), start=1
    ):
        columns = list(
            dict.fromkeys([actor_col, "repo_id", entity_col, action_col, available_col])
        )
        if table == "Issues":
            columns.append("is_pull_request")
        raw = pd.read_parquet(db_dir / f"{table}.parquet", columns=columns)
        mask = raw[actor_col].isin(cohort) & raw[action_col].notna()
        if table == "Issues":
            mask &= ~raw["is_pull_request"].fillna(False)
        rows = raw.loc[mask].copy()
        rows["user_id"] = rows[actor_col]
        rows["entity_id"] = rows[entity_col]
        rows["action_at"] = rows[action_col]
        rows["available_at"] = rows[available_col]
        rows = rows.dropna(
            subset=["user_id", "repo_id", "entity_id", "action_at", "available_at"]
        )
        rows = rows[["user_id", "repo_id", "entity_id", "action_at", "available_at"]]
        rows = rows.merge(
            entity_availability[node_type],
            left_on="entity_id",
            right_on="entity_row_id",
            how="left",
            validate="many_to_one",
        )
        rows = rows.dropna(subset=["entity_available_at"])
        rows["relation_id"] = relation_id
        rows["node_type"] = node_type
        rows["user_id"] = rows["user_id"].astype(np.int32)
        rows["repo_id"] = rows["repo_id"].astype(np.int32)
        rows["entity_id"] = rows["entity_id"].astype(np.int32)
        rows["global_entity_id"] = _entity_global_ids(
            node_type, rows["entity_id"].to_numpy(), vocabulary
        ).astype(np.int32)
        for column in ("action_at", "available_at", "entity_available_at"):
            rows[column] = pd.to_datetime(rows[column])
        frames.append(
            rows[
                [
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
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _empty_tokens(user_ids: np.ndarray) -> GraphTokens:
    count = len(user_ids)
    return GraphTokens(
        user_ids=user_ids,
        entity_ids=np.zeros((count, MAX_NEIGHBORS), dtype=np.int32),
        node_type_ids=np.zeros((count, MAX_NEIGHBORS), dtype=np.uint8),
        relation_ids=np.zeros((count, MAX_NEIGHBORS), dtype=np.uint8),
        hop_ids=np.zeros((count, MAX_NEIGHBORS), dtype=np.uint8),
        node_features=np.zeros((count, MAX_NEIGHBORS, 4), dtype=np.float32),
        relative_days=np.zeros((count, MAX_NEIGHBORS), dtype=np.float32),
    )


def snapshot_tokens(
    events: pd.DataFrame,
    cutoff: pd.Timestamp,
    user_ids: list[int] | np.ndarray,
    vocabulary: EntityVocabulary,
    eligible_only: bool = False,
    lookback_days: int = LOOKBACK_DAYS,
) -> GraphTokens:
    """Build a temporal user→entity→repository neighborhood at a cutoff."""
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive.")
    ordered_user_ids = np.asarray(user_ids, dtype=np.int32)
    tokens = _empty_tokens(ordered_user_ids)
    if not len(ordered_user_ids):
        return tokens

    start = cutoff - pd.Timedelta(days=lookback_days)
    known = events.loc[
        (events["available_at"] < cutoff)
        & (events["entity_available_at"] < cutoff)
        & (events["action_at"] < cutoff)
        & (events["action_at"] >= start)
    ]
    if known.empty:
        return tokens

    recent = known.loc[known["user_id"].isin(ordered_user_ids)].assign(
        active_day=lambda rows: rows["action_at"].dt.floor("D")
    )
    if recent.empty:
        return tokens
    group_columns = [
        "user_id",
        "repo_id",
        "entity_id",
        "global_entity_id",
        "node_type",
        "relation_id",
    ]
    grouped = (
        recent.groupby(group_columns, sort=False)
        .agg(
            event_count=("action_at", "size"),
            active_days=("active_day", "nunique"),
            last_action=("action_at", "max"),
        )
        .reset_index()
    )
    if eligible_only:
        eligible = grouped.groupby("user_id")["event_count"].sum()
        eligible_ids = set(eligible.loc[eligible >= 5].index.astype(int))
        ordered_user_ids = np.asarray(
            [uid for uid in ordered_user_ids if int(uid) in eligible_ids],
            dtype=np.int32,
        )
        tokens = _empty_tokens(ordered_user_ids)
        if not len(ordered_user_ids):
            return tokens
        grouped = grouped.loc[grouped["user_id"].isin(eligible_ids)]
        recent = recent.loc[recent["user_id"].isin(eligible_ids)]

    row_for_user = {int(uid): row for row, uid in enumerate(ordered_user_ids)}
    entity_structure = (
        known.groupby(["node_type", "entity_id"], sort=False)
        .agg(
            entity_event_count=("action_at", "size"),
            entity_user_count=("user_id", "nunique"),
            entity_relation_count=("relation_id", "nunique"),
        )
        .reset_index()
    )
    object_neighbors = grouped.merge(
        entity_structure, on=["node_type", "entity_id"], validate="many_to_one"
    )
    object_neighbors = object_neighbors.sort_values(
        ["user_id", "relation_id", "last_action", "event_count"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    object_neighbors = object_neighbors.groupby(
        ["user_id", "relation_id"], sort=False, group_keys=False
    ).head(MAX_OBJECTS_PER_RELATION)
    object_neighbors["row"] = object_neighbors["user_id"].map(row_for_user)
    object_neighbors["rank"] = object_neighbors.groupby(
        ["user_id", "relation_id"], sort=False
    ).cumcount()
    object_neighbors["slot"] = (
        (object_neighbors["relation_id"].astype(np.int32) - 1)
        * MAX_OBJECTS_PER_RELATION
        + object_neighbors["rank"].astype(np.int32)
    )
    object_neighbors["hop"] = 1
    object_neighbors["feature_count"] = np.log1p(
        object_neighbors["event_count"].astype(np.float32)
    )
    object_neighbors["feature_active_days"] = np.log1p(
        object_neighbors["active_days"].astype(np.float32)
    )
    object_neighbors["feature_degree"] = np.log1p(
        object_neighbors["entity_event_count"].astype(np.float32)
    )
    object_neighbors["feature_diversity"] = (
        object_neighbors["entity_user_count"].astype(np.float32)
        / np.maximum(object_neighbors["entity_relation_count"], 1)
    )

    repo_structure = (
        known.groupby("repo_id", sort=False)
        .agg(
            repo_event_count=("action_at", "size"),
            repo_user_count=("user_id", "nunique"),
            repo_relation_count=("relation_id", "nunique"),
        )
        .reset_index()
    )
    user_repositories = (
        recent.groupby(["user_id", "repo_id"], sort=False)
        .agg(
            event_count=("action_at", "size"),
            active_days=("active_day", "nunique"),
            last_action=("action_at", "max"),
        )
        .reset_index()
        .merge(repo_structure, on="repo_id", validate="many_to_one")
    )
    user_repositories = user_repositories.sort_values(
        ["user_id", "last_action", "event_count"],
        ascending=[True, False, False],
        kind="stable",
    )
    user_repositories = user_repositories.groupby(
        "user_id", sort=False, group_keys=False
    ).head(MAX_REPOSITORY_NEIGHBORS)
    user_repositories["row"] = user_repositories["user_id"].map(row_for_user)
    user_repositories["slot"] = (
        MAX_OBJECT_NEIGHBORS
        + user_repositories.groupby("user_id", sort=False).cumcount().astype(np.int32)
    )
    user_repositories["node_type"] = 3
    user_repositories["relation_id"] = REPO_MEMBERSHIP_RELATION
    user_repositories["hop"] = 2
    user_repositories["global_entity_id"] = (
        user_repositories["repo_id"].astype(np.int64)
        + vocabulary.offset_for_type(3)
    )
    user_repositories["feature_count"] = np.log1p(
        user_repositories["event_count"].astype(np.float32)
    )
    user_repositories["feature_active_days"] = np.log1p(
        user_repositories["active_days"].astype(np.float32)
    )
    user_repositories["feature_degree"] = np.log1p(
        user_repositories["repo_event_count"].astype(np.float32)
    )
    user_repositories["feature_diversity"] = (
        user_repositories["repo_user_count"].astype(np.float32)
        / np.maximum(user_repositories["repo_relation_count"], 1)
    )

    for neighbors in (object_neighbors, user_repositories):
        rows = neighbors["row"].to_numpy(dtype=np.intp)
        slots = neighbors["slot"].to_numpy(dtype=np.intp)
        tokens.entity_ids[rows, slots] = neighbors[
            "global_entity_id"
        ].to_numpy(dtype=np.int32)
        tokens.node_type_ids[rows, slots] = neighbors["node_type"].to_numpy(
            dtype=np.uint8
        )
        tokens.relation_ids[rows, slots] = neighbors["relation_id"].to_numpy(
            dtype=np.uint8
        )
        tokens.hop_ids[rows, slots] = neighbors["hop"].to_numpy(dtype=np.uint8)
        tokens.node_features[rows, slots, :] = neighbors[
            [
                "feature_count",
                "feature_active_days",
                "feature_degree",
                "feature_diversity",
            ]
        ].to_numpy(dtype=np.float32)
        elapsed = (cutoff - neighbors["last_action"]).dt.total_seconds() / 86_400
        tokens.relative_days[rows, slots] = np.clip(
            elapsed.to_numpy(dtype=np.float32), 0, 3650
        )
    return tokens


def labels_for_cutoff(
    events: pd.DataFrame, cutoff: pd.Timestamp, user_ids: np.ndarray
) -> np.ndarray:
    """Label user actions in the seven days following a historical cutoff."""
    labels = np.zeros((len(user_ids), len(LABELS)), dtype=np.uint8)
    if not len(user_ids):
        return labels

    end = cutoff + pd.Timedelta(days=HORIZON_DAYS)
    future = events.loc[
        (events["action_at"] >= cutoff)
        & (events["action_at"] < end)
        & events["user_id"].isin(user_ids),
        ["user_id", "relation_id"],
    ].drop_duplicates()
    if future.empty:
        return labels
    row_for_user = {int(user_id): row for row, user_id in enumerate(user_ids)}
    rows = future["user_id"].map(row_for_user).to_numpy(dtype=np.intp)
    columns = future["relation_id"].to_numpy(dtype=np.intp) - 1
    labels[rows, columns] = 1
    return labels


def build_training_examples(
    db_dir: Path,
    start_cutoff: pd.Timestamp,
    vocabulary: EntityVocabulary,
    lookback_days: int = LOOKBACK_DAYS,
    eligibility_lookback_days: int | None = None,
    end_cutoff: pd.Timestamp | None = None,
) -> TrainingExamples:
    """Create weekly training samples with chronological labels.

    When ``end_cutoff`` is given, events after it are discarded, so neither
    snapshots nor labels use later data.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive.")
    if eligibility_lookback_days is None:
        eligibility_lookback_days = lookback_days
    if eligibility_lookback_days < lookback_days:
        raise ValueError(
            "eligibility_lookback_days must be at least lookback_days."
        )
    users = pd.read_parquet(
        db_dir / "Users.parquet", columns=["user_id", "is_cohort"]
    )
    cohort_ids = users.loc[users["is_cohort"], "user_id"].astype(np.int32).to_numpy()
    events = load_events(db_dir, cohort_ids.tolist(), vocabulary)
    if end_cutoff is not None:
        events = events[events["action_at"] <= end_cutoff]

    coverage_end = events.groupby("relation_id")["action_at"].max().min()
    last_cutoff = coverage_end - pd.Timedelta(days=HORIZON_DAYS)
    cutoffs = pd.date_range(start=start_cutoff, end=last_cutoff, freq="7D")
    if len(cutoffs) < 10:
        raise ValueError(
            f"Only {len(cutoffs)} complete weekly samples are available from "
            f"{start_cutoff} through {last_cutoff}. Choose an earlier "
            "--start-cutoff or use a newer data release."
        )

    token_batches: list[GraphTokens] = []
    label_batches: list[np.ndarray] = []
    cutoff_batches: list[np.ndarray] = []
    for index, cutoff in enumerate(cutoffs, start=1):
        eligible_tokens = snapshot_tokens(
            events,
            cutoff,
            cohort_ids,
            vocabulary,
            eligible_only=True,
            lookback_days=eligibility_lookback_days,
        )
        if lookback_days == eligibility_lookback_days:
            tokens = eligible_tokens
        else:
            tokens = snapshot_tokens(
                events,
                cutoff,
                eligible_tokens.user_ids,
                vocabulary,
                lookback_days=lookback_days,
            )
        labels = labels_for_cutoff(events, cutoff, tokens.user_ids)
        token_batches.append(tokens)
        label_batches.append(labels)
        cutoff_batches.append(
            np.full(
                len(tokens.user_ids),
                cutoff.value // pd.Timedelta(days=1).value,
                dtype=np.int32,
            )
        )
        print(
            f"Built weekly graph {index}/{len(cutoffs)}: {cutoff} "
            f"({len(tokens.user_ids):,} eligible users)"
        )

    combined_tokens = GraphTokens(
        user_ids=np.concatenate([batch.user_ids for batch in token_batches]),
        entity_ids=np.concatenate([batch.entity_ids for batch in token_batches], axis=0),
        node_type_ids=np.concatenate(
            [batch.node_type_ids for batch in token_batches], axis=0
        ),
        relation_ids=np.concatenate(
            [batch.relation_ids for batch in token_batches], axis=0
        ),
        hop_ids=np.concatenate([batch.hop_ids for batch in token_batches], axis=0),
        node_features=np.concatenate(
            [batch.node_features for batch in token_batches], axis=0
        ),
        relative_days=np.concatenate(
            [batch.relative_days for batch in token_batches], axis=0
        ),
    )
    return TrainingExamples(
        tokens=combined_tokens,
        labels=np.concatenate(label_batches, axis=0),
        cutoff_days=np.concatenate(cutoff_batches),
        num_entities=vocabulary.num_entities,
    )
