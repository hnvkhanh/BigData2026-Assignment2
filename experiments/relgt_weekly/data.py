"""Week-vector inputs: each user-week sample is the last W weeks of activity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.relgt.data import (
    LABELS,
    labels_for_cutoff,
    load_entity_vocabulary,
    load_events,
    weekly_cutoffs,
)
from experiments.relgt.run import _data_signature

WEEK_DAYS = 7
NUM_RELATIONS = len(LABELS)
# Per relation and week: log1p of event count, distinct entities, distinct repositories.
STATS_PER_RELATION = 3
NUM_FEATURES = NUM_RELATIONS * STATS_PER_RELATION
MIN_ELIGIBLE_EVENTS = 5
DAY_NS = pd.Timedelta(days=1).value


@dataclass
class WeeklyExamples:
    """``features[i, w]`` is user i's vector for the week ending ``w`` weeks before the cutoff."""

    user_ids: np.ndarray
    features: np.ndarray  # (N, weeks, NUM_FEATURES) float32; w=0 is the most recent week
    labels: np.ndarray
    cutoff_days: np.ndarray


def weekly_features(
    events: pd.DataFrame, cutoff: pd.Timestamp, weeks: int, eligibility_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Eligible users and their per-week activity vectors at one cutoff.

    Week 0 is ``[cutoff - 7d, cutoff)``, week 1 the week before, and so on, so
    the bins line up with the 7-day label window. Events count only once their
    action, row and target entity are all known strictly before the cutoff.
    A user is eligible with at least five such events in ``eligibility_days``.
    """
    window = pd.Timedelta(days=max(weeks * WEEK_DAYS, eligibility_days))
    known = events.loc[
        (events["available_at"] < cutoff)
        & (events["entity_available_at"] < cutoff)
        & (events["action_at"] < cutoff)
        & (events["action_at"] >= cutoff - window)
    ]
    empty = (
        np.empty(0, dtype=np.int32),
        np.zeros((0, weeks, NUM_FEATURES), dtype=np.float32),
    )
    if known.empty:
        return empty
    recent = known.loc[
        known["action_at"] >= cutoff - pd.Timedelta(days=eligibility_days)
    ]
    recent_counts = recent.groupby("user_id").size()
    eligible = np.sort(
        recent_counts.index[recent_counts >= MIN_ELIGIBLE_EVENTS].to_numpy(np.int32)
    )
    if not len(eligible):
        return empty

    rows = known.loc[known["user_id"].isin(eligible)]
    age_weeks = np.floor(
        (cutoff - rows["action_at"]).dt.total_seconds() / (WEEK_DAYS * 86_400)
    ).astype(np.int64)
    rows = rows.assign(week=age_weeks.to_numpy()).loc[lambda r: r["week"] < weeks]
    grouped = (
        rows.groupby(["user_id", "week", "relation_id"], sort=False)
        .agg(
            count=("action_at", "size"),
            entities=("global_entity_id", "nunique"),
            repos=("repo_id", "nunique"),
        )
        .reset_index()
    )
    features = np.zeros(
        (len(eligible), weeks, NUM_RELATIONS, STATS_PER_RELATION), dtype=np.float32
    )
    users = np.searchsorted(eligible, grouped["user_id"].to_numpy(np.int32))
    week = grouped["week"].to_numpy(np.int64)
    relation = grouped["relation_id"].to_numpy(np.int64) - 1
    for stat, column in enumerate(("count", "entities", "repos")):
        features[users, week, relation, stat] = np.log1p(
            grouped[column].to_numpy(np.float32)
        )
    return eligible, features.reshape(len(eligible), weeks, NUM_FEATURES)


def build_weekly_examples(
    db_dir: Path, start_cutoff: pd.Timestamp, weeks: int, eligibility_days: int
) -> WeeklyExamples:
    users = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    cohort_ids = users.loc[users["is_cohort"], "user_id"].astype(np.int32).tolist()
    events = load_events(db_dir, cohort_ids, load_entity_vocabulary(db_dir))
    cutoffs = weekly_cutoffs(events, start_cutoff)

    user_batches, feature_batches, label_batches, cutoff_batches = [], [], [], []
    for index, cutoff in enumerate(cutoffs, start=1):
        user_ids, features = weekly_features(events, cutoff, weeks, eligibility_days)
        user_batches.append(user_ids)
        feature_batches.append(features)
        label_batches.append(labels_for_cutoff(events, cutoff, user_ids))
        cutoff_batches.append(
            np.full(len(user_ids), cutoff.value // DAY_NS, dtype=np.int32)
        )
        print(
            f"Built weekly vectors {index}/{len(cutoffs)}: {cutoff} "
            f"({len(user_ids):,} eligible users)"
        )
    return WeeklyExamples(
        user_ids=np.concatenate(user_batches),
        features=np.concatenate(feature_batches, axis=0),
        labels=np.concatenate(label_batches, axis=0),
        cutoff_days=np.concatenate(cutoff_batches),
    )


def load_or_build_weekly_examples(
    db_dir: Path,
    output_dir: Path,
    start_cutoff: pd.Timestamp,
    weeks: int,
    eligibility_days: int,
) -> WeeklyExamples:
    cache_path = (
        output_dir / "cache" / f"weekly_vectors_{weeks}w_eligible_{eligibility_days}d.npz"
    )
    metadata_path = cache_path.with_suffix(".json")
    signature: dict[str, Any] = {
        **_data_signature(db_dir, start_cutoff, weeks * WEEK_DAYS, eligibility_days),
        "kind": "weekly_vectors",
        "weeks": weeks,
        "num_features": NUM_FEATURES,
    }
    if cache_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == signature:
            with np.load(cache_path, allow_pickle=False) as stored:
                print(f"Loaded cached weekly vectors from {cache_path}")
                return WeeklyExamples(
                    user_ids=stored["user_ids"],
                    features=stored["features"],
                    labels=stored["labels"],
                    cutoff_days=stored["cutoff_days"],
                )

    examples = build_weekly_examples(db_dir, start_cutoff, weeks, eligibility_days)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    np.savez(
        temporary_path,
        user_ids=examples.user_ids,
        features=examples.features,
        labels=examples.labels,
        cutoff_days=examples.cutoff_days,
    )
    temporary_path.replace(cache_path)
    metadata_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    print(f"Cached weekly vectors at {cache_path}")
    return examples
