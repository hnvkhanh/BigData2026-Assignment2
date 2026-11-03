"""Measure how strongly candidate relationships predict next-week actions.

Uses only the training weeks of the chronological split (the oldest 70% of
weekly cutoffs), so exploring never looks at calibration or holdout weeks.
For every relational feature and every label it reports:

- ``lift``: P(label | feature > 0) / P(label | feature = 0);
- ``new_lift``: the same lift among user-weeks where the user did *not* do the
  label in the 7 days before the cutoff. A high ``new_lift`` means the
  relationship predicts actions the user's own recent activity does not.

It also writes the action-to-action transition lift (did X in the last 7 days
-> does Y next week).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.lightgbm_baseline.run import (
    DEFAULT_DB_DIR,
    DEFAULT_START_CUTOFF,
    HORIZON_DAYS,
    LABELS,
    chronological_splits,
    eligible_users_for_cutoff,
    labels_for_horizon,
    load_histories,
)
from experiments.relational_lightgbm.features import (
    FEATURE_GROUPS,
    feature_names,
    load_relational_tables,
    relational_features,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MIN_SUPPORT = 200


def recent_actions(
    cutoff: pd.Timestamp, histories: dict[str, pd.DataFrame], user_ids: list[int]
) -> pd.DataFrame:
    """Whether each user did each label in the 7 days before ``cutoff``."""
    frame = pd.DataFrame({"user_id": user_ids})
    start = cutoff - pd.Timedelta(days=HORIZON_DAYS)
    for label in LABELS:
        history = histories[label]
        done = history.loc[
            (history["available_at"] < cutoff)
            & (history["action_at"] < cutoff)
            & (history["action_at"] >= start),
            "user_id",
        ].unique()
        frame[f"recent_{label}"] = frame["user_id"].isin(done).astype(np.int8)
    return frame


def _rate(y: np.ndarray, mask: np.ndarray) -> float:
    return float(y[mask].mean()) if mask.any() else float("nan")


def lift_table(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        active = panel[feature].to_numpy() > 0
        for label in LABELS:
            y = panel[label].to_numpy()
            fresh = panel[f"recent_{label}"].to_numpy() == 0
            with_rate, without_rate = _rate(y, active), _rate(y, ~active)
            new_with, new_without = _rate(y, active & fresh), _rate(y, ~active & fresh)
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "support": int(active.sum()),
                    "new_support": int((active & fresh).sum()),
                    "rate_with": with_rate,
                    "rate_without": without_rate,
                    "lift": with_rate / without_rate if without_rate else np.nan,
                    "new_rate_with": new_with,
                    "new_rate_without": new_without,
                    "new_lift": new_with / new_without if new_without else np.nan,
                }
            )
    return pd.DataFrame(rows)


def transition_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Lift of doing ``to`` next week given ``from`` was done in the last 7 days."""
    table = pd.DataFrame(index=pd.Index(LABELS, name="from"), columns=list(LABELS), dtype=float)
    for source in LABELS:
        did = panel[f"recent_{source}"].to_numpy() == 1
        for target in LABELS:
            y = panel[target].to_numpy()
            table.loc[source, target] = _rate(y, did) / _rate(y, ~did)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF))
    args = parser.parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR / (
        "explore_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    user_ids = users.loc[users["is_cohort"], "user_id"].astype(int).tolist()
    histories = load_histories(args.db_dir, user_ids)
    tables = load_relational_tables(args.db_dir)
    coverage_end = min(histories[label]["action_at"].max() for label in LABELS)
    cutoffs = list(
        pd.date_range(args.start_cutoff, coverage_end - pd.Timedelta(days=HORIZON_DAYS), freq="7D")
    )
    train_end, _ = chronological_splits(cutoffs)
    train_cutoffs = [cutoff for cutoff in cutoffs if cutoff < train_end]

    groups = list(FEATURE_GROUPS)
    panels = []
    for index, cutoff in enumerate(train_cutoffs, start=1):
        eligible = eligible_users_for_cutoff(cutoff, histories, user_ids)
        frame = (
            relational_features(cutoff, tables, histories, user_ids, groups)
            .merge(labels_for_horizon(cutoff, histories, user_ids), on="user_id")
            .merge(recent_actions(cutoff, histories, user_ids), on="user_id")
        )
        panels.append(frame.loc[eligible].assign(cutoff=cutoff))
        print(f"Explored training week {index}/{len(train_cutoffs)}: {cutoff}")
    panel = pd.concat(panels, ignore_index=True)

    lifts = lift_table(panel, feature_names(groups))
    lifts.to_csv(output_dir / "relationship_lift.csv", index=False)
    transitions = transition_table(panel)
    transitions.to_csv(output_dir / "transition_lift.csv")

    pd.set_option("display.width", 220)
    print(f"\n{len(panel):,} user-weeks from {len(train_cutoffs)} training weeks (< {train_end.date()})")
    print("\nAction transitions (lift of next-week column given row in last 7 days):")
    print(transitions.round(2).to_string())
    strongest = (
        lifts.loc[lifts["new_support"] >= MIN_SUPPORT]
        .sort_values("new_lift", ascending=False)
        .groupby("label")
        .head(6)
        .sort_values(["label", "new_lift"], ascending=[True, False])
    )
    print(f"\nStrongest relationships per label among users who did not do it last week (support >= {MIN_SUPPORT}):")
    print(
        strongest[
            ["label", "feature", "new_support", "new_rate_with", "new_rate_without", "new_lift", "lift"]
        ].round(3).to_string(index=False)
    )
    print(f"\nSaved {output_dir / 'relationship_lift.csv'} and {output_dir / 'transition_lift.csv'}")


if __name__ == "__main__":
    main()
