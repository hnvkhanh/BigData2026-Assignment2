from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments.lightgbm_baseline.run import (
    _lifecycle_history,
    load_histories,
    row_for_cutoff,
    training_data_before,
)


class LifecycleFeatureTests(unittest.TestCase):
    def test_lifecycle_events_require_event_and_row_availability_before_cutoff(self) -> None:
        raw = pd.DataFrame(
            {
                "user_id": [7, 7, 7],
                "repo_id": [1, 1, 1],
                "merged_at": pd.to_datetime(
                    ["2024-01-03", "2024-01-12", "2024-01-03"]
                ),
                "first_seen_at": pd.to_datetime(
                    ["2024-01-03", "2024-01-03", "2024-01-15"]
                ),
            }
        )
        history = _lifecycle_history(
            raw, "user_id", "merged_at", "first_seen_at", {7}
        )

        features = row_for_cutoff(
            pd.Timestamp("2024-01-10"),
            {"pr_merged_authored": history},
            [7],
        )

        self.assertEqual(int(features.loc[0, "pr_merged_authored_count_90d"]), 1)

    def test_load_histories_builds_lifecycle_features_for_the_right_users(self) -> None:
        tables = {
            "PullRequests.parquet": pd.DataFrame(
                {
                    "author_id": [7],
                    "merged_by_id": [8],
                    "repo_id": [1],
                    "created_at": pd.to_datetime(["2024-01-01"]),
                    "first_seen_at": pd.to_datetime(["2024-01-01"]),
                    "merged_at": pd.to_datetime(["2024-01-03"]),
                    "closed_at": pd.to_datetime(["2024-01-03"]),
                }
            ),
            "Issues.parquet": pd.DataFrame(
                {
                    "author_id": [7],
                    "repo_id": [1],
                    "created_at": pd.to_datetime(["2024-01-01"]),
                    "first_seen_at": pd.to_datetime(["2024-01-01"]),
                    "is_pull_request": [False],
                    "closed_at": pd.to_datetime(["2024-01-04"]),
                    "state_reason": ["completed"],
                }
            ),
            "IssueComments.parquet": pd.DataFrame(
                {
                    "author_id": [7],
                    "repo_id": [1],
                    "created_at": pd.to_datetime(["2024-01-02"]),
                }
            ),
            "PRReviews.parquet": pd.DataFrame(
                {
                    "reviewer_id": [7],
                    "repo_id": [1],
                    "submitted_at": pd.to_datetime(["2024-01-02"]),
                    "first_seen_at": pd.to_datetime(["2024-01-02"]),
                    "dismissed_at": pd.to_datetime(["2024-01-05"]),
                }
            ),
            "PRReviewComments.parquet": pd.DataFrame(
                {
                    "author_id": [7],
                    "repo_id": [1],
                    "created_at": pd.to_datetime(["2024-01-02"]),
                }
            ),
            "Pushes.parquet": pd.DataFrame(
                {
                    "user_id": [7],
                    "repo_id": [1],
                    "created_at": pd.to_datetime(["2024-01-02"]),
                }
            ),
        }

        def read_table(path: Path, columns: list[str]) -> pd.DataFrame:
            return tables[path.name][columns]

        with patch(
            "experiments.lightgbm_baseline.run.pd.read_parquet",
            side_effect=read_table,
        ):
            histories = load_histories(Path("unused"), [7, 8])

        cutoff = pd.Timestamp("2024-01-10")
        authored = row_for_cutoff(cutoff, histories, [7, 8]).set_index("user_id")

        self.assertEqual(int(authored.loc[7, "pr_merged_authored_count_90d"]), 1)
        self.assertEqual(int(authored.loc[8, "pr_merged_by_count_90d"]), 1)
        self.assertEqual(
            int(authored.loc[7, "issue_closed_reason_completed_count_90d"]), 1
        )
        self.assertEqual(int(authored.loc[7, "pr_review_dismissed_count_90d"]), 1)

    def test_training_examples_require_complete_labels_before_prediction(self) -> None:
        data = pd.DataFrame(
            {
                "cutoff": pd.to_datetime(
                    ["2024-01-01", "2024-01-08", "2024-01-15"]
                ),
                "label": [0, 1, 0],
            }
        )

        filtered = training_data_before(data, pd.Timestamp("2024-01-15"))

        self.assertEqual(
            filtered["cutoff"].tolist(),
            [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")],
        )


if __name__ == "__main__":
    unittest.main()
