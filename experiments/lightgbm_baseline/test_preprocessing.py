from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import pandas as pd

from experiments.lightgbm_baseline.run import (
    _lifecycle_history,
    LABELS,
    DEFAULT_THRESHOLDS,
    eligible_users_for_cutoff,
    fit_frequency_nn_models,
    fit_models,
    frequency_feature_columns,
    load_histories,
    predict_frequency_nn_probabilities,
    row_for_cutoff,
    save_models,
    training_data_before,
)


class LifecycleFeatureTests(unittest.TestCase):
    def test_fixed_thresholds_are_the_defaults(self) -> None:
        self.assertEqual(set(DEFAULT_THRESHOLDS), set(LABELS))
        self.assertEqual(DEFAULT_THRESHOLDS["push"], 0.45)
        self.assertTrue(
            all(DEFAULT_THRESHOLDS[label] == 0.30 for label in LABELS if label != "push")
        )

    def test_frequency_nn_uses_only_frequency_inputs_and_returns_probabilities(self) -> None:
        feature_columns = [
            f"{label}_frequency_per_day_7d"
            for label in LABELS
        ]
        columns = ["user_id", "cutoff", *LABELS, *feature_columns]
        selected_features = frequency_feature_columns(columns)
        self.assertEqual(selected_features, feature_columns)
        self.assertNotIn("user_id", selected_features)
        self.assertFalse(any(label in selected_features for label in LABELS))

        train = pd.DataFrame(
            {
                column: [float((row + index) % 5) for row in range(24)]
                for index, column in enumerate(feature_columns)
            }
        )
        for label_index, label in enumerate(LABELS):
            train[label] = [(row + label_index) % 2 for row in range(24)]

        models = fit_frequency_nn_models(train, selected_features)
        probabilities = predict_frequency_nn_probabilities(
            models, train, selected_features
        )

        self.assertEqual(set(probabilities), set(LABELS))
        for values in probabilities.values():
            self.assertEqual(values.shape, (24,))
            self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_frequency_nn_can_share_one_six_label_output_layer(self) -> None:
        feature_columns = [
            f"{label}_frequency_per_day_7d"
            for label in LABELS
        ]
        train = pd.DataFrame(
            {
                column: [float((row + index) % 5) for row in range(24)]
                for index, column in enumerate(feature_columns)
            }
        )
        for label_index, label in enumerate(LABELS):
            train[label] = [(row + label_index) % 2 for row in range(24)]

        model = fit_frequency_nn_models(train, feature_columns, "multi_label")
        probabilities = predict_frequency_nn_probabilities(
            model, train, feature_columns, "multi_label"
        )

        self.assertEqual(set(probabilities), set(LABELS))
        for values in probabilities.values():
            self.assertEqual(values.shape, (24,))
            self.assertTrue(((values >= 0) & (values <= 1)).all())

        with TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            save_models(
                model,
                {},
                output_dir,
                {},
                temporal_model="frequency_nn",
                nn_output_mode="multi_label",
            )
            saved_model = joblib.load(
                output_dir / "models" / "frequency_nn_multilabel.joblib"
            )
            saved_probabilities = predict_frequency_nn_probabilities(
                saved_model, train, feature_columns, "multi_label"
            )
        for label in LABELS:
            self.assertEqual(
                probabilities[label].tolist(), saved_probabilities[label].tolist()
            )

    def test_action_totals_shares_and_frequencies_are_cutoff_safe(self) -> None:
        cutoff = pd.Timestamp("2024-04-01")
        histories = {
            label: pd.DataFrame(
                {
                    "user_id": [7] * len(action_dates),
                    "repo_id": [1] * len(action_dates),
                    "action_at": pd.to_datetime(action_dates),
                    "available_at": pd.to_datetime(available_dates),
                }
            )
            for label, action_dates, available_dates in (
                (
                    "open_pr",
                    ["2024-03-10"],
                    ["2024-03-10"],
                ),
                (
                    "open_issue",
                    [],
                    [],
                ),
                (
                    "issue_comment",
                    ["2024-03-05", "2024-03-15"],
                    ["2024-03-05", "2024-03-15"],
                ),
                (
                    "pr_review",
                    [],
                    [],
                ),
                (
                    "pr_review_comment",
                    [],
                    [],
                ),
                (
                    "push",
                    [
                        "2024-03-10",
                        "2024-03-11",
                        "2024-03-12",
                        "2024-03-20",
                    ],
                    [
                        "2024-03-10",
                        "2024-03-11",
                        "2024-03-12",
                        "2024-04-02",
                    ],
                ),
            )
        }

        features = row_for_cutoff(cutoff, histories, [7, 8]).set_index("user_id")

        self.assertAlmostEqual(float(features.loc[7, "push_share_30d"]), 0.5)
        self.assertAlmostEqual(float(features.loc[7, "open_pr_share_30d"]), 1 / 6)
        self.assertAlmostEqual(
            float(features.loc[7, "issue_comment_share_30d"]), 2 / 6
        )
        self.assertEqual(float(features.loc[8, "push_share_30d"]), 0.0)
        self.assertIn("total_actions_count_7d", features.columns)
        self.assertNotIn("total_actions_count_30d", features.columns)
        self.assertFalse(
            any(column.endswith("_count_30d") for column in features.columns)
        )
        self.assertAlmostEqual(
            float(features.loc[7, "push_frequency_per_day_7d"]), 0.0
        )
        self.assertTrue(
            all(
                f"{label}_frequency_per_day_7d" in features.columns
                for label in LABELS
            )
        )
        self.assertFalse(
            any("_frequency_per_day_30d" in column for column in features.columns)
        )
        self.assertFalse(
            any("90d" in column or "365d" in column for column in features.columns)
        )

    def test_median_daily_activity_includes_zero_days_and_respects_availability(self) -> None:
        cutoff = pd.Timestamp("2024-04-08")
        histories = {
            "push": pd.DataFrame(
                {
                    "user_id": [7] * 5,
                    "repo_id": [1] * 5,
                    "action_at": pd.to_datetime(
                        [
                            "2024-04-01",
                            "2024-04-02",
                            "2024-04-03",
                            "2024-04-04",
                            "2024-04-07",
                        ]
                    ),
                    "available_at": pd.to_datetime(
                        [
                            "2024-04-01",
                            "2024-04-02",
                            "2024-04-03",
                            "2024-04-04",
                            "2024-04-09",
                        ]
                    ),
                }
            ),
            "issue_comment": pd.DataFrame(
                {
                    "user_id": [7] * 8,
                    "repo_id": [1] * 8,
                    "action_at": pd.to_datetime(
                        [
                            "2024-04-04",
                            "2024-04-04",
                            "2024-04-05",
                            "2024-04-05",
                            "2024-04-06",
                            "2024-04-06",
                            "2024-04-07",
                            "2024-04-07",
                        ]
                    ),
                    "available_at": pd.to_datetime(
                        [
                            "2024-04-04",
                            "2024-04-04",
                            "2024-04-05",
                            "2024-04-05",
                            "2024-04-06",
                            "2024-04-06",
                            "2024-04-07",
                            "2024-04-07",
                        ]
                    ),
                }
            ),
        }

        features = row_for_cutoff(cutoff, histories, [7, 8]).set_index("user_id")

        self.assertEqual(
            float(features.loc[7, "push_median_actions_per_day_7d"]), 1.0
        )
        self.assertEqual(
            float(features.loc[7, "issue_comment_median_actions_per_day_7d"]), 2.0
        )
        self.assertEqual(
            float(features.loc[7, "total_median_actions_per_day_7d"]), 2.0
        )
        self.assertEqual(float(features.loc[8, "total_median_actions_per_day_7d"]), 0)

    def test_90_day_sample_filter_is_cutoff_safe_and_not_a_feature(self) -> None:
        histories = {
            label: pd.DataFrame(
                {
                    "user_id": user_ids,
                    "repo_id": [1] * len(user_ids),
                    "action_at": pd.to_datetime(action_dates),
                    "available_at": pd.to_datetime(available_dates),
                }
            )
            for label, user_ids, action_dates, available_dates in (
                ("open_pr", [7, 8], ["2024-01-01"] * 2, ["2024-01-01"] * 2),
                ("open_issue", [7, 8], ["2024-01-02"] * 2, ["2024-01-02"] * 2),
                (
                    "issue_comment",
                    [7, 8],
                    ["2024-01-03"] * 2,
                    ["2024-01-03"] * 2,
                ),
                (
                    "pr_review",
                    [7, 8],
                    ["2024-01-04"] * 2,
                    ["2024-01-11", "2024-01-04"],
                ),
                (
                    "pr_review_comment",
                    [7, 8],
                    ["2024-01-05"] * 2,
                    ["2024-01-05"] * 2,
                ),
                ("push", [], [], []),
            )
        }
        cutoff = pd.Timestamp("2024-01-10")

        eligible = eligible_users_for_cutoff(cutoff, histories, [7, 8])
        features = row_for_cutoff(cutoff, histories, [7, 8])

        self.assertEqual(eligible.tolist(), [False, True])
        self.assertFalse(
            any("90d" in column or "365d" in column for column in features.columns)
        )

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

        self.assertEqual(int(features.loc[0, "pr_merged_authored_count_7d"]), 1)

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

        self.assertEqual(int(authored.loc[7, "pr_merged_authored_count_7d"]), 1)
        self.assertEqual(int(authored.loc[8, "pr_merged_by_count_7d"]), 1)
        self.assertEqual(
            int(authored.loc[7, "issue_closed_reason_completed_count_7d"]), 1
        )
        self.assertEqual(int(authored.loc[7, "pr_review_dismissed_count_7d"]), 1)

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
