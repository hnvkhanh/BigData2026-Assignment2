from __future__ import annotations

import sys
import unittest

import numpy as np
import pandas as pd

from experiments.drift_lightgbm.data_quality import (
    anomaly_days,
    label_window_flags,
    push_coverage_features,
)
from experiments.drift_lightgbm.features import history_features, regime_features
from experiments.drift_lightgbm.run import (
    VARIANTS,
    Variant,
    delay_pushes,
    joint_thresholds,
    parse_args,
    training_frame,
    weekly_score,
)
from experiments.lightgbm_baseline.run import LABELS, ROOT

sys.path.insert(0, str(ROOT / "assignment2" / "evaluation"))
import evaluate as official_eval  # noqa: E402


def _history(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [row[0] for row in rows],
            "repo_id": [1] * len(rows),
            "action_at": pd.to_datetime([row[1] for row in rows]),
            "available_at": pd.to_datetime([row[2] for row in rows]),
        }
    )


def _empty_histories() -> dict[str, pd.DataFrame]:
    return {label: _history([]) for label in LABELS}


class DataQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        days = pd.date_range("2026-01-01", periods=56, freq="D")
        counts = pd.DataFrame({label: 100.0 for label in LABELS}, index=days)
        non_push = [label for label in LABELS if label != "push"]
        counts.loc["2026-02-10", "push"] = 0.0
        counts.loc["2026-02-12", non_push] = 900.0  # every non-push table at once
        counts.loc["2026-02-14", "issue_comment"] = 900.0  # one table only: user behaviour
        self.outage, self.burst = anomaly_days(counts)

    def test_flags_push_outages_and_systemic_non_push_bursts(self) -> None:
        self.assertTrue(self.outage.loc["2026-02-10", "push"])
        self.assertTrue(self.burst.loc["2026-02-12", "issue_comment"])
        self.assertFalse(self.burst.loc["2026-02-12", "push"])
        self.assertFalse(self.burst.loc["2026-02-14"].any())
        self.assertEqual(int(self.outage.to_numpy().sum()), 1)
        self.assertEqual(int(self.burst.to_numpy().sum()), 5)
        # The first four weeks have no baseline and are never flagged.
        self.assertFalse(self.outage.loc[:"2026-01-28"].to_numpy().any())

    def test_label_window_flags_cover_every_day_the_window_touches(self) -> None:
        flags = label_window_flags(
            [pd.Timestamp("2026-02-03 16:00"), pd.Timestamp("2026-02-10 16:00"),
             pd.Timestamp("2026-02-17 16:00")],
            self.outage,
            self.burst,
        ).set_index("cutoff")
        # [02-03 16:00, 02-10 16:00) includes part of 02-10.
        self.assertEqual(flags.loc["2026-02-03 16:00", "flag_push"], 1)
        self.assertEqual(flags.loc["2026-02-10 16:00", "flag_push"], 1)
        self.assertEqual(flags.loc["2026-02-10 16:00", "flag_issue_comment"], 1)
        self.assertEqual(flags.loc["2026-02-10 16:00", "flag_open_pr"], 1)
        self.assertEqual(flags.loc["2026-02-03 16:00", "flag_issue_comment"], 0)
        self.assertEqual(int(flags.loc["2026-02-17 16:00"].sum()), 0)

    def test_coverage_uses_only_full_days_before_the_cutoff(self) -> None:
        self.assertEqual(
            push_coverage_features(pd.Timestamp("2026-02-10 16:00"), self.outage)["push_outage_days_7d"],
            0,
        )
        later = push_coverage_features(pd.Timestamp("2026-02-11 16:00"), self.outage)
        self.assertEqual(later["push_outage_days_7d"], 1)
        lagged = push_coverage_features(pd.Timestamp("2026-02-20 16:00"), self.outage, lag_days=3)
        self.assertEqual(lagged["push_outage_days_7d"], 3)


class FeatureTests(unittest.TestCase):
    def test_history_features_are_cutoff_safe(self) -> None:
        cutoff = pd.Timestamp("2026-03-01 16:00")
        histories = _empty_histories()
        histories["push"] = _history(
            [
                (1, "2026-02-28", "2026-02-28"),  # week 0
                (1, "2026-02-20", "2026-02-20"),  # week 1
                (1, "2026-02-12", "2026-02-12"),  # week 2
                (1, "2026-02-27", "2026-03-05"),  # not yet available
                (1, "2026-03-02", "2026-03-02"),  # after the cutoff
                (2, "2026-01-01", "2026-01-01"),
            ]
        )
        frame = history_features(cutoff, histories, [1, 2, 3]).set_index("user_id")
        self.assertEqual(frame.loc[1, "push_active_weeks_4w"], 3)
        self.assertEqual(frame.loc[1, "push_active_week_minus3"], 1)
        self.assertEqual(frame.loc[1, "push_active_week_minus4"], 0)
        self.assertEqual(frame.loc[1, "push_count_30d"], 3)
        self.assertEqual(frame.loc[1, "any_active_streak_weeks"], 3)
        self.assertEqual(frame.loc[1, "any_active_days_7d"], 1)
        self.assertEqual(frame.loc[2, "push_count_90d"], 1)
        self.assertEqual(frame.loc[2, "any_active_streak_weeks"], 0)
        self.assertAlmostEqual(frame.loc[2, "days_since_first_action"], 59 + 16 / 24, places=4)
        self.assertEqual(frame.loc[3, "days_since_first_action"], 0)

    def test_regime_features_measure_the_eligible_cohort(self) -> None:
        cutoff = pd.Timestamp("2026-03-01 16:00")
        histories = _empty_histories()
        histories["open_pr"] = _history(
            [(1, "2026-02-28", "2026-02-28"), (2, "2026-02-20", "2026-02-20"),
             (9, "2026-02-28", "2026-02-28")]
        )
        values = regime_features(cutoff, histories, [1, 2, 3, 4])
        self.assertAlmostEqual(values["cohort_rate_open_pr_1w"], 0.25)
        self.assertAlmostEqual(values["cohort_rate_open_pr_4w"], 0.125)
        self.assertEqual(values["cohort_rate_push_1w"], 0.0)

    def test_push_delay_changes_availability_only(self) -> None:
        histories = _empty_histories()
        histories["push"] = _history([(1, "2026-02-28", "2026-02-28")])
        delayed = delay_pushes(histories, 6)
        self.assertEqual(delayed["push"]["available_at"].iloc[0], pd.Timestamp("2026-03-06"))
        self.assertEqual(delayed["push"]["action_at"].iloc[0], pd.Timestamp("2026-02-28"))
        self.assertIs(delay_pushes(histories, 0), histories)


class ProtocolTests(unittest.TestCase):
    def test_training_rows_end_before_the_target_week_and_decay(self) -> None:
        cutoffs = pd.date_range("2026-01-07 16:00", periods=6, freq="7D")
        data = pd.DataFrame({"cutoff": cutoffs, "user_id": range(6)})
        end = cutoffs[4]
        rows, weights = training_frame(data, end, Variant("t", "", half_life_weeks=1.0))
        self.assertEqual(rows["cutoff"].max(), cutoffs[3])
        np.testing.assert_allclose(weights, [0.125, 0.25, 0.5, 1.0])
        rows, _ = training_frame(data, end, Variant("t", "", last_weeks=2))
        self.assertEqual(list(rows["cutoff"]), list(cutoffs[2:4]))

    def test_weekly_score_matches_the_official_evaluator(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=(40, len(LABELS)))
        predicted = rng.integers(0, 2, size=(40, len(LABELS)))
        weeks = np.repeat([0, 1], 20)
        expected = []
        for week in (0, 1):
            users = {"UserID": np.arange(20)}
            truth = pd.DataFrame(y[weeks == week], columns=list(LABELS)).assign(**users)
            guess = pd.DataFrame(predicted[weeks == week], columns=list(LABELS)).assign(**users)
            expected.append(official_eval.scores(guess, truth)["score"])
        self.assertAlmostEqual(weekly_score(y, predicted, weeks), float(np.mean(expected)))

    def test_masked_weekly_score_drops_flagged_labels(self) -> None:
        y = np.array([[1, 0], [0, 1]] * 2)
        predicted = np.array([[1, 1], [0, 0]] * 2)
        weeks = np.array([0, 0, 1, 1])
        keep = np.array([[True, False]] * 2 + [[True, True]] * 2)
        # Week 0 keeps only the perfect first label; week 1 has F1 1 and 0.
        self.assertAlmostEqual(weekly_score(y, predicted, weeks, keep), (1.0 + (0.5 + 0.5) / 2) / 2)

    def test_joint_thresholds_improve_the_score(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=(200, len(LABELS)))
        probabilities = np.clip(y * 0.15 + rng.uniform(0, 0.2, size=y.shape), 0, 1)
        weeks = np.repeat([0, 1], 100)
        tuned = joint_thresholds(y, probabilities, weeks)
        default = weekly_score(y, probabilities >= np.array([0.3] * 5 + [0.45]), weeks)
        best = weekly_score(y, probabilities >= np.array([tuned[label] for label in LABELS]), weeks)
        self.assertGreater(best, default)

    def test_cli_accepts_variant_lists(self) -> None:
        args = parse_args(["backtest", "--variants", "base,mask"])
        self.assertEqual([variant.name for variant in args.variants], ["base", "mask"])
        self.assertEqual(len({variant.name for variant in VARIANTS}), len(VARIANTS))
        with self.assertRaises(SystemExit):
            parse_args(["backtest", "--variants", "nope"])


if __name__ == "__main__":
    unittest.main()
