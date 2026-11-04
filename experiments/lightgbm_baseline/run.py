"""Train, backtest, and predict weekly GitHub activity with temporal models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "assignment2" / "evaluation"))
import evaluate as official_eval  # noqa: E402

DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_TEST_INPUT = ROOT / "assignment2" / "evaluation" / "test_input.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

HORIZON_DAYS = 7
LOOKBACK_DAYS = (7, 30)
FREQUENCY_LOOKBACK_DAYS = (7,)
ELIGIBILITY_WINDOW_DAYS = 90
# Weekly cutoffs share the test window's phase (Wednesday 16:00 UTC), so every
# training label covers the same weekly span as the official 7-day target.
DEFAULT_START_CUTOFF = "2026-03-11 16:00:00"
DEFAULT_PREDICTION_CUTOFF = "2026-11-11 16:00:00"
USER_COLUMN = "UserID"
USER_FEATURE_COLUMN = "user_id_feature"
LABELS = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)
DEFAULT_THRESHOLDS = {
    label: 0.45 if label == "push" else 0.30
    for label in LABELS
}
ISSUE_STATE_REASONS = ("completed", "not_planned", "duplicate", "other")

# action_time identifies when the activity happened; available_time identifies
# when that row could first be used in a prediction snapshot.
ACTION_SPECS = {
    "open_pr": ("PullRequests", "author_id", "created_at", "first_seen_at"),
    "open_issue": ("Issues", "author_id", "created_at", "first_seen_at"),
    "issue_comment": ("IssueComments", "author_id", "created_at", "created_at"),
    "pr_review": ("PRReviews", "reviewer_id", "submitted_at", "first_seen_at"),
    "pr_review_comment": (
        "PRReviewComments",
        "author_id",
        "created_at",
        "created_at",
    ),
    "push": ("Pushes", "user_id", "created_at", "created_at"),
}


def load_histories(
    db_dir: Path, cohort_user_ids: list[int]
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    cohort_ids = set(cohort_user_ids)

    for label, (table, user_col, action_col, available_col) in ACTION_SPECS.items():
        columns = list(dict.fromkeys([user_col, "repo_id", action_col, available_col]))
        if table == "Issues":
            columns.extend(["is_pull_request", "closed_at", "state_reason"])
        elif table == "PullRequests":
            columns.extend(["merged_at", "merged_by_id", "closed_at"])
        elif table == "PRReviews":
            columns.append("dismissed_at")
        raw = pd.read_parquet(db_dir / f"{table}.parquet", columns=columns)
        mask = raw[user_col].isin(cohort_ids) & raw[action_col].notna()
        if table == "Issues":
            mask &= ~raw["is_pull_request"].fillna(False)

        history = pd.DataFrame(
            {
                "user_id": raw.loc[mask, user_col].astype("int64"),
                "repo_id": raw.loc[mask, "repo_id"].astype("int64"),
                "action_at": pd.to_datetime(raw.loc[mask, action_col]),
                "available_at": pd.to_datetime(raw.loc[mask, available_col]),
            }
        )
        history = history.dropna(subset=["action_at", "available_at"])
        histories[label] = history.reset_index(drop=True)

        if table == "PullRequests":
            histories["pr_merged_authored"] = _lifecycle_history(
                raw, "author_id", "merged_at", "first_seen_at", cohort_ids
            )
            histories["pr_merged_by"] = _lifecycle_history(
                raw, "merged_by_id", "merged_at", "first_seen_at", cohort_ids
            )
            histories["pr_closed_authored"] = _lifecycle_history(
                raw, "author_id", "closed_at", "first_seen_at", cohort_ids
            )
        elif table == "Issues":
            histories["issue_closed_authored"] = _lifecycle_history(
                raw, "author_id", "closed_at", "first_seen_at", cohort_ids
            )
            closed = raw.loc[raw["closed_at"].notna()]
            reasons = closed["state_reason"]
            known_reasons = set(ISSUE_STATE_REASONS) - {"other"}
            for reason in ISSUE_STATE_REASONS:
                if reason == "other":
                    reason_mask = reasons.notna() & ~reasons.isin(known_reasons)
                else:
                    reason_mask = reasons.eq(reason)
                reason_rows = closed.loc[reason_mask]
                histories[f"issue_closed_reason_{reason}"] = _lifecycle_history(
                    reason_rows,
                    "author_id",
                    "closed_at",
                    "first_seen_at",
                    cohort_ids,
                )
        elif table == "PRReviews":
            histories["pr_review_dismissed"] = _lifecycle_history(
                raw, "reviewer_id", "dismissed_at", "first_seen_at", cohort_ids
            )

    return histories


def _lifecycle_history(
    raw: pd.DataFrame,
    actor_column: str,
    event_column: str,
    row_available_column: str,
    cohort_user_ids: set[int],
) -> pd.DataFrame:
    """Represent a lifecycle transition only after both it and its row are known."""
    mask = (
        raw[actor_column].isin(cohort_user_ids)
        & raw[event_column].notna()
        & raw[row_available_column].notna()
        & raw["repo_id"].notna()
    )
    rows = raw.loc[mask]
    return pd.DataFrame(
        {
            "user_id": rows[actor_column].astype("int64"),
            "repo_id": rows["repo_id"].astype("int64"),
            "action_at": pd.to_datetime(rows[event_column]),
            "available_at": pd.to_datetime(rows[row_available_column]),
        }
    ).reset_index(drop=True)


def _median_daily_count(
    daily_counts: pd.Series,
    days: int,
    user_ids: list[int],
) -> np.ndarray:
    medians = np.zeros(len(user_ids), dtype=np.float32)
    user_positions = {user_id: index for index, user_id in enumerate(user_ids)}

    for user_id, counts in daily_counts.groupby(level="user_id"):
        sorted_counts = np.sort(counts.to_numpy(dtype=np.int32))
        zero_days = days - len(sorted_counts)
        lower_index = (days - 1) // 2
        upper_index = days // 2
        lower = (
            0
            if lower_index < zero_days
            else sorted_counts[lower_index - zero_days]
        )
        upper = (
            0
            if upper_index < zero_days
            else sorted_counts[upper_index - zero_days]
        )
        medians[user_positions[int(user_id)]] = (lower + upper) / 2

    return medians


def row_for_cutoff(
    cutoff: pd.Timestamp,
    histories: dict[str, pd.DataFrame],
    user_ids: list[int],
) -> pd.DataFrame:
    """Create one feature row per user using only rows available before cutoff."""
    feature_values: dict[str, object] = {"user_id": user_ids}
    user_index = pd.Index(user_ids)
    action_counts: dict[str, dict[int, np.ndarray]] = {
        label: {} for label in LABELS
    }
    daily_action_counts: dict[str, dict[int, pd.Series]] = {
        label: {} for label in LABELS
    }
    for label, history in histories.items():
        known = history.loc[
            (history["available_at"] < cutoff)
            & (history["action_at"] < cutoff)
        ]

        for days in LOOKBACK_DAYS:
            window = known.loc[
                known["action_at"] >= cutoff - pd.Timedelta(days=days)
            ]
            counts = window.groupby("user_id").size().reindex(user_index, fill_value=0)
            count_values = counts.to_numpy(dtype=np.int32)
            if days != 30:
                feature_values[f"{label}_count_{days}d"] = count_values
            if label in action_counts:
                action_counts[label][days] = count_values
                day_bins = np.floor(
                    np.nextafter(
                        (cutoff - window["action_at"]).dt.total_seconds().to_numpy()
                        / 86_400,
                        -np.inf,
                    )
                ).astype(np.int16)
                daily_action_counts[label][days] = (
                    window.assign(_day_bin=day_bins)
                    .groupby(["user_id", "_day_bin"])
                    .size()
                )

        last_action = known.groupby("user_id")["action_at"].max().reindex(user_index)
        recency = (cutoff - last_action).dt.total_seconds() / 86_400
        feature_values[f"{label}_days_since_last"] = (
            recency.fillna(3660).clip(upper=3660).to_numpy(dtype=np.float32)
        )

    for days in LOOKBACK_DAYS:
        counts_by_action = {
            label: action_counts[label].get(
                days, np.zeros(len(user_ids), dtype=np.int32)
            )
            for label in LABELS
        }
        total_actions = np.sum(
            list(counts_by_action.values()), axis=0, dtype=np.int64
        )
        if days != 30:
            feature_values[f"total_actions_count_{days}d"] = total_actions
        total_daily_counts: pd.Series | None = None
        for label, counts in counts_by_action.items():
            feature_values[f"{label}_share_{days}d"] = np.divide(
                counts,
                total_actions,
                out=np.zeros(len(user_ids), dtype=np.float32),
                where=total_actions > 0,
            )
            if days in FREQUENCY_LOOKBACK_DAYS:
                feature_values[f"{label}_frequency_per_day_{days}d"] = (
                    counts.astype(np.float32) / float(days)
                )
            daily_counts = daily_action_counts[label].get(days)
            if daily_counts is not None:
                total_daily_counts = (
                    daily_counts
                    if total_daily_counts is None
                    else total_daily_counts.add(daily_counts, fill_value=0)
                )
            action_daily_counts = daily_action_counts[label].get(days)
            if action_daily_counts is None:
                action_daily_counts = pd.Series(
                    dtype=np.int64,
                    index=pd.MultiIndex.from_arrays(
                        [[], []], names=["user_id", "_day_bin"]
                    ),
                )
            feature_values[f"{label}_median_actions_per_day_{days}d"] = (
                _median_daily_count(action_daily_counts, days, user_ids)
            )

        if total_daily_counts is None:
            total_daily_counts = pd.Series(
                dtype=np.int64,
                index=pd.MultiIndex.from_arrays(
                    [[], []], names=["user_id", "_day_bin"]
                ),
            )
        feature_values[f"total_median_actions_per_day_{days}d"] = (
            _median_daily_count(total_daily_counts, days, user_ids)
        )

    feature_values["cutoff"] = cutoff
    return pd.DataFrame(feature_values)


def labels_for_horizon(
    cutoff: pd.Timestamp,
    histories: dict[str, pd.DataFrame],
    user_ids: list[int],
) -> pd.DataFrame:
    """Label whether each user performs each action in [cutoff, cutoff + 7d)."""
    labels = pd.DataFrame({"user_id": user_ids})
    user_index = pd.Index(user_ids)
    horizon_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)

    for label in LABELS:
        history = histories[label]
        future = history.loc[
            (history["action_at"] >= cutoff)
            & (history["action_at"] < horizon_end)
        ]
        acted = future.groupby("user_id").size().reindex(user_index, fill_value=0)
        labels[label] = (acted.to_numpy() > 0).astype(np.int8)

    return labels


def eligible_users_for_cutoff(
    cutoff: pd.Timestamp,
    histories: dict[str, pd.DataFrame],
    user_ids: list[int],
) -> np.ndarray:
    """Keep the assignment's five-actions-in-90-days sample filter."""
    user_index = pd.Index(user_ids)
    eligible_counts = np.zeros(len(user_ids), dtype=np.int32)
    start = cutoff - pd.Timedelta(days=ELIGIBILITY_WINDOW_DAYS)
    for label in LABELS:
        history = histories[label]
        recent = history.loc[
            (history["available_at"] < cutoff)
            & (history["action_at"] < cutoff)
            & (history["action_at"] >= start)
        ]
        counts = recent.groupby("user_id").size().reindex(user_index, fill_value=0)
        eligible_counts += counts.to_numpy(dtype=np.int32)
    return eligible_counts >= 5


def build_training_data(
    db_dir: Path,
    start_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[int]]:
    users = pd.read_parquet(
        db_dir / "Users.parquet",
        columns=[
            "user_id",
            "github_user_id",
            "is_cohort",
        ],
    )
    cohort = users.loc[users["is_cohort"]].copy()
    user_ids = cohort["user_id"].astype(int).tolist()
    histories = load_histories(db_dir, user_ids)

    # All six labels need a fully observed seven-day target window.
    coverage_end = min(histories[label]["action_at"].max() for label in LABELS)
    last_cutoff = coverage_end - pd.Timedelta(days=HORIZON_DAYS)
    cutoffs = pd.date_range(start=start_cutoff, end=last_cutoff, freq="7D")
    if len(cutoffs) < 10:
        raise ValueError(
            f"Only {len(cutoffs)} complete weekly samples are available from "
            f"{start_cutoff} through {last_cutoff}. Check the data snapshot or "
            "choose an earlier --start-cutoff."
        )

    panels: list[pd.DataFrame] = []
    for index, cutoff in enumerate(cutoffs, start=1):
        x = row_for_cutoff(cutoff, histories, user_ids)
        y = labels_for_horizon(cutoff, histories, user_ids)
        panel = x.merge(y, on="user_id", validate="one_to_one")
        eligible = eligible_users_for_cutoff(cutoff, histories, user_ids)
        panels.append(panel.loc[eligible])
        print(f"Built weekly sample {index}/{len(cutoffs)}: {cutoff}")

    return pd.concat(panels, ignore_index=True), histories, user_ids


def training_data_before(
    data: pd.DataFrame, prediction_cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Keep only examples whose full seven-day label window precedes prediction."""
    label_window_ends = data["cutoff"] + pd.Timedelta(days=HORIZON_DAYS)
    return data.loc[label_window_ends <= prediction_cutoff].copy()


def frequency_feature_columns(columns: list[str]) -> list[str]:
    """Return only the per-action frequencies, excluding identifiers and labels."""
    expected = [
        f"{label}_frequency_per_day_{days}d"
        for days in FREQUENCY_LOOKBACK_DAYS
        for label in LABELS
    ]
    missing = [column for column in expected if column not in columns]
    if missing:
        raise ValueError(f"Missing frequency features: {', '.join(missing)}")
    return expected


def add_user_feature(frame: pd.DataFrame, categories: np.ndarray) -> pd.DataFrame:
    """Add the user ID as a categorical feature; users outside ``categories`` become NaN."""
    return frame.assign(
        **{
            USER_FEATURE_COLUMN: pd.Categorical(
                frame["user_id"], categories=categories
            )
        }
    )


def frequency_preprocessor() -> Any:
    return StandardScaler()


def fit_models(
    train: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, int | float],
) -> dict[str, lgb.LGBMClassifier | None]:
    models: dict[str, lgb.LGBMClassifier | None] = {}
    x_train = train[feature_columns]

    for label in LABELS:
        y_train = train[label]
        if y_train.nunique() < 2:
            models[label] = None
            print(f"{label}: only one class in training data; using constant predictions.")
            continue

        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **model_params,
        )
        model.fit(x_train, y_train)
        models[label] = model
        print(f"{label}: trained on {int(y_train.sum()):,} positive rows.")

    return models


def fit_frequency_nn_models(
    train: pd.DataFrame,
    feature_columns: list[str],
    output_mode: str = "per_label",
) -> Any:
    """Fit frequency-only neural classifiers with separate or shared outputs."""
    if output_mode not in {"per_label", "multi_label"}:
        raise ValueError(f"Unknown frequency NN output mode: {output_mode}")

    x_train = train[feature_columns]
    if output_mode == "multi_label":
        model = make_pipeline(
            frequency_preprocessor(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=min(256, len(train)),
                learning_rate_init=1e-3,
                max_iter=100,
                random_state=42,
                early_stopping=False,
            ),
        )
        model.fit(x_train, train[list(LABELS)].to_numpy(dtype=np.int8))
        print("Trained one frequency NN with a six-label output vector.")
        return model

    models: dict[str, Any] = {}

    for label in LABELS:
        y_train = train[label]
        if y_train.nunique() < 2:
            models[label] = None
            print(f"{label}: only one class in training data; using constant predictions.")
            continue

        model = make_pipeline(
            frequency_preprocessor(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=min(256, len(train)),
                learning_rate_init=1e-3,
                max_iter=100,
                random_state=42,
                early_stopping=False,
            ),
        )
        model.fit(x_train, y_train)
        models[label] = model
        print(f"{label}: trained frequency NN on {int(y_train.sum()):,} positive rows.")

    return models


def predict_frequency_nn_probabilities(
    models: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_mode: str = "per_label",
) -> dict[str, np.ndarray]:
    if output_mode == "multi_label":
        values = models.predict_proba(frame[feature_columns])
        if values.shape != (len(frame), len(LABELS)):
            raise ValueError(
                "Multi-label frequency NN returned probabilities with shape "
                f"{values.shape}; expected {(len(frame), len(LABELS))}."
            )
        return {label: values[:, index] for index, label in enumerate(LABELS)}
    if output_mode != "per_label":
        raise ValueError(f"Unknown frequency NN output mode: {output_mode}")

    probabilities = {}
    for label, model in models.items():
        if model is None:
            probabilities[label] = np.zeros(len(frame), dtype=np.float32)
        else:
            positive_index = list(model.classes_).index(1)
            probabilities[label] = model.predict_proba(frame[feature_columns])[
                :, positive_index
            ]
    return probabilities


def predict_probabilities(
    models: dict[str, lgb.LGBMClassifier | None],
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, np.ndarray]:
    probabilities = {}
    for label, model in models.items():
        if model is None:
            probabilities[label] = np.zeros(len(frame), dtype=np.float32)
        else:
            probabilities[label] = model.predict_proba(frame[feature_columns])[:, 1]
    return probabilities


def fit_temporal_model(
    temporal_model: str,
    train: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, int | float],
    nn_output_mode: str = "per_label",
) -> Any:
    if temporal_model == "lightgbm":
        return fit_models(train, feature_columns, model_params)
    if temporal_model == "frequency_nn":
        return fit_frequency_nn_models(train, feature_columns, nn_output_mode)
    raise ValueError(f"Unknown temporal model: {temporal_model}")


def predict_temporal_probabilities(
    temporal_model: str,
    models: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    nn_output_mode: str = "per_label",
) -> dict[str, np.ndarray]:
    if temporal_model == "lightgbm":
        return predict_probabilities(models, frame, feature_columns)
    if temporal_model == "frequency_nn":
        return predict_frequency_nn_probabilities(
            models, frame, feature_columns, nn_output_mode
        )
    raise ValueError(f"Unknown temporal model: {temporal_model}")


def f1_score(y_true: np.ndarray, predicted: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (predicted == 1)))
    fp = int(np.sum((y_true == 0) & (predicted == 1)))
    fn = int(np.sum((y_true == 1) & (predicted == 0)))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 1.0


def evaluate_predictions(
    truth: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Score like the official evaluator: one whole-week submission per cutoff.

    Each holdout week is a separate submission (one row per user, 7-day labels),
    scored with ``evaluate.scores``; the reported metrics are the mean over weeks.
    """
    predicted = pd.DataFrame(
        {
            label: (probabilities[label] >= thresholds[label]).astype(np.int64)
            for label in LABELS
        }
    )
    predicted.insert(0, USER_COLUMN, truth["user_id"].to_numpy())
    actual = truth[["user_id", *LABELS]].rename(columns={"user_id": USER_COLUMN})
    actual = actual.astype({label: np.int64 for label in LABELS})
    cutoffs = truth["cutoff"].to_numpy()

    weekly = []
    for cutoff in np.unique(cutoffs):
        in_week = cutoffs == cutoff
        week_truth = actual.loc[in_week].reset_index(drop=True)
        week_pred = predicted.loc[in_week].reset_index(drop=True)
        week = official_eval.scores(week_pred, week_truth)
        week["per_action_f1"] = dict(
            zip(LABELS, official_eval.f1_per_event(week_pred, week_truth))
        )
        week["cutoff"] = str(pd.Timestamp(cutoff))
        weekly.append(week)

    mean = lambda key: float(np.mean([week[key] for week in weekly]))
    return {
        "macro_f1": mean("macro_f1"),
        "micro_f1": mean("micro_f1"),
        "score": mean("score"),
        "per_action_f1": {
            label: float(np.mean([week["per_action_f1"][label] for week in weekly]))
            for label in LABELS
        },
        "per_week": weekly,
        "thresholds": thresholds,
    }


def chronological_splits(cutoffs: list[pd.Timestamp]) -> tuple[pd.Timestamp, pd.Timestamp]:
    if len(cutoffs) < 10:
        raise ValueError("At least 10 weekly cutoffs are needed for chronological splits.")
    train_end = max(1, int(len(cutoffs) * 0.70))
    calibration_end = max(train_end + 1, int(len(cutoffs) * 0.85))
    if calibration_end >= len(cutoffs):
        calibration_end = len(cutoffs) - 1
    return cutoffs[train_end], cutoffs[calibration_end]


def save_models(
    models: Any,
    thresholds: dict[str, float],
    output_dir: Path,
    metadata: dict[str, Any],
    temporal_model: str = "lightgbm",
    nn_output_mode: str = "per_label",
) -> None:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if temporal_model == "frequency_nn":
        import joblib

        if nn_output_mode == "multi_label":
            joblib.dump(models, model_dir / "frequency_nn_multilabel.joblib")
        else:
            for label, model in models.items():
                if model is not None:
                    joblib.dump(model, model_dir / f"{label}.joblib")
    else:
        for label, model in models.items():
            if model is not None:
                model.booster_.save_model(str(model_dir / f"{label}.txt"))
    metadata["thresholds"] = thresholds
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def create_run_output_dir(command: str, temporal_model: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = DEFAULT_OUTPUT_DIR / f"{command}_{temporal_model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def apply_train_fraction(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    fraction: float,
    seed: int,
    discard: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on a random ``fraction`` of rows.

    The unused rows join calibration, or are dropped when ``discard`` is set.
    """
    if fraction >= 1.0:
        return train, calibration
    keep = max(1, round(len(train) * fraction))
    chosen = np.zeros(len(train), dtype=bool)
    chosen[np.random.default_rng(seed).choice(len(train), size=keep, replace=False)] = True
    print(
        f"Using {keep} of {len(train)} training rows (--train-fraction {fraction}); "
        + ("the rest are discarded" if discard else "the rest go to calibration")
    )
    if discard:
        return train.iloc[chosen], calibration
    return train.iloc[chosen], pd.concat([calibration, train.iloc[~chosen]])


def run_backtest(
    db_dir: Path,
    output_dir: Path,
    start_cutoff: pd.Timestamp,
    model_params: dict[str, int | float],
    temporal_model: str = "lightgbm",
    nn_output_mode: str = "per_label",
    train_fraction: float = 1.0,
    seed: int = 42,
    user_embedding: bool = False,
    discard_unused_train: bool = False,
) -> None:
    data, _, _ = build_training_data(db_dir, start_cutoff)
    unique_cutoffs = sorted(data["cutoff"].unique())
    cutoffs = [pd.Timestamp(value) for value in unique_cutoffs]
    train_end, calibration_end = chronological_splits(cutoffs)
    feature_columns = [column for column in data.columns if column not in {"user_id", "cutoff", *LABELS}]
    if temporal_model == "frequency_nn":
        feature_columns = frequency_feature_columns(feature_columns)

    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[
        (data["cutoff"] >= train_end) & (data["cutoff"] < calibration_end)
    ]
    holdout = data.loc[data["cutoff"] >= calibration_end]
    train, calibration = apply_train_fraction(
        train, calibration, train_fraction, seed, discard_unused_train
    )
    if user_embedding:
        user_categories = np.unique(train["user_id"])
        train, calibration, holdout = (
            add_user_feature(frame, user_categories)
            for frame in (train, calibration, holdout)
        )
        feature_columns = [*feature_columns, USER_FEATURE_COLUMN]
    print(
        f"\nChronological split: train < {train_end.date()}, "
        f"calibrate < {calibration_end.date()}, holdout >= {calibration_end.date()}"
    )

    models = fit_temporal_model(
        temporal_model, train, feature_columns, model_params, nn_output_mode
    )
    thresholds = DEFAULT_THRESHOLDS.copy()
    holdout_probabilities = predict_temporal_probabilities(
        temporal_model, models, holdout, feature_columns, nn_output_mode
    )
    results = evaluate_predictions(holdout, holdout_probabilities, thresholds)
    results["split"] = {
        "train_weeks": int(train["cutoff"].nunique()),
        "calibration_weeks": int(calibration["cutoff"].nunique()),
        "holdout_weeks": int(holdout["cutoff"].nunique()),
        "holdout_start": str(calibration_end),
    }
    results["temporal_model"] = temporal_model
    results["train_fraction"] = train_fraction
    results["user_embedding"] = user_embedding
    results["threshold_mode"] = "fixed_default"
    if temporal_model == "lightgbm":
        results["model_params"] = model_params
    else:
        results["model_params"] = {
            "architecture": [
                len(feature_columns),
                32,
                len(LABELS) if nn_output_mode == "multi_label" else 1,
            ],
            "output_mode": nn_output_mode,
            "input_features": "per-action daily frequencies over 7 days only",
            "user_id_used_as_feature": False,
            "max_iter": 100,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    holdout_output = holdout[["user_id", "cutoff", *feature_columns]].copy()
    for label in LABELS:
        holdout_output[f"actual_{label}"] = holdout[label].to_numpy(dtype=np.int8)
        holdout_output[f"{temporal_model}_{label}"] = holdout_probabilities[
            label
        ].astype(np.float32)
    holdout_output.to_csv(output_dir / "holdout_probabilities.csv", index=False)
    (output_dir / "backtest_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nHOLDOUT RESULTS (thresholds: fixed defaults)")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {output_dir / 'backtest_metrics.json'}")
    print(f"Saved holdout probabilities to {output_dir / 'holdout_probabilities.csv'}")


def run_prediction(
    db_dir: Path,
    test_input: Path,
    output_dir: Path,
    cutoff: pd.Timestamp,
    start_cutoff: pd.Timestamp,
    model_params: dict[str, int | float],
    temporal_model: str = "lightgbm",
    nn_output_mode: str = "per_label",
    train_fraction: float = 1.0,
    seed: int = 42,
    user_embedding: bool = False,
    discard_unused_train: bool = False,
) -> None:
    data, histories, cohort_user_ids = build_training_data(db_dir, start_cutoff)
    data = training_data_before(data, cutoff)
    cutoffs = [pd.Timestamp(value) for value in sorted(data["cutoff"].unique())]
    train_end, _ = chronological_splits(cutoffs)
    feature_columns = [column for column in data.columns if column not in {"user_id", "cutoff", *LABELS}]
    if temporal_model == "frequency_nn":
        feature_columns = frequency_feature_columns(feature_columns)

    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[data["cutoff"] >= train_end]
    train, calibration = apply_train_fraction(
        train, calibration, train_fraction, seed, discard_unused_train
    )
    if user_embedding:
        user_categories = np.unique(train["user_id"])
        train, calibration = (
            add_user_feature(frame, user_categories) for frame in (train, calibration)
        )
        feature_columns = [*feature_columns, USER_FEATURE_COLUMN]
    models = fit_temporal_model(
        temporal_model, train, feature_columns, model_params, nn_output_mode
    )
    thresholds = DEFAULT_THRESHOLDS.copy()

    target_users = pd.read_csv(test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{test_input} contains duplicate UserIDs.")
    cohort_users = pd.read_parquet(
        db_dir / "Users.parquet",
        columns=[
            "user_id",
            "github_user_id",
            "is_cohort",
        ],
    )
    cohort_users = cohort_users.loc[cohort_users["is_cohort"]]
    user_lookup = cohort_users.set_index("github_user_id")["user_id"]
    mapped_ids = target_users["UserID"].map(user_lookup)
    if mapped_ids.isna().any():
        raise ValueError(
            f"{int(mapped_ids.isna().sum())} test UserID(s) are not in the cohort "
            "for this data release."
        )
    internal_ids = mapped_ids.astype("int64").tolist()

    inference = row_for_cutoff(cutoff, histories, cohort_user_ids)
    inference = inference.set_index("user_id").loc[internal_ids].reset_index()
    if user_embedding:
        inference = add_user_feature(inference, user_categories)
    probabilities = predict_temporal_probabilities(
        temporal_model, models, inference, feature_columns, nn_output_mode
    )
    output = pd.DataFrame({USER_COLUMN: target_users[USER_COLUMN]})
    for label in LABELS:
        output[label] = (probabilities[label] >= thresholds[label]).astype("int8")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)
    if temporal_model == "frequency_nn":
        probability_output = pd.DataFrame({USER_COLUMN: target_users[USER_COLUMN]})
        for label in LABELS:
            probability_output[label] = probabilities[label]
        probability_output.to_csv(output_dir / "probabilities.csv", index=False)
    metadata = {
        "temporal_model": temporal_model,
        "threshold_mode": "fixed_default",
        "prediction_cutoff_utc": str(cutoff),
        "training_start_cutoff_utc": str(start_cutoff),
        "training_weeks": int(train["cutoff"].nunique()),
        "calibration_weeks": int(calibration["cutoff"].nunique()),
        "feature_columns": feature_columns,
        "user_embedding": user_embedding,
        "labels": list(LABELS),
    }
    if temporal_model == "lightgbm":
        metadata["model_params"] = model_params
    else:
        metadata["model_params"] = {
            "architecture": [
                len(feature_columns),
                32,
                len(LABELS) if nn_output_mode == "multi_label" else 1,
            ],
            "output_mode": nn_output_mode,
            "input_features": "per-action daily frequencies over 7 days only",
            "user_id_used_as_feature": False,
            "max_iter": 100,
        }
    save_models(
        models, thresholds, output_dir, metadata, temporal_model, nn_output_mode
    )
    print(f"Saved predictions for {len(output):,} users to {prediction_path}")
    print(f"Saved model files and thresholds to {output_dir}")
    print("Thresholds:", json.dumps(thresholds, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backtest", "predict"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
        subparser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="output folder (default: a new timestamped folder under outputs/)",
        )
        subparser.add_argument(
            "--temporal-model",
            choices=("lightgbm", "frequency_nn"),
            default="lightgbm",
            help="model",
        )
        subparser.add_argument(
            "--nn-output-mode",
            choices=("per_label", "multi_label"),
            default="per_label",
            help="frequency_nn: use six models or one six-output model",
        )
        subparser.add_argument(
            "--start-cutoff",
            type=pd.Timestamp,
            default=pd.Timestamp(DEFAULT_START_CUTOFF),
            help="first weekly prediction cutoff used to construct training samples",
        )
        subparser.add_argument("--n-estimators", type=int, default=300)
        subparser.add_argument("--learning-rate", type=float, default=0.03)
        subparser.add_argument("--num-leaves", type=int, default=15)
        subparser.add_argument("--min-child-samples", type=int, default=40)
        subparser.add_argument("--subsample", type=float, default=0.85)
        subparser.add_argument("--colsample-bytree", type=float, default=0.85)
        subparser.add_argument("--reg-lambda", type=float, default=2.0)
        subparser.add_argument(
            "--train-fraction",
            type=float,
            default=1.0,
            help=(
                "fraction of training rows to train on, sampled at random "
                "(seeded by --seed); the unused remainder is added to calibration"
            ),
        )
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument(
            "--discard-unused-train",
            action="store_true",
            help="with --train-fraction below 1, drop the unused rows instead of adding them to calibration",
        )
        subparser.add_argument(
            "--user-embedding",
            action="store_true",
            help=(
                "lightgbm only: give the model the user ID as a categorical feature; "
                "users unseen in training get no user signal"
            ),
        )
    predict_parser = subparsers.choices["predict"]
    predict_parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    predict_parser.add_argument(
        "--cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF),
        help="UTC time at which to make the prediction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nn_output_mode != "per_label" and args.temporal_model != "frequency_nn":
        raise ValueError("--nn-output-mode applies only to --temporal-model frequency_nn.")
    if args.user_embedding and args.temporal_model != "lightgbm":
        raise ValueError("--user-embedding applies only to --temporal-model lightgbm.")
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1].")
    output_dir = args.output_dir or create_run_output_dir(
        args.command, args.temporal_model
    )
    model_params: dict[str, int | float] = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
    }
    if args.command == "backtest":
        run_backtest(
            args.db_dir,
            output_dir,
            args.start_cutoff,
            model_params,
            args.temporal_model,
            args.nn_output_mode,
            args.train_fraction,
            args.seed,
            args.user_embedding,
            args.discard_unused_train,
        )
    else:
        run_prediction(
            args.db_dir,
            args.test_input,
            output_dir,
            args.cutoff,
            args.start_cutoff,
            model_params,
            args.temporal_model,
            args.nn_output_mode,
            args.train_fraction,
            args.seed,
            args.user_embedding,
            args.discard_unused_train,
        )


if __name__ == "__main__":
    main()
