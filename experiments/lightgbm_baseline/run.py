"""Train, backtest, and predict weekly GitHub activity with temporal models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_TEST_INPUT = ROOT / "assignment2" / "evaluation" / "test_input.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

HORIZON_DAYS = 7
LOOKBACK_DAYS = (7, 30, 90, 365)
DEFAULT_START_CUTOFF = "2026-03-01 00:00:00"
DEFAULT_PREDICTION_CUTOFF = "2026-11-11 16:00:00"
USER_COLUMN = "UserID"
LABELS = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)

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
            columns.append("is_pull_request")
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

    return histories


def row_for_cutoff(
    cutoff: pd.Timestamp,
    histories: dict[str, pd.DataFrame],
    user_ids: list[int],
) -> pd.DataFrame:
    """Create one feature row per user using only rows available before cutoff."""
    features = pd.DataFrame({"user_id": user_ids})
    user_index = pd.Index(user_ids)
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
            features[f"{label}_count_{days}d"] = counts.to_numpy(dtype=np.int32)

        recent = known.loc[
            known["action_at"] >= cutoff - pd.Timedelta(days=90)
        ]
        repo_counts = (
            recent.groupby("user_id")["repo_id"]
            .nunique()
            .reindex(user_index, fill_value=0)
        )
        features[f"{label}_repos_90d"] = repo_counts.to_numpy(dtype=np.int32)

        active_days = (
            recent.assign(active_day=recent["action_at"].dt.floor("D"))
            .groupby("user_id")["active_day"]
            .nunique()
            .reindex(user_index, fill_value=0)
        )
        features[f"{label}_active_days_90d"] = active_days.to_numpy(dtype=np.int16)

        last_action = known.groupby("user_id")["action_at"].max().reindex(user_index)
        recency = (cutoff - last_action).dt.total_seconds() / 86_400
        features[f"{label}_days_since_last"] = recency.fillna(3660).clip(upper=3660).to_numpy(
            dtype=np.float32
        )

    features["cutoff"] = cutoff
    return features


def labels_for_horizon(
    cutoff: pd.Timestamp,
    histories: dict[str, pd.DataFrame],
    user_ids: list[int],
) -> pd.DataFrame:
    """Label whether each user performs each action in [cutoff, cutoff + 7d)."""
    labels = pd.DataFrame({"user_id": user_ids})
    user_index = pd.Index(user_ids)
    horizon_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)

    for label, history in histories.items():
        future = history.loc[
            (history["action_at"] >= cutoff)
            & (history["action_at"] < horizon_end)
        ]
        acted = future.groupby("user_id").size().reindex(user_index, fill_value=0)
        labels[label] = (acted.to_numpy() > 0).astype(np.int8)

    return labels


def build_training_data(
    db_dir: Path,
    start_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[int]]:
    users = pd.read_parquet(
        db_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"],
    )
    cohort = users.loc[users["is_cohort"]].copy()
    user_ids = cohort["user_id"].astype(int).tolist()
    histories = load_histories(db_dir, user_ids)

    # All six labels need a fully observed seven-day target window.
    coverage_end = min(history["action_at"].max() for history in histories.values())
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
        # The official test list is restricted to users with at least five
        # activity records in the preceding 90 days.
        eligible_count = panel[
            [f"{label}_count_90d" for label in LABELS]
        ].sum(axis=1)
        panels.append(panel.loc[eligible_count >= 5])
        print(f"Built weekly sample {index}/{len(cutoffs)}: {cutoff}")

    return pd.concat(panels, ignore_index=True), histories, user_ids


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.02, 0.98, 97)
    scores = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (y_true == 1)))
        fp = int(np.sum(predicted & (y_true == 0)))
        fn = int(np.sum(~predicted & (y_true == 1)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 1.0)
    return float(candidates[int(np.argmax(scores))])


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


def load_s2p2_adapter() -> tuple[Any, Any]:
    if __package__:
        from .s2p2 import S2P2TemporalModel, select_device
    else:
        from s2p2 import S2P2TemporalModel, select_device
    return S2P2TemporalModel, select_device


def fit_temporal_model(
    temporal_model: str,
    train: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    feature_columns: list[str],
    model_params: dict[str, int | float],
    s2p2_params: dict[str, int],
    device_name: str,
) -> tuple[Any, Any]:
    if temporal_model == "lightgbm":
        return fit_models(train, feature_columns, model_params), None
    if temporal_model != "s2p2":
        raise ValueError(f"Unknown temporal model: {temporal_model}")

    s2p2_class, select_s2p2_device = load_s2p2_adapter()
    model = s2p2_class(
        device=select_s2p2_device(device_name),
        **s2p2_params,
    )
    return model.fit(train, histories), model


def predict_temporal_probabilities(
    temporal_model: str,
    models: Any,
    s2p2_model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    histories: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    if temporal_model == "lightgbm":
        return predict_probabilities(models, frame, feature_columns)
    if temporal_model == "s2p2":
        return s2p2_model.predict(frame, histories)
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
    true_values = truth[list(LABELS)].to_numpy(dtype=np.int8)
    predicted_values = np.column_stack(
        [probabilities[label] >= thresholds[label] for label in LABELS]
    ).astype(np.int8)
    per_action = {
        label: f1_score(true_values[:, index], predicted_values[:, index])
        for index, label in enumerate(LABELS)
    }
    micro = f1_score(true_values.ravel(), predicted_values.ravel())
    macro = float(np.mean(list(per_action.values())))
    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "score": (macro + micro) / 2,
        "per_action_f1": per_action,
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
    s2p2_model: Any = None,
) -> None:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if temporal_model == "s2p2":
        s2p2_model.save(model_dir / "s2p2.pt")
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


def run_backtest(
    db_dir: Path,
    output_dir: Path,
    start_cutoff: pd.Timestamp,
    model_params: dict[str, int | float],
    temporal_model: str = "lightgbm",
    s2p2_params: dict[str, int] | None = None,
    device_name: str = "auto",
) -> None:
    data, histories, _ = build_training_data(db_dir, start_cutoff)
    unique_cutoffs = sorted(data["cutoff"].unique())
    cutoffs = [pd.Timestamp(value) for value in unique_cutoffs]
    train_end, calibration_end = chronological_splits(cutoffs)
    feature_columns = [column for column in data.columns if column not in {"user_id", "cutoff", *LABELS}]

    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[
        (data["cutoff"] >= train_end) & (data["cutoff"] < calibration_end)
    ]
    holdout = data.loc[data["cutoff"] >= calibration_end]
    print(
        f"\nChronological split: train < {train_end.date()}, "
        f"calibrate < {calibration_end.date()}, holdout >= {calibration_end.date()}"
    )

    s2p2_options = s2p2_params or {}
    models, s2p2_model = fit_temporal_model(
        temporal_model,
        train,
        histories,
        feature_columns,
        model_params,
        s2p2_options,
        device_name,
    )
    calibration_probabilities = predict_temporal_probabilities(
        temporal_model,
        models,
        s2p2_model,
        calibration,
        feature_columns,
        histories,
    )
    thresholds = {
        label: choose_threshold(
            calibration[label].to_numpy(dtype=np.int8),
            calibration_probabilities[label],
        )
        for label in LABELS
    }
    holdout_probabilities = predict_temporal_probabilities(
        temporal_model, models, s2p2_model, holdout, feature_columns, histories
    )
    results = evaluate_predictions(holdout, holdout_probabilities, thresholds)
    results["split"] = {
        "train_weeks": int(train["cutoff"].nunique()),
        "calibration_weeks": int(calibration["cutoff"].nunique()),
        "holdout_weeks": int(holdout["cutoff"].nunique()),
        "holdout_start": str(calibration_end),
    }
    results["temporal_model"] = temporal_model
    if temporal_model == "lightgbm":
        results["model_params"] = model_params
    else:
        results["s2p2_params"] = {
            **s2p2_options,
            "device": str(s2p2_model.device),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backtest_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nHOLDOUT RESULTS (thresholds selected on earlier calibration weeks)")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {output_dir / 'backtest_metrics.json'}")


def run_prediction(
    db_dir: Path,
    test_input: Path,
    output_dir: Path,
    cutoff: pd.Timestamp,
    start_cutoff: pd.Timestamp,
    model_params: dict[str, int | float],
    temporal_model: str = "lightgbm",
    s2p2_params: dict[str, int] | None = None,
    device_name: str = "auto",
) -> None:
    data, histories, cohort_user_ids = build_training_data(db_dir, start_cutoff)
    cutoffs = [pd.Timestamp(value) for value in sorted(data["cutoff"].unique())]
    train_end, _ = chronological_splits(cutoffs)
    feature_columns = [column for column in data.columns if column not in {"user_id", "cutoff", *LABELS}]

    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[data["cutoff"] >= train_end]
    s2p2_options = s2p2_params or {}
    models, s2p2_model = fit_temporal_model(
        temporal_model,
        train,
        histories,
        feature_columns,
        model_params,
        s2p2_options,
        device_name,
    )
    calibration_probabilities = predict_temporal_probabilities(
        temporal_model,
        models,
        s2p2_model,
        calibration,
        feature_columns,
        histories,
    )
    thresholds = {
        label: choose_threshold(
            calibration[label].to_numpy(dtype=np.int8),
            calibration_probabilities[label],
        )
        for label in LABELS
    }

    target_users = pd.read_csv(test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{test_input} contains duplicate UserIDs.")
    cohort_users = pd.read_parquet(
        db_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"],
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
    probabilities = predict_temporal_probabilities(
        temporal_model,
        models,
        s2p2_model,
        inference,
        feature_columns,
        histories,
    )
    output = pd.DataFrame({USER_COLUMN: target_users[USER_COLUMN]})
    for label in LABELS:
        output[label] = (probabilities[label] >= thresholds[label]).astype("int8")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)
    metadata = {
        "temporal_model": temporal_model,
        "prediction_cutoff_utc": str(cutoff),
        "training_start_cutoff_utc": str(start_cutoff),
        "training_weeks": int(train["cutoff"].nunique()),
        "calibration_weeks": int(calibration["cutoff"].nunique()),
        "feature_columns": feature_columns,
        "labels": list(LABELS),
    }
    if temporal_model == "lightgbm":
        metadata["model_params"] = model_params
    else:
        metadata["s2p2_params"] = {
            **s2p2_options,
            "device": str(s2p2_model.device),
        }
    save_models(
        models,
        thresholds,
        output_dir,
        metadata,
        temporal_model,
        s2p2_model,
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
            choices=("lightgbm", "s2p2"),
            default="lightgbm",
            help="temporal feature model; S2P2 requires its optional dependencies",
        )
        subparser.add_argument(
            "--device",
            choices=("auto", "cpu", "cuda"),
            default="auto",
            help="S2P2 device; auto selects the least-loaded CUDA GPU or CPU",
        )
        subparser.add_argument("--s2p2-epochs", type=int, default=3)
        subparser.add_argument("--s2p2-batch-size", type=int, default=128)
        subparser.add_argument("--s2p2-max-sequence-length", type=int, default=128)
        subparser.add_argument("--s2p2-hidden-size", type=int, default=32)
        subparser.add_argument("--s2p2-state-size", type=int, default=8)
        subparser.add_argument("--s2p2-num-layers", type=int, default=2)
        subparser.add_argument("--s2p2-integral-samples", type=int, default=8)
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
    s2p2_params = {
        "epochs": args.s2p2_epochs,
        "batch_size": args.s2p2_batch_size,
        "max_sequence_length": args.s2p2_max_sequence_length,
        "hidden_size": args.s2p2_hidden_size,
        "state_size": args.s2p2_state_size,
        "num_layers": args.s2p2_num_layers,
        "integral_samples": args.s2p2_integral_samples,
    }
    if args.command == "backtest":
        run_backtest(
            args.db_dir,
            output_dir,
            args.start_cutoff,
            model_params,
            args.temporal_model,
            s2p2_params,
            args.device,
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
            s2p2_params,
            args.device,
        )


if __name__ == "__main__":
    main()
