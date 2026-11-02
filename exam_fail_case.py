"""Run the LightGBM holdout and export misclassified examples for inspection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.lightgbm_baseline import run as baseline


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "lightgbm_baseline" / "outputs"


def build_failure_cases(
    holdout: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    github_user_ids: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Return one row per misclassified user-week-label, with its input features."""
    failures: list[pd.DataFrame] = []
    for label in baseline.LABELS:
        actual = holdout[label].to_numpy(dtype=np.int8)
        predicted_probability = probabilities[label]
        threshold = thresholds[label]
        predicted = (predicted_probability >= threshold).astype(np.int8)
        error_mask = actual != predicted
        if not error_mask.any():
            continue

        rows = holdout.loc[
            error_mask, ["user_id", "cutoff", *feature_columns]
        ].copy()
        rows.insert(
            1,
            "github_user_id",
            rows["user_id"].map(github_user_ids),
        )
        rows["label"] = label
        rows["actual"] = actual[error_mask]
        rows["predicted"] = predicted[error_mask]
        rows["probability"] = predicted_probability[error_mask]
        rows["threshold"] = threshold
        rows["failure_type"] = np.where(
            actual[error_mask] == 1, "false_negative", "false_positive"
        )
        failures.append(rows)

    if not failures:
        return pd.DataFrame(
            columns=[
                "user_id",
                "github_user_id",
                "cutoff",
                "label",
                "actual",
                "predicted",
                "probability",
                "threshold",
                "failure_type",
                *feature_columns,
            ]
        )
    return pd.concat(failures, ignore_index=True)


def analyze_failure_cases(failure_cases: pd.DataFrame) -> None:
    """TODO: Analyze failure patterns and explain why the model was wrong."""
    pass


def run_lightgbm(
    db_dir: Path,
    output_dir: Path,
    start_cutoff: pd.Timestamp,
    model_params: dict[str, int | float],
) -> None:
    data, _, _ = baseline.build_training_data(db_dir, start_cutoff)
    cutoffs = [pd.Timestamp(value) for value in sorted(data["cutoff"].unique())]
    train_end, calibration_end = baseline.chronological_splits(cutoffs)
    feature_columns = [
        column
        for column in data.columns
        if column not in {"user_id", "cutoff", *baseline.LABELS}
    ]

    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[
        (data["cutoff"] >= train_end) & (data["cutoff"] < calibration_end)
    ]
    holdout = data.loc[data["cutoff"] >= calibration_end]
    print(
        f"Chronological split: train < {train_end.date()}, "
        f"calibrate < {calibration_end.date()}, "
        f"holdout >= {calibration_end.date()}"
    )

    models = baseline.fit_models(train, feature_columns, model_params)
    calibration_probabilities = baseline.predict_probabilities(
        models, calibration, feature_columns
    )
    thresholds = {
        label: baseline.choose_threshold(
            calibration[label].to_numpy(dtype=np.int8),
            calibration_probabilities[label],
        )
        for label in baseline.LABELS
    }
    holdout_probabilities = baseline.predict_probabilities(
        models, holdout, feature_columns
    )
    metrics = baseline.evaluate_predictions(holdout, holdout_probabilities, thresholds)
    metrics.update(
        {
            "temporal_model": "lightgbm",
            "model_params": model_params,
            "split": {
                "train_weeks": int(train["cutoff"].nunique()),
                "calibration_weeks": int(calibration["cutoff"].nunique()),
                "holdout_weeks": int(holdout["cutoff"].nunique()),
                "holdout_start": str(calibration_end),
            },
        }
    )

    users = pd.read_parquet(
        db_dir / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"],
    )
    cohort = users.loc[users["is_cohort"]]
    github_user_ids = cohort.set_index("user_id")["github_user_id"]
    failure_cases = build_failure_cases(
        holdout,
        holdout_probabilities,
        thresholds,
        github_user_ids,
        feature_columns,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "failure_cases.csv"
    metrics_path = output_dir / "backtest_metrics.json"
    failure_cases.to_csv(failure_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved {len(failure_cases):,} failure rows to {failure_path}")
    print(f"Saved holdout metrics to {metrics_path}")
    analyze_failure_cases(failure_cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=baseline.DEFAULT_DB_DIR,
        help="directory containing the assignment Parquet tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output folder (default: a timestamped folder under LightGBM outputs)",
    )
    parser.add_argument(
        "--start-cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(baseline.DEFAULT_START_CUTOFF),
        help="first weekly prediction cutoff used to construct training samples",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR / f"failure_analysis_{timestamp}"
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
    run_lightgbm(args.db_dir, output_dir, args.start_cutoff, model_params)


if __name__ == "__main__":
    main()