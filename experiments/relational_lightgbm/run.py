"""LightGBM with cross-action relational features (ownership, authorship, reviews, assignments)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.lightgbm_baseline.run import (
    DEFAULT_DB_DIR,
    DEFAULT_PREDICTION_CUTOFF,
    DEFAULT_THRESHOLDS,
    DEFAULT_START_CUTOFF,
    DEFAULT_TEST_INPUT,
    LABELS,
    USER_COLUMN,
    build_training_data,
    chronological_splits,
    evaluate_predictions,
    fit_models,
    predict_probabilities,
    row_for_cutoff,
    save_models,
    training_data_before,
)
from experiments.relational_lightgbm.features import (
    FEATURE_GROUPS,
    HYPOTHESIS_GROUPS,
    RelationalTables,
    feature_names,
    hypothesis_lift,
    load_relational_tables,
    relational_features,
)

EXPERIMENT_NAME = "relational_lightgbm"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FEATURE_VERSION = 2


def add_relational_features(
    data: pd.DataFrame,
    tables: RelationalTables,
    histories: dict[str, pd.DataFrame],
    user_ids: list[int],
    groups: list[str],
) -> pd.DataFrame:
    """Join relational features onto every (user_id, cutoff) row of ``data``."""
    if not groups:
        return data
    parts = []
    cutoffs = sorted(pd.Timestamp(value) for value in data["cutoff"].unique())
    for index, cutoff in enumerate(cutoffs, start=1):
        frame = relational_features(cutoff, tables, histories, user_ids, groups)
        parts.append(frame.assign(cutoff=cutoff))
        print(f"Built relational features {index}/{len(cutoffs)}: {cutoff}")
    relational = pd.concat(parts, ignore_index=True)
    merged = data.merge(relational, on=["user_id", "cutoff"], how="left", validate="one_to_one")
    if len(merged) != len(data):
        raise ValueError("Relational feature join changed the number of examples.")
    return merged


def feature_columns_for(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column not in {"user_id", "cutoff", *LABELS}]


def feature_importance(models: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame({"feature": feature_columns})
    for label, model in models.items():
        gain = (
            model.booster_.feature_importance(importance_type="gain")
            if model is not None
            else np.zeros(len(feature_columns))
        )
        table[label] = gain / gain.sum() if gain.sum() else gain
    table["relational"] = table["feature"].isin(feature_names(list(FEATURE_GROUPS)))
    return table.sort_values("feature").reset_index(drop=True)


def create_run_output_dir(command: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = DEFAULT_OUTPUT_DIR / f"{command}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_examples(
    db_dir: Path, start_cutoff: pd.Timestamp, groups: list[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[int], RelationalTables]:
    data, histories, user_ids = build_training_data(db_dir, start_cutoff)
    tables = load_relational_tables(db_dir)
    data = add_relational_features(data, tables, histories, user_ids, groups)
    return data, histories, user_ids, tables


def run_backtest(
    db_dir: Path,
    output_dir: Path,
    start_cutoff: pd.Timestamp,
    model_params: dict[str, int | float],
    groups: list[str],
) -> dict[str, Any]:
    data, _, _, _ = build_examples(db_dir, start_cutoff, groups)
    cutoffs = [pd.Timestamp(value) for value in sorted(data["cutoff"].unique())]
    train_end, calibration_end = chronological_splits(cutoffs)
    feature_columns = feature_columns_for(data)

    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[(data["cutoff"] >= train_end) & (data["cutoff"] < calibration_end)]
    holdout = data.loc[data["cutoff"] >= calibration_end]
    for name, part in (("train", train), ("calibration", calibration), ("holdout", holdout)):
        if part.empty:
            raise ValueError(f"The chronological {name} partition is empty.")
    print(
        f"\nChronological split: train < {train_end.date()}, "
        f"calibrate < {calibration_end.date()}, holdout >= {calibration_end.date()}"
    )

    models = fit_models(train, feature_columns, model_params)
    thresholds = DEFAULT_THRESHOLDS.copy()
    holdout_probabilities = predict_probabilities(models, holdout, feature_columns)
    results = evaluate_predictions(holdout, holdout_probabilities, thresholds)
    results.update(
        {
            "experiment": EXPERIMENT_NAME,
            "temporal_model": EXPERIMENT_NAME,
            "feature_version": FEATURE_VERSION,
            "relational_groups": groups,
            "relational_features": feature_names(groups),
            "threshold_mode": "fixed_default",
            "model_params": model_params,
            "seed": 42,
            "split": {
                "train_weeks": int(train["cutoff"].nunique()),
                "calibration_weeks": int(calibration["cutoff"].nunique()),
                "holdout_weeks": int(holdout["cutoff"].nunique()),
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "holdout_rows": len(holdout),
                "calibration_start": str(train_end),
                "holdout_start": str(calibration_end),
            },
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Hypothesis lift is measured on training weeks only, so it never peeks at the holdout.
    lift = hypothesis_lift(train)
    lift.to_csv(output_dir / "hypothesis_lift.csv", index=False)
    results["hypothesis_lift_train"] = lift.to_dict(orient="records")
    feature_importance(models, feature_columns).to_csv(output_dir / "feature_importance.csv", index=False)

    holdout_output = holdout[["user_id", "cutoff", *feature_columns]].copy()
    for label in LABELS:
        holdout_output[f"actual_{label}"] = holdout[label].to_numpy(dtype=np.int8)
        holdout_output[f"{EXPERIMENT_NAME}_{label}"] = holdout_probabilities[label].astype(np.float32)
    holdout_output.to_csv(output_dir / "holdout_probabilities.csv", index=False)
    save_models(models, thresholds, output_dir, {"feature_columns": feature_columns, **results})
    (output_dir / "backtest_metrics.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"\nHOLDOUT score {results['score']:.4f} (macro {results['macro_f1']:.4f}, micro {results['micro_f1']:.4f})")
    for label in LABELS:
        print(f"  {label:<18} F1 {results['per_action_f1'][label]:.4f}")
    print("\nHypothesis lift on training weeks:")
    print(lift.to_string(index=False))
    print(f"\nSaved metrics to {output_dir / 'backtest_metrics.json'}")
    return results


def run_prediction(
    db_dir: Path,
    test_input: Path,
    output_dir: Path,
    cutoff: pd.Timestamp,
    start_cutoff: pd.Timestamp,
    model_params: dict[str, int | float],
    groups: list[str],
) -> None:
    data, histories, cohort_user_ids, tables = build_examples(db_dir, start_cutoff, groups)
    data = training_data_before(data, cutoff)
    cutoffs = [pd.Timestamp(value) for value in sorted(data["cutoff"].unique())]
    train_end, _ = chronological_splits(cutoffs, holdout=False)
    feature_columns = feature_columns_for(data)
    train = data.loc[data["cutoff"] < train_end]
    calibration = data.loc[data["cutoff"] >= train_end]

    models = fit_models(train, feature_columns, model_params)
    thresholds = DEFAULT_THRESHOLDS.copy()

    target_users = pd.read_csv(test_input, dtype={USER_COLUMN: "int64"})
    if target_users[USER_COLUMN].duplicated().any():
        raise ValueError(f"{test_input} contains duplicate UserIDs.")
    users = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"])
    lookup = users.loc[users["is_cohort"]].set_index("github_user_id")["user_id"]
    mapped = target_users[USER_COLUMN].map(lookup)
    if mapped.isna().any():
        raise ValueError(f"{int(mapped.isna().sum())} test UserID(s) are not in this release's cohort.")
    internal_ids = mapped.astype("int64").tolist()

    inference = row_for_cutoff(cutoff, histories, cohort_user_ids)
    if groups:
        relational = relational_features(cutoff, tables, histories, cohort_user_ids, groups)
        inference = inference.merge(relational, on="user_id", validate="one_to_one")
    inference = inference.set_index("user_id").loc[internal_ids].reset_index()
    probabilities = predict_probabilities(models, inference, feature_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({USER_COLUMN: target_users[USER_COLUMN]})
    probability_output = predictions.copy()
    for label in LABELS:
        predictions[label] = (probabilities[label] >= thresholds[label]).astype("int8")
        probability_output[label] = probabilities[label]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    probability_output.to_csv(output_dir / "probabilities.csv", index=False)
    save_models(
        models,
        thresholds,
        output_dir,
        {
            "experiment": EXPERIMENT_NAME,
            "feature_version": FEATURE_VERSION,
            "prediction_cutoff_utc": str(cutoff),
            "training_start_cutoff_utc": str(start_cutoff),
            "training_weeks": int(train["cutoff"].nunique()),
            "calibration_weeks": int(calibration["cutoff"].nunique()),
            "threshold_mode": "fixed_default",
            "relational_groups": groups,
            "feature_columns": feature_columns,
            "model_params": model_params,
            "labels": list(LABELS),
        },
    )
    print(f"Saved predictions for {len(predictions):,} users to {output_dir / 'predictions.csv'}")


def parse_groups(value: str) -> list[str]:
    keyword = value.strip().lower()
    if keyword == "none":
        return []
    if keyword == "all":
        return list(FEATURE_GROUPS)
    if keyword == "hypotheses":
        return list(HYPOTHESIS_GROUPS)
    groups = [part.strip() for part in value.split(",") if part.strip()]
    try:
        feature_names(groups)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{error}. Choose from: {', '.join(FEATURE_GROUPS)}, 'all', 'hypotheses' or 'none'."
        ) from error
    return groups


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backtest", "predict"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
        sub.add_argument("--output-dir", type=Path, default=None,
                         help="output folder (default: a new timestamped folder under outputs/)")
        sub.add_argument("--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF))
        sub.add_argument(
            "--groups",
            type=parse_groups,
            default=list(FEATURE_GROUPS),
            help=f"comma-separated relational feature groups ({', '.join(FEATURE_GROUPS)}) or 'none' "
                 "for the plain LightGBM baseline features; 'hypotheses' = the first six groups (default: all)",
        )
        sub.add_argument("--n-estimators", type=int, default=300)
        sub.add_argument("--learning-rate", type=float, default=0.03)
        sub.add_argument("--num-leaves", type=int, default=15)
        sub.add_argument("--min-child-samples", type=int, default=40)
        sub.add_argument("--subsample", type=float, default=0.85)
        sub.add_argument("--colsample-bytree", type=float, default=0.85)
        sub.add_argument("--reg-lambda", type=float, default=2.0)
    predict = subparsers.choices["predict"]
    predict.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    predict.add_argument("--cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF))
    args = parser.parse_args(argv)
    if args.n_estimators < 1 or args.num_leaves < 2 or args.learning_rate <= 0:
        parser.error("--n-estimators must be >= 1, --num-leaves >= 2 and --learning-rate > 0.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or create_run_output_dir(args.command)
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
        run_backtest(args.db_dir, output_dir, args.start_cutoff, model_params, args.groups)
    else:
        run_prediction(args.db_dir, args.test_input, output_dir, args.cutoff, args.start_cutoff,
                       model_params, args.groups)


if __name__ == "__main__":
    main()
