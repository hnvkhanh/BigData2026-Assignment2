"""LightGBM variants that handle drifting activity levels and ingestion gaps.

Examples share the LightGBM baseline's definition (Wednesday 16:00 UTC cutoffs,
5-actions-in-90-days eligibility, 7-day labels) and add four feature sets:
relational features, longer-history features, cohort-regime features and push
coverage features. Each named variant changes one thing (features, training
window, recency weights, anomaly masking, refitting, seeds, thresholds or a
simulated push ingestion lag) and is scored twice:

* validation: fit on the training weeks, score the calibration weeks;
* holdout: score the last 15% of weeks with the variant's fitting scheme.

Variants are chosen on validation; holdout is reported for confirmation, both
over all holdout weeks (official score) and as a "masked" score that drops,
week by week, the labels whose target window touches an ingestion anomaly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.drift_lightgbm.data_quality import (
    anomaly_days,
    daily_counts,
    label_window_flags,
    push_coverage_features,
)
from experiments.drift_lightgbm.features import history_features, regime_features
from experiments.lightgbm_baseline.run import (
    DEFAULT_DB_DIR,
    DEFAULT_PREDICTION_CUTOFF,
    DEFAULT_START_CUTOFF,
    DEFAULT_TEST_INPUT,
    DEFAULT_THRESHOLDS,
    LABELS,
    USER_COLUMN,
    chronological_splits,
    eligible_users_for_cutoff,
    evaluate_predictions,
    labels_for_horizon,
    load_histories,
    row_for_cutoff,
)
from experiments.relational_lightgbm.features import (
    FEATURE_GROUPS,
    RelationalTables,
    load_relational_tables,
    relational_features,
)

EXPERIMENT_NAME = "drift_lightgbm"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"
CACHE_DIR = PACKAGE_DIR / "cache"
FEATURE_VERSION = 2
FEATURE_SETS = ("base", "relational", "history", "regime", "coverage")
FLAG_COLUMNS = tuple(f"flag_{label}" for label in LABELS)
MODEL_PARAMS: dict[str, int | float] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
}
# Cutoffs whose target weeks fall in the September surge (5-10x the usual
# number of users with non-push activity); see README.
SURGE_CUTOFFS = (pd.Timestamp("2026-09-02 16:00:00"), pd.Timestamp("2026-09-09 16:00:00"))
THRESHOLD_GRID = np.round(np.arange(0.05, 0.801, 0.025), 3)
WEEK = pd.Timedelta(days=7)


# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
@dataclass
class Sources:
    histories: dict[str, pd.DataFrame]
    tables: RelationalTables
    user_ids: list[int]
    outage: pd.DataFrame
    burst: pd.DataFrame


def load_sources(db_dir: Path) -> Sources:
    users = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    user_ids = users.loc[users["is_cohort"], "user_id"].astype(int).tolist()
    histories = load_histories(db_dir, user_ids)
    outage, burst = anomaly_days(daily_counts(histories))
    return Sources(histories, load_relational_tables(db_dir), user_ids, outage, burst)


def delay_pushes(histories: dict[str, pd.DataFrame], lag_days: int) -> dict[str, pd.DataFrame]:
    """Make each push usable only ``lag_days`` after it happened (an ingestion lag)."""
    if not lag_days:
        return histories
    delayed = dict(histories)
    push = histories["push"]
    delayed["push"] = push.assign(available_at=push["action_at"] + pd.Timedelta(days=lag_days))
    return delayed


def feature_rows(
    cutoff: pd.Timestamp,
    sources: Sources,
    eligible: np.ndarray,
    push_lag_days: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """All feature sets for the eligible users at one cutoff."""
    histories = delay_pushes(sources.histories, push_lag_days)
    tables = sources.tables
    if push_lag_days:
        visible = tables.pushes["at"] < cutoff - pd.Timedelta(days=push_lag_days)
        tables = replace(tables, pushes=tables.pushes.loc[visible])

    base = row_for_cutoff(cutoff, histories, sources.user_ids).drop(columns="cutoff")
    relational = relational_features(cutoff, tables, histories, sources.user_ids, list(FEATURE_GROUPS))
    history = history_features(cutoff, histories, sources.user_ids)
    eligible_ids = [uid for uid, keep in zip(sources.user_ids, eligible) if keep]
    regime = regime_features(cutoff, histories, eligible_ids)
    coverage = push_coverage_features(cutoff, sources.outage, push_lag_days)

    frame = base.merge(relational, on="user_id", validate="one_to_one")
    frame = frame.merge(history, on="user_id", validate="one_to_one").loc[eligible]
    frame = frame.assign(**regime, **coverage).reset_index(drop=True)
    columns = {
        "base": [c for c in base.columns if c != "user_id"],
        "relational": [c for c in relational.columns if c != "user_id"],
        "history": [c for c in history.columns if c != "user_id"],
        "regime": list(regime),
        "coverage": list(coverage),
    }
    return frame, columns


def coverage_end(histories: dict[str, pd.DataFrame]) -> pd.Timestamp:
    return min(histories[label]["action_at"].max() for label in LABELS)


def _cache_key(db_dir: Path, start_cutoff: pd.Timestamp, push_lag_days: int) -> str:
    tables = sorted(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in db_dir.glob("*.parquet")
    )
    payload = json.dumps([FEATURE_VERSION, str(start_cutoff), push_lag_days, tables])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_examples(
    db_dir: Path,
    start_cutoff: pd.Timestamp,
    push_lag_days: int,
    sources: Sources | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Labelled weekly examples (cached under ``cache/``) and the columns of each feature set."""
    key = _cache_key(db_dir, start_cutoff, push_lag_days)
    data_path = CACHE_DIR / f"examples_lag{push_lag_days}_{key}.parquet"
    columns_path = data_path.with_suffix(".json")
    if data_path.exists() and columns_path.exists():
        print(f"Loaded cached examples from {data_path}")
        return pd.read_parquet(data_path), json.loads(columns_path.read_text(encoding="utf-8"))

    sources = sources or load_sources(db_dir)
    last_cutoff = coverage_end(sources.histories) - WEEK
    cutoffs = list(pd.date_range(start=start_cutoff, end=last_cutoff, freq="7D"))
    flags = label_window_flags(cutoffs, sources.outage, sources.burst)
    parts = []
    columns: dict[str, list[str]] = {}
    for index, cutoff in enumerate(cutoffs, start=1):
        # Eligibility and labels use the true histories; only features see the lag.
        eligible = eligible_users_for_cutoff(cutoff, sources.histories, sources.user_ids)
        frame, columns = feature_rows(cutoff, sources, eligible, push_lag_days)
        labels = labels_for_horizon(cutoff, sources.histories, sources.user_ids)
        frame = frame.merge(labels, on="user_id", validate="one_to_one").assign(cutoff=cutoff)
        parts.append(frame)
        print(f"Built week {index}/{len(cutoffs)}: {cutoff} ({len(frame):,} users)")
    data = pd.concat(parts, ignore_index=True).merge(flags, on="cutoff", validate="many_to_one")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(data_path, index=False)
    columns_path.write_text(json.dumps(columns, indent=2) + "\n", encoding="utf-8")
    return data, columns


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    feature_sets: tuple[str, ...] = ("base", "relational")
    fit: str = "fixed"  # fixed: train weeks only; refit: + calibration weeks; rolling: weekly refit
    half_life_weeks: float | None = None
    last_weeks: int | None = None
    mask_anomalies: bool = False
    drop_surge: bool = False
    seeds: int = 1
    thresholds: str = "fixed"  # fixed defaults or "joint" (chosen on validation weeks)
    train_push_lag: int = 0
    serve_push_lag: int = 0


ALL_FEATURES = FEATURE_SETS
_DATA = dict(feature_sets=ALL_FEATURES, mask_anomalies=True)
_FINAL = dict(**_DATA, fit="rolling", half_life_weeks=8.0)
_REL = dict(feature_sets=("base", "relational"), mask_anomalies=True)
VARIANTS: tuple[Variant, ...] = (
    # Feature sets (fit on training weeks only, fixed thresholds).
    Variant("base", "baseline LightGBM features", feature_sets=("base",)),
    Variant("relational", "+ relational features (relational_lightgbm)"),
    Variant("history", "+ longer-history features", feature_sets=("base", "relational", "history")),
    Variant("regime", "+ cohort-regime features",
            feature_sets=("base", "relational", "history", "regime")),
    Variant("all_features", "+ push coverage features", feature_sets=ALL_FEATURES),
    # Data quality and drift.
    Variant("mask", "all features, anomalous label weeks masked", **_DATA),
    Variant("drop_surge", "mask + drop the September surge weeks", **_DATA, drop_surge=True),
    Variant("half_life_4", "mask + recency weights, half-life 4 weeks", **_DATA, half_life_weeks=4.0),
    Variant("half_life_8", "mask + recency weights, half-life 8 weeks", **_DATA, half_life_weeks=8.0),
    Variant("half_life_16", "mask + recency weights, half-life 16 weeks", **_DATA, half_life_weeks=16.0),
    Variant("last_8_weeks", "mask + train on the last 8 weeks only", **_DATA, last_weeks=8),
    Variant("last_12_weeks", "mask + train on the last 12 weeks only", **_DATA, last_weeks=12),
    # Fitting scheme (holdout only; validation is identical to the fixed fit).
    Variant("refit", "mask + refit on train + calibration weeks", **_DATA, fit="refit"),
    Variant("rolling", "mask + refit before every holdout week", **_DATA, fit="rolling"),
    Variant("rolling_half_life_8", "rolling + half-life 8 weeks", **_FINAL),
    # Ensembling and thresholds.
    Variant("seeds_5", "rolling + half-life 8 + 5-seed bagging", **_FINAL, seeds=5),
    Variant("joint_thresholds", "rolling + half-life 8 + thresholds tuned on validation",
            **_FINAL, thresholds="joint"),
    # Push ingestion lag at prediction time (6 days, as in the 2026-11-01 release).
    Variant("lag6_naive", "rolling + half-life 8, trained without lag, served with a 6-day push lag",
            **_FINAL, serve_push_lag=6),
    Variant("lag6_matched", "rolling + half-life 8, trained and served with a 6-day push lag",
            **_FINAL, train_push_lag=6, serve_push_lag=6),
    # Round 2: history/regime/coverage features lowered validation mAP, so the drift
    # handling is repeated on the baseline + relational features only.
    Variant("rel_mask", "relational features, anomalous label weeks masked", **_REL),
    Variant("rel_half_life_4", "rel_mask + half-life 4 weeks", **_REL, half_life_weeks=4.0),
    Variant("rel_half_life_8", "rel_mask + half-life 8 weeks", **_REL, half_life_weeks=8.0),
    Variant("rel_refit", "rel_mask + refit on train + calibration weeks", **_REL, fit="refit"),
    Variant("rel_refit_drop_surge", "rel_refit without the September surge weeks",
            **_REL, fit="refit", drop_surge=True),
    Variant("rel_refit_half_life_4", "rel_refit + half-life 4 weeks",
            **_REL, fit="refit", half_life_weeks=4.0),
    Variant("rel_refit_half_life_8", "rel_refit + half-life 8 weeks",
            **_REL, fit="refit", half_life_weeks=8.0),
    Variant("rel_rolling_half_life_8", "rel_mask + weekly refit + half-life 8 weeks",
            **_REL, fit="rolling", half_life_weeks=8.0),
    Variant("rel_final", "rel_rolling_half_life_8 + 5-seed bagging",
            **_REL, fit="rolling", half_life_weeks=8.0, seeds=5),
    Variant("rel_final_lag6", "rel_final trained and served with a 6-day push lag",
            **_REL, fit="rolling", half_life_weeks=8.0, seeds=5, train_push_lag=6,
            serve_push_lag=6),
)
VARIANTS_BY_NAME = {variant.name: variant for variant in VARIANTS}


def feature_columns_for(variant: Variant, columns: dict[str, list[str]]) -> list[str]:
    return [column for name in variant.feature_sets for column in columns[name]]


# --------------------------------------------------------------------------
# Training and scoring
# --------------------------------------------------------------------------
def training_frame(
    data: pd.DataFrame, end: pd.Timestamp, variant: Variant
) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows whose label window ends by ``end`` and their recency weights."""
    rows = data.loc[data["cutoff"] + WEEK <= end]
    if variant.last_weeks:
        rows = rows.loc[rows["cutoff"] > end - WEEK * (variant.last_weeks + 1)]
    if variant.drop_surge:
        rows = rows.loc[~rows["cutoff"].isin(SURGE_CUTOFFS)]
    if rows.empty:
        raise ValueError(f"No training rows end before {end}.")
    weights = np.ones(len(rows), dtype=np.float64)
    if variant.half_life_weeks:
        age_weeks = (rows["cutoff"].max() - rows["cutoff"]) / WEEK
        weights = np.power(0.5, age_weeks.to_numpy(dtype=np.float64) / variant.half_life_weeks)
    return rows, weights


def fit_predict(
    train: pd.DataFrame,
    weights: np.ndarray,
    feature_columns: list[str],
    targets: list[pd.DataFrame],
    variant: Variant,
) -> list[dict[str, np.ndarray]]:
    """Fit one classifier per label (averaged over seeds) and score each target frame."""
    outputs: list[dict[str, np.ndarray]] = [{} for _ in targets]
    for label in LABELS:
        keep = np.ones(len(train), dtype=bool)
        if variant.mask_anomalies:
            keep = train[f"flag_{label}"].to_numpy() == 0
        x, y, w = train.loc[keep, feature_columns], train.loc[keep, label], weights[keep]
        sums = [np.zeros(len(target)) for target in targets]
        for seed in range(variant.seeds):
            model = lgb.LGBMClassifier(
                objective="binary", random_state=42 + seed, n_jobs=-1, verbosity=-1, **MODEL_PARAMS
            )
            model.fit(x, y, sample_weight=w)
            for index, target in enumerate(targets):
                sums[index] += model.predict_proba(target[feature_columns])[:, 1]
        for index in range(len(targets)):
            outputs[index][label] = sums[index] / variant.seeds
    return outputs


def weekly_score(
    y: np.ndarray,
    predicted: np.ndarray,
    weeks: np.ndarray,
    label_mask: np.ndarray | None = None,
) -> float:
    """Mean over weeks of the official (macro-F1 + micro-F1) / 2.

    ``label_mask`` (rows x labels, constant within a week) keeps only the
    labels marked True in each week, for both macro- and micro-F1; weeks with
    no label left are skipped.
    """
    scores = []
    for week in np.unique(weeks):
        in_week = weeks == week
        keep = np.ones(y.shape[1], dtype=bool) if label_mask is None else label_mask[in_week][0]
        if not keep.any():
            continue
        truth, guess = y[in_week][:, keep] == 1, predicted[in_week][:, keep] == 1
        tp = (truth & guess).sum(0)
        fp = (~truth & guess).sum(0)
        fn = (truth & ~guess).sum(0)
        scores.append((_macro(tp, fp, fn) + _f1(tp.sum(), fp.sum(), fn.sum())) / 2)
    return float(np.mean(scores))


def _f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def _macro(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    return float(np.mean([_f1(*counts) for counts in zip(tp, fp, fn)]))


def joint_thresholds(
    y: np.ndarray, probabilities: np.ndarray, weeks: np.ndarray, passes: int = 2
) -> dict[str, float]:
    """Coordinate ascent on the weekly official score, starting from the defaults."""
    thresholds = np.array([DEFAULT_THRESHOLDS[label] for label in LABELS])
    for _ in range(passes):
        for index in range(len(LABELS)):
            best_value, best_score = thresholds[index], -1.0
            for candidate in THRESHOLD_GRID:
                thresholds[index] = candidate
                score = weekly_score(y, probabilities >= thresholds, weeks)
                if score > best_score + 1e-9:
                    best_value, best_score = candidate, score
            thresholds[index] = best_value
    return {label: float(value) for label, value in zip(LABELS, thresholds)}


def _matrix(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([probabilities[label] for label in LABELS])


def mean_average_precision(frame: pd.DataFrame, probabilities: dict[str, np.ndarray]) -> float:
    return float(
        np.mean([average_precision_score(frame[label], probabilities[label]) for label in LABELS])
    )


def score_frame(
    frame: pd.DataFrame, probabilities: dict[str, np.ndarray], thresholds: dict[str, float]
) -> dict[str, Any]:
    """Official per-week metrics, plus the score without labels hit by ingestion anomalies."""
    metrics = evaluate_predictions(frame.reset_index(drop=True), probabilities, thresholds)
    predicted = _matrix(probabilities) >= np.array([thresholds[label] for label in LABELS])
    metrics["masked_score"] = weekly_score(
        frame[list(LABELS)].to_numpy(),
        predicted,
        frame["cutoff"].to_numpy(),
        frame[list(FLAG_COLUMNS)].to_numpy() == 0,
    )
    metrics["mean_average_precision"] = mean_average_precision(frame, probabilities)
    return metrics


def run_variant(
    variant: Variant,
    datasets: dict[int, pd.DataFrame],
    columns: dict[str, list[str]],
    validation_start: pd.Timestamp,
    holdout_start: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.time()
    train_data = datasets[variant.train_push_lag]
    serve_data = datasets[variant.serve_push_lag]
    feature_columns = feature_columns_for(variant, columns)
    validation = serve_data.loc[
        (serve_data["cutoff"] >= validation_start) & (serve_data["cutoff"] < holdout_start)
    ].reset_index(drop=True)
    holdout = serve_data.loc[serve_data["cutoff"] >= holdout_start].reset_index(drop=True)

    train, weights = training_frame(train_data, validation_start, variant)
    targets = [validation] + ([holdout] if variant.fit == "fixed" else [])
    predictions = fit_predict(train, weights, feature_columns, targets, variant)
    validation_probabilities = predictions[0]

    if variant.fit == "fixed":
        holdout_probabilities = predictions[1]
    elif variant.fit == "refit":
        train, weights = training_frame(train_data, holdout_start, variant)
        holdout_probabilities = fit_predict(train, weights, feature_columns, [holdout], variant)[0]
    elif variant.fit == "rolling":
        holdout_probabilities = {label: np.zeros(len(holdout)) for label in LABELS}
        for cutoff in sorted(holdout["cutoff"].unique()):
            in_week = (holdout["cutoff"] == cutoff).to_numpy()
            train, weights = training_frame(train_data, pd.Timestamp(cutoff), variant)
            week = fit_predict(train, weights, feature_columns, [holdout.loc[in_week]], variant)[0]
            for label in LABELS:
                holdout_probabilities[label][in_week] = week[label]
    else:
        raise ValueError(f"Unknown fit scheme: {variant.fit}")

    if variant.thresholds == "joint":
        thresholds = joint_thresholds(
            validation[list(LABELS)].to_numpy(),
            _matrix(validation_probabilities),
            validation["cutoff"].to_numpy(),
        )
    elif variant.thresholds == "fixed":
        thresholds = DEFAULT_THRESHOLDS.copy()
    else:
        raise ValueError(f"Unknown threshold mode: {variant.thresholds}")

    validation_metrics = score_frame(validation, validation_probabilities, thresholds)
    holdout_metrics = score_frame(holdout, holdout_probabilities, thresholds)
    result = {
        **holdout_metrics,
        "validation": {
            key: validation_metrics[key]
            for key in ("score", "macro_f1", "micro_f1", "masked_score", "mean_average_precision")
        },
        "variant": asdict(variant),
        "feature_count": len(feature_columns),
        "seconds": round(time.time() - started, 1),
    }
    output = holdout[["user_id", "cutoff", *LABELS]].rename(
        columns={label: f"actual_{label}" for label in LABELS}
    )
    for label in LABELS:
        output[f"{EXPERIMENT_NAME}_{label}"] = holdout_probabilities[label].astype(np.float32)
    return result, output


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def create_run_output_dir(command: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = DEFAULT_OUTPUT_DIR / f"{command}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def load_datasets(
    db_dir: Path,
    start_cutoff: pd.Timestamp,
    variants: list[Variant],
    sources: Sources | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[str, list[str]]]:
    lags = sorted({v.train_push_lag for v in variants} | {v.serve_push_lag for v in variants})
    datasets, columns = {}, {}
    for lag in lags:
        key = _cache_key(db_dir, start_cutoff, lag)
        if sources is None and not (CACHE_DIR / f"examples_lag{lag}_{key}.parquet").exists():
            sources = load_sources(db_dir)
        datasets[lag], columns = build_examples(db_dir, start_cutoff, lag, sources)
    return datasets, columns


def write_report(output_dir: Path, results: dict[str, dict[str, Any]], split: dict[str, Any]) -> None:
    pct = lambda value: "-" if value is None else f"{100 * value:.2f}"  # noqa: E731
    lines = [
        "# drift_lightgbm backtest",
        "",
        f"Validation weeks: {split['validation_start']} .. < {split['holdout_start']} "
        f"({split['validation_weeks']} weeks). Holdout: >= {split['holdout_start']} "
        f"({split['holdout_weeks']} weeks).",
        "Scores are the official (macro-F1 + micro-F1) / 2 averaged over weeks, in %. "
        "Masked scores leave out labels whose target week touches an ingestion anomaly: "
        + "; ".join(
            f"{cutoff[:10]}: {', '.join(labels)}"
            for cutoff, labels in split["flagged_label_weeks"].items()
            if cutoff >= split["validation_start"]
        )
        + ".",
        "",
        "| Variant | Val score | Val masked | Val mAP | Holdout score | Holdout masked | Holdout mAP | "
        + " | ".join(LABELS) + " | Description |",
        "|---|---|---|---|---|---|---|" + "---|" * len(LABELS) + "---|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {pct(result['validation']['score'])} | "
            f"{pct(result['validation']['masked_score'])} | "
            f"{pct(result['validation']['mean_average_precision'])} | **{pct(result['score'])}** | "
            f"{pct(result['masked_score'])} | {pct(result['mean_average_precision'])} | "
            + " | ".join(pct(result["per_action_f1"][label]) for label in LABELS)
            + f" | {result['variant']['description']} |"
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_backtest(
    db_dir: Path, output_dir: Path, start_cutoff: pd.Timestamp, variants: list[Variant], headline: str
) -> dict[str, Any]:
    datasets, columns = load_datasets(db_dir, start_cutoff, variants)
    weekly_flags = next(iter(datasets.values())).groupby("cutoff")[list(FLAG_COLUMNS)].max()
    cutoffs = [pd.Timestamp(value) for value in weekly_flags.index]
    validation_start, holdout_start = chronological_splits(cutoffs)
    flagged = {
        str(cutoff): [label for label in LABELS if flags[f"flag_{label}"]]
        for cutoff, flags in weekly_flags.iterrows()
    }
    split = {
        "train_weeks": sum(c < validation_start for c in cutoffs),
        "validation_weeks": sum(validation_start <= c < holdout_start for c in cutoffs),
        "holdout_weeks": sum(c >= holdout_start for c in cutoffs),
        "validation_start": str(validation_start),
        "holdout_start": str(holdout_start),
        "flagged_label_weeks": {cutoff: labels for cutoff, labels in flagged.items() if labels},
    }
    print(f"Split: {json.dumps(split, indent=2)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variants").mkdir(exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        print(f"\n=== {variant.name}: {variant.description}")
        result, probabilities = run_variant(variant, datasets, columns, validation_start, holdout_start)
        results[variant.name] = result
        probabilities.to_csv(output_dir / "variants" / f"{variant.name}_holdout.csv", index=False)
        print(
            f"    validation {result['validation']['score']:.4f} | holdout {result['score']:.4f} "
            f"(masked {result['masked_score']:.4f}, "
            f"mAP {result['mean_average_precision']:.4f}) in {result['seconds']}s"
        )
        write_report(output_dir, results, split)

    headline_name = headline if headline in results else next(iter(results))
    metrics = {
        **results[headline_name],
        "experiment": EXPERIMENT_NAME,
        "temporal_model": EXPERIMENT_NAME,
        "headline_variant": headline_name,
        "feature_version": FEATURE_VERSION,
        "model_params": MODEL_PARAMS,
        "split": split,
        "branch_metrics": {name: result for name, result in results.items() if name != headline_name},
    }
    (output_dir / "backtest_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"\nSaved {output_dir / 'RESULTS.md'} and backtest_metrics.json")
    return metrics


def run_prediction(
    db_dir: Path,
    test_input: Path,
    output_dir: Path,
    cutoff: pd.Timestamp,
    start_cutoff: pd.Timestamp,
    variant: Variant,
) -> None:
    """Train on every week whose label window ends by ``cutoff`` and predict the test users."""
    sources = load_sources(db_dir)
    datasets, columns = load_datasets(
        db_dir, start_cutoff, [replace(variant, serve_push_lag=variant.train_push_lag)], sources
    )
    data = datasets[variant.train_push_lag]
    feature_columns = feature_columns_for(variant, columns)

    push_end = sources.histories["push"]["action_at"].max()
    other_end = max(sources.histories[label]["action_at"].max() for label in LABELS if label != "push")
    lag_days = (other_end - push_end) / pd.Timedelta(days=1)
    print(f"Push data ends {push_end}; other tables end {other_end} ({lag_days:.1f} days later).")
    if lag_days > 1 and variant.train_push_lag == 0:
        print("WARNING: pushes lag the other tables; consider a variant trained with a matching push lag.")

    thresholds = DEFAULT_THRESHOLDS.copy()
    if variant.thresholds == "joint":
        cutoffs = sorted(data.loc[data["cutoff"] + WEEK <= cutoff, "cutoff"].unique())
        validation_start, _ = chronological_splits([pd.Timestamp(c) for c in cutoffs])
        validation = data.loc[
            (data["cutoff"] >= validation_start) & (data["cutoff"] + WEEK <= cutoff)
        ].reset_index(drop=True)
        train, weights = training_frame(data, validation_start, variant)
        probabilities = fit_predict(train, weights, feature_columns, [validation], variant)[0]
        thresholds = joint_thresholds(
            validation[list(LABELS)].to_numpy(), _matrix(probabilities), validation["cutoff"].to_numpy()
        )

    target_users = pd.read_csv(test_input, dtype={USER_COLUMN: "int64"})
    if target_users[USER_COLUMN].duplicated().any():
        raise ValueError(f"{test_input} contains duplicate UserIDs.")
    users = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"])
    lookup = users.loc[users["is_cohort"]].set_index("github_user_id")["user_id"]
    mapped = target_users[USER_COLUMN].map(lookup)
    if mapped.isna().any():
        raise ValueError(f"{int(mapped.isna().sum())} test UserID(s) are not in this release's cohort.")
    internal_ids = mapped.astype("int64").tolist()

    # Regime features describe the eligible cohort; the test users are scored whether or not
    # they pass the 90-day filter at this cutoff.
    eligible = eligible_users_for_cutoff(cutoff, sources.histories, sources.user_ids)
    everyone = np.ones(len(sources.user_ids), dtype=bool)
    inference, _ = feature_rows(cutoff, sources, everyone, variant.train_push_lag)
    regime = regime_features(
        cutoff,
        delay_pushes(sources.histories, variant.train_push_lag),
        [uid for uid, keep in zip(sources.user_ids, eligible) if keep],
    )
    inference = inference.assign(**regime).set_index("user_id").loc[internal_ids].reset_index()

    train, weights = training_frame(data, cutoff, variant)
    probabilities = fit_predict(train, weights, feature_columns, [inference], variant)[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({USER_COLUMN: target_users[USER_COLUMN]})
    probability_output = predictions.copy()
    for label in LABELS:
        predictions[label] = (probabilities[label] >= thresholds[label]).astype("int8")
        probability_output[label] = probabilities[label]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    probability_output.to_csv(output_dir / "probabilities.csv", index=False)
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "variant": asdict(variant),
        "prediction_cutoff_utc": str(cutoff),
        "training_weeks": int(train["cutoff"].nunique()),
        "last_training_cutoff": str(train["cutoff"].max()),
        "push_data_end": str(push_end),
        "other_tables_end": str(other_end),
        "thresholds": thresholds,
        "feature_columns": feature_columns,
        "model_params": MODEL_PARAMS,
        "predicted_positive_rate": {label: float(predictions[label].mean()) for label in LABELS},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions for {len(predictions):,} users to {output_dir / 'predictions.csv'}")


def parse_variants(value: str) -> list[Variant]:
    if value.strip().lower() == "all":
        return list(VARIANTS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in VARIANTS_BY_NAME]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variant(s) {', '.join(unknown)}; choose from {', '.join(VARIANTS_BY_NAME)}"
        )
    return [VARIANTS_BY_NAME[name] for name in names]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backtest", "predict"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
        sub.add_argument("--output-dir", type=Path, default=None,
                         help="output folder (default: a new timestamped folder under outputs/)")
        sub.add_argument("--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF))
    backtest = subparsers.choices["backtest"]
    backtest.add_argument("--variants", type=parse_variants, default=list(VARIANTS),
                          help="comma-separated variant names, or 'all' (default)")
    backtest.add_argument("--headline", default="relational",
                          help="variant reported as the headline score in backtest_metrics.json")
    predict = subparsers.choices["predict"]
    predict.add_argument("--variant", choices=list(VARIANTS_BY_NAME), default="relational")
    predict.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    predict.add_argument("--cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or create_run_output_dir(args.command)
    if args.command == "backtest":
        run_backtest(args.db_dir, output_dir, args.start_cutoff, args.variants, args.headline)
    else:
        run_prediction(args.db_dir, args.test_input, output_dir, args.cutoff, args.start_cutoff,
                       VARIANTS_BY_NAME[args.variant])


if __name__ == "__main__":
    main()
