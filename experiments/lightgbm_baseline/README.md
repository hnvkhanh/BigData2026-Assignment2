# LightGBM baseline

This experiment trains six independent LightGBM classifiers for the six weekly
activity labels. It builds historical user-week examples, uses trailing activity
counts/recency/repository breadth as features, and keeps validation chronological.
Historical user-week rows are restricted to users with at least five recorded
activity rows in the preceding 90 days, matching the test-list activity filter.

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run a time-based backtest

```bash
python experiments/lightgbm_baseline/run.py backtest
```

The script uses weekly forecast cutoffs beginning on 2026-03-01. It trains on the
oldest 70% of weeks, selects per-label F1 thresholds on the next 15%, and reports
the untouched final 15% holdout. Metrics are written to
`experiments/lightgbm_baseline/outputs/backtest_metrics.json`.

## Run and compare experiments

Give each experiment its own output folder so its metrics are not overwritten.
For example, compare the default model with a smaller and a larger tree:

```bash
python experiments/lightgbm_baseline/run.py backtest \
  --output-dir experiments/lightgbm_baseline/outputs/default

python experiments/lightgbm_baseline/run.py backtest \
  --num-leaves 7 --min-child-samples 80 \
  --output-dir experiments/lightgbm_baseline/outputs/smaller_trees

python experiments/lightgbm_baseline/run.py backtest \
  --num-leaves 31 --min-child-samples 30 --n-estimators 500 \
  --output-dir experiments/lightgbm_baseline/outputs/larger_trees
```

Each run writes `backtest_metrics.json`, including the parameters, per-label F1,
macro-F1, micro-F1, combined score, and selected thresholds. Compare the `score`
field across runs; it is the assignment's average of macro-F1 and micro-F1.
Change one or two settings at a time, keep the same cutoff/data/split, and use
the chronological holdout only for comparing a small number of final candidates.
Repeatedly tuning to that holdout can overfit it.

Supported LightGBM options are `--n-estimators`, `--learning-rate`,
`--num-leaves`, `--min-child-samples`, `--subsample`, `--colsample-bytree`,
and `--reg-lambda`. The baseline defaults are 300, 0.03, 15, 40, 0.85, 0.85,
and 2.0 respectively. Row subsampling is enabled with a sampling frequency of 1.

## Generate predictions

```bash
python experiments/lightgbm_baseline/run.py predict
```

This reads `assignment2/evaluation/test_input.csv`, uses a default prediction
cutoff of `2026-11-11 16:00:00 UTC`, and writes:

- `experiments/lightgbm_baseline/outputs/predictions.csv`
- One native LightGBM model per label under `outputs/models/`
- `outputs/metadata.json`, including feature names and calibrated thresholds

Override data, dates, input users, or output location with flags:

```bash
python experiments/lightgbm_baseline/run.py backtest --start-cutoff "2026-02-01 00:00:00"
python experiments/lightgbm_baseline/run.py predict --cutoff "2026-11-11 16:00:00" --output-dir experiments/lightgbm_baseline/outputs/run_01
python experiments/lightgbm_baseline/run.py predict --num-leaves 31 --output-dir experiments/lightgbm_baseline/outputs/prediction_candidate
```

The CSV uses the evaluator's required `UserID` and six binary label columns. Check
it with:

```bash
python assignment2/evaluation/evaluate.py \
  --pred experiments/lightgbm_baseline/outputs/predictions.csv \
  --input assignment2/evaluation/test_input.csv
```

## Leakage and reproducibility notes

- Each feature snapshot filters by both the action timestamp and the table's
  documented row-availability timestamp. PRs, ordinary issues, and reviews use
  `first_seen_at` for availability; comments and pushes use `created_at`.
- Labels use actual action timestamps: PR/issue `created_at`, review
  `submitted_at`, and comment/push `created_at`. Each label covers
  `[cutoff, cutoff + 7 days)`.
- PR conversations in `Issues` are excluded from `open_issue`.
- Model features deliberately exclude user IDs. Only cohort users are included
  because other users' activity is incomplete.
- The default prediction date is the assignment's target date. Run prediction
  against the latest data release available before that target window.
- Generated outputs are ignored by Git so experiments do not clutter the source
  tree. Use a distinct `--output-dir` to retain separate runs locally.
