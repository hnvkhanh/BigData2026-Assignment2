# Temporal activity models

This experiment supports two selectable temporal models for the six weekly
activity labels: LightGBM (the default) and S2P2, the state-space marked temporal
point process from UCI DataLab's EasyTPP repository. Both use historical
user-week examples and chronological validation. Historical user-week rows are
restricted to users with at least five recorded activity rows in the preceding
90 days, matching the test-list activity filter.

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
```

S2P2 is optional; install its pinned upstream implementation only when selecting
that model:

```bash
python -m pip install -r experiments/lightgbm_baseline/s2p2-requirements.txt
```

The upstream S2P2 README recommends CPU or CUDA and warns that MPS can produce
incorrect results. S2P2 therefore uses CPU if CUDA is unavailable; with CUDA it
reports the available devices and selects the least-utilized GPU (using memory
use if utilization is unavailable). CPU training uses one PyTorch thread to
avoid a native CPU crash observed with the upstream scan implementation.
`--device cpu` and `--device cuda` can override automatic selection.

## Run a time-based backtest

```bash
python experiments/lightgbm_baseline/run.py backtest
```

Choose S2P2 instead of LightGBM with:

```bash
python experiments/lightgbm_baseline/run.py backtest \
  --temporal-model s2p2 --device auto \
  --s2p2-epochs 3 --output-dir experiments/lightgbm_baseline/outputs/s2p2
```

The script uses weekly forecast cutoffs beginning on 2026-03-01. It trains on the
oldest 70% of weeks, selects per-label F1 thresholds on the next 15%, and reports
the untouched final 15% holdout. Metrics are written to
`experiments/lightgbm_baseline/outputs/backtest_<model>_<timestamp>/backtest_metrics.json`.
Each run gets a new timestamped output folder by default. Supply `--output-dir`
to choose a fixed folder explicitly.

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
S2P2 options are `--s2p2-epochs`, `--s2p2-batch-size`,
`--s2p2-max-sequence-length`, `--s2p2-hidden-size`, `--s2p2-state-size`,
`--s2p2-num-layers`, and `--s2p2-integral-samples`.

S2P2 represents each user's six activity types as marked events. It trains on
historical sequences containing the prior 90-day context and the labeled
seven-day window. Weekly probabilities are estimated as
`1 - exp(-integrated mark intensity)` over the seven days starting at the
forecast cutoff, then per-label thresholds are calibrated on earlier weeks.
This is a point-process-to-weekly-label approximation; compare its chronological
holdout results with LightGBM rather than assuming it will perform better.

## Generate predictions

```bash
python experiments/lightgbm_baseline/run.py predict
```

This reads `assignment2/evaluation/test_input.csv`, uses a default prediction
cutoff of `2026-11-11 16:00:00 UTC`, and writes to a new timestamped folder under
`experiments/lightgbm_baseline/outputs/`:

- `predictions.csv`
- Model files under `models/` (six LightGBM models or S2P2 weights)
- `metadata.json`, including feature names and calibrated thresholds

To generate predictions with S2P2, use `--temporal-model s2p2`, for example:

```bash
python experiments/lightgbm_baseline/run.py predict \
  --temporal-model s2p2 --device auto \
  --output-dir experiments/lightgbm_baseline/outputs/s2p2_prediction
```

For S2P2, the model file is `models/s2p2.pt`. The default LightGBM model choice
remains unchanged.

Override data, dates, input users, or output location with flags:

```bash
python experiments/lightgbm_baseline/run.py backtest --start-cutoff "2026-02-01 00:00:00"
python experiments/lightgbm_baseline/run.py predict --cutoff "2026-11-11 16:00:00" --output-dir experiments/lightgbm_baseline/outputs/run_01
python experiments/lightgbm_baseline/run.py predict --num-leaves 31 --output-dir experiments/lightgbm_baseline/outputs/prediction_candidate
```

The CSV uses the evaluator's required `UserID` and six binary label columns.
Check the `predictions.csv` in the run's output folder with:

```bash
python assignment2/evaluation/evaluate.py \
  --pred experiments/lightgbm_baseline/outputs/<run-folder>/predictions.csv \
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
  tree. Each default run gets its own timestamped folder; provide a distinct
  `--output-dir` to choose output paths explicitly. The
  upstream S2P2 implementation is available at
  https://github.com/UCIDataLab/state_space_point_process and is pinned in the
  optional requirements file.
