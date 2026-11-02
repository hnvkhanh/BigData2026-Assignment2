# Temporal activity models

This experiment supports LightGBM (the default), a frequency-only neural
network, and S2P2, the state-space marked temporal point process from UCI
DataLab's EasyTPP repository. All predict the six weekly activity labels using
historical user-week examples and chronological validation. Historical
user-week rows are restricted to users with at least five recorded activity
rows in the preceding 90 days, matching the test-list activity filter. That
90-day period is used only for sample eligibility, not as a model feature.

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

Choose the small neural network with:

```bash
python experiments/lightgbm_baseline/run.py backtest \
  --temporal-model frequency_nn
```

The network has one 32-unit hidden layer and trains one binary classifier per
activity label. It receives only the six per-action 7-day frequencies; 30-day
frequency features, 90- and 365-day features, user IDs, counts, shares, and
other features are excluded. Training, calibration, and holdout remain
chronological. By default, it trains six separate classifiers. To train one
network with a shared hidden layer and a six-value output vector instead, use:

```bash
python experiments/lightgbm_baseline/run.py backtest \
  --temporal-model frequency_nn --nn-output-mode multi_label
```

The same option is available for `predict`. Each output value is the
probability for one activity label; the selected fixed or calibrated
thresholds turn those probabilities into the six binary predictions. The
single model is saved as `models/frequency_nn_multilabel.joblib`.

The script uses weekly forecast cutoffs beginning on 2026-03-01. It trains on the
oldest 70% of weeks, reserves the next 15% for calibration, and reports the
untouched final 15% holdout. By default, decisions use fixed thresholds:
`0.30` for all labels except `push`, which uses `0.45`. Add
`--calibrated-threshold` to choose per-label thresholds on the calibration
weeks instead. Metrics are written to
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

# Opt in to per-label threshold calibration
python experiments/lightgbm_baseline/run.py backtest --calibrated-threshold
```

Each run writes `backtest_metrics.json`, including the parameters, per-label F1,
macro-F1, micro-F1, combined score, threshold mode, and thresholds. Compare the
`score` field across runs; it is the assignment's average of macro-F1 and
micro-F1.
Change one or two settings at a time, keep the same cutoff/data/split, and use
the chronological holdout only for comparing a small number of final candidates.
Repeatedly tuning to that holdout can overfit it.

LightGBM uses `scale_pos_weight = negative_count / positive_count` separately
for each label by default, calculated from the training split only. Pass
`--no-scale-pos-weight` to disable class weighting for either `backtest` or
`predict`.

## Export LightGBM holdout failures

From the repository root, run:

```bash
python exam_fail_case.py
```

The script uses the same chronological train/calibration/holdout split as the
LightGBM backtest. It saves holdout metrics and `failure_cases.csv` under a new
timestamped folder in `experiments/lightgbm_baseline/outputs/`. The CSV has one
row per misclassified user-week-label, including actual/predicted values,
probability, threshold, failure type, user IDs, cutoff, and feature values.
It uses the default fixed thresholds; pass `--calibrated-threshold` to opt in to
calibration-based thresholds. Override the output folder or LightGBM parameters
with the script's CLI flags. Per-label `scale_pos_weight` is enabled by
default; pass `--no-scale-pos-weight` to disable it.
Failure interpretation is intentionally left as a TODO in the script.

To analyze a previously completed backtest from any supported experiment
(including LightGBM, RelGT, and GRU), point the analyzer at that run directory:

```bash
python exam_fail_case.py --run-dir experiments/lightgbm_baseline/outputs/backtest_RUN --no-dashboard
```

This exports failure and false-positive CSVs under the run's
`failure_analysis/` folder. LightGBM backtests save `holdout_probabilities.csv`
with embedded actual labels for this purpose; rerun older LightGBM backtests
that do not contain this file. For multi-branch runs, select a branch with
`--model-branch`, for example `--model-branch relgt`.

Supported LightGBM options are `--n-estimators`, `--learning-rate`,
`--num-leaves`, `--min-child-samples`, `--subsample`, `--colsample-bytree`,
and `--reg-lambda`. The baseline defaults are 300, 0.03, 15, 40, 0.85, 0.85,
and 2.0 respectively. Row subsampling is enabled with a sampling frequency of 1.
S2P2 options are `--s2p2-epochs`, `--s2p2-batch-size`,
`--s2p2-max-sequence-length`, `--s2p2-hidden-size`, `--s2p2-state-size`,
`--s2p2-num-layers`, and `--s2p2-integral-samples`.

For each historical cutoff, LightGBM receives the total action count across
the six activity types for the preceding 7 days, each action's share of the
total for the preceding 7 and 30 days, and per-action daily frequencies for
the 7-day window. Raw 30-day count features are excluded. Shares are zero when
the user has no actions in that window. These features use only actions
available before the cutoff and are included automatically in LightGBM
training, prediction, and RelGT+LightGBM ensembles.

LightGBM also receives each action's median daily count and the combined
median daily count across all six actions for the same 7- and 30-day windows.
These medians include zero-activity days, use complete 24-hour bins anchored to
the prediction cutoff, and exclude events or rows unavailable at that cutoff.
The frequency-only neural network continues to use only its per-action daily
frequency inputs.

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
- `metadata.json`, including feature names, thresholds, and threshold mode

Prediction uses the same fixed thresholds by default. Add
`--calibrated-threshold` to calibrate thresholds from the earlier validation
weeks, for example:

```bash
python experiments/lightgbm_baseline/run.py predict --calibrated-threshold
```

To generate predictions with S2P2, use `--temporal-model s2p2`, for example:

```bash
python experiments/lightgbm_baseline/run.py predict \
  --temporal-model s2p2 --device auto \
  --output-dir experiments/lightgbm_baseline/outputs/s2p2_prediction
```

To generate neural-network predictions and probabilities:

```bash
python experiments/lightgbm_baseline/run.py predict \
  --temporal-model frequency_nn
```

For one six-output neural network, add `--nn-output-mode multi_label`.

This writes evaluator-format binary labels to `predictions.csv` and per-label
probabilities to `probabilities.csv`. The six trained models are saved as
`models/<label>.joblib`; multi-label mode saves one
`models/frequency_nn_multilabel.joblib` model instead.

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
- Lifecycle summaries are treated as events at their transition time:
  PullRequest merges and closures, issue closures and their `state_reason`, and
  review dismissals contribute only when the transition and source row were
  both available before the feature cutoff. PR merge counts are attributed to
  both the PR author and `merged_by_id`; user IDs are not model features.
- Labels use actual action timestamps: PR/issue `created_at`, review
  `submitted_at`, and comment/push `created_at`. Each label covers
  `[cutoff, cutoff + 7 days)`.
- Training, calibration, and holdout are split chronologically by weekly
  cutoff. Prediction-time training and calibration examples are further
  limited to label windows that end on or before the requested prediction
  cutoff.
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
