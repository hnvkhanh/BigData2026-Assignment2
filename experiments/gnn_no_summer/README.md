# Relational GNN without summer training weeks

Same model and data as `python -m experiments.relgt.run backtest --temporal-model gnn`
(relation-aware message passing over 14-day sampled user ego graphs, see
`experiments/relgt/gnn_model.py`). The only change is the training set: weeks
whose labels fall in summer are dropped.

Why: `data_analyze.py` shows the cohort's activity drops sharply in July-August
and recovers in September. The holdout and test weeks (October, November) look
nothing like that trough, so summer weeks may teach the model a base rate that
does not transfer.

## Definition

- Prediction unit: one eligible user (>= 5 events in the 14 days before the
  cutoff, or `--eligibility-days`) at a weekly cutoff, starting 2026-03-11 16:00.
- Labels: the six shared `LABELS` over `[cutoff, cutoff + 7 days)`.
- Inputs: the shared cutoff-safe graph snapshots from `experiments/relgt/data.py`.
  Summer events are still used as *history* for September cutoffs; only summer
  *examples* are removed.
- Exclusion rule: a training example is dropped when any day of its label window
  falls in `--summer-months` (default `6,7,8`). With the default data this drops
  the cutoffs 2026-05-27 through 2026-08-26 (14 weeks) and keeps 16 training weeks.
- Split: the shared chronological split (latest week = holdout, the week before
  = calibration, earlier weeks = train). The filter applies to the training
  partition only. Calibration and holdout are never filtered, so holdout scores
  are directly comparable with the `gnn` experiment.
- `--calibration-months M,M` (optional) replaces the single calibration week.
  Calibration becomes every usable week whose *cutoff* falls in those months.
  In a backtest that means weeks before the holdout; in predict, weeks whose
  label window is complete before `--cutoff`. Every other usable week goes to
  the training pool, then the summer filter is applied. For example,
  `--calibration-months 9` calibrates on the five September weeks, and the
  2026-10-07 week moves into training. Calibration weeks may sit before
  training weeks; the holdout always stays last. The command fails if no usable
  week matches. The chosen weeks are saved under `calibration` in the metrics or
  metadata. A summer month given here is used for calibration, not dropped.
- `--train-fraction` subsamples what is left after the summer weeks are removed.
- Thresholds: per label, chosen on calibration only. Holdout score: the official
  `evaluate.py` scoring, run once per holdout week.
- `--summer-months none` runs the same code with no filter (the control).

## Commands

```bash
python -m experiments.gnn_no_summer.run backtest
python -m experiments.gnn_no_summer.run backtest --summer-months 7,8
python -m experiments.gnn_no_summer.run backtest --summer-months none   # control
python -m experiments.gnn_no_summer.run backtest --calibration-months 9
python -m experiments.gnn_no_summer.run predict --calibration-months 9,10
python -m experiments.gnn_no_summer.run predict
python -m pytest experiments/gnn_no_summer -q
```

## Outputs

Backtest: `backtest_metrics.json` (with a `train_filter` block that lists the excluded
and kept training weeks), `holdout_probabilities.csv` (`actual_<label>` and
`gnn_no_summer_<label>`), and `gnn_no_summer_model.pt`.
Predict: `predictions.csv`, `probabilities.csv`, `metadata.json`, and the checkpoint.
