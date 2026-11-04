# Relational GNN trained on September only

Same model, examples, split, calibration and scoring as `experiments/gnn_no_summer`
(and `relgt.run --temporal-model gnn`). Training keeps only weeks whose **cutoff**
falls in `--train-months` (default `9`). The idea: September is the period closest
to the October holdout and November test week, after the summer trough described
in `data_analyze.py`.

- With the default data this trains on the cutoffs 2026-09-02 through 2026-09-30
  (5 weeks, about 6.5k examples). The 2026-09-30 label window runs to 2026-10-06.
- Calibration: the week before the holdout (2026-10-07) by default, or every week
  in `--calibration-months`. That flag must not overlap `--train-months`.
- Holdout: the latest week (2026-10-14), never filtered, scored with the official
  `evaluate.py` once per week, so results compare directly with `gnn`.
- Graph history is unchanged. September snapshots still see the 14 days before
  each cutoff, which reaches into August.
- `--train-fraction` subsamples the September pool.
- The run logic lives in `experiments/gnn_no_summer/run.py`. This package only
  adds the month filter.

```bash
python -m experiments.gnn_september.run backtest
python -m experiments.gnn_september.run backtest --train-months 9,10
python -m experiments.gnn_september.run predict
python -m pytest experiments/gnn_september -q
```

Outputs: `backtest_metrics.json` (a `train_filter` block lists kept and excluded
weeks), `holdout_probabilities.csv` (`gnn_september_<label>`) and
`gnn_september_model.pt`. Predict mode writes `predictions.csv`,
`probabilities.csv` and `metadata.json`.
