# Ensemble: all-weeks GNN + September-only GNN

Two relational GNNs (`experiments/relgt/gnn_model.py`), trained independently on
the same examples and graph snapshots:

| Branch | Training weeks |
|---|---|
| `gnn_all` | every week before calibration, summer included (same as `gnn`) |
| `gnn_september` | only weeks whose cutoff falls in `--train-months` (default `9`), same as `gnn_september` |

The ensemble averages their probabilities, `p = w * p_all + (1 - w) * p_september`.
`w` is searched over 0, 0.1, ..., 1 together with the per-label thresholds, on
the calibration weeks only, maximising the calibration score. `--fusion-weight W`
fixes `w` instead. This is an ensemble of two separate models; their
representations are not combined.

- Calibration: the week before the holdout (2026-10-07) by default, or
  `--calibration-months` (which must not overlap `--train-months`).
- Holdout: the latest week (2026-10-14), scored with the official `evaluate.py`
  once per week.
- Both branches use the same seed and model settings. `--train-fraction` is not
  supported, so that both branches calibrate on exactly the same rows.
- `backtest_metrics.json` reports the ensemble as `holdout_metrics`. Under
  `branch_holdout` it also reports each branch alone, scored with its own
  calibration thresholds. `run_all_experiments.py` puts all three in its report.

```bash
python -m experiments.gnn_september_ensemble.run backtest
python -m experiments.gnn_september_ensemble.run backtest --fusion-weight 0.5
python -m experiments.gnn_september_ensemble.run predict
python -m pytest experiments/gnn_september_ensemble -q
```

Outputs: `holdout_probabilities.csv` (`gnn_all_<label>`, `gnn_september_<label>`,
`ensemble_<label>`) and `gnn_all_model.pt` / `gnn_september_model.pt`.
Predict writes `predictions.csv` (the ensemble), `probabilities.csv` (ensemble
plus per-branch columns) and `metadata.json` (with the fusion weight).
