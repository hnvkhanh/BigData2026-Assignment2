# GRU + relational attention + PLE

This experiment combines three model components to predict all six weekly
GitHub activity labels:

- A GRU encodes 90 days of daily per-action counts. Counts are log-transformed,
  ordered oldest to newest, and include only events and event rows available
  before each weekly cutoff.
- The RelGT-inspired graph encoder applies typed, time-aware local and global
  relational attention to the cutoff-safe user neighborhood.
- A two-stage Progressive Layered Extraction (PLE) head routes shared and
  label-specific expert representations to six separate prediction towers.

The model is trained as a single multi-task network with binary
cross-entropy. It uses chronological train/calibration/holdout periods; label
thresholds are selected on calibration data and reported metrics are measured
on the untouched holdout. This is an experimental combination, not a claim
that it improves on the individual models.

## Install

From the repository root, install the base requirements and PyTorch:

```powershell
python -m pip install -r requirements.txt -r experiments/relgt/requirements.txt
```

## Run a backtest

```powershell
python -m experiments.gru_relational_ple.run
```

For a quick smoke backtest:

```powershell
python -m experiments.gru_relational_ple.run --epochs 1 --batch-size 64
```

Options include `--db-dir`, `--start-cutoff`, `--device`, `--epochs`,
`--batch-size`, `--hidden-dim`, `--num-heads`, `--num-graph-layers`,
`--num-gru-layers`, `--dropout`, and `--learning-rate`. The default output is
a unique timestamped folder under `experiments/gru_relational_ple/outputs/`;
pass `--output-dir` to choose another folder.

Each run saves `backtest_metrics.json`, calibration-selected
`holdout_probabilities.csv`, and `gru_relational_ple_model.pt`. Weekly graph
examples are cached in the run folder using the existing RelGT cache format.
