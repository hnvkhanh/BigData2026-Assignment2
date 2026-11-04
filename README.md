# Assignment 2: Predicting GitHub User Activity

This repository contains schema-inspection utilities and experiments for
predicting six weekly GitHub activity labels: opening pull requests or issues,
issue comments, PR reviews, PR review comments, and pushes. The experiments
use historical user-week examples, time-ordered splits, and cutoff-aware
features.

## Repository contents

- [`understand_schema.py`](./understand_schema.py) inspects the assignment's
  Parquet tables and writes a schema report. By default, it reads from
  `assignment2/data/db/` and saves `schema_summary.txt` in the repository root.
- [`experiments/lightgbm_baseline/`](./experiments/lightgbm_baseline/README.md)
  contains the temporal tabular baseline:
  - **LightGBM** is the default model. It trains one binary classifier for each
    of the six labels.
  - **frequency_nn** is a small neural network that uses only per-action
    frequencies to predict all six labels; it never uses user IDs as inputs.
  - `s2p2.py` holds the S2P2 marked temporal point-process adapter (pinned
    EasyTPP dependency) used by RelGT's hybrid modes.
- [`experiments/drift_lightgbm/`](./experiments/drift_lightgbm/README.md)
  tests the LightGBM follow-ups (relational/history/regime features, ingestion
  anomaly masking, recency weights, refits, bagging, threshold tuning, push
  ingestion lag) as named variants under one protocol, compares the graph
  models on the official 90-day test population, and documents the
  recommended submission model. Start there for current results.
- [`experiments/relgt/`](./experiments/relgt/README.md) contains a compact,
  assignment-specific **RelGT-inspired** graph model. It is not a drop-in
  execution of Stanford's RelGT/RelBench implementation.
  - Its optional **hybrid** mode combines RelGT and S2P2 probabilities.
- [`experiments/gru_relational_ple/`](./experiments/gru_relational_ple/README.md)
  contains an experimental **GRU + relational attention + PLE** model that
  combines daily activity sequences and graph neighborhoods for six-label
  prediction.

The temporal experiments use chronological training/calibration/holdout
periods rather than random row splits. Prediction runs train and calibrate only
on examples whose complete seven-day label window ends by the requested
prediction cutoff. Feature snapshots also exclude actions, rows, and lifecycle
transitions that were not yet available at each historical cutoff. LightGBM
also uses per-action and combined median daily activity counts over its
lookback windows; inactive days count as zero.

## Setup

Run commands from the repository root. Install the base dependencies and the
PyTorch dependency for RelGT:

```powershell
python -m pip install -r requirements.txt -r experiments/relgt/requirements.txt
```

S2P2 is optional. Install it only when running RelGT's `hybrid` or
`relgt_lightgbm_s2p2` modes:

```powershell
python -m pip install -r experiments/lightgbm_baseline/s2p2-requirements.txt
```

The assignment data should be available as Parquet files under
`assignment2/data/db/`. Prediction also expects
`assignment2/evaluation/test_input.csv`.

## Run experiments

Run a LightGBM chronological backtest:

```powershell
python experiments/lightgbm_baseline/run.py backtest
```

The fixed decision thresholds are `0.30` for all labels except push (`0.45`).

Run the frequency-only neural network:

```powershell
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn --nn-output-mode multi_label
```

By default, it trains six classifiers; `--nn-output-mode multi_label` trains
one network that predicts a six-label probability vector. Its prediction run
also saves probabilities in `probabilities.csv`.

Run RelGT, or combine it with S2P2, LightGBM, or both:

```powershell
python -m experiments.relgt.run backtest
python -m experiments.relgt.run backtest --temporal-model hybrid
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm_s2p2
python -m experiments.gru_relational_ple.run
```

The three-model mode requires the optional S2P2/EasyTPP dependency. To
generate its predictions:

```powershell
python -m experiments.relgt.run predict --temporal-model relgt_lightgbm_s2p2
```

Generate prediction files using the default model for each experiment:

```powershell
python experiments/lightgbm_baseline/run.py predict
python -m experiments.relgt.run predict
```

See the experiment READMEs for model-specific options, device selection,
validation details, and S2P2 configuration.

## Where outputs are saved

Unless `--output-dir` is supplied, each run creates a new timestamped folder
under its experiment's `outputs/` directory:

- `experiments/lightgbm_baseline/outputs/`
- `experiments/relgt/outputs/`
- `experiments/gru_relational_ple/outputs/`

These output directories are ignored by Git. Specify `--output-dir` to choose a
stable location; for example:

```powershell
python -m experiments.relgt.run predict --output-dir experiments/relgt/outputs/my_run
```

Typical outputs:

| Experiment and command | Files saved |
|---|---|
| LightGBM backtest | `backtest_metrics.json` with holdout metrics, split details, thresholds, and model settings |
| LightGBM prediction | `predictions.csv`, `metadata.json`, and files under `models/` (`<label>.txt` for trained LightGBM labels) |
| RelGT prediction | `predictions.csv`, `probabilities.csv`, `model.pt`, and `metadata.json` |
| RelGT + S2P2 prediction | The RelGT prediction files, with `relgt_model.pt` and `s2p2_model.pt` saved as model weights |
| RelGT + LightGBM prediction | Fused prediction/probability files, `relgt_model.pt`, `lightgbm_<label>.txt` files, and `metadata.json` |
| RelGT + LightGBM + S2P2 prediction | Fused and branch probability files, `relgt_model.pt`, `s2p2_model.pt`, LightGBM model files, and `metadata.json` |
| RelGT backtest | `backtest_metrics.json`; combined runs also save branch/fused holdout probabilities and both model weights |
| GRU + relational attention + PLE backtest | `backtest_metrics.json`, `holdout_probabilities.csv`, and `gru_relational_ple_model.pt` |

RelGT runs also cache weekly graph examples under the run folder at
`cache/weekly_graph_examples.npz`, with a neighboring JSON file describing the
data signature. Use separate output folders for separate experiments to keep
their outputs and caches distinct.

To inspect the assignment tables and save a schema report to a chosen path:

```powershell
python understand_schema.py --output schema_summary.txt
```
