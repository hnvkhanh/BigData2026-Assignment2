# Run from the repository root.

# LightGBM (fixed thresholds: 0.30 for all labels except push at 0.45)
# Per-label scale_pos_weight class weighting is enabled by default.
python experiments/lightgbm_baseline/run.py backtest

# Disable class weighting for a LightGBM run (also supported by RelGT+LightGBM)
python experiments/lightgbm_baseline/run.py backtest --no-scale-pos-weight

# Opt in to per-label threshold calibration for LightGBM
python experiments/lightgbm_baseline/run.py backtest --calibrated-threshold

# S2P2 baseline (requires optional S2P2 dependencies)
python experiments/lightgbm_baseline/run.py backtest --temporal-model s2p2

# Frequency neural network (six independent models)
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn

# Frequency neural network (one six-output model)
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn --nn-output-mode multi_label

# GRU + relational attention + PLE experiment
python -m experiments.gru_relational_ple.run

# Quick smoke backtest
python -m experiments.gru_relational_ple.run --epochs 1 --batch-size 64

# Example with explicit model/device settings and output directory
python -m experiments.gru_relational_ple.run --device auto --epochs 5 --hidden-dim 64 --num-heads 4 --num-graph-layers 2 --num-gru-layers 1 --output-dir experiments/gru_relational_ple/outputs/gru_relational_ple_run

# Relational GNN specialists: compare all-non-push with PR/issue grouping
python -m experiments.relgt_action_specialists.run

# Run only the grouped PR and issue action specialists
python -m experiments.relgt_action_specialists.run --variant grouped

# Quick specialist smoke backtest
python -m experiments.relgt_action_specialists.run --epochs 1 --batch-size 64

# Activity-chunk GNNs: see activity percentiles and chunk sizes before training
python -m experiments.relgt_activity_chunks.run --inspect-only --chunk-thresholds 20,200

# Three GNNs by prior-action count (low < 20 <= mid < 200 <= high) plus a global baseline
python -m experiments.relgt_activity_chunks.run --chunk-thresholds 20,200

# Same, with tertile thresholds from the training partition and no global baseline
python -m experiments.relgt_activity_chunks.run --no-global-baseline

# Count only the last 90 days of activity; quick smoke run
python -m experiments.relgt_activity_chunks.run --activity-window-days 90 --epochs 1 --batch-size 64

# RelGT
python -m experiments.relgt.run backtest

# Standalone relational GNN experiment
python -m experiments.relgt.run backtest --temporal-model gnn

# Calibrated 30/14-day RelGT ensemble
python -m experiments.relgt.run backtest --temporal-model relgt_ensemble

# Calibrated 30/14-day GNN ensemble
python -m experiments.relgt.run backtest --temporal-model gnn_ensemble

# Compare a different pair of windows (also supports 90,30 or 90,14)
python -m experiments.relgt.run backtest --temporal-model relgt_ensemble --ensemble-lookbacks 90,14

# Set the first lookback model's ensemble weight manually (second gets 1 - weight)
python -m experiments.relgt.run backtest --temporal-model relgt_ensemble --fusion-weight 0.7
python -m experiments.relgt.run predict --temporal-model gnn_ensemble --fusion-weight 0.6

# RelGT + S2P2 (requires optional S2P2 dependencies)
python -m experiments.relgt.run backtest --temporal-model hybrid
python -m experiments.relgt.run backtest --temporal-model hybrid --fusion-weight 0.65

# RelGT + LightGBM
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm --fusion-weight 0.7

# RelGT + LightGBM + S2P2 (requires optional S2P2 dependencies)
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm_s2p2
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm_s2p2 --fusion-weights relgt=0.4,lightgbm=0.4,s2p2=0.2

# Generate predictions with the default LightGBM and RelGT models
python experiments/lightgbm_baseline/run.py predict
python -m experiments.relgt.run predict

# Generate predictions with the relational GNN
python -m experiments.relgt.run predict --temporal-model gnn

# Generate predictions with the 30/14-day RelGT or GNN ensemble
python -m experiments.relgt.run predict --temporal-model relgt_ensemble
python -m experiments.relgt.run predict --temporal-model gnn_ensemble

# Generate LightGBM predictions with calibrated thresholds
python experiments/lightgbm_baseline/run.py predict --calibrated-threshold