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

# Train LightGBM / S2P2 / frequency NN on a random 30% of training rows (seeded by --seed);
# the unused 70% is added to calibration. Works for backtest and predict.
python experiments/lightgbm_baseline/run.py backtest --train-fraction 0.3
python experiments/lightgbm_baseline/run.py backtest --temporal-model s2p2 --train-fraction 0.3
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn --train-fraction 0.3 --seed 7

# Let the baselines see the user ID: categorical feature (lightgbm), one-hot/learned embedding
# input (frequency_nn), per-user logit offset on top of S2P2 (s2p2). Unseen users get no signal.
python experiments/lightgbm_baseline/run.py backtest --user-embedding
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn --user-embedding
python experiments/lightgbm_baseline/run.py backtest --temporal-model s2p2 --user-embedding
python experiments/lightgbm_baseline/run.py predict --user-embedding --train-fraction 0.5

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

# Specialists that see the user ID (learned per-user embedding on the root node)
python -m experiments.relgt_action_specialists.run --user-embedding

# Activity-chunk GNNs: see activity percentiles and chunk sizes before training
python -m experiments.relgt_activity_chunks.run --inspect-only --chunk-thresholds 20,200

# Three GNNs by prior-action count (low < 20 <= mid < 200 <= high) plus a global baseline
python -m experiments.relgt_activity_chunks.run --chunk-thresholds 20,200

# Same, with tertile thresholds from the training partition and no global baseline
python -m experiments.relgt_activity_chunks.run --no-global-baseline

# Count only the last 90 days of activity; quick smoke run
python -m experiments.relgt_activity_chunks.run --activity-window-days 90 --epochs 1 --batch-size 64

# Activity-chunk GNNs with a different graph lookback per chunk (low 180d, mid 21d, high 7d)
python -m experiments.relgt_lookback_chunks.run --chunk-thresholds 20,200

# Custom lookbacks (LOW,MID,HIGH), reusing cached graphs via a fixed output dir
python -m experiments.relgt_lookback_chunks.run --chunk-thresholds 20,200 --lookbacks 90,30,7 --output-dir experiments/relgt_lookback_chunks/outputs/run1

# Same options as the chunk experiment: --train-fraction, --balance-chunks, --user-embedding,
# --no-global-baseline, --global-lookback 14, --inspect-only
python -m experiments.relgt_lookback_chunks.run --chunk-thresholds 20,200 --train-fraction 0.5 --user-embedding

# Train on only 25% of the training partition (random, seeded by --seed); the unused 75% is added to calibration
python -m experiments.relgt_activity_chunks.run --chunk-thresholds 20,200 --train-fraction 0.25

# Equal training size per chunk: downsample larger chunks to the smallest chunk's size
# (the global baseline still trains on everything)
python -m experiments.relgt_activity_chunks.run --chunk-thresholds 20,200 --balance-chunks

# Give each model a learned per-user embedding (users unseen in training get a zero vector)
python -m experiments.relgt_activity_chunks.run --chunk-thresholds 20,200 --user-embedding

# All three options together
python -m experiments.relgt_activity_chunks.run --chunk-thresholds 20,200 --train-fraction 0.5 --balance-chunks --user-embedding

# RelGT
python -m experiments.relgt.run backtest

# Standalone relational GNN experiment
python -m experiments.relgt.run backtest --temporal-model gnn

# RelGT / GNN with a learned per-user embedding (both architectures and their ensembles;
# users unseen in training get a zero vector). The LightGBM/S2P2 branches of hybrid modes
# do not use it.
python -m experiments.relgt.run backtest --temporal-model gnn --user-embedding
python -m experiments.relgt.run backtest --temporal-model gnn_ensemble --user-embedding
python -m experiments.relgt.run predict --temporal-model gnn --user-embedding
python -m experiments.relgt.run predict --temporal-model gnn_ensemble --user-embedding
python -m experiments.relgt.run backtest --user-embedding
python -m experiments.relgt.run backtest --temporal-model relgt_ensemble --user-embedding

# Train on only a random fraction of the training examples (seeded by --seed); the unused rest
# is added to calibration. Works for every relgt backtest/predict mode, including hybrids.
python -m experiments.relgt.run backtest --train-fraction 0.25
python -m experiments.relgt.run backtest --temporal-model gnn --train-fraction 0.5
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm --train-fraction 0.5
python -m experiments.relgt.run predict --temporal-model gnn_ensemble --train-fraction 0.5

# Combine both
python -m experiments.relgt.run backtest --temporal-model gnn --train-fraction 0.5 --user-embedding

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

# RelGT applied directly to the relational database: all 8 tables become a pkey/fkey graph,
# every column is encoded by type, and K rows are sampled from each user's 2-hop neighbourhood
# (port of snap-stanford/relgt). Same user-weeks, labels and split as the relgt mode.
python -m experiments.relgt.run backtest --temporal-model relgt_db

# Quick relgt_db smoke backtest (fewer weeks, capped steps)
python -m experiments.relgt.run backtest --temporal-model relgt_db --epochs 1 --max-steps-per-epoch 30 --start-cutoff "2026-07-01"

# relgt_db with explicit model settings; reuse the output dir to skip token sampling next time
python -m experiments.relgt.run backtest --temporal-model relgt_db --epochs 10 --num-neighbors 256 --hidden-dim 128 --num-heads 4 --num-layers 2 --num-centroids 512 --db-lr 5e-4 --output-dir experiments/relgt/outputs/relgt_db_k256

# relgt_db attention variants and upstream-style ablations (drop one token element:
# type, hop, time, tfs = column features, gnn = subgraph positional encoding)
python -m experiments.relgt.run backtest --temporal-model relgt_db --conv-type local
python -m experiments.relgt.run backtest --temporal-model relgt_db --ablate gnn
python -m experiments.relgt.run backtest --temporal-model relgt_db --ablate tfs

# relgt_db on a random fraction of the training examples (the rest goes to calibration)
python -m experiments.relgt.run backtest --temporal-model relgt_db --train-fraction 0.5

# Failure analysis of a relgt_db backtest
python exam_fail_case.py --run-dir experiments/relgt/outputs/<run-folder> --model-branch relgt_db

# Generate predictions with relgt_db
python -m experiments.relgt.run predict --temporal-model relgt_db

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