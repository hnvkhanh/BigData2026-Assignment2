# Run from the repository root.

# LightGBM (fixed thresholds: 0.30 for all labels except push at 0.45)
python experiments/lightgbm_baseline/run.py backtest

# By default LightGBM / frequency NN train and test only on the users in
# assignment2/evaluation/test_input.csv (--test-input to change the file).
# Use --train-users cohort to train on every cohort user instead. Works for backtest and predict.
python experiments/lightgbm_baseline/run.py backtest --train-users cohort
python experiments/lightgbm_baseline/run.py predict --temporal-model frequency_nn --train-users cohort

# Frequency neural network (six independent models)
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn

# Frequency neural network (one six-output model)
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn --nn-output-mode multi_label

# Train LightGBM / frequency NN on a random 30% of training rows (seeded by --seed);
# pass --discard-unused-train to drop the unused 70% instead of adding it to calibration. Works for backtest and predict.
python experiments/lightgbm_baseline/run.py backtest --train-fraction 0.3
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn --train-fraction 0.3 --seed 7

# Let LightGBM see the user ID as a categorical feature. Unseen users get no signal.
python experiments/lightgbm_baseline/run.py backtest --user-embedding
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

# Train on only 25% of the training partition (random, seeded by --seed); the unused 75% is added to calibration (or dropped with --discard-unused-train)
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

# Relational GNN trained without summer weeks (label window touching Jun-Aug).
# Calibration/holdout are unchanged; --summer-months none is the unfiltered control.
python -m experiments.gnn_no_summer.run backtest
python -m experiments.gnn_no_summer.run backtest --summer-months 7,8
python -m experiments.gnn_no_summer.run backtest --summer-months none
python -m experiments.gnn_no_summer.run predict
# Calibrate thresholds on chosen months (weeks whose cutoff is in them) instead of the last week
python -m experiments.gnn_no_summer.run backtest --calibration-months 9
python -m experiments.gnn_no_summer.run predict --calibration-months 9,10

# Relational GNN trained only on September weeks (cutoff in September); calibration/holdout unchanged
python -m experiments.gnn_september.run backtest
python -m experiments.gnn_september.run backtest --train-months 9,10
python -m experiments.gnn_september.run predict

# Ensemble: GNN on all weeks + GNN on September only; weight w*p_all + (1-w)*p_sep picked on calibration
python -m experiments.gnn_september_ensemble.run backtest
python -m experiments.gnn_september_ensemble.run backtest --fusion-weight 0.5
python -m experiments.gnn_september_ensemble.run predict

# DM2026-Final sequence forecasters (LSTM / Informer / FEDformer) on 91-day daily activity;
# blend weights and thresholds picked on calibration
python -m experiments.sequence_forecasters.run backtest
python -m experiments.sequence_forecasters.run backtest --epochs 1
python -m experiments.sequence_forecasters.run backtest --fusion-weights lstm=0.4,informer=0.2,fedformer=0.4
python -m experiments.sequence_forecasters.run backtest --models lstm,fedformer
python -m experiments.sequence_forecasters.run predict
python -m pytest experiments/sequence_forecasters -q

# RelGT / GNN with a learned per-user embedding (users unseen in training get a zero vector).
# The LightGBM/S2P2 branches of hybrid modes do not use it.
python -m experiments.relgt.run backtest --temporal-model gnn --user-embedding
python -m experiments.relgt.run predict --temporal-model gnn --user-embedding
python -m experiments.relgt.run backtest --user-embedding

# RelGT / GNN on the official test population (>= 5 events in 90 days instead of 14)
python -m experiments.relgt.run backtest --eligibility-days 90
python -m experiments.relgt.run backtest --temporal-model gnn --eligibility-days 90

# GNN on every cohort user at every cutoff (no activity filter; inactive users have an empty graph
# and are still trained on and scored). Prediction already scores every test user.
python -m experiments.relgt.run backtest --temporal-model gnn --eligibility-days 0
python -m experiments.relgt.run predict --temporal-model gnn --eligibility-days 0

# GNN trained, calibrated and backtested only on the users in assignment2/evaluation/test_input.csv,
# at every cutoff with no activity filter (the population that predict scores)
python -m experiments.relgt.run backtest --temporal-model gnn --test-users
python -m experiments.relgt.run predict --temporal-model gnn --test-users

# GNN graph lookback (days of history per user; default 14). --eligibility-days, if given, must be >= it.
python -m experiments.relgt.run backtest --temporal-model gnn --lookback-days 30
python -m experiments.relgt.run backtest --temporal-model gnn --test-users --lookback-days 30
python -m experiments.relgt.run predict --temporal-model gnn --test-users --lookback-days 30

# Train on only a random fraction of the training examples (seeded by --seed); the unused rest
# is added to calibration. Works for every relgt backtest/predict mode, including hybrids.
python -m experiments.relgt.run backtest --train-fraction 0.25
python -m experiments.relgt.run backtest --temporal-model gnn --train-fraction 0.5
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm --train-fraction 0.5
python -m experiments.relgt.run predict --temporal-model gnn_ensemble --train-fraction 0.5

# Combine both
python -m experiments.relgt.run backtest --temporal-model gnn --train-fraction 0.5 --user-embedding

# Calibrated 30/14-day GNN ensemble
python -m experiments.relgt.run backtest --temporal-model gnn_ensemble

# Set the first lookback model's ensemble weight manually (second gets 1 - weight)
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

# Failure analysis of a relgt_db backtest
python exam_fail_case.py --run-dir experiments/relgt/outputs/<run-folder> --model-branch relgt_db

# Generate predictions with relgt_db
python -m experiments.relgt.run predict --temporal-model relgt_db

# Generate predictions with the default LightGBM and RelGT models
python experiments/lightgbm_baseline/run.py predict
python -m experiments.relgt.run predict

# Generate predictions with the relational GNN
python -m experiments.relgt.run predict --temporal-model gnn

# Generate predictions with the 30/14-day GNN ensemble
python -m experiments.relgt.run predict --temporal-model gnn_ensemble

# LightGBM + cross-action relational features (owner / author / reviewer / assignee / sequence)
python -m experiments.relational_lightgbm.run backtest
# Ablation: drop all relational features (same code path as the plain baseline features)
python -m experiments.relational_lightgbm.run backtest --groups none
# Ablation: only some groups (owner, sequence, assigned, reviewer, author, repo_activity)
python -m experiments.relational_lightgbm.run backtest --groups reviewer,sequence
python -m experiments.relational_lightgbm.run predict
# Measure candidate relationships on the training weeks (writes relationship_lift.csv, transition_lift.csv)
python -m experiments.relational_lightgbm.explore

# RelGT / GNN with the relational features added to the user root token (--groups none = control)
python -m experiments.relgt_relational.run backtest
python -m experiments.relgt_relational.run backtest --architecture gnn
python -m experiments.relgt_relational.run backtest --architecture gnn --groups none
python -m experiments.relgt_relational.run predict --architecture gnn

# RelGT / GNN on the 14-day graph plus typed relationship edges (owns, assigned, mentioned_in,
# review_candidate, open_authored_*, thread, reviewed_repo, unpaired_branch, interacted users)
python -m experiments.relgt_relgraph.run backtest --select-epoch
python -m experiments.relgt_relgraph.run backtest --architecture gnn --select-epoch
python -m experiments.relgt_relgraph.run backtest --architecture gnn --relations none --select-epoch  # control
python -m experiments.relgt_relgraph.run backtest --relations interacted,review_candidate --select-epoch
python -m experiments.relgt_relgraph.run predict --architecture gnn --select-epoch

# relgt_db on the database plus relationship tables (RepoOwners, Notifications, Collaborations);
# --reserve keeps N of the K tokens for relationship rows and the rows they point to
python -m experiments.relgt_db_relational.run backtest --epochs 10
python -m experiments.relgt_db_relational.run backtest --epochs 10 --reserve 32
python -m experiments.relgt_db_relational.run backtest --epochs 10 --tables none  # control (plain relgt_db)
python -m experiments.relgt_db_relational.run backtest --epochs 10 --tables Notifications --reserve 32
python -m experiments.relgt_db_relational.run predict --epochs 10 --reserve 32

# Relational Transformer (stanford-star/relational-transformer port): every cell is a token,
# attention by column / row + foreign-key parents / referencing rows; target is a masked
# UserWeeks task row, earlier closed weeks of the same user are labelled in-context rows
python -m experiments.relational_transformer.run backtest
python -m experiments.relational_transformer.run backtest --epochs 1 --max-steps-per-epoch 30 --start-cutoff "2026-07-01 16:00:00"  # smoke
python -m experiments.relational_transformer.run backtest --attention col,feat,nbr,full
python -m experiments.relational_transformer.run backtest --history-weeks 0   # ablation: no in-context labels
python -m experiments.relational_transformer.run backtest --mtp-weight 0.1    # + masked-token objective
python -m experiments.relational_transformer.run backtest --num-neighbors 128 --max-cells 1024 --batch-size 16
python -m experiments.relational_transformer.run predict
python -m pytest experiments/relational_transformer -q

# Drift-robust LightGBM: all LightGBM improvements as named variants scored under one protocol
# (validation = calibration weeks, holdout = last week); results in outputs/<run>/RESULTS.md
python -m experiments.drift_lightgbm.run backtest
python -m experiments.drift_lightgbm.run backtest --variants relational,mask,rolling_half_life_8
python -m experiments.drift_lightgbm.run predict --variant relational

# Run every experiment and write a ranked comparison (experiments/results/<tag>/RESULTS.md)
python run_all_experiments.py --list
python run_all_experiments.py --quick --tag smoke
python run_all_experiments.py --tag full
