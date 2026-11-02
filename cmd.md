# LightGBM
python experiments/lightgbm_baseline/run.py backtest

# S2P2
python experiments/lightgbm_baseline/run.py backtest --temporal-model s2p2

# Frequency neural network (six independent models)
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn

# Frequency neural network (one six-output model)
python experiments/lightgbm_baseline/run.py backtest --temporal-model frequency_nn --nn-output-mode multi_label

# RelGT
python -m experiments.relgt.run backtest

# RelGT + S2P2
python -m experiments.relgt.run backtest --temporal-model hybrid

# RelGT + LightGBM
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm

# RelGT + LightGBM + S2P2
python -m experiments.relgt.run backtest --temporal-model relgt_lightgbm_s2p2