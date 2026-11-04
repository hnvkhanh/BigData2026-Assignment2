# Sequence forecasters (DM2026-Final architecture)

This experiment ports the models from
[hnvkhanh/DM2026-Final](https://github.com/hnvkhanh/DM2026-Final) to this task.
DM2026 forecast weekly disaster severity from 91-day daily weather windows per
region. Here, each user-week example uses a 91-day window of the user's daily
GitHub actions. Three forecasters are trained independently, and their
probabilities are blended. This is a weighted probability average of separate
models, as in DM2026's `ensemble.py`. No representations are shared.

| Branch | Ported from | Change for this task |
|---|---|---|
| `lstm` | `lstm_v2.py` (`LSTMRegressor`) | The horizon embedding becomes a label embedding. The head runs once per label and outputs one logit each. It also uses the cutoff calendar features (`aux`). |
| `informer` | `informer_updated.py` (`InformerRegressor`) | Same ProbSparse attention, distilling encoder and start-token decoder. It decodes six label tokens instead of five horizon tokens, using the original causal mask in `LABELS` order. |
| `fedformer` | `fedformer_updated.py` | Same seasonal/trend decomposition and Fourier-mode encoder. The head emits six logits. |

Changes that apply to every branch:

- The region embedding becomes a per-user embedding (`--user-emb-dim`, default
  32, `0` disables it). Users unseen in training share a zero vector.
- The MSE/Huber regression loss becomes `BCEWithLogitsLoss`.
- DM2026 kept the best validation epoch. Here each branch keeps the epoch with
  the lowest BCE on the calibration week.
- The optimiser defaults follow DM2026: LSTM 6 epochs at lr 1e-3, Informer 8
  epochs at lr 1e-4, FEDformer 10 epochs at lr 2e-4, all with AdamW and
  gradient clipping at 1.0.

DM2026's experiment 3 (a meta-learner on frozen base predictions) is not ported.
Training it without leakage would need labelled rows outside the model-training
weeks, and this split has only one calibration week. That week already selects
the thresholds, the blend weights and the epoch.

## Inputs

These inputs replace DM2026's weather feature engineering.

- **Examples, eligibility and labels** are shared with `experiments.relgt`
  (`load_or_build_examples`). The default eligibility is at least 5 events in the
  14 days before the cutoff (`--eligibility-days`). The six labels cover the 7
  days `[cutoff, cutoff + 7 days)`. Graph tokens are built (and cached in the
  run folder) only to obtain these rows. The models do not read them.
- **Sequence** (`float32 (n, window_days, 10)`, oldest day first). The window
  is split into 24-hour bins that end exactly at the cutoff (16:00 UTC), and
  bin `i` covers `[cutoff - (W - i) days, cutoff - (W - i - 1) days)`. An action
  counts only if `action_at`, `available_at` and `entity_available_at` are all
  before the cutoff. Each bin has 10 channels:
  - `log1p` count of each of the six actions;
  - `log1p` total count;
  - an any-activity flag;
  - day-of-week sin/cos.

  The first 8 channels are standardised with mean/std from the training rows
  only (as DM2026 scales with training statistics). The scaler is saved in the
  metrics/metadata.
- **Aux** (LSTM only): month, day-of-year and ISO week of the cutoff as sin/cos.

## Split, blending and scoring

The split follows the repository convention:

- holdout is the latest weekly cutoff (2026-10-14);
- calibration is the week before it;
- every earlier week trains.

Blend weights are searched on a simplex grid (`--fusion-step`, default 0.1)
together with the per-label thresholds, using calibration only.
`--fusion-weights lstm=0.4,informer=0.2,fedformer=0.4` fixes the weights;
those values are DM2026's default blend. Holdout is scored once per week with
the official `evaluate.py` (`calculate_metrics`). `branch_holdout` also scores
each branch alone with its own calibration thresholds.

## Run

```bash
python -m experiments.sequence_forecasters.run backtest
python -m experiments.sequence_forecasters.run backtest --epochs 1            # smoke
python -m experiments.sequence_forecasters.run backtest --models lstm,fedformer
python -m experiments.sequence_forecasters.run backtest --fusion-weights lstm=0.4,informer=0.2,fedformer=0.4
python -m experiments.sequence_forecasters.run backtest --attn full --no-distil
python -m experiments.sequence_forecasters.run predict
python -m pytest experiments/sequence_forecasters -q
```

Run `--help` to see every hyperparameter (`--window-days`, `--label-days`,
`--d-model`, `--n-heads`, `--fed-modes`, `--moving-avg`, `--<branch>-lr`, …).

## Outputs

Each run writes to a new timestamped folder under `outputs/`, or to `--output-dir`.

- Backtest: `backtest_metrics.json`, which contains the ensemble as
  `holdout_metrics`, a per-branch `branch_holdout`, epoch histories, the fusion
  weights, the thresholds and the feature scaler. It also writes
  `holdout_probabilities.csv`, with `actual_<label>`, `<branch>_<label>` and
  `ensemble_<label>` columns, and `<branch>_model.pt`.
- Predict: `predictions.csv` (the ensemble), `probabilities.csv` (the ensemble
  plus per-branch columns), `metadata.json` and `<branch>_model.pt`. Predict
  trains on every week whose label window ends before `--cutoff` and
  calibrates on the latest such week.
