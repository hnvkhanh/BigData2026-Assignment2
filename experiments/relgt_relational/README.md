# RelGT / GNN with relational root features

This experiment brings the cross-action relationships from
[`experiments/relational_lightgbm`](../relational_lightgbm/README.md) into the
graph models. Those relationships are ownership, PR/issue authorship, reviewer
history, assignment, push → PR sequences, @mentions, thread activity, the review
cycle and 8-week habits.

Everything except the root features is `experiments.relgt.run` in its `relgt` and
`gnn` modes: the same 14-day graph snapshots, five-action 14-day eligibility,
weekly cutoffs from 2026-03-11 16:00 UTC, 70/15/15 chronological split, model
sizes, optimiser, per-label threshold calibration on calibration weeks, and
per-week official scoring. With `--groups none` the model is the unchanged
`RelGTInspired` / `RelationalGNN`, and with the same seed it reproduces the
`relgt` / `gnn` results exactly (0.5658 / 0.5580 at seed 42).

## How the features enter the model

For every user-week sample the 42 relational features (`--groups all`) are
computed at that sample's exact cutoff with
`experiments.relational_lightgbm.features.relational_features`. They follow the
same rule as everywhere else: rows must be acted and available before the cutoff.
The vector is `log1p`-scaled and standardised with means and standard deviations
from the **training weeks only** (saved under `root_feature_scaler` in the
metrics). A linear layer projects it to the hidden size.

- **RelGT:** the projection is added to the user root token before the local
  transformer, so every attention layer and the centroid/global readout are
  conditioned on it.
- **GNN:** the projection is added to the root state before message passing,
  so the relation-specific messages update a root that already carries it.

This is representation fusion inside one jointly trained network, not an
ensemble. The base classes gained an optional `num_root_features` argument
(default 0, so all other experiments are unchanged).

The relational features also give the graph models information beyond their
14-day window. Examples are `reviewed_repos_90d`, open items first seen within
90 days, and the 8-week habit counts. The plain graph models cannot see that far back.

## Run

```bash
python -m experiments.relgt_relational.run backtest                                 # RelGT + all groups
python -m experiments.relgt_relational.run backtest --architecture gnn
python -m experiments.relgt_relational.run backtest --architecture gnn --groups none  # control
python -m experiments.relgt_relational.run backtest --groups hypotheses               # only the first six groups
python -m experiments.relgt_relational.run backtest --architecture gnn --select-epoch  # keep best calibration epoch
python -m experiments.relgt_relational.run predict --architecture gnn
python -m pytest experiments/relgt_relational
```

A backtest writes `backtest_metrics.json`, `holdout_probabilities.csv` (branch
`relgt_relational` / `gnn_relational`, or `relgt` / `gnn` for the control) and
the model weights. The graph samples and the root features are cached in the run
folder's `cache/`, keyed by the data files, start cutoff, feature version and
groups. Copying a previous run's `cache/` into a new `--output-dir` skips
rebuilding them. A stale cache is detected and rebuilt.

`predict` trains on the weeks whose label window ends by `--cutoff`. It
calibrates thresholds on the latest of those weeks, builds the 14-day snapshot
and root features at `--cutoff` for the users in `test_input.csv`, and writes
`predictions.csv`, `probabilities.csv`, `metadata.json` and the weights.

## Results

The data release has pushes up to 2026-10-26. The split is 22 train / 5 calibration /
5 holdout weeks (holdout from 2026-09-16), with 5 epochs. Each cell is the
mean over seeds 42 and 7. "Best calib" (`--select-epoch`) keeps the epoch with
the best calibration-week score; "last" keeps epoch 5.

| Model | Epoch | Root features | Score | Macro-F1 | Micro-F1 | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GNN | best calib | none | **0.5657** | 0.5189 | **0.6126** | 0.452 | 0.478 | 0.436 | 0.480 | 0.439 | 0.829 |
| GNN | best calib | all | **0.5657** | **0.5265** | 0.6049 | **0.485** | 0.488 | 0.440 | 0.483 | 0.439 | 0.825 |
| RelGT | best calib | none | 0.5624 | 0.5184 | 0.6063 | 0.467 | 0.491 | 0.436 | 0.469 | 0.423 | 0.824 |
| RelGT | best calib | all | 0.5607 | 0.5204 | 0.6010 | 0.478 | 0.485 | 0.429 | 0.477 | 0.425 | 0.829 |
| GNN | last | none | 0.5615 | 0.5131 | 0.6099 | 0.452 | 0.481 | 0.431 | 0.471 | 0.420 | 0.824 |
| GNN | last | all | 0.5600 | 0.5188 | 0.6013 | 0.465 | 0.481 | 0.429 | 0.475 | 0.437 | 0.827 |
| RelGT | last | none | 0.5565 | 0.5107 | 0.6023 | 0.458 | 0.478 | 0.413 | 0.474 | 0.413 | 0.829 |
| RelGT | last | all | 0.5479 | 0.5074 | 0.5884 | 0.460 | 0.483 | 0.407 | 0.465 | 0.400 | 0.830 |

`python -m experiments.relgt_relational.summarize` prints each seed and this table.

What the results show:

- **The score does not improve.** With epoch selection the relational models tie
  the controls (GNN 0.5657 vs 0.5657, RelGT 0.5607 vs 0.5624). Without it, the
  relational RelGT is worse (0.548 vs 0.557): it reaches a lower training loss,
  so the extra 42 dense inputs make it easier to overfit the training weeks.
- **Macro-F1 rises and micro-F1 falls**, in all four comparisons. The rare,
  non-push actions get better: macro-F1 improves by up to +0.008 and `open_pr`
  by +0.01 to +0.03. Micro-F1, which is dominated by the frequent `push` label,
  drops by 0.005 to 0.009. Because the score averages the two, it stays flat.
  This fits the motivation: the relationships help the actions that are hard to
  predict, but thresholds chosen per label for F1 trade away some precision on
  the frequent ones.
- **Seed variation is about as large as the effect.** For example, the RelGT
  control scores 0.566 at seed 42 and 0.547 at seed 7. Treat any single-run
  difference below about 0.01 as noise.
- Epoch selection on calibration weeks helps every configuration (+0.004 to
  +0.013). It is worth using whether or not root features are added.
