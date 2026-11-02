# Relational GNN activity-chunk models

This experiment splits user-week examples into three chunks by how active the
user already was, then trains one `RelationalGNN` per chunk. Each model
predicts all six labels. A single model trained on every user (`global`) is
trained alongside by default, so the chunked result can be compared on the same
examples, cutoffs, and chronological split.

## Chunking

The activity measure is the number of a user's actions **known before the
example's cutoff**: the action time, row `available_at`, and target
`entity_available_at` must all be strictly earlier than the cutoff. By default
the whole cutoff-safe history is counted; `--activity-window-days N` counts
only actions that became known in the last `N` days. A user can therefore move
between chunks from one week to the next.

You choose the two thresholds with `--chunk-thresholds LOW,HIGH`:

| Chunk | Rule |
|---|---|
| `low` | count `< LOW` |
| `mid` | `LOW <= count < HIGH` |
| `high` | count `>= HIGH` |

With `--chunk-thresholds auto` (the default) the thresholds are the tertiles of
the **training partition** counts; calibration and holdout are never used.

Pick thresholds by looking at the distribution first. `--inspect-only` prints
training-partition percentiles and the example count in each chunk and
partition, then exits without training:

```powershell
python -m experiments.relgt_activity_chunks.run --inspect-only --chunk-thresholds 20,200
```

The run fails with an actionable error if any chunk has no train, calibration,
or holdout examples.

## Models and evaluation

- Inputs: 14-day graph snapshots, shared with the other RelGT experiments
  (`load_or_build_examples`).
- Split: oldest 70% of cutoffs train, next 15% calibration, latest 15% holdout.
- Each chunk model trains on its chunk's training rows only. Its probability
  thresholds are chosen on its own chunk's calibration rows only.
- `global` (skip with `--no-global-baseline`) trains on all training rows and
  picks thresholds on all calibration rows.
- The `chunked` output routes every holdout row to the one model that owns its
  chunk and uses that model's probabilities and thresholds. This is routing,
  not an ensemble: the other chunks' models do not contribute to the row, and
  nothing is averaged or weighted. The three models together cover each row
  exactly once.

## Outputs

Written to a timestamped folder under `experiments/relgt_activity_chunks/outputs/`
(or `--output-dir`):

- `backtest_metrics.json`: chunk thresholds and their source, examples per chunk
  and partition, split boundaries, settings, seed, per-model thresholds,
  `holdout_metrics` (chunked), `branch_holdout` (chunked and global), and
  `per_chunk_holdout` (the chunk model against the global model on each chunk's
  holdout rows).
- `holdout_probabilities.csv`: `user_id`, `cutoff`, `prior_actions`, `chunk`,
  `actual_<label>`, and `<branch>_<label>` for `chunked`, `low`, `mid`, `high`,
  and `global`. Every model scores every holdout row; `chunked` takes each row
  from its own chunk's model.
- `low_model.pt`, `mid_model.pt`, `high_model.pt`, `global_model.pt`.

This experiment is backtest-only; there is no `predict` command.

## Run

```powershell
python -m experiments.relgt_activity_chunks.run --chunk-thresholds 20,200
python -m experiments.relgt_activity_chunks.run --epochs 1 --batch-size 64
```

Options: `--db-dir`, `--output-dir`, `--start-cutoff`, `--device`,
`--chunk-thresholds`, `--activity-window-days`, `--inspect-only`,
`--no-global-baseline`, `--epochs`, `--batch-size`, `--hidden-dim`,
`--num-layers`, `--dropout`, `--learning-rate`, and `--seed`.
