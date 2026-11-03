# Relational GNN activity chunks with per-chunk lookbacks

Same idea as `relgt_activity_chunks`: split user-week examples into `low`,
`mid`, and `high` chunks by prior activity and train one `RelationalGNN`
(architecture `gnn`) per chunk. The difference is the **graph lookback window**:
each chunk model sees a different amount of history.

| Chunk | Default lookback |
|---|---|
| `low` | 180 days |
| `mid` | 21 days |
| `high` | 7 days |

Set them with `--lookbacks LOW,MID,HIGH`. The optional `global` baseline (skip
with `--no-global-baseline`) trains on every user with `--global-lookback`
(default 14).

## Alignment

Eligibility uses the **longest** lookback for every model, so all example sets
contain the same (user, cutoff) rows and labels; only each row's graph
neighbourhood differs. The run fails if the sets are not aligned. As a result
the row set can differ from `relgt_activity_chunks`, whose eligibility uses a
14-day window.

Chunking (`--chunk-thresholds`, `--activity-window-days`, `--inspect-only`),
the chronological split, per-model threshold calibration, and chunk routing
work exactly as in `relgt_activity_chunks`. The options `--train-fraction`
(unused training rows join calibration), `--balance-chunks`, and
`--user-embedding` are shared too.

## Outputs

Written to a timestamped folder under
`experiments/relgt_lookback_chunks/outputs/` (or `--output-dir`), with the same
files as `relgt_activity_chunks`. `backtest_metrics.json` additionally records
`lookback_days` per model and `eligibility_lookback_days`. Pass the same
`--output-dir` on later runs to reuse the cached per-lookback graphs.
