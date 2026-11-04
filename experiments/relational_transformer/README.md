# Relational Transformer (RT) on the GitHub database

A port of the **Relational Transformer** ([stanford-star/relational-transformer](https://github.com/stanford-star/relational-transformer),
Ranjan et al., 2025, arXiv:2510.06377) to the assignment's six-label weekly
task. RT differs from `relgt_db` (`experiments/relgt/db_run.py`) in its unit of
attention. `relgt_db` encodes each sampled *row* into one token. RT makes every
*cell* (one value in one column of one row) a token. Structure enters only
through attention masks built from columns and foreign keys.

It is trained from scratch. Upstream's pretrained checkpoints (RT-J, RT-PluRel)
expect upstream's preprocessing and frozen sentence-embedding text features, so
they are not used. See [Differences from upstream](#differences-from-upstream).

## Experiment definition

| | |
|---|---|
| Prediction unit | cohort user at a weekly cutoff (Wednesday 16:00 UTC grid from `--start-cutoff`) |
| Targets | `open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push` in `[cutoff, cutoff + 7 days)` |
| Eligibility | at least 5 cutoff-safe actions in the 14 days before the cutoff (same as `relgt_db`; this is a filter, not a feature) |
| Split | chronological: latest week = holdout, the week before = calibration, earlier weeks = training |
| Epoch / thresholds | best calibration macro AUPRC epoch; per-label F1 thresholds on calibration (`relgt` `choose_thresholds`) |
| Modes | `backtest` and `predict` |

The examples are exactly `relgt_db`'s: the same user-weeks, the same labels, and
the same sampled rows for a given `--num-neighbors`/`--context-mode`/`--seed`.
The holdout scores are therefore directly comparable.

## Input: one cell sequence per user-week

1. **Database rows.** All 8 tables, loaded with `relgt_db`'s schema
   (`experiments/relgt/database.py`). The `relgt_db` temporal 1/2-hop sampler
   picks `--num-neighbors` rows (default 64, seed user first). Rows sampled more
   than once are kept once.
2. **Task table `UserWeeks`.** This is a virtual table with a foreign key to
   `Users`, a `cutoff` datetime cell and one number cell per label:
   - the row being predicted has its six label cells **masked** (these are the targets);
   - `--history-weeks` (default 8) earlier rows of the same user have their
     labels visible. These are RT's in-context labelled task rows. Only weeks
     whose window `[w, w + 7 days)` closed by the cutoff are used. A history
     label is 1 only if a matching action was **known (row and target entity
     ingested) before the cutoff**, so ingestion lag cannot leak a late action.
3. **Cells.** Every non-key, non-missing value is one cell
   ([`cells.py`](cells.py)). It carries its column id (`"<column> of <table>"`)
   and one of four semantic types:
   - number: the standardized `log1p` value from `relgt_db`;
   - category: a learned embedding per (column, value);
   - text: mean-pooled hashed word pieces, as in `relgt_db`;
   - datetime: `log1p` of the age in days before the cutoff, plus time-of-day
     and day-of-week phase.

   A row with no known values gets one `<row>` marker cell.
4. **Order and length.** Cells are ordered task row, history rows, then
   database rows in sampler order. They are cut at `--max-cells` (default
   512), so the six targets always survive.

**Leakage rules:**
- Rows and edges must be known before the cutoff (sampler).
- Late columns are hidden until their `known_at` time; for example,
  `merged_at`, `merged_by_id` and `state` are hidden until the PR merges.
- Timestamps at or after the cutoff are dropped.
- A foreign-key parent is linked only once that edge is known.
- History labels follow the availability rule above.

## Architecture ([`model.py`](model.py))

Each cell's input is `RMSNorm(column embedding) + RMSNorm(value encoder)`.
Target cells (and cells masked for MTP) use a learned per-type mask embedding
instead of their value. Each of `--num-blocks` blocks then applies pre-norm
residual masked attention, in upstream's order:

| Attention | A cell attends to |
|---|---|
| `col` | cells of the same column (and table), across all rows in the context |
| `feat` | cells of its own row and of the rows it references (foreign-key parents) |
| `nbr` | cells of the rows that reference its row (children) |
| `full` (optional, upstream v1) | every cell |

After the attentions comes a SwiGLU FFN. Attention details follow upstream:
- QK-RMSNorm;
- attention scale `1 / head_dim`;
- the query is multiplied by a learned per-head `scale × log(kv_size)`;
- a `2·sigmoid` output gate;
- residual outputs and decoders are zero-initialized.

The final hidden states of the six target cells go through the shared number
decoder. The resulting logits are trained with BCE.

How the target sees the data:
- **`col` attention** lets the masked `push` cell read last week's `push`
  labels from the history rows.
- **`feat` attention** links it to the user row.
- **`nbr` attention** lets the user row gather the user's PRs, pushes,
  comments, and so on. Their own `feat` attention brings in repositories and
  PRs.

Optionally, `--mtp-weight > 0` adds upstream's masked-token objective as an
auxiliary loss. It hides a `--mtp-prob` share of number cells (including
visible history labels) and reconstructs them with a separate decoder (Huber
loss).

## Differences from upstream

| Upstream | Here | Why |
|---|---|---|
| Frozen sentence-transformer embeddings for column names, categories and text | Learned column/category embeddings; hashed-token text bags | One database, no zero-shot transfer, no extra model download |
| Pretrained on many RelBench DBs, then ICL / fine-tune | Supervised from scratch, optional MTP auxiliary loss | No checkpoint shares this preprocessing |
| Rust BFS context sampler | `relgt_db`'s temporal 1/2-hop sampler + same-user history task rows | Comparability with `relgt_db`, existing cutoff-safety |
| Absolute normalized datetimes | Age before the cutoff + phase | Stationary across weeks, cutoff-relative |
| `flex_attention` + `torch.compile` | Dense boolean masks with SDPA | Runs on Windows/CPU; sequences are ≤ 512 cells |
| Regression/Huber targets | BCE on six binary target cells | Task is multi-label classification |

## Run

```powershell
python -m experiments.relational_transformer.run backtest
python -m experiments.relational_transformer.run backtest --epochs 1 --max-steps-per-epoch 30 --start-cutoff "2026-07-01 16:00:00"   # smoke
python -m experiments.relational_transformer.run backtest --attention col,feat,nbr,full
python -m experiments.relational_transformer.run backtest --history-weeks 0          # ablation: no in-context labels
python -m experiments.relational_transformer.run backtest --mtp-weight 0.1           # + masked-token objective
python -m experiments.relational_transformer.run backtest --num-neighbors 128 --max-cells 1024 --batch-size 16
python -m experiments.relational_transformer.run predict
python -m pytest experiments/relational_transformer -q
```

Main options (see `--help`):
- context: `--num-neighbors`, `--context-mode r0|r1|r2|r3`, `--history-weeks`, `--max-cells`;
- model: `--d-model 128`, `--num-heads 4`, `--num-blocks 4`, `--d-ff 256`, `--attention`;
- training: `--epochs 5`, `--batch-size 32`, `--lr 3e-4`, `--weight-decay 0.01`,
  `--max-steps-per-epoch`, `--amp/--no-amp` (bfloat16 on CUDA);
- `--train-fraction`, `--discard-unused-train`, `--seed`.

## Outputs

Each run writes to `outputs/<command>_<timestamp>_<pid>/` or `--output-dir`.

- `backtest`:
  - `backtest_metrics.json` contains the per-label metrics, macro/micro F1,
    score, thresholds, split boundaries and example counts, all settings, the
    epoch history (BCE, mean cells per sequence, calibration AUPRC, seconds)
    and the selected epoch.
  - `holdout_probabilities.csv` contains `user_id`, `cutoff`,
    `actual_<label>` and `relational_transformer_<label>`.
  - `relational_transformer_model.pt` is the trained model.
  - `cache/` holds the `relgt_db` sampled rows, keyed by a data/settings
    signature.
- `predict`: `predictions.csv`, `probabilities.csv`, `metadata.json` and the
  model checkpoint.
