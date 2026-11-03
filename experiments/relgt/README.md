# RelGT-inspired graph branch

This experiment adds a relational graph model alongside the existing LightGBM
baseline. It is an assignment-specific implementation inspired by the
[Relational Graph Transformer (RelGT)](https://github.com/snap-stanford/relgt)
paper and its public implementation; it is **not** a drop-in run of Stanford's
RelBench training code.

The design references the upstream [`model.py`](https://github.com/snap-stanford/relgt/blob/main/model.py),
[`encoders.py`](https://github.com/snap-stanford/relgt/blob/main/encoders.py),
and [`local_module.py`](https://github.com/snap-stanford/relgt/blob/main/local_module.py):
typed/time-aware neighborhood tokens, local self-attention, and global
centroid-attention readout are adapted to the assignment-specific snapshots.

## What this implementation models

At each historical weekly cutoff, it creates a temporal heterogeneous graph
from cohort actions in the configured lookback window (30 days by default):

- Target users connect to typed Pull Request, Issue, and Repository nodes through
  six activity relations (open, comment, review, inline review comment, push).
- PR and Issue nodes connect to their Repository at hop 2. Repository nodes are
  also reached directly by Push relations.
- The target node and sampled neighbors are tokenized with entity features,
  node type, relation type, hop distance, relative time, and local structure.
  Degree and diversity features use all cohort events available at the cutoff,
  independent of which users are included in a prediction batch.
  At most four recent entity neighbors per activity relation and 24 repositories
  are retained for each user.
- A local Transformer encodes each sampled neighborhood. Learnable centroids
  attend to that neighborhood, and a global attention readout combines the
  centroid representation with the target user representation.
- One model predicts all six next-week labels with binary cross-entropy.

This follows RelGT's five-component tokenization and local/global attention
design, with a much smaller sampled graph and PyTorch-only implementation so it
can run on Apple Silicon MPS. It does not reproduce RelGT's vector-quantized
centroid codebook, full RelBench feature encoders, or all cross-table edges.
The branch is therefore a transparent RelGT-inspired experiment, not a claim of
an exact RelGT reproduction. For a version that does all three and reads the
database tables directly, use `--temporal-model relgt_db` (next section).

## RelGT on the relational database (`relgt_db`)

`--temporal-model relgt_db` applies the upstream RelGT method to the database
itself instead of to hand-built activity snapshots. It is a PyTorch-only port
of [`model.py`](https://github.com/snap-stanford/relgt/blob/main/model.py),
[`local_module.py`](https://github.com/snap-stanford/relgt/blob/main/local_module.py),
[`codebook.py`](https://github.com/snap-stanford/relgt/blob/main/codebook.py),
[`encoders.py`](https://github.com/snap-stanford/relgt/blob/main/encoders.py),
and the sampler in [`utils.py`](https://github.com/snap-stanford/relgt/blob/main/utils.py).
It needs no PyTorch Geometric, torch_frame, or RelBench.

**From database to graph** ([`database.py`](database.py)). All 8 tables are
read, using a schema declared from `assignment2/data/db/SCHEMA.md`
(`GITHUB_SCHEMA`: primary keys, foreign keys, row-time columns, and "known
only later" columns). As in RelBench's `make_pkey_fkey_graph`:

- every row is a node (2.58M), its table is the node type, and every
  foreign key is an undirected edge (5.1M), e.g.
  `PRReviews.pr_id -> PullRequests`, `Issues.pull_request_id -> PullRequests`,
  `PRReviewComments.in_reply_to_id -> PRReviewComments`;
- every other column is typed and encoded: numbers (signed log, standardized),
  booleans and columns with at most 512 values (categorical), free text such as
  titles, bodies, branches, paths, and diff hunks (hashed word pieces),
  `Issues.labels` (multicategorical), and timestamps (whether known, age at the
  cutoff, hour of day, day of week). GitHub ids, event ids, commit hashes,
  `head_repo_id`/`org_id` (not row indices), constant columns, and
  `Pushes.ref` (a copy of `ref_type` + `branch`) are dropped.

**Leakage rules.** A row is known from its row time (`first_seen_at` or
`created_at`), and an edge once both of its rows are known. The late key
`merged_by_id` is linked only from `merged_at`. Late values (`merged_at`,
`closed_at`, `Issues.state_reason` (known from `closed_at`), `dismissed_at`)
are masked for each token unless they were known before that sample's cutoff.
Every comparison is strictly earlier than the cutoff. Unlike upstream, the
fallback tokens below are also drawn only from rows known before the cutoff.

**Tokens** ([`db_sampler.py`](db_sampler.py)). The seed is the cohort user's
`Users` row at the weekly cutoff. As upstream, the sampler collects the 1-hop
neighbors known before the cutoff (at most 5,000) and their neighbors as 2-hop
(at most 1,000 per 1-hop node). It then draws `K - 1` tokens uniformly from
both, with replacement when there are fewer than `K - 1`. A seed with no
neighbors gets random known rows marked hop 3. The full history before the
cutoff is used, with no lookback window. Each token has the RelGT elements:
the row's encoded columns, its table, its hop, its age (cutoff minus row time),
and a GIN positional encoding over the edges among the sampled rows.

**Model** ([`db_model.py`](db_model.py)). As upstream: per-table column
encoders feed a ResNet (torch_frame-style). The five elements are each
layer-normalized, concatenated, and mixed. A local Transformer over the `K`
tokens reads out to the seed through attention. Global attention goes to an
EMA vector-quantized codebook of seed representations (`full` = both). A
batch-norm feed-forward network and an MLP head give six logits. The model is
trained with BCE, gradient clipping at 1.0, and Adam; the epoch with the best
calibration-week macro AUPRC is kept, as upstream keeps its best validation
epoch.

**Same task as `relgt`.** The user-weeks, labels, five-action 14-day
eligibility filter, weekly cutoffs, 70/15/15 chronological split, per-label
threshold calibration, and output files are the same as in the `relgt` mode.
On the current release, the 102,524 user-weeks and labels match the `relgt`
cache exactly, so holdout scores compare directly.

**Differences from upstream**, adapted to this task and machine:

- Text uses a trainable hashed word-piece vocabulary shared by all text
  columns, instead of frozen GloVe sentence embeddings.
- The defaults are smaller: `K=128` instead of 300, 64 channels instead of 512,
  and 512 centroids instead of 4,096. The learning rate is 1e-3 (upstream used
  1e-4) to match this repository's other graph runs.
- The codebook assignment buffer covers `Users` rows (the seeds) rather than
  every node.
- Attention dropout is applied only in training; upstream also applies it at
  evaluation.
- One head predicts all six labels.

**Relational-context configurations** (`--context-mode`). Only the token
candidates change; everything else is identical:

| Mode | Candidates for the K-1 tokens |
| --- | --- |
| `r0` | the user's own rows only (hop 1) |
| `r1` | `r0` plus other rows that directly touch them (hop 2, no `Repositories`) |
| `r2` (default) | upstream full 1/2-hop context, including `Repositories` |
| `r3` | control: random rows known before the cutoff, as many as `r2` has candidates (capped at K-1), marked hop 3 |

Run it:

```bash
python -m experiments.relgt.run backtest --temporal-model relgt_db
python -m experiments.relgt.run backtest --temporal-model relgt_db --context-mode r0
python -m experiments.relgt.run predict --temporal-model relgt_db
```

Useful options: `--num-neighbors` (K, default 128), `--hidden-dim`,
`--num-heads`, `--num-layers` (local Transformer layers), `--num-centroids`,
`--conv-type full|local|global`, `--ablate none|type|hop|time|tfs|gnn` (drop
one token element, as in the upstream ablations), `--text-buckets`, `--db-lr`,
`--weight-decay`, `--max-steps-per-epoch`, `--epochs`, `--batch-size`,
`--dropout`, `--train-fraction`, and `--seed`. `--user-embedding` is rejected
because RelGT has no per-entity id embedding. The fusion modes
(`relgt_lightgbm`, `hybrid`, ...) still use the `relgt` snapshot branch.

Backtests write `backtest_metrics.json` (holdout metrics, settings, and
per-epoch history), `holdout_probabilities.csv` (branch `relgt_db`, usable
with `exam_fail_case.py --model-branch relgt_db`), and `relgt_db_model.pt`.
Predictions write `predictions.csv`, `probabilities.csv`, `metadata.json`,
and the model. Token sets are cached in the run folder's `cache/`, keyed by
the data files, start cutoff, `K`, and seed; reuse `--output-dir` to skip
sampling. Loading and graph construction take about 15 s, and sampling about
0.3 ms per user-week on one CPU core.

## GNN comparison

The standalone `gnn` mode provides a message-passing baseline over the same
cutoff-safe snapshots. It embeds each typed neighbor, applies relation-specific
messages, mean-aggregates them into a user root, and updates the neighbors from
that root over `--num-layers` steps. Hop IDs distinguish direct activity nodes
from sampled repository context. The current snapshot format does not retain
explicit entity-to-repository edge indices, so this experiment uses the
user-centered sampled edges rather than reconstructing those links. It requires
only the PyTorch dependency already used by this experiment and does not change
the default RelGT model or ensemble modes.

Historical snapshots require both the event row and its referenced node to be
available before the cutoff. PRs, Issues, and PRReviews use `first_seen_at`;
IssueComments, PRReviewComments, and Pushes use `created_at`. Action timestamps
are used for history windows and future labels. PR conversations are excluded
from the `open_issue` label, matching the assignment definition.

## Environment

The root `requirements.txt` contains the existing data and LightGBM
dependencies. This branch additionally requires PyTorch:

```bash
python -m pip install -r requirements.txt -r experiments/relgt/requirements.txt
```

On an Apple Silicon Mac, a recent standard PyTorch install includes MPS support.
At startup the script reports the number of visible CUDA GPUs and, when
`nvidia-smi` is available, each GPU's utilization and memory use. `--device
auto` selects the CUDA GPU with the lowest reported utilization (breaking ties
by memory use); if CUDA is unavailable, it selects MPS, then CPU. You can
explicitly use `--device cuda` to select the least-loaded CUDA GPU, or choose
`--device mps` / `--device cpu`. If `nvidia-smi` cannot report utilization, it
warns and uses free GPU memory as the selection fallback. The code, including
`relgt_db`, does not require PyTorch Geometric, torch_frame, RelBench, or a
Stanford RelGT clone.

The optional `hybrid` mode also requires the pinned S2P2/EasyTPP dependency:

```bash
python -m pip install -r experiments/lightgbm_baseline/s2p2-requirements.txt
```

In hybrid mode RelGT learns from the temporal graph snapshots while S2P2 learns
from each user's six-mark action sequence. Both branches train on the same
chronological training weeks. A shared probability-fusion weight is selected
from `0.0` to `1.0` on calibration weeks, along with per-label thresholds; the
holdout is used only for final evaluation. S2P2 uses CPU when CUDA is
unavailable because its upstream implementation warns against MPS. On CUDA,
automatic device selection chooses the least-utilized GPU.

The `relgt_lightgbm` mode combines RelGT with the LightGBM baseline and uses
the root `requirements.txt` dependencies; it does not require EasyTPP. It builds
the baseline's cutoff-safe tabular features, aligns them to RelGT user-week
examples, and selects a shared probability-fusion weight and label thresholds
using calibration weeks only. The LightGBM features include 7-day total
actions, per-action shares over 7- and 30-day windows, 7-day per-action
frequencies, and median daily activity over both windows. Raw 30-day count
features are excluded.
The baseline's 90-day eligibility filter remains, but 90- and 365-day windows
are not used as tabular model features. The frequency-only neural network in
the baseline remains restricted to per-action frequency inputs.

The `relgt_lightgbm_s2p2` mode combines all three branches. It requires both
the base/RelGT dependencies and the optional pinned S2P2/EasyTPP dependency.
The three non-negative probability weights are selected on a simplex grid with
0.25 increments using calibration weeks only, together with per-label
thresholds. Alternatively, set all weights explicitly with
`--fusion-weights relgt=0.4,lightgbm=0.4,s2p2=0.2`. Manual weights must be
non-negative and sum to 1. In either mode, thresholds are calibrated using only
the calibration split. These options work for both backtest and predict. The
S2P2 dependency is checked before training starts.

## Run a chronological backtest

From the project root:

```bash
python -m experiments.relgt.run backtest
```

The default settings train for five epochs, calibrate per-label F1 thresholds
on later weeks, and score a final chronological holdout. Every invocation
creates a unique timestamped run folder, such as
`experiments/relgt/outputs/backtest_20261102_001234_12345/`. Its report and
temporal graph cache are kept inside that folder, so concurrent runs do not
overwrite one another. The backtest saves calibration-thresholded holdout
metrics in `backtest_metrics.json` and per-user-week actual labels and model
probabilities in `holdout_probabilities.csv`.

Analyze a saved holdout with `exam_fail_case.py` (use the run folder printed
when the backtest starts):

```bash
python exam_fail_case.py \
  --run-dir experiments/relgt/outputs/backtest_20261102_001234_12345 \
  --model-branch relgt
```

For the standalone GNN, use `--model-branch gnn`. Add `--no-dashboard` to
export failure-case CSVs without launching the interactive dashboard.

For a quick first run with fewer training epochs:

```bash
python -m experiments.relgt.run backtest --epochs 1 --batch-size 128
```

Run the standalone GNN comparison with:

```bash
python -m experiments.relgt.run backtest \
  --temporal-model gnn \
  --epochs 5 \
  --hidden-dim 64 \
  --num-layers 2
```

It uses the same chronological split, calibration thresholds, and sampled
graph cache as RelGT. The report identifies the message-passing architecture,
and the run saves its weights as `gnn_model.pt`.

## Lookback ensembles

The `relgt_ensemble` and `gnn_ensemble` modes independently train two models
from the selected lookback windows, then choose their probability-fusion weight
and per-label thresholds using calibration weeks only. The default pair is
30 and 14 days; `--ensemble-lookbacks` accepts any two distinct values from
14, 30, and 90. Both branches use the same users, labels, and chronological
split; eligibility is determined using the longer selected window.

```bash
python -m experiments.relgt.run backtest --temporal-model relgt_ensemble
python -m experiments.relgt.run backtest --temporal-model gnn_ensemble
python -m experiments.relgt.run backtest \
  --temporal-model relgt_ensemble \
  --ensemble-lookbacks 90,14
```

Backtests save one checkpoint per lookback and branch probabilities named
`relgt_30d_<label>` / `relgt_14d_<label>` (or `gnn_*`), plus `fused_<label>`,
in the shared holdout CSV format. `backtest_metrics.json` includes holdout
metrics and the selected weights for each branch. To set the first
branch's weight manually (the second branch receives the remainder), pass
`--fusion-weight` to either backtest or predict:

```bash
python -m experiments.relgt.run backtest \
  --temporal-model relgt_ensemble \
  --fusion-weight 0.7
```

When omitted, the weight is still selected on calibration weeks. Thresholds
are always calibrated on calibration probabilities using the selected or
manually specified blend. The same ensemble
modes are available with `predict`. Analyze the fused holdout or either branch
with `exam_fail_case.py --run-dir <run-folder> --model-branch fused` or, for
example, `--model-branch relgt_14d`.

Run the combined RelGT + S2P2 model with:

```bash
python -m experiments.relgt.run backtest \
  --temporal-model hybrid \
  --device auto \
  --epochs 1 \
  --s2p2-epochs 1
```

The run report includes the calibration-selected RelGT/S2P2 fusion weights;
`--fusion-weight` can manually set the RelGT weight (S2P2 receives the
remainder). The same option is available for RelGT + LightGBM.
`holdout_probabilities.csv` records the aligned RelGT, S2P2, and fused
probabilities for each holdout user-week, and the run folder contains both
trained model weights. S2P2 training settings can be adjusted with
`--s2p2-batch-size`, `--s2p2-max-sequence-length`, `--s2p2-hidden-size`,
`--s2p2-state-size`, `--s2p2-num-layers`, and `--s2p2-integral-samples`.

Run RelGT combined with LightGBM with:

```bash
python -m experiments.relgt.run backtest \
  --temporal-model relgt_lightgbm \
  --epochs 1 \
  --n-estimators 300
```

Fusion is calibrated on the same chronological calibration weeks used for
threshold selection; the holdout is reserved for final evaluation. The report
includes the fused score and separate RelGT and LightGBM holdout metrics.
`holdout_probabilities.csv` contains the aligned branch and fused
probabilities. The run saves `relgt_model.pt` and one
`lightgbm_<label>.txt` file per trained label. LightGBM settings can be
adjusted with `--n-estimators`, `--learning-rate`, `--num-leaves`,
`--min-child-samples`, `--subsample`, `--colsample-bytree`, and `--reg-lambda`.

Run all three models together with:

```bash
python -m experiments.relgt.run backtest \
  --temporal-model relgt_lightgbm_s2p2 \
  --epochs 1 \
  --s2p2-epochs 1 \
  --n-estimators 300
```

The backtest report records all three fusion weights and the individual branch
holdout metrics. `holdout_probabilities.csv` includes RelGT, LightGBM, S2P2,
and fused probabilities. Both secondary branch weights and the RelGT weights
are saved in the run folder.

Each run gets its own cache to keep concurrent runs isolated. Use
`--output-dir` to choose a specific folder when you want to organize or rerun
an experiment:

```bash
python -m experiments.relgt.run backtest \
  --epochs 5 --hidden-dim 64 --num-heads 4 --num-layers 2 \
  --output-dir experiments/relgt/outputs/baseline_64d
```

Other useful options include `--start-cutoff`, `--device`, `--batch-size`,
`--hidden-dim`, `--num-heads`, `--num-layers`, `--dropout`, and `--seed`.
`--hidden-dim` must be divisible by `--num-heads`.

## Generate predictions

Use the latest assignment data release and its `test_input.csv`:

```bash
python -m experiments.relgt.run predict
```

To train the same GNN architecture and generate predictions, pass
`--temporal-model gnn`; the output folder contains `gnn_model.pt` and metadata
identifying the architecture:

```bash
python -m experiments.relgt.run predict --temporal-model gnn
```

By default this uses the assignment cutoff `2026-11-11 16:00:00 UTC` and
creates a unique timestamped folder under `experiments/relgt/outputs/`. It
writes:

- `predictions.csv`: evaluator-format binary labels.
- `probabilities.csv`: unthresholded per-label
  probabilities, for later comparison or probability-level fusion.
- `model.pt` for RelGT-only runs, or `relgt_model.pt` plus the selected
  secondary branch's weights for combined runs, and `metadata.json`.

Select the RelGT + LightGBM ensemble for prediction with:

```bash
python -m experiments.relgt.run predict \
  --temporal-model relgt_lightgbm \
  --output-dir experiments/relgt/outputs/relgt_lightgbm_prediction
```

The output includes fused predictions and probabilities, separate
`relgt_<label>_probability` and `lightgbm_<label>_probability` columns,
`relgt_model.pt`, LightGBM model files, and metadata containing the calibrated
fusion weights and LightGBM settings.

To generate predictions with all three models, use:

```bash
python -m experiments.relgt.run predict \
  --temporal-model relgt_lightgbm_s2p2 \
  --output-dir experiments/relgt/outputs/relgt_lightgbm_s2p2_prediction
```

The probabilities file includes branch-specific RelGT, LightGBM, and S2P2
columns as well as fused probabilities. The output folder contains
`relgt_model.pt`, `s2p2_model.pt`, LightGBM model files, and metadata with the
calibration-selected three-branch weights.

Training and calibration examples for prediction are restricted to weekly
samples whose complete seven-day label window ends on or before the requested
prediction cutoff. RelGT's graph snapshots use only actions and entity rows
available before each sample cutoff.

Override the cutoff, test-user input, or output location with flags:

```bash
python -m experiments.relgt.run predict \
  --cutoff "2026-11-11 16:00:00" \
  --test-input assignment2/evaluation/test_input.csv \
  --output-dir experiments/relgt/outputs/candidate
```

Pass an explicit `--output-dir` to both the prediction and evaluator commands
when you want a stable, known path. Omit it for timestamped, collision-free
outputs when launching multiple experiments.

Validate the output format and user IDs:

```bash
python assignment2/evaluation/evaluate.py \
  --pred experiments/relgt/outputs/predictions.csv \
  --input assignment2/evaluation/test_input.csv
```

Run final prediction only after the assignment's latest data release is
available. Use a separate output directory to keep candidate metrics and
predictions from overwriting one another.

## Compare with LightGBM

Compare LightGBM, RelGT, RelGT+S2P2, RelGT+LightGBM, and
RelGT+LightGBM+S2P2 runs using the same cutoff dates and chronological split.
Compare the assignment score
`(macro-F1 + micro-F1) / 2` on the untouched holdout. Combined models are
experiments, not assumed improvements; retain one only if it improves the
measured holdout score over the individual branches.

## Weekly scoring

Weekly cutoffs now start at `2026-03-11 16:00` (the test window's weekday and
hour). Holdout metrics (`calculate_metrics(..., cutoffs=...)`) score every
holdout week as a separate submission with `assignment2/evaluation/evaluate.py`
and report the mean over weeks, with the per-week scores under `per_week`.
The same applies to the GNN, DB, GRU, action-specialist and chunk experiments.
