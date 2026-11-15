# Assignment 2: Predicting GitHub User Activity

Predict, for each user in `test_input.csv`, which of 6 activities (`open_pr`, `open_issue`,
`issue_comment`, `pr_review`, `pr_review_comment`, `push`) they perform during
2026-11-12 00:00 – 11-19 00:00 Taiwan time (2026-11-11 16:00 UTC + 7 days).

## Setup

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
```

`assignment2/` (the course data) is not committed; it must sit at the repo root.

## Run

```bash
python experiments/baseline/run.py
python assignment2/evaluation/evaluate.py --pred experiments/baseline/outputs/none/output.csv --input assignment2/evaluation/test_input.csv
```

Writes `experiments/baseline/outputs/<gap-mode>/output.csv` (rename to
`output_{studentID}.csv` for submission) and `metrics.json` with validation scores. Takes
about 30 s. Options: `--gap-mode {none,mask,drop,downweight,recent}`, `--gap-weight`,
`--seed` and the data-period options below (see `--help`).

**Data period** (all three runners). A week counts as inside a period when its label window
lies fully in `[start, end]` (UTC). `--start`/`--end` bound every stage; `--train-start/-end`,
`--val-start/-end` and `--test-start/-end` override them for one stage (`test` = the weeks
the final models are fitted on). Without `--val-start/--val-end`, validation is the last
`--val-weeks` weeks; training weeks never overlap validation.

```bash
python experiments/baseline/run.py --start 2026-03-01 --end 2026-10-28
python experiments/baseline/run.py --train-start 2026-09-09 --val-start 2026-10-07 --val-end 2026-10-28
```

## Interactive EDA (`eda.py`)

```bash
python eda.py        # then open http://127.0.0.1:8050 (startup ~2 s)
```

A Dash + Plotly app over the cohort's activity, with the same activity definitions and
test-aligned weeks as the baseline. Global filters (date range, all / test / non-test users,
activities) drive the tabs: daily activity per table with the outages shaded, weekly label
rates and co-occurrence, persistence (P(active) by weeks since last active, week-to-week
transitions), per-user concentration, time-of-day heatmaps, top repositories, data-quality
checks (staleness before the test week, how late rows appear, null rates), a per-table column
explorer and a per-user timeline. The shifted test selection rule reproduces the 3,960 test
users exactly.

## Baseline (`experiments/baseline/run.py`)

**Samples.** One row per (cohort user, week start `s`), with weeks aligned to the test week
(Wednesday 16:00 UTC). Label = the user does the activity in `[s, s + 7d)`, using
`created_at` / `submitted_at` (reproduces `examples/sample_truth.csv` at >99.3% agreement).
Rows are kept only for users meeting the test selection rule shifted to that week
(≥ 5 records in the 90 days before `s − 14d4h`); at the test week this rule yields exactly the
3,960 test users.

**No future leakage.** The data ends ~10 days before the test week (pushes ~16 days), so each
table gets a lag = test start − its last timestamp, and features of a week starting at `s`
only use rows visible before `s − lag` (by `first_seen_at` for PRs/issues/reviews, `created_at`
otherwise). `closed_at` is used only when it is before the cutoff. Training and validation
are split by time, and training labels end before the validation weeks start.

**Features (60).** For each of the 6 activities and for *received* activity (comments and
reviews other cohort users leave on the user's own issues/PRs, joined via
IssueComments → Issues and PRReviews/PRReviewComments → PullRequests): counts in the last
7/14/30/90 days and overall, days since last event, active days and distinct repositories in
the last 30 days. Plus the number of the user's PRs and issues still open, days since any
activity and the number of activity types used in the last 30 days.

**Model.** One LightGBM binary classifier per activity. Trained on weeks 2026-01-14 … 09-30,
validated on 10-07, 10-14, 10-21 (early stopping, per-activity threshold maximising F1). The
final models are retrained on all labelled weeks; the validation share of predicted positives
is reused as the decision rule because retraining shifts the probability scale.

**Validation score** (Score = (macro-F1 + micro-F1) / 2, weeks with all 6 labels):

| Method | 10-07 | 10-14 | Mean |
|---|---|---|---|
| Active in the last 30 days (rule) | 37.26 | 33.44 | 35.35 |
| LightGBM baseline | 52.11 | 47.22 | **49.66** |

The thresholds are tuned on the same validation weeks, so the LightGBM number is somewhat
optimistic.

## Experiment: training weeks hit by collection outages

Daily cohort event counts show outages (listed in `GAPS` in `run.py`):

- **Non-push, 2026-06-01 → 09-07:** issues, PRs, comments and reviews drop to ~5–20% of
  their usual level and end with a backfill burst (~1,400 events/day on 9/4–9/6).
- **Push:** almost nothing 08-18 → 09-12 (≤ ~20% of the usual volume); days missing on 10-07, 10-15 → 10-18 and 10-23.
- October is quieter than spring for every activity but steadily so; the test week is in this
  regime, so it is not treated as an outage.

A training week is *affected* if its label week overlaps an outage, or if more than 25% of its
30-day feature window does. Modes (`--gap-mode`):

| Mode | Training weeks used | Validation score, mean of 3 seeds (min–max) |
|---|---|---|
| `none`: keep everything | 38 | **49.60** (49.46–49.69) |
| `downweight`: affected weeks weight 0.2 | 38 | 49.44 (49.24–49.74) |
| `mask`: drop a week only for labels whose label week has an outage | 23 non-push / 32 push | 49.25 (49.09–49.44) |
| `drop`: drop affected weeks | 19 | 49.19 (48.94–49.46) |
| `recent`: only weeks after the summer outage (09-09 on) | 4 | 49.17 (49.14–49.19) |

**Result: none of the treatments beats training on all weeks.** All differences (≤ 0.4) are
inside the seed-to-seed spread (up to 0.5), so `none` stays the default. A likely reason: the
model mostly ranks users by their own recent activity, and the decision rule is set on recent
weeks (threshold / predicted share), so a week-wide drop in activity shifts every user's
features and labels together and still gives useful ranking examples. Even 4 post-outage weeks
reach almost the same score, so more weeks add little.

Caveats: only 2 validation weeks (~4.5k users each) are scored, and those weeks contain push
outages themselves (10-07, 10-15 → 10-18), so push labels there are slightly understated.

## Experiment: GNN, all labels vs push only (`experiments/gnn/run.py`)

```bash
python experiments/gnn/run.py --task all    # one model, 6 outputs -> outputs/all/output.csv
python experiments/gnn/run.py --task push   # one model, push only -> outputs/push/push.csv
```

Needs PyTorch (no PyTorch Geometric; message passing is written with `index_add_`). Uses the
GPU if available, ~1 min per run.

**Graph per sample week** (same weeks, cutoffs, lags, eligibility and split as the baseline):

- *User nodes*: the 9,000 cohort users, with the baseline's 60 features (log1p, standardised
  on training weeks).
- *Repository nodes*: repos touched in the 90 days before the cutoff (~25k in the test week),
  with per-edge-type 30/90-day event counts, number of distinct cohort users, and whether the
  repo belongs to an organisation (`Repositories.org_id`).
- *7 edge types user–repo*: open_pr (PullRequests), open_issue (Issues), issue_comment
  (IssueComments), pr_review (PRReviews), pr_review_comment (PRReviewComments), push (Pushes),
  and received (IssueComments → Issues, PRReviews / PRReviewComments → PullRequests).
  Edge features: log events in the last 7/30/90 days, days since the last one / 90.

**Model.** MLP encoders (dim 64), then 2 rounds of per-edge-type GraphSAGE-style message passing
user → repo → user (mean aggregation, residual + LayerNorm), then an MLP head. One week is one
full-graph batch. AdamW (lr 2e-3), early stopping on validation loss (patience 5), thresholds and
the final retrain on all weeks as in the baseline. The two variants differ only in the head and
loss: `all` has 6 outputs and averages the 6 BCE losses, `push` has 1 output.

**Push F1 on validation** (mean of weeks 10-07 and 10-14; 5 seeds for the GNNs):

| Model | Push F1, mean (min–max) | Overall score |
|---|---|---|
| GNN, push only | 0.6206 (0.6168–0.6268) | – |
| GNN, all 6 labels | **0.6240** (0.6206–0.6269) | 48.03 (47.16–48.63) |
| LightGBM baseline (seed 0, for reference) | 0.6293 | 49.66 |
| Rule: pushed in the last 30 days | 0.5960 | 35.35 |

**Result: the multi-task GNN is slightly better at push than the push-only GNN** (+0.003 F1 on
average and higher for every seed), so learning the other five activities helps the shared
user representation rather than hurting push. The gap is small, about one seed standard
deviation, so it is a weak effect. Both GNNs stay below LightGBM on push and on the overall
score. Early stopping picks epoch 2–7, i.e. the GNN overfits within a few passes over 38 weekly
graphs.

## Experiment: Relational Graph Transformer (`experiments/relgt/run.py`)

```bash
python experiments/relgt/run.py    # -> outputs/default/output.csv, ~35 min on an RTX 3060
```

A port of the official RelGT ([Dwivedi et al. 2025](https://arxiv.org/abs/2505.10960),
[code](https://github.com/snap-stanford/relgt)) in plain PyTorch. Same examples, labels,
eligibility rule, split, thresholds and final retrain as the baseline.

- *Graph*: every row of all 8 tables is a node, every foreign key an undirected edge (2.6M
  nodes, 5.0M edges). `PullRequests.merged_by_id` is left out (known only after the merge).
- *No leakage*: a node of table T is visible to week `s` only if its row time is before
  `s - lag_T`, the staleness that table has at the test week.
- *Tokens*: the user plus K-1 = 99 nodes sampled uniformly from its visible 1- and 2-hop
  neighbours. Each token combines node type, hop, time relative to `s`, the row's columns
  (per-table residual MLP; merged / closed / dismissed only as "before the cutoff" flags) and a
  GIN encoding of the sampled subgraph.
- *Model*: local transformer over the tokens with attention readout to the user, global
  attention to an EMA k-means codebook, feed-forward block, 6-output head. Channels 128,
  1 layer, 4 heads, dropout 0.3, Adam, batch 512. Tokens are cached in `experiments/relgt/cache/`.

Differences from the official code (torch_frame / GloVe column encoders replaced by
hand-picked columns, neighbour caps applied after the time filter, smaller model) are listed
in the script's docstring.

**Validation score** (seed 0, same weeks as above):

| Run | Val mean AP (best epoch) | 10-07 | 10-14 | Mean |
|---|---|---|---|---|
| Paper settings: lr 1e-4, 10 epochs (`outputs/lr1e-4_e10`) | 0.330 (10) | 46.48 | 42.32 | 44.40 |
| lr 3e-4, 20 epochs (default, `outputs/default`) | 0.356 (20) | 47.93 | 44.12 | **46.02** |

**Result: RelGT learns from the raw rows alone but stays below LightGBM (49.66) and the GNN
(48.03).** With lr 1e-4 it was still improving after 10 epochs; at 3e-4 validation AP levels
off after ~14 epochs. Push F1 is 0.613 (LightGBM 0.629). The model sees 99 tokens sampled
uniformly over the user's whole history, so recent activity, the strongest signal (see
Persistence in the EDA), is only partly visible. In the baseline it is counted exactly over
7/14/30 days. Only one seed was run, so differences of about 0.5 points are within noise.

## Data notes

- Only cohort users' own activity is collected, so *received* activity only covers
  interactions between cohort users and is sparse.
