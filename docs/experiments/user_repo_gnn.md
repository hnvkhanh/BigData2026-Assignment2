# Experiment: GNN (user–repo graph), all labels vs push only (`experiments/gnn/run.py --graph-structure user-repo`)

This was the default before the ego model. Every command in this section needs
`--graph-structure user-repo`.

```bash
python experiments/gnn/run.py --graph-structure user-repo --task all    # one model, 6 outputs -> outputs/all/output.csv
python experiments/gnn/run.py --graph-structure user-repo --task push   # one model, push only -> outputs/push/push.csv
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

**Time encoding** (`--time-enc edge`, `--time-dim 16`). Without it the model sees time only
through the 7/30/90-day counts and a linear "days since last / 90". With it, the days since the
last and the first event of each user–repo edge (within the 90-day window) go through RelGT's
sinusoidal `PositionalEncoding` and a linear layer shared by the edge types, and the two
encodings are appended to the edge features of every message.

```bash
python experiments/gnn/run.py --graph-structure user-repo --time-enc edge                 # -> outputs/all_time/
python experiments/gnn/run.py --graph-structure user-repo --actions push --time-enc edge  # -> outputs/push_time/
```

**Results on the revised assignment** (test week 2026-11-21, 39 training weeks; validation
weeks 10-17, 10-24 and 10-31, but only 10-17 has all 6 labels, so the scores below are for that
week; 5 seeds each, runs `ref_default`/`ref_s1-4`, `time_s0-4`, `push`/`push_s1-4`,
`push_time_s0-4`):

| Model | Push F1, mean (min–max) | Overall score, mean (min–max) |
|---|---|---|
| GNN, all 6 labels | 0.6248 (0.6237–0.6274) | 47.42 (46.68–47.90) |
| GNN, all 6 labels, time encoding | 0.6248 (0.6214–0.6268) | **47.76** (47.17–48.41) |
| GNN, push only | **0.6253** (0.6230–0.6300) | – |
| GNN, push only, time encoding | 0.6247 (0.6205–0.6297) | – |

Mean F1 per activity, all 6 labels, without → with time encoding: open_pr 0.399 → 0.399,
open_issue 0.438 → 0.448, issue_comment 0.344 → 0.351, pr_review 0.368 → 0.382,
pr_review_comment 0.360 → 0.355, push 0.625 → 0.625.

**Result: the time encoding gives at most a small gain, within seed noise.** The overall score
rises by 0.34 on average (+0.99, +0.82 and +0.04 for seeds 3, 2 and 4; −0.07 and −0.07 for seeds 0
and 1), mostly from open_issue, issue_comment and pr_review; push does not change at all. The
seed-to-seed spread (~1.2 points) is larger than the gain and only one week is scored, so this is
not a reliable improvement. A likely reason: the 60 user features already count recent activity
exactly over 7/14/30/90 days, and the edges already carry windowed counts and recency, so a finer
encoding of edge age adds little. Push-only and all-label GNNs are now equal on push (0.625 vs
0.625), so the earlier multi-task gain (+0.003) does not hold up on the new data. Early stopping
still picks epoch 2–17.

Earlier setup, for reference (test week 11-12, validation weeks 10-07 and 10-14, not rerun so
not directly comparable): GNN all labels push F1 0.6240 / score 48.03, GNN push only 0.6206,
LightGBM 0.6293 / 49.66, rule "pushed in the last 30 days" 0.5960 / 35.35.

**Short history** (`--history-days 14`). The model only sees rows of the last 14 days before each
cutoff, in training, validation and test alike: user features are recomputed from those rows
(39 instead of 60: the 30d / 90d / all-time counts are dropped; open PRs / issues count only those
opened in the window), the user–repo edges and repo features cover 14 days (edge counts over 7 and
14 days). Eligibility and labels are unchanged, so it is scored on the same users.

```bash
python experiments/gnn/run.py --graph-structure user-repo --history-days 14   # -> outputs/all_h14/
```

Seed 0, validation week 10-17, 14 days vs full history (`ref_default`, same seed): overall score
46.58 vs 47.90; F1 open_pr 0.387 vs 0.407, open_issue 0.441 vs 0.457, issue_comment 0.344 vs
0.344, pr_review 0.366 vs 0.371, pr_review_comment 0.341 vs 0.365, push 0.610 vs 0.625. Best
epoch 7 in both. **Result: 14 days keep most of the signal. The score drops 1.3 points, below
the lowest of the 5 full-history seeds (46.68), but this is one seed scored on one week.**
