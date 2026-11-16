# Experiment: ego-graph GNN, the default of `experiments/gnn/run.py`

```bash
python experiments/gnn/run.py                    # -> experiments/gnn/outputs/all_ego/output.csv
python experiments/gnn/run.py --user-features    # -> experiments/gnn/outputs/all_ego_uf/
```

The relation-aware GNN of `archive/old_experiments/relgt_inspired` (`gnn_model.py`), ported to the
current protocol. It uses the same weeks, per-table lags, eligibility, split, thresholds and
final retrain as the baseline. Its old backtest score (0.596) used its own user selection and
holdout weeks, so that number can't be compared with the results here (see *Rerun of the old
backtest* below).

- *One star graph per example* (eligible user, week). It has a learned user node and 48 tokens
  from the last 14 days (`--ego-lookback`) before each table's cutoff:
  - the 4 latest objects of each activity: the PR or issue acted on, or for a push its
    repository (hop 1);
  - the 24 latest repositories of any activity (hop 2, relation "member").

  A row is used only if both the row and the PR, issue or repo it points to are visible before
  that table's cutoff.
- *Token*: a learned entity-id embedding (599k PRs, issues and repos), node type, relation and
  hop, plus 4 counts:
  - the user's events on it;
  - active days;
  - events on it by all cohort users;
  - users per relation.

  A sinusoidal encoding of the log age is added, then an MLP mixes all of it.
- *Message passing* (2 rounds, dim 64):
  - token → user, with one weight per relation;
  - user → token;
  - push repo ↔ the member token of the same repo.

  Each is mean-aggregated and added to a self transform. The head reads the user node plus the
  mean of all nodes.
- *Training*: mini-batches of 256, AdamW lr 1e-3, early stopping on validation loss (patience 5,
  up to 20 epochs).
- The old model has no user features. `--user-features` adds an MLP of the baseline's 60
  features to the user node.

Over all weeks, 28% of the (user, week) examples have no activity in the 14-day window. Without
`--user-features`, these users all look the same to the model: they are scored from the learned
user node alone.

**Results** (validation week 10-17, 5 seeds, runs `all_ego_s0-4` and `all_ego_uf_s0-4`):

| Model | Overall score, mean (min–max) | Push F1, mean (min–max) | Best epoch |
|---|---|---|---|
| Ego GNN (faithful port) | 46.29 (45.82–46.69) | 0.6086 (0.6079–0.6095) | 1 for every seed |
| Ego GNN + user features | **48.22** (47.73–48.54) | **0.6277** (0.6252–0.6312) | 1–2 |
| User–repo GNN, for reference ([user_repo_gnn.md](user_repo_gnn.md)) | 47.42 (46.68–47.90) | 0.6248 (0.6237–0.6274) | 2–17 |

Mean F1 per activity:

| Activity | Ego GNN | Ego GNN + user features | User–repo GNN |
|---|---|---|---|
| open_pr | 0.366 | 0.403 | 0.399 |
| open_issue | 0.413 | 0.464 | 0.438 |
| issue_comment | 0.349 | 0.347 | 0.344 |
| pr_review | 0.369 | 0.376 | 0.368 |
| pr_review_comment | 0.360 | 0.368 | 0.360 |
| push | 0.609 | 0.628 | 0.625 |

**Result: the ego GNN with user features is the best GNN so far.**
- It scores 48.22 on average, 0.8 above the user–repo GNN. Its lowest seed (47.73) is about
  equal to the user–repo GNN's best seed (47.90).
- Most of the gain comes from open_issue (+0.026) and push (+0.003).
- It is still below LightGBM (49.66 in the earlier setup).

On its own, the faithful port is 1.1 points below the user–repo GNN. It loses most on push and
open_pr / open_issue, the activities that the 60 counts over 7, 14, 30 and 90 days predict best.
It only sees 14 days, and 28% of examples have no tokens at all.

The model overfits immediately: validation loss is lowest after epoch 1–2. The likely cause is
the entity-id embedding (599k × 64 ≈ 38M parameters), which can memorise individual
PRs, issues and repos. As with the user–repo GNN, only one validation week is scored.

The old backtest scored 0.596 against 0.46 here. That is not a regression. The old run:
- kept only users with at least 5 events in the last 14 days (about 1,500 per holdout week,
  1,200–1,740; positive rates of 12–24%, and 74% for push, against 3–6% and 33% in week 10-17
  here);
- used no data lags, so its features ran right up to the cutoff. Here the cutoff is about 7
  days earlier for most tables and 25.7 days earlier for push;
- pooled 6 holdout weeks (09-13 → 10-18), including 09-13, the week after the September
  backfill burst, which alone scores 0.67.

### Rerun of the old backtest on the current data

```bash
python -m archive.old_experiments.relgt_inspired.run backtest --temporal-model gnn    # or relgt
```

Unchanged code, default seed 42, one run each (`archive/old_experiments/relgt_inspired/outputs/rerun_gnn`,
`rerun_relgt`; the original runs are `backtest_gnn_full`, `backtest_relgt_full` and the two in
`results/archive_old_data`). The old code always holds out the last 6 weeks, which are now
09-27 → 11-01, so the comparison is made on the 4 weeks both holdouts share, with each run's own
thresholds:

| Model | Run | Holdout weeks | Holdout score | Score on 09-27 → 10-18 |
|---|---|---|---|---|
| GNN | old data (2 runs) | 09-13 → 10-18 | 0.596 / 0.600 | 0.583 / 0.588 |
| GNN | current data | 09-27 → 11-01 | 0.569 | **0.593** |
| RelGT | old data (2 runs) | 09-13 → 10-18 | 0.583 / 0.587 | 0.574 / 0.575 |
| RelGT | current data | 09-27 → 11-01 | 0.569 | **0.591** |

**Result: the old score still holds; on the same weeks the rerun is equal or slightly better**
(+0.005 to +0.010 for the GNN, +0.016 for RelGT, one seed). The rerun has 25 training weeks
instead of 23. Its lower headline score comes from the two new holdout weeks: pushes are mostly
missing from GH Archive after 10-27, so only 46% (10-25) and 21% (11-01) of these users have a
push label against ~72% before. Push F1 falls to 0.63 and 0.34, and those weeks score 0.49 and
0.44. The old code has no label masking for such weeks.

## Separate models for quiet and active users (`--segment activity`)

```bash
python experiments/gnn/run.py --user-features --segment activity   # -> outputs/all_ego_uf_seg/
python experiments/gnn/run.py --segment activity                   # -> outputs/all_ego_seg/
```

An example is *quiet* if it has no ego token, i.e. no visible activity in the 14 days before
the cutoffs (28% of examples; 1,714 of the 4,465 test users). Otherwise it is *active*. The
split uses only data before the cutoff, so it works the same way in the test week. Two ego GNNs
are trained, one per segment. Each is early-stopped on its own segment's validation loss and
retrained for its own best epoch. The quiet model always gets the 60 user features, because its
tokens are all padding. The active model follows `--user-features`. Thresholds are tuned on the
pooled validation predictions, as for one model. In the test week, the validation share of
predicted positives is applied per segment, because each final model has its own probability
scale. Every ego run, split or not, also stores the validation F1 per segment
(`metrics.json` → `per_segment`).

Names below are `outputs/<name>_s0-4`; `_seg` = `--segment activity`.

**Results** (validation week 10-17, 5 seeds, 2,034 quiet and 3,131 active validation users):

| Model | One model | Segmented |
|---|---|---|
| Ego GNN | 46.29 (45.82–46.69) | 46.19 (45.78–46.62) |
| Ego GNN + user features | **48.22** (47.73–48.54) | 47.90 (47.39–48.50) |

Segmented, with user features, mean F1 over the 6 activities: quiet 0.029, active 0.447. A quiet
user is almost never predicted to do anything except push (push F1 0.13, others 0.00–0.04).

**Result: separate models do not help.** The segmented score is 0.1–0.3 below the single model,
inside the seed spread. `--active-min-events 5` (quiet = fewer than 5 events in the window,
3,918 quiet users; `all_ego_uf_seg5`, 4 seeds) gives 47.83 with user features, also no better.

## Holdout week 11-07 (`--holdout`)

```bash
python experiments/gnn/run.py --holdout --user-features --seed 0   # -> outputs/all_ego_uf_ho/metrics.json (+ holdout.npz: per-user probabilities, since 2026-11-16)
CONFIGS="all_ego all_ego_uf all_ego_seg all_ego_uf_seg user_repo" bash runs/gnn_ho1107.sh   # the 5 configs below x seeds 0-2 on 2 GPUs, logs in outputs/logs/, prints the summary
```

The validation numbers above come from one week (10-17), which also sets early stopping and the
thresholds. `--holdout` scores a week that nothing is tuned on:
- training weeks: labels ending by 10-24;
- tuning week 10-31: early stopping and the decision rule (share of predicted positives, per
  segment with `--segment`);
- holdout week 11-07 (4,670 eligible users; 1,460 quiet, 3,210 active), labelled up to the end of
  the data (11-14 13:35 UTC, ~2 h short of the full week). It is scored with the official scorer
  on the 5 non-push activities: push data ends 10-27, so neither 10-31 nor 11-07 has push labels;
- the model trained up to 10-24 is scored directly (no final retrain, no `output.csv`).

These are not comparable with the 6-activity scores above, nor with the missing-data holdout
(tuned on 10-17, scored on 10-24 and 10-31, LightGBM).

**Results** (3 seeds, runs `<name>_ho_s0-2`):

| Model | Run | Holdout score, mean (min–max) | Tuning-week score | Best epoch |
|---|---|---|---|---|
| Ego GNN | `all_ego` | 36.53 (35.88–37.09) | 38.88 | 1, 1, 2 |
| Ego GNN + user features | `all_ego_uf` | 38.80 (37.70–39.68) | 43.40 | 1, 1, 2 |
| Ego GNN, segmented | `all_ego_seg` | 35.21 (34.49–35.77) | 38.97 | quiet 8 / 6 / 3, active 2 / 3 / 2 |
| Ego GNN + user features, segmented | `all_ego_uf_seg` | 38.66 (38.50–38.81) | 43.31 | quiet 8 / 6 / 3, active 2 |
| User–repo GNN | `user_repo` | **38.92** (38.09–39.80) | 43.17 | 11, 5, 12 |

Mean holdout F1 per activity, and the mean F1 over the 5 activities per segment:

| Activity | Ego | Ego + UF | Ego, seg | Ego + UF, seg | User–repo |
|---|---|---|---|---|---|
| open_pr | 0.346 | **0.377** | 0.329 | 0.370 | 0.371 |
| open_issue | 0.417 | 0.424 | 0.410 | 0.415 | **0.441** |
| issue_comment | 0.337 | 0.382 | 0.327 | **0.395** | 0.369 |
| pr_review | 0.399 | 0.391 | 0.391 | 0.399 | **0.400** |
| pr_review_comment | 0.317 | **0.363** | 0.293 | 0.346 | 0.360 |
| quiet users (1,460) | 0.000 | 0.020 | 0.033 | 0.039 | – |
| active users (3,210) | 0.374 | 0.400 | 0.360 | 0.396 | – |

**Result: on the holdout week, the ego GNN with user features and the user–repo GNN are tied**
(38.80 vs 38.92, well inside the seed spread of ~2 points). The 0.8-point lead of the ego model
on 10-17 does not carry over.
- User features are still the main gain for the ego model: +2.3 points, on every activity
  except pr_review (open_pr +0.031, issue_comment +0.045, pr_review_comment +0.046).
- Segmenting again does not help: −1.3 without user features, −0.1 with them. It gives the
  quiet users slightly better F1 (0.039 vs 0.020) but costs more on the active ones. Its only
  benefit is a smaller seed spread (38.50–38.81 against 37.70–39.68).
- Quiet users remain unpredictable: every model's F1 on them is at most 0.04.
- The ego models still stop after 1–2 epochs (the active segment model too); the user–repo GNN
  trains for 5–12.
- The tuning-week score is 4–5 points above the holdout for every model with user features, so
  part of what is tuned on 10-31 does not transfer to the next week.
- Only one holdout week, 3 seeds.
