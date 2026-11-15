# Assignment 2: Predicting GitHub User Activity

Predict, for each user in `test_input.csv`, which of 6 activities (`open_pr`, `open_issue`,
`issue_comment`, `pr_review`, `pr_review_comment`, `push`) they perform during
2026-11-22 00:00 – 11-29 00:00 Taiwan time (2026-11-21 16:00 UTC + 7 days).

## Setup

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
```

`assignment2/` (the course data) is not committed; it must sit at the repo root.

## Run

```bash
python experiments/baseline/run.py
python assignment2/scripts/evaluate.py --pred experiments/baseline/outputs/none/output.csv --input assignment2/submission/test_input.csv
```

Writes `experiments/baseline/outputs/<gap-mode>/output.csv` (rename to
`output_{studentID}.csv` for submission) and `metrics.json` with validation scores. Takes
about 30 s. Options: `--gap-mode {none,mask,drop,downweight,recent}`, `--gap-weight`,
`--seed` and the data-period options below (see `--help`).

**Data period** (baseline, GNN, GNN-review, relations and RelGT runners). A week counts as inside a period when its label window
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
python eda.py        # then open http://127.0.0.1:8050 (startup ~3 s)
```

A Dash + Plotly app over the cohort's activity, with the same activity definitions and
test-aligned weeks as the baseline. Global filters (date range, all / test / non-test users,
activities) drive the tabs: daily activity per table with the outages shaded, weekly label
rates and co-occurrence, persistence (P(active) by weeks since last active, week-to-week
transitions), quiet users, per-user concentration, time-of-day heatmaps, top repositories,
data-quality checks (staleness before the test week, how late rows appear, null rates), a
per-table column explorer and a per-user timeline. The shifted test selection rule reproduces
the 4,465 test users exactly.

The *Quiet users* tab covers the quiet segment of `experiments/gnn/run.py --segment activity`.
It uses the models' examples and per-table cutoffs and reproduces the 1,714 quiet test users.
The window and minimum number of rows can be changed (`--ego-lookback`,
`--active-min-events`). It shows:
- quiet share and return rates per week;
- label rates and the share of positives, quiet vs active;
- what predicts a return, against a break-even line (F1 / 2);
- activity hidden between the cutoff and the week start;
- how often each user is quiet;
- how the test week's quiet users compare.

Weeks distorted by the May cohort sampling or the summer outage are left out by default.

## Baseline (`experiments/baseline/run.py`)

**Samples.** One row per (cohort user, week start `s`), with weeks aligned to the test week
(Wednesday 16:00 UTC). Label = the user does the activity in `[s, s + 7d)`, using
`created_at` / `submitted_at` (reproduces `examples/sample_truth.csv` at >99.3% agreement).
Rows are kept only for users meeting the test selection rule shifted to that week
(≥ 3 records in the 90 days before `s − 17d4h`, the test week's selection time 2026-11-04 12:00);
at the test week this rule yields exactly the 4,465 test users.

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

Earlier setup (test week 11-12, validation weeks 10-07 / 10-14), not rerun. On the current GH
Archive data the outages are re-measured, and these modes compared again, in the next section.

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

## Experiment: handling GH Archive's missing events (`experiments/missing_data/run.py`)

```bash
python experiments/missing_data/run.py        # 14 approaches x 3 seeds, ~13 min -> outputs/summary.csv, results.json
python experiments/missing_data/run.py --approaches baseline relative --seeds 0
```

**The data.** All 8,366 hour files since 2025-12-01 exist (`data/gharchive/missing_hours.txt` is
empty), but the hours contain fewer events since June. Cohort events per day, as a share of the
Jan–May level:

| Period | Non-push | Push |
|---|---|---|
| 2025-12 → 2026-05 | ~100% | ~100% (more in May, when the cohort was sampled) |
| 06-01 → 09-03 | 25%, falling to 3% | ~120% in June, falling to ~0% by early September |
| 09-04 → 09-17 | 100–250% (a burst) | 5–30% |
| 09-18 → 10-26 | 10–30% | 30–60% |
| 10-27 → 11-14 | ~12% | 0–4%, most hours have no push |

The full GH Archive files shrink the same way (the 14:00 UTC hour: 26 MB in March, 11 MB on
10-20, 3–4 MB in November), so the archive is losing events; the cohort has not gone quiet.
Events are lost one at a time and evenly over the hours:
- positive rates fall less than event counts (open_pr: 22.7% → 6.6% of eligible users a week,
  while events fall to 15%);
- most active weeks now have a single event (open_pr: 50% → 71%);
- dropping spring events at random reproduces the recent positive rates within ~1 point.

Part of the drop is probably real: even in the full-capture weeks of 09-05 and 09-12 the rates
are only 60–70% of spring (the cohort was drawn from users active in May). The answer key of the test week comes from the same archive, so no approach
fills in events. They change which weeks the model learns from, or how the features are scaled.

**Protocol** (the same for every approach):
- the baseline's features, labels, per-table lags and selection rule;
- training weeks: labels ending by 10-17 (week starts up to 10-10, 39 weeks);
- tuning week 10-17: early stopping and the decision rule (the share of predicted positives with
  the best F1, carried to other weeks as in the baseline's final step);
- holdout weeks 10-24 and 10-31 (~3.8k users each), seen by nothing above, scored with the
  official scorer on the 5 non-push activities. Push data ends 10-27, so no recent week has push
  labels; push is simulated separately below;
- 3 seeds per approach (LightGBM seed and thinning draw). A paired bootstrap (2,000 resamples of
  users within each holdout week, the same draws for every approach, averaged over seeds) gives
  the 95% interval of the difference to the baseline.

**Approaches:**
- `baseline`: every training week.
- `rule_from_train`: baseline models, with the threshold that maximises F1 on the training weeks
  instead of the tuning week.
- `mask_burst`: non-push labels of weeks overlapping 09-04 → 09-17 get weight 0.
- `gaps_mask` / `gaps_drop` / `gaps_downweight`: the baseline's `--gap-mode` options with `GAPS`
  re-measured (non-push 06-01 → 09-18; push 08-18 → 09-17 and 10-23).
- `recent`: only weeks from 09-19 on (4 training weeks).
- `relative`: every count feature divided by the cohort mean of the same week, plus the log
  cohort totals of the 7- and 30-day counts of each source (14 features). `relative_no_volume`
  leaves out the totals.
- `rank`: count features replaced by their percentile rank among the cohort users of the week
  (0 stays 0).
- `thin`: the 19 weeks before June rebuilt from events kept at random with probability p, with
  labels and the selection rule recomputed from the kept events. p = events per eligible
  user-week in 09-19 … 10-17 ÷ spring: open_pr 0.18, open_issue 0.41, issue_comment 0.20,
  pr_review 0.19, pr_review_comment 0.18. This brings the spring positive rates close to the
  recent ones (open_pr 22.8% → 8.3%, recent weeks 7.5%). `thin_x3`: three draws, weight 1/3 each.
- Combinations: `relative_mask_burst`, `thin_x3_mask_burst`.

**Results** (holdout score, 5 non-push activities; the last column carries over the raw
probability threshold instead of the share of predicted positives):

| Approach | Score, mean of 3 seeds (min–max) | vs baseline [95% interval] | 10-24 | 10-31 | Threshold carried over |
|---|---|---|---|---|---|
| `relative` | **39.24** (38.72–39.62) | **+1.49** [+0.51, +2.44] | 37.94 | 40.55 | 38.80 |
| `relative_mask_burst` | 39.19 (38.96–39.54) | +1.44 [+0.45, +2.46] | 37.25 | 41.13 | 35.44 |
| `mask_burst` | 38.74 (38.70–38.80) | +0.99 [+0.26, +1.76] | 36.50 | 40.98 | 37.06 |
| `relative_no_volume` | 38.72 (38.41–38.91) | +0.96 [−0.23, +2.08] | 37.61 | 39.84 | 38.47 |
| `gaps_downweight` | 38.71 (38.19–39.54) | +0.95 [+0.21, +1.68] | 36.60 | 40.81 | 38.67 |
| `thin_x3_mask_burst` | 38.58 (38.23–39.05) | +0.82 [−0.06, +1.66] | 36.26 | 40.89 | 37.07 |
| `gaps_drop` | 38.01 (37.65–38.38) | +0.24 [−0.75, +1.20] | 36.45 | 39.57 | 37.16 |
| `baseline` | 37.75 (37.39–38.40) | – | 35.21 | 40.29 | 37.35 |
| `gaps_mask` | 37.71 (37.20–38.14) | −0.05 [−0.96, +0.79] | 35.68 | 39.73 | 37.55 |
| `rank` | 37.47 (36.86–37.81) | −0.28 [−1.40, +0.80] | 35.86 | 39.07 | 37.16 |
| `thin` | 37.32 (37.11–37.56) | −0.44 [−1.34, +0.45] | 35.67 | 38.96 | 36.16 |
| `thin_x3` | 37.10 (36.39–37.48) | −0.66 [−1.47, +0.15] | 35.34 | 38.86 | 36.21 |
| `rule_from_train` | 36.80 (36.31–37.55) | −0.97 [−2.14, +0.09] | 35.33 | 38.27 | – |
| `recent` | 36.18 (35.89–36.48) | −1.58 [−2.63, −0.48] | 33.97 | 38.38 | 35.85 |

Mean F1 per activity, baseline → `relative`: open_pr 0.410 → 0.421, open_issue 0.425 → 0.429,
issue_comment 0.396 → 0.404, pr_review 0.391 → 0.409, pr_review_comment 0.238 → 0.268.

**Result: `relative` is best, 1.5 points above the baseline (95% interval +0.5 to +2.4).** Its
lowest seed (38.72) beats the baseline's best (38.40). Most of the gain is on pr_review_comment
and pr_review.
- Both parts of `relative` help: without the cohort totals the gain drops to +0.96 and its
  interval includes 0.
- Masking the September burst helps on its own (+0.99, the most stable across seeds) but adds
  nothing on top of `relative` (39.19). Masking all of 06-01 → 09-18 (`gaps_mask`) gains nothing:
  only the burst weeks hurt, and dropping the low-capture summer weeks as well cancels the gain.
- Thinning the spring weeks does not help (−0.44 and −0.66), and adds nothing to `mask_burst`
  (38.58 vs 38.74), although it matches the recent positive rates. A likely reason: the long
  feature windows of recent weeks mix capture levels (the 90-day and all-time counts include the
  full spring and the burst), which thinned weeks do not reproduce.
- Training only on weeks from 09-19 on is worst (−1.58, the only interval entirely below 0):
  4 weeks are too few.
- The decision rule matters. Taking the threshold from the training weeks costs ~1 point; the
  baseline already sets it on the latest weeks. Carrying over the share of predicted positives
  beats carrying over the probability threshold for every approach.

**Push, if it is scored.** Simulated on the week of 11-07, labelled up to the end of the data
(~2 hours short of the full week). Push is ~1% of its usual volume, but 5.2% of eligible users
have a visible push. 6-activity score with the baseline models, 3 seeds:

| Push rule | Users predicted to push | Score (min–max) |
|---|---|---|
| Share from the tuning week 10-17 (what the pipeline does now) | 40–47% | 28.8–30.4 |
| Nobody | 0% | 30.0–31.5 |
| Share of users with a push in the previous week (10-31) | 5.5% | 34.6–36.0 |
| Best share in hindsight | 4–9% | 34.6–36.0 |

**Result: if push is scored, predict it for as many users as pushed in the last labelled week.**
That matches the best share in hindsight and is 5–6 points above the current rule. If push is not
scored, the rule changes nothing.

Caveats:
- Only 2 holdout weeks are scored, and the decision rule is set on 1 week.
- These scores (5 activities, holdout not used for tuning) are not comparable with the 6-activity
  validation scores elsewhere in this README.
- Neither `relative` nor the push rule is in `experiments/baseline/run.py` yet.
- LightGBM runs with `--threads 8` (default). With all 28 cores it was ~170× slower under WSL
  (100 rounds: 187 s against 1.1 s).

## Experiment: ego-graph GNN, the default of `experiments/gnn/run.py`

```bash
python experiments/gnn/run.py                    # -> experiments/gnn/outputs/all_ego/output.csv
python experiments/gnn/run.py --user-features    # -> experiments/gnn/outputs/all_ego_uf/
```

The relation-aware GNN of `old_experiments/relgt_inspired` (`gnn_model.py`), ported to the
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
| User–repo GNN, for reference (below) | 47.42 (46.68–47.90) | 0.6248 (0.6237–0.6274) | 2–17 |

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

#### Rerun of the old backtest on the current data

```bash
python -m old_experiments.relgt_inspired.run backtest --temporal-model gnn    # or relgt
```

Unchanged code, default seed 42, one run each (`old_experiments/relgt_inspired/outputs/rerun_gnn`,
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

### Separate models for quiet and active users (`--segment activity`)

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

## Experiment: GNN (user–repo graph), all labels vs push only (`experiments/gnn/run.py --graph-structure user-repo`)

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

**Validation score** (earlier setup: test week 11-12, seed 0, weeks 10-07 / 10-14; not rerun on the revised assignment):

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

## Other experiments and tools

- `experiments/gnn/run.py --graph-structure chain`: RelGT-style chain graph for the GNN.
- `experiments/gnn_review/`, `experiments/relations/`: see `ACTION_RELATIONSHIPS.md`.
- `scripts/download_gharchive.py` (fetch GH Archive hours), `scripts/collect_scores.py` (gather
  every metrics file into one JSON), `scripts/delete_archive_pt.py` (deletes `.pt` files and cache dirs under `results/archive_old_data`),
  `schema_graph.py` (draws `figures/schema_graph.*`), `relgt_graph.py` (draws token graphs).

## Data notes

- Only cohort users' own activity is collected, so *received* activity only covers
  interactions between cohort users and is sparse.
