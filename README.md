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
- Neither `relative` nor the push rule is in `experiments/baseline/run.py` yet. Both are flags of the
  GNN runner. For the GNN, the push rule helps (+6.0 on 11-07) and `relative` does not (see
  "The LightGBM missing-data fixes in the GNN").
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

### Holdout week 11-07 (`--holdout`)

```bash
python experiments/gnn/run.py --holdout --user-features --seed 0   # -> outputs/all_ego_uf_ho/metrics.json
CONFIGS="all_ego all_ego_uf all_ego_seg all_ego_uf_seg user_repo" bash script.sh   # the 5 configs below x seeds 0-2 on 2 GPUs, logs in outputs/logs/, prints the summary
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

### Holdout week 09-27 → 10-04 with push (`--holdout-week`)

Week 11-07 cannot score push (GH Archive has no PushEvents after 10-26). The week 09-27 → 10-04
(Taiwan time; 09-26 16:00 UTC) has labels for all 6 activities, so `--holdout-week` runs the same
protocol there and scores the 6-activity official score:
- training weeks: labels ending by 09-19;
- tuning week 09-19 → 09-26 (early stopping and the share of predicted positives);
- holdout week 09-26 16:00 UTC, no retrain, no `output.csv`.

```bash
python experiments/gnn/run.py --holdout --holdout-week "2026-09-26 16:00" --user-features   # -> outputs/all_ego_uf_ho0926/
bash script_ho0927.sh   # het, het_notime, all_ego_uf, user_repo x seeds 0-2 -> outputs/<cfg>_ho0926_s*
```

Not comparable with the 11-07 holdout: a different week, push included, and the non-push activities
were then at ~10-30% of their usual volume.

**Results** (3 seeds, runs `<cfg>_ho0926_s0-2`, 6,064 eligible users, 6-activity score):

| Model | Run | Holdout score, mean (min–max) | Best epoch |
|---|---|---|---|
| Ego GNN + user features | `all_ego_uf` | **53.02** (52.42–53.35) | 1, 1, 1 |
| Temporal hetero GNN | `het` | 52.77 (52.21–53.69) | 17, 6, 16 |
| Hetero, time embedding off | `het_notime` | 52.34 (51.96–52.84) | 5, 12, 5 |
| User–repo GNN | `user_repo` | 52.29 (52.07–52.47) | 13, 12, 5 |

Mean holdout F1 per activity:

| Activity | all_ego_uf | het | het_notime | user_repo |
|---|---|---|---|---|
| open_pr | **0.436** | 0.411 | 0.405 | 0.417 |
| open_issue | 0.415 | **0.445** | **0.445** | 0.435 |
| issue_comment | 0.417 | **0.425** | 0.395 | 0.403 |
| pr_review | 0.474 | **0.486** | 0.456 | 0.467 |
| pr_review_comment | 0.448 | **0.461** | 0.458 | 0.447 |
| push | **0.678** | 0.669 | 0.675 | 0.670 |

**Result: the four models are tied on this week.**
- The means span 0.73 points (52.29–53.02), and the seed ranges overlap. The seed spread of each
  model is 0.4–1.5 points, so no ranking is supported.
- The time embedding of the hetero GNN is worth +0.43 here (52.77 against 52.34), against +1.3 on
  week 11-07; the ranges overlap. The 11-07 gain does not show clearly on this week.
- Push F1 is 0.669–0.678 for every model, so the push part of the score is the same everywhere.
  The models differ only on the 5 other activities. The hetero model with time embedding has the
  lowest push F1 (0.669) and the best on pr_review (0.486), issue_comment and pr_review_comment.
- The ego model with user features stops after epoch 1 on every seed and is as good as models
  that train for 5–17 epochs.
- The scores are 14–15 points above the 11-07 holdout (38–39), mainly because push (F1 ~0.67)
  now counts and the activities are less thin. They are not comparable with the 5-activity tables.
- One holdout week, 3 seeds.

#### Chain-graph models on the same week (`script_chain_ho0927.sh`)

The chain graph of `experiments/relgt/README.md` (a user's rows of each table linked in time order,
the user linked to the newest one; the 16 latest actions are the seeds), scored on the same week:
- `relgt_chain`: `experiments/relgt/run.py --graph-structure chain --chain-seed latest --num-seeds 16
  --epochs 15`, with `--end`, `--holdout-weeks 1` and `--val-start / --val-end` giving the same
  train / tune / holdout weeks. It reports two scores: the *validation model* (trained to 09-19,
  thresholds of 09-19) and the *final model* (retrained through 09-26, the validation share of
  predicted positives).
- `gnn_chain`: `experiments/gnn/run.py --graph-structure chain` with the same token settings and
  `--holdout --holdout-week` (trained to 09-19, the tune week's share of predicted positives).

The decision rules are not identical to the GNN holdout's, so the two RelGT scores bracket it.

```bash
bash script_chain_ho0927.sh   # relgt_chain on GPU 0, gnn_chain on GPU 1, seeds 0-2; summary next to script_ho0927.sh
SEQUENTIAL=1 bash script_chain_ho0927.sh   # both on GPU 0, one after another (less RAM)
```

**Results** (3 seeds, runs `relgt_chain_ho0926_s0-2` and `gnn_chain_ho0926_s0-2`, the same 6,064
users, 6-activity score; the last four rows are the table above):

| Model | Run | Holdout score, mean (min–max) | Macro / micro F1 | Best epoch |
|---|---|---|---|---|
| RelGT chain, final model | `relgt_chain` | **53.12** (52.72–53.59) | 47.97 / 58.28 | 11, 8, 8 |
| RelGT chain, validation model | `relgt_chain` | 52.01 (51.32–52.80) | 46.28 / 57.74 | 11, 8, 8 |
| GNN chain | `gnn_chain` | 52.48 (51.90–52.94) | 47.38 / 57.57 | 1, 1, 1 |
| Ego GNN + user features | `all_ego_uf` | 53.02 (52.42–53.35) | 47.79 / 58.25 | 1, 1, 1 |
| Temporal hetero GNN | `het` | 52.77 (52.21–53.69) | 48.28 / 57.26 | 17, 6, 16 |
| Hetero, time embedding off | `het_notime` | 52.34 (51.96–52.84) | 47.24 / 57.44 | 5, 12, 5 |
| User–repo GNN | `user_repo` | 52.29 (52.07–52.47) | 47.32 / 57.25 | 13, 12, 5 |

Mean holdout F1 per activity:

| Activity | relgt final | relgt validation | gnn_chain | all_ego_uf | het | het_notime | user_repo |
|---|---|---|---|---|---|---|---|
| open_pr | 0.395 | 0.373 | 0.434 | **0.436** | 0.411 | 0.405 | 0.417 |
| open_issue | **0.470** | 0.442 | 0.416 | 0.415 | 0.445 | 0.445 | 0.435 |
| issue_comment | 0.407 | 0.360 | 0.392 | 0.417 | **0.425** | 0.395 | 0.403 |
| pr_review | 0.481 | 0.470 | 0.475 | 0.474 | **0.486** | 0.456 | 0.467 |
| pr_review_comment | 0.442 | 0.454 | 0.453 | 0.448 | **0.461** | 0.458 | 0.447 |
| push | **0.683** | 0.678 | 0.674 | 0.678 | 0.669 | 0.675 | 0.670 |

**Result: the chain graph does not beat the other graphs on this week.**
- The RelGT final model has the highest mean (53.12), but it is only 0.10 above `all_ego_uf` (53.02),
  and the seed ranges overlap. Leaving out the RelGT validation model, the seven means span 0.83 points
  (52.29–53.12). The seed spreads are 0.4–1.5 points, so no ranking is supported.
- `gnn_chain` (52.48) is in the middle of the GNN results. Like `all_ego_uf`, it stops after epoch 1
  on every seed. Its per-activity F1 is close to `all_ego_uf`'s (open_pr 0.434 against 0.436). Only
  issue_comment is lower (0.392 against 0.417).
- The RelGT final model scores 1.11 above its validation model, and it is ahead on every seed (+0.79,
  +1.15, +1.40). It has two differences: it is trained on one more week (through 09-26), and it
  reuses the share of predicted positives instead of the thresholds. These runs cannot tell which
  one helps. Most of the gain is on issue_comment (0.360 → 0.407), open_issue (0.442 → 0.470) and
  open_pr (0.373 → 0.395). pr_review_comment falls (0.454 → 0.442).
- The GNN holdout uses the validation model's training weeks and the final model's decision rule.
  `gnn_chain` (52.48) falls between the two RelGT scores, as expected.
- The validation model chooses its threshold rule on the tune week, one choice per seed: joint
  thresholds on seeds 0 and 1, per-label F1 thresholds on seed 2.
- Push F1 is again 0.669–0.683 for every model, so the models differ only on the other 5 activities.
  RelGT has the best open_issue F1 of all models (0.470) and the lowest open_pr F1 (0.373–0.395).
- One holdout week, 3 seeds.

#### Quiet users copy their previous week (`--quiet-rule prev-week`, `script_qprev_ho0927.sh`)

```bash
bash script_qprev_ho0927.sh                         # active = >= 5 events (old experiment's rule), seeds 0-2 -> outputs/ego_uf_qprev5_ho0926_s*
CONFIGS="ego_uf_qprev" bash script_qprev_ho0927.sh  # active = >= 1 event -> outputs/ego_uf_qprev_ho0926_s*
```

The `all_ego_uf` GNN predicts almost nothing for quiet users. A user is quiet when they have no
visible activity in the 14-day ego window, the same split as `--segment activity`. On this week
2,822 of the 6,064 users are quiet, and their mean F1 is 0.025. Here the two groups are predicted
in different ways:
- **Active users (3,242):** one ego GNN with user features is trained, early-stopped and tuned on
  active examples only. The share of active users predicted positive in the tune week (09-20 →
  09-27) is applied to the active users of the holdout week.
- **Quiet users:** the prediction is what the user really did in the previous week (labels of
  09-20 → 09-27; in the tune week, labels of 09-13 → 09-20).

The predictions of both groups are joined and scored once over all 6,064 users with the official
scorer: score = (macro-F1 + micro-F1) / 2 over the 6 activities. The F1 for quiet and active users
(`holdout.per_segment` in `metrics.json`) shows where the gains come from. It is not part of the
score.

**The previous week is not visible in the test week.** Non-push data lags about 7 days and push
about 26, so the run needs `--holdout`. Its score is an upper bound for this rule, and the rule
cannot be used for the submission.

**Results** (3 seeds, 6,064 users, 6-activity score; mean F1 per activity):

| Run | Score, mean (min–max) | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | F1 quiet | F1 active |
|---|---|---|---|---|---|---|---|---|---|
| `all_ego_uf` (model for everyone) | 53.02 (52.42–53.35) | **0.436** | 0.415 | 0.417 | **0.474** | **0.448** | 0.678 | 0.025 | **0.505** |
| `ego_uf_qprev` (quiet users copy the previous week) | **53.08** (52.63–53.70) | 0.388 | **0.423** | **0.418** | 0.448 | 0.445 | 0.699 | **0.226** | 0.493 |
| `ego_uf_qprev5` (same, active = ≥ 5 events) | 52.25 (52.20–52.28) | 0.367 | 0.408 | 0.384 | 0.453 | 0.423 | **0.714** | 0.221 | 0.554 * |
| Everyone copies the previous week (no model) | 50.00 | 0.394 | 0.411 | 0.385 | 0.369 | 0.360 | 0.697 | – | – |
| Everyone copies the last 7 *visible* days (no model) | 42.86 | 0.381 | 0.404 | 0.418 | 0.461 | 0.427 | 0.457 | – | – |

\* Over 1,508 active users, not 3,242; see below. Best epoch is 1 on every seed of the three GNN runs. The last two rows are deterministic rules,
scored on the same week and users (macro / micro: 43.60 / 56.40 and 42.46 / 43.26).

**Result: copying the previous week for quiet users does not raise the score** (53.08 against
53.02; the seed ranges overlap).
- Quiet users gain a lot: their mean F1 rises from 0.025 to 0.226. The copied week does best on
  push (0.489 against 0.151) and worst on open_pr (0.146) and pr_review (0.114).
- The model for active users gets worse: its mean F1 falls from 0.505 to 0.493. Most of the loss
  is on open_pr (0.459 → 0.415) and pr_review (0.489 → 0.466), which cancels the quiet gain on
  those two activities. Training on active examples only also changes the share of positives
  tuned on the tune week: open_pr 0.077 → 0.103 of users, push 0.514 → 0.791. (The two shares are
  taken over different user sets: all users against active users only.)
- Even this upper bound only ties the plain model. A rule that the test week can use, such as
  copying the last visible days, would do worse.
- One holdout week, 3 seeds.

**Closest match to the old GNN's user filter (`ego_uf_qprev5`, `--active-min-events 5`).** The
old experiment (`old_experiments/relgt_inspired/data.py`, `eligible_only`) kept a user as an
example only with ≥ 5 visible events in the 14-day lookback. It dropped everyone else from
training, calibration and the backtest score. Here the same ≥ 5 rule picks the users of the GNN
(119,532 training examples; on the holdout week 1,508 active, 4,556 quiet). The quiet users still
copy their previous week, so the score covers all 6,064 users.
- Score 52.25 (macro / micro 45.82 / 58.69), 0.8 below `ego_uf_qprev` (53.08) and `all_ego_uf`
  (53.02). The seed range (52.20–52.28) is narrow and lies below both.
- The active F1 of 0.554 is the number to compare with the old backtest, which scored only these
  users. It does **not** mean a better model than the 0.493 of `ego_uf_qprev`: it is over the
  1,508 most active users, whose labels are denser (positive rate 0.739 for push and 0.259 for
  open_pr, against 0.603 and 0.158 over the 3,242 users with ≥ 1 event).
- The 3,048 users with 1–4 events move from the model to the copy rule. Quiet F1 stays at 0.221
  but now covers 4,556 users instead of 2,822. Overall F1 falls on issue_comment (0.418 → 0.384)
  and pr_review_comment (0.445 → 0.423); only push gains (0.699 → 0.714).
- So the ≥ 5 filter makes the old backtest's numbers look higher, but it does not help when every
  user is scored. On the full user set, ≥ 1 event (or the model for everyone) is the better split.

#### Where the score is lost: quiet vs active users (`experiments/segments/run.py`)

```bash
python experiments/segments/run.py                 # all_ego_uf_ho0926_s0-2 + the capture-drop tables -> experiments/segments/outputs/
python experiments/segments/run.py --skip-drops --runs all_ego_uf_ho0926_s0   # error split only, any ego run
```

Quiet / active is the split above: no visible row in the 14 days before the cutoffs. The runs save
no predictions, so the script rebuilds the hits, false alarms and misses of each group from
`metrics.json`. Per activity it uses the overall F1, each group's F1 and positive rate
(`holdout.per_segment`), and the number of users predicted positive (the validation share; push:
`holdout_push.tune.share`). The rebuilt counts give back the reported scores (53.21, 53.33, 52.40).
Then the score is recomputed with one kind of error removed: a removed miss becomes a hit, a
removed false alarm a true negative. Only ego runs write `per_segment`, so `het` / `het_ev` can't be
split this way.

**Score if one kind of error is removed** (`all_ego_uf`, week 09-26, mean of 3 seeds):

| Removed | Score | Gain |
|---|---|---|
| Nothing (now) | 52.98 | – |
| Every error on quiet users (2,822) | 61.48 | +8.50 |
| ↳ false alarms on quiet users | 53.48 | +0.50 |
| ↳ misses on quiet users | 60.95 | +7.97 |
| Every error on active users (3,242) | 92.85 | **+39.87** |
| ↳ false alarms on active users | 67.65 | +14.67 |
| ↳ misses on active users | 75.76 | +22.78 |

Counts per seed, over the 6 activities: quiet users ~790 errors (~120 false alarms, ~670 misses),
active users ~3,150 (~1,960 false alarms, ~1,190 misses).

**Result: active users cost the score far more than quiet users.** They make about 80% of the
errors.
- **Quiet users: only push matters.** The model predicts no non-push activity for them, but they
  have only 12–53 positives per activity (5–11% of the positives), so this costs 0.03–0.08 F1 per
  activity. The 8.5 points are almost all push: 579 quiet users push (23% of all pushers), and
  about 530 of them are missed.
- **Active users, non-push:** false alarms and misses are about equal (115–310 of each per
  activity), at F1 ~0.45. The model ranks active users poorly on these activities; the share of
  users predicted positive is not the problem.
- **Active users, push:** ~1,100 false alarms against ~100 misses. About 51% of all users are
  predicted to push, nearly all of them active, so almost every active user is predicted to push,
  including the ~40% who don't.
- So the most room is in ranking active users on the 5 non-push activities, and in moving part of
  the push share from active non-pushers to likely quiet pushers. A perfect fix of quiet non-push
  is worth less than 1 point.

**Which users the capture drops hit.** The EDA and the missing-data experiment show activity
falling in the summer and again from mid-September. These are GH Archive capture losses: the
archive drops a share of events at random (see *Experiment: handling GH Archive's missing events*). The script measures them in two ways:
- by the model's quiet / active split, which itself moves with the capture;
- by a fixed type from spring. A user is *heavy* with an event in at least half of the weeks
  Feb 7 – Apr 25 (4,320 users), *light* otherwise (4,680). Spring stops before May because the cohort
  was sampled among users active in May.

Per-week activity as a fraction of spring (all 9,000 cohort users; periods: summer 07-04 → 08-22,
Sep burst 08-29 → 09-12, recent 09-19 → 10-17, Nov 10-24 → 10-31):

| Spring type | Summer | Sep burst | Recent | Nov |
|---|---|---|---|---|
| Heavy, non-push events | **0.05** | 0.91 | **0.22** | 0.13 |
| Light, non-push events | 0.27 | 3.50 | 0.81 | 0.51 |
| Heavy, push events | 0.29 | 0.10 | **0.24** | 0.04 |
| Light, push events | 1.09 | 0.36 | 0.70 | 0.11 |

The model's split, over the eligible examples of the same weeks:

| | Spring | Summer | Sep burst | Recent | Nov |
|---|---|---|---|---|---|
| Quiet share | 18% | 40% | 54% | **46%** | 30% |
| Quiet users who were heavy in spring | 27% | 26% | 35% | **49%** | 54% |
| Rate of any non-push activity, active | 0.447 | 0.077 | 0.396 | 0.278 | 0.200 |
| Rate of any non-push activity, quiet | 0.110 | 0.010 | 0.081 | 0.046 | 0.036 |
| Push rate, active | 0.690 | 0.436 | 0.311 | 0.534 | – |
| Push rate, quiet | 0.318 | 0.089 | 0.063 | 0.175 | – |

**Result: the drops hit active users, and they turn many of them into quiet ones.**
- Heavy users fall to 5% of their spring non-push events in the summer and to 22% recently. Light
  users keep 27% and 81%, and their push does not drop in the summer at all. A heavy user has many
  events for the thinning to remove; a light user had few. Only the push collapse after 10-27 hits
  both groups (4% and 11%). Part of the gap is regression to the mean (light users were partly
  inactive before the May sampling), but that can't explain 5% against 27%.
- In the model's split, label rates fall in both groups, a little more for quiet users relative to
  their spring rate, but from a much lower base. The bigger effect is that the drop moves users
  from active to quiet: with rows missing, a heavy user's 14-day window is empty. The quiet share
  rises from 18% to 46%, and half of the recent quiet users were heavy in spring.
- On week 09-26, 1,308 of the 2,822 quiet users were heavy in spring. Their push rate is 0.286,
  against 0.135 for the 1,514 light ones, and they are 374 of the 579 quiet pushers. So two thirds
  of the quiet pushers the model misses are active users hidden by the capture loss.
- A long-memory signal, such as spring activity or the share of past weeks with activity, could
  find them. The 90-day and all-time user features may already carry part of it; not tested yet.
- One holdout week for the error split, 3 seeds. The capture-drop tables cover the whole data.

### The LightGBM missing-data fixes in the GNN (`--relative-features`, `--push-share prev-week`)

The two fixes that worked for LightGBM in the missing-data experiment, ported to `experiments/gnn/run.py`
as opt-in flags (defaults unchanged: a default run gives the same scores as before):
- `--relative-features`: the user features become the `relative` set of `experiments/missing_data/run.py`.
  Each count is divided by the cohort mean of the same week, and the log of the cohort totals of the 7- and
  30-day counts is added (60 → 74 features). The code is shared through
  `experiments/baseline/run.py` (`relative_features`). It works for every graph structure. The ego tokens keep their raw counts.
- `--push-share prev-week`: push is predicted for the share of users with a push in the last 7 days of
  raw push data before the week (not cut at 10-27), instead of the validation share. For the test week
  this is 11-07 → 11-14: 5.5% of the test users, against ~47% with the validation share.
- Every `--holdout` run now also writes `holdout_push` to `metrics.json` (the `HOLDOUT_PUSH` line in the
  log): the 6-activity score with push decided by each rule, the non-push predictions unchanged. Week
  11-07 has no push labels in the pipeline (push data ends 10-27), so it uses the raw push rows to the end
  of the data (positive rate 5.2%). The rules: `tune` is the validation share. Week 11-07's tune week has
  no push labels, so `tune_proxy` stands in for it: the best-F1 share on 10-17, the last week with push labels
  (a training week). `prev_week` is the new rule. `none` predicts nobody, and `best_in_hindsight` is the best
  share chosen with the labels.

```bash
GPU=0 HO_WEEK="2026-09-26 16:00" CONFIGS="all_ego_uf all_ego_uf_rel" bash script_ho0927.sh
GPU=1 HO_WEEK="2026-11-07 16:00" CONFIGS="all_ego_uf all_ego_uf_rel" bash script_ho0927.sh
```

The `*_pprev` configs of `script_ho0927.sh` are not needed for these numbers: a run's
`holdout_push.prev_week` uses the same model as its `--push-share prev-week` twin.

**Relative features** (3 seeds, `holdout` score; 09-26: 6 activities, 11-07: 5 activities):

| Week | `all_ego_uf` | `all_ego_uf_rel` | Change |
|---|---|---|---|
| 09-26 (push capture 30–60%) | 52.98 (52.39–53.33) | 53.06 (52.85–53.47) | +0.08 |
| 11-07 (non-push ~12–20%) | **38.58** (38.36–38.90) | 37.13 (35.72–38.04) | **−1.45** |

Mean F1 per activity on 11-07:

| Activity | `all_ego_uf` | `all_ego_uf_rel` |
|---|---|---|
| open_pr | 0.380 | 0.386 |
| open_issue | 0.409 | 0.411 |
| issue_comment | **0.382** | 0.344 |
| pr_review | 0.393 | 0.405 |
| pr_review_comment | **0.361** | 0.294 |

**Result: the relative features do not carry over to the GNN.**
- They tie on 09-26 (+0.08, the ranges overlap) and lose 1.45 on 11-07, with a wider seed spread
  (2.3 points against 0.5).
- The loss on 11-07 is on issue_comment (−0.038) and pr_review_comment (−0.067). It is the opposite of
  LightGBM, where most of the gain was on pr_review_comment and pr_review. pr_review (+0.012) and the two
  opening activities change little.
- The model with relative features trains for 2–3 epochs instead of 1–2.
- The ego tokens still carry raw counts, so the model sees both scales. These runs do not show why the
  features hurt here.
- 3 seeds, one week each. The `all_ego_uf` reruns are within the noise of the earlier runs: 52.98 against
  53.02 on 09-26, and 38.58 against 38.80 for the `_ho` runs of 11-07.

**Push rule** (6-activity score, mean of 3 seeds, the same models; share = users predicted to push):

| Rule | 09-26, `all_ego_uf` | 09-26, `_rel` | 11-07, `all_ego_uf` | 11-07, `_rel` |
|---|---|---|---|---|
| Validation share (`tune`; 11-07: `tune_proxy`) | **52.98** (51.4%) | **53.06** (47.2%) | 31.75 (43.2%) | 31.48 (38.3%) |
| Last 7 days of push data (`prev_week`) | 51.60 (39.2%) | 52.19 (39.2%) | **37.73** (5.7%) | **36.26** (5.7%) |
| Nobody (`none`) | 31.33 | 31.81 | 33.43 | 32.20 |
| Best share in hindsight | 53.20 (60.2%) | 53.57 (58.8%) | 37.80 (4.3%) | 36.59 (8.7%) |

Push positive rate: 41.8% on 09-26, 5.2% on 11-07.

**Result: on week 11-07 the prev-week rule is worth +6.0 points and matches the best share in hindsight.**
- On 11-07, predicting push for the validation share (43%) scores below predicting it for nobody
  (31.75 against 33.43). The prev-week rule (37.73, push F1 0.338) is 0.07 below the best share in
  hindsight. This matches LightGBM (~29 → ~35).
- On 09-26 it costs 1.38 points (−0.87 with relative features). There push was common, and the best share
  (~60%) is about 1.5× last week's push rate (39%). On 11-07 the best share (4–9%) is about the same as
  the rate (5.7%). So scaling the rate up would only help when push is common.
- The test week looks like 11-07: push has been at ~1–7% of its usual volume since 10-27. **For the
  submission, use `--push-share prev-week`.** If GH Archive's push capture recovers before the test week,
  the validation share would be better again.
- One week per regime, 3 seeds. The `tune_proxy` share is taken on a training week, so it only
  approximates what the final pipeline does.

**For the submission:** `python experiments/gnn/run.py --user-features --push-share prev-week`.

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

## Experiment: temporal heterogeneous GNN (`experiments/gnn/run.py --graph-structure hetero`)

The user–repo GNN reduces every user–repo pair to one edge of windowed counts, and its time
encoding (first / last age of the pair) adds at most noise. This model keeps the events and the
objects they act on, TGAT-style. One full graph per week, built only from rows visible before
each table's cutoff (same weeks, lags, eligibility and split as the baseline):

- *Nodes*: the 9,000 cohort users (the 60 baseline features), and the PRs, issues and
  repositories reached by an edge (test week: 24k PRs, 17k issues, 22k repos). An object's
  features are its visible cohort events per activity and its distinct cohort users, plus days
  since first seen for a PR / issue and the organisation flag for a repo.
- *Edges, one per event* (90-day window, `--history-days`):
  - user → PR / issue / pushed repo per activity, keeping the latest 32 per user and activity
    (`--het-max-events`);
  - repo → user `received`;
  - PR / issue → its repository.

  Every relation also runs reversed with its own weights (~300k edges per week).
- *Time embedding*: each edge's age (days before its cutoff; for object → repo, days since the
  object was first seen) goes through a learned `cos(w·log(1+age) + b)` encoding
  (`--het-time-dim 16`, one encoder each for events, received and membership).
- *Model* (`TemporalHeteroGNN`), 2 layers, dim 64:
  - message `W_r[h_src ‖ φ(age)]`;
  - attention `a_r·LeakyReLU(Q h_dst + m)`, softmaxed over a node's incoming edges of each
    relation;
  - relations summed per node type, then residual + LayerNorm;
  - head on the user nodes.

  Training (AdamW lr 2e-3, early stopping, thresholds) is the same as for the user–repo GNN.
- `--het-time none` feeds zeros in place of φ. The architecture stays the same, so the
  comparison isolates the time embedding.

```bash
python experiments/gnn/run.py --graph-structure hetero --holdout                  # -> outputs/all_het_ho/
python experiments/gnn/run.py --graph-structure hetero --het-time none --holdout  # -> outputs/all_het_notime_ho/
bash script.sh   # het and het_notime x seeds 0-2 under --holdout -> outputs/het_ho_s*, het_notime_ho_s*
```

**Results** (holdout protocol above, 3 seeds, runs `het_ho_s0-2` and `het_notime_ho_s0-2`):

| Model | Run | Holdout score, mean (min–max) | Best epoch |
|---|---|---|---|
| Temporal hetero GNN | `het` | **38.71** (37.98–39.59) | 11, 5, 12 |
| Same, time embedding off | `het_notime` | 37.42 (36.90–37.78) | 11, 5, 12 |

Mean holdout F1 per activity:

| Activity | het | het_notime | Difference |
|---|---|---|---|
| open_pr | 0.380 | 0.371 | +0.009 |
| open_issue | 0.432 | 0.418 | +0.014 |
| issue_comment | 0.393 | 0.382 | +0.011 |
| pr_review | 0.386 | 0.383 | +0.003 |
| pr_review_comment | 0.335 | 0.303 | +0.032 |

**Result: the time embedding is worth about 1.3 points, but the hetero GNN does not beat the
earlier models.**
- With the time embedding the score is 38.71, against 37.42 without it. It is higher on every
  activity, mostly on pr_review_comment (+0.032); pr_review barely moves (+0.003). The seed
  ranges do not overlap (37.98–39.59 against 36.90–37.78), but with 3 seeds and a seed spread
  of ~1–2 points this is suggestive, not conclusive.
- Against the models of the holdout table above, 38.71 is level with the user–repo GNN (38.92)
  and the ego GNN with user features (38.80), within the seed spread. Keeping the individual
  events and objects instead of one windowed edge per user–repo pair brings no measurable gain.
- Without the time embedding the hetero GNN is about as good as the plain ego GNN (36.53)
  and 1.5 points under the user–repo GNN.
- Both variants have the same best epochs (11, 5, 12) per seed, and train for as long as the
  user–repo GNN (5–12 epochs), unlike the ego models.
- One holdout week, 3 seeds; tuning-week scores were not recorded here.

### 14-day window (`--history-days 14`)

The same two models when they only see the last 14 days before each cutoff:
- event edges, received edges and object / repo counts come from the 14 days;
- a PR's or issue's days since first seen are clipped at 14;
- the user features are recomputed from the 14 days (39 instead of 60: the 30d / 90d / all-time
  counts are dropped).

Eligibility and labels are unchanged, so the runs are scored on the same users as above. The
test-week graph shrinks to 2.2k PRs, 1.7k issues and 6.2k repos (~77k edges per week on average,
against ~300k).

```bash
bash script_h14.sh   # het_h14 and het_notime_h14 x seeds 0-2 under --holdout; summary next to het / het_notime
```

**Results** (holdout protocol, 3 seeds, runs `het_h14_ho_s0-2` and `het_notime_h14_ho_s0-2`):

| Model | 90 days | 14 days | Change | Best epoch (14 days) |
|---|---|---|---|---|
| Temporal hetero GNN | **38.71** (37.98–39.59) | 36.01 (35.54–36.43) | -2.70 | 11, 6, 4 |
| Same, time embedding off | 37.42 (36.90–37.78) | 36.67 (36.33–36.89) | -0.75 | 4, 6, 12 |

Mean holdout F1 per activity:

| Activity | het | het_h14 | het_notime | het_notime_h14 |
|---|---|---|---|---|
| open_pr | 0.380 | 0.357 | 0.371 | 0.350 |
| open_issue | 0.432 | 0.403 | 0.418 | 0.405 |
| issue_comment | 0.393 | 0.352 | 0.382 | 0.374 |
| pr_review | 0.386 | 0.375 | 0.383 | 0.390 |
| pr_review_comment | 0.335 | 0.290 | 0.303 | 0.300 |

**Result: the 14-day window hurts, and it removes the benefit of the time embedding.**
- The time embedding model loses 2.7 points (38.71 to 36.01), on every activity: most on
  pr_review_comment (-0.045) and issue_comment (-0.041). The seed ranges do not overlap.
- Without the embedding the loss is 0.75 points (37.42 to 36.67) and the ranges overlap, so
  that drop is within the seed spread. pr_review even gains slightly (+0.007).
- At 14 days the order flips: the model with time embedding scores 0.66 under the one without
  (36.01 against 36.67, ranges overlapping, so no real difference). The 1.3-point gain at 90 days
  disappears. A 14-day window leaves little age variation for the embedding to use, and the
  longer history it exploited at 90 days is gone.
- The extra history is useful, mainly to the model that can place it in time. A shorter window
  gives no gain and no cheaper way to match the 90-day score, although the graph is about 4x
  smaller.
- Best epochs are early and uneven (4-12). One holdout week, 3 seeds.

### Event graph: links between actions (`--het-graph event`)

The graph above only joins actions where their edges meet at the same user or object. Nothing
records that a review comment belongs to a review, that a push updated a PR, or which action
came after which. `--het-graph event` (`hetero_event_graph`) makes every kept event a node of
its own: the same latest 32 per user and activity, under the same visibility rules. Events can
then link to each other directly. Week 2026-09-26 has 9,000 users, 20k PRs, 13k issues,
29k repos and 177k events, with about 550k edges:

| Relation | From → to | Edge age | Edges (09-26) |
|---|---|---|---|
| `<activity>` (6) | user → own event | days before the cutoff | 177k |
| `<activity>_on` (6) | event → PR / issue / repo | days before the cutoff | 175k |
| `issue_comment_on_pr` | issue comment on a PR → the PR node | days before the cutoff | 1.8k |
| `push_to_pr` | push → the PR it likely updates | days before the cutoff | 6.7k |
| `next` | a user's event → that user's next event | gap (days) | 170k |
| `in_review` | review comment → its review | gap (days) | 2.5k |
| `reply_to` | review comment → the comment it answers | gap (days) | 77 |
| `received` | another user's event on my PR / issue → me | days before the `received` cutoff | 80 |
| `pr_in`, `issue_in` | PR / issue → its repo | days since first seen | 33k |

- **Where the links come from.**
  - `in_review` comes from `PRReviewComments.pull_request_review_id`, and `reply_to` from
    `in_reply_to_id`. Both need the two events in the graph.
  - An issue comment on a PR moves to the PR node when `Issues.pull_request_id` names the PR
    (about a third of PR issues do).
  - Text is masked in the data, so `push_to_pr` is a heuristic: a same-repo PR by the pusher
    whose masked head branch equals the pushed branch and that is open at the push. It links
    9% of pushes, and 37% of those links are ambiguous (the closest PR is taken).
- **Not linked.** An issue and the PR that fixes it can't be joined: PR bodies are masked
  (`#000`) and there is no closing-reference column.
- **Event node features.** One-hot activity, log age, whether it is on the user's own PR or
  issue, and whether it has a link (moved to its PR, linked to a PR, or part of a kept review).
- **Model.** Same `TemporalHeteroGNN` as above, with a fourth time-encoder family for gaps
  between events. It uses **4 layers** by default, because user → event → PR → event → other
  user is 4 hops. `--het-layers N` overrides this, and the folder name gets `_l<N>`.
- **Memory.** The 4 layers store about 12 GB of per-edge activations, which leaves no headroom on a
  12 GB card, so `--het-checkpoint on` (the default with `--het-graph event`) keeps only each layer's
  input and recomputes its activations in the backward pass. Same model and losses, peak 5.55 GB.
  Run one job per GPU.

```bash
# one run (holdout week 09-27 -> 10-04, all 6 activities, push included) -> outputs/all_het_ev_ho0926/
python experiments/gnn/run.py --graph-structure hetero --het-graph event --holdout --holdout-week "2026-09-26 16:00"
# ablation without the time embedding -> outputs/all_het_ev_notime_ho0926/
python experiments/gnn/run.py --graph-structure hetero --het-graph event --het-time none --holdout --holdout-week "2026-09-26 16:00"
# 3 seeds each, next to the object graph; prints the summary table at the end
CONFIGS="het_ev het_ev_notime" bash script_ho0927.sh   # -> outputs/het_ev[_notime]_ho0926_s{0,1,2}/
# both GPUs at once: one config per card
GPU=0 CONFIGS="het_ev" bash script_ho0927.sh &  GPU=1 CONFIGS="het_ev_notime" bash script_ho0927.sh
# draw one user's 4-hop field (event-to-event links dashed)
python model_graph.py --model hetero-event --week "2026-09-26 16:00"          # app at http://127.0.0.1:8052
python model_graph.py --html --model hetero-event --week "2026-09-26 16:00"   # -> figures/graph_hetero-event_<login>_<date>.html
```

**Results** (holdout week 09-27 → 10-04, all 6 activities, 3 seeds, finished 2026-11-16; score = mean, range in brackets):

| Config | Score | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | Best epochs |
|---|---|---|---|---|---|---|---|---|
| `het_ev` | 51.71 (50.85-52.40) | 0.410 | 0.411 | 0.422 | 0.461 | 0.445 | 0.666 | 1, 6, 12 |
| `het_ev_notime` | 51.97 (51.45-52.26) | 0.403 | 0.434 | 0.408 | 0.459 | 0.444 | 0.668 | 1, 6, 12 |
| `het` (object graph, for comparison) | 52.77 (52.21-53.69) | 0.411 | 0.445 | 0.425 | 0.486 | 0.461 | 0.669 | 17, 6, 16 |
| `het_notime` | 52.34 (51.96-52.84) | 0.405 | 0.445 | 0.395 | 0.456 | 0.458 | 0.675 | 5, 12, 5 |

6-activity score by push rule (`het_ev` / `het_ev_notime`): tune 51.71 / 51.97, prev_week 50.89 / 50.93,
none 31.09 / 30.70.

- **The event graph does not help.** `het_ev` is about 1 point below `het`, and `het_ev_notime` about
  0.4 below `het_notime`. The gap to `het` is the size of the seed ranges, but the direction is the same
  with and without the time embedding, and no activity gains clearly (pr_review is lowest, 0.461 vs 0.486).
  Direct event links (`in_review`, `reply_to`, `push_to_pr`) cover few events (2.5k, 77, 6.7k of 177k),
  so they add little signal for the extra size.
- **Time embedding.** With the event graph, removing it changes the score by +0.26, within noise
  (`het`: -0.43 the other way). No clear effect in either graph.
- **Training is erratic.** Best epochs jump between 1 and 12, so the selected epoch is noisy.
- **Push rule.** `tune` is best, `prev_week` loses about 0.8-1.0, and `none` falls to about 31, so push
  carries the score here too.

For the best of the other GNNs on this week, `all_ego_uf` scores 52.98, `all_ego_uf_rel` 53.06
and `user_repo` 52.29 (3 seeds each). The top four configurations overlap within seed noise.

## Experiment: ego GNN with the hetero model's time encoding (`--ego-time func`)

The ego GNN encodes time inside each token: a fixed sinusoidal encoding of the log age, mixed
with the other token features by the MLP. `--ego-time func` moves time onto the edges, as in
the temporal hetero GNN:
- the token has no time part (5 parts instead of 6);
- every token ↔ user edge carries `φ = cos(w·log(1+age) + b)`, learned, of the token's age
  (`--ego-time-dim 16`);
- message token → user `W_r[h_token ‖ φ]`, attention `a_r·LeakyReLU(Q h_user + m)`, softmaxed
  over the tokens of each relation, relations summed;
- message user → token `W[h_user ‖ φ]`. A token has one incoming edge from the user, so there is
  no attention;
- the push repo ↔ member link, the self transform, the head and the training stay as before.

`--ego-time none` keeps that architecture with φ = 0, the ablation (like `--het-time none`).
The default `--ego-time sin` is the original model.

```bash
python experiments/gnn/run.py --user-features --ego-time func --holdout   # -> outputs/all_ego_uf_ftime_ho/
bash script_ego_time.sh   # ego_uf_ftime, ego_uf_notime x seeds 0-2 on one GPU (GPU=0); summary also lists all_ego_uf (sin)
```

**Results** (holdout protocol, 3 seeds): TODO.

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
- `experiments/segments/run.py`: quiet vs active error split of a holdout run and the capture drops
  by user type; see *Where the score is lost* above.
- `scripts/download_gharchive.py` (fetch GH Archive hours), `scripts/collect_scores.py` (gather
  every metrics file into one JSON), `scripts/delete_archive_pt.py` (deletes `.pt` files and cache dirs under `results/archive_old_data`),
  `schema_graph.py` (draws `figures/schema_graph.*`), `relgt_graph.py` (draws token graphs).

## Data notes

- Only cohort users' own activity is collected, so *received* activity only covers
  interactions between cohort users and is sparse.
