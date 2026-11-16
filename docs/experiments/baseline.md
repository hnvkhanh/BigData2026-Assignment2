# Baseline (`experiments/baseline/run.py`)

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
