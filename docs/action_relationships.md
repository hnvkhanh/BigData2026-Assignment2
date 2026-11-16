# Relationships between actions and tables

Findings from exploring how the six predicted actions relate to each other and to the other
columns of the database, and whether those relationships improve the prediction. Short
version: the relationships are real and strong, but most of what they carry is already in the
baseline's per-action count/recency features. The one targeted change that helped the GNN is
worth about +0.7 score points; the cross-table feature groups are worth at most ~+0.3 together
on LightGBM.

## 1. Setup

> **Status:** the descriptive statistics and model scores below were computed on the earlier
> setup (test week 11-12, validation weeks 10-07 / 10-14) and have not been recomputed.

- **Unit:** (cohort user, week), weeks start Wednesday 16:00 UTC, aligned to the test week. A
  user-week counts only if the user passes the test selection rule shifted to that week
  (≥ 5 records in the 90 days before) as in `experiments/baseline/run.py` at the time of this
  analysis. The baseline has since moved to the revised assignment (test week 2026-11-22 Taiwan
  time, ≥ 3 records, 4,465 test users).
- **No leakage:** every signal is computed from rows visible before `s − lag` of its own table
  (lag = test week start − end of that table's data, ~10 days, ~16 for pushes).
- **Descriptive statistics (sections 2–4):** 15 weeks, 2026-02-11 … 05-20, i.e. weeks whose
  label window and previous 30 days avoid the collection outages; 82,227 user-weeks.
  Exception: the review / review-comment overlap in section 3 uses all rows of the two tables.
- **Model results (sections 5–6):** the baseline split, training weeks 2026-01-14 … 09-30,
  validation 10-07, 10-14, 10-21. Scores are (macro-F1 + micro-F1) / 2, averaged over the
  validation weeks whose six labels are all known (10-07 and 10-14; pushes of 10-21 are not in
  the data yet).

## 2. How the actions relate

### Same week: P(row action | column action), base rate on the diagonal

|                   | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| open_pr           | **0.23** | 0.44 | 0.46 | 0.44 | 0.46 | 0.33 |
| open_issue        | 0.19 | **0.10** | 0.30 | 0.19 | 0.20 | 0.13 |
| issue_comment     | 0.32 | 0.49 | **0.16** | 0.41 | 0.45 | 0.21 |
| pr_review         | 0.27 | 0.28 | 0.37 | **0.14** | 0.74 | 0.19 |
| pr_review_comment | 0.20 | 0.21 | 0.28 | 0.52 | **0.10** | 0.13 |
| push              | 0.96 | 0.90 | 0.91 | 0.91 | 0.92 | **0.68** |

Example: in a week where a user leaves a review comment, they also submit a review 74% of the
time (base rate 14%).

### Across weeks: lift of "did X in the last 7 visible days" on label Y

Lift = P(Y | X) / P(Y). Rows are the signal X, columns the label Y. This is what a model can
actually use, because of the staleness of the data.

|                   | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| open_pr           | 2.03 | 1.59 | 1.70 | 1.77 | 1.84 | 1.24 |
| open_issue        | 1.64 | 4.13 | 2.45 | 1.91 | 2.06 | 1.22 |
| issue_comment     | 1.64 | 2.33 | 2.98 | 2.32 | 2.52 | 1.24 |
| pr_review         | 1.63 | 1.71 | 2.26 | 3.59 | 3.72 | 1.25 |
| pr_review_comment | 1.68 | 1.86 | 2.46 | 3.72 | 4.55 | 1.26 |
| push              | 1.22 | 1.19 | 1.24 | 1.26 | 1.26 | 1.15 |
| received *        | 2.05 | 2.60 | 3.45 | 3.44 | 3.81 | 1.21 |

\* comments / reviews other cohort users left on the user's own issues and PRs.

**Takeaways**

- **Collaboration cluster.** `pr_review`, `pr_review_comment`, `issue_comment` and `open_issue`
  predict each other 2–4×. `pr_review` ↔ `pr_review_comment` is the tightest pair (3.6–3.7×
  each way, close to each action's own persistence).
- **`open_pr` is the most independent** non-push action: other signals lift it only 1.6–2.0×.
- **Push is uninformative in both directions.** It lifts every other action ~1.2×, and
  90%+ of users doing any other action also push that week; push mostly marks "active user".
- **Received activity is the strongest cross signal** for comments and reviews (3.4–3.8×),
  but it fires on only 0.5% of user-weeks: only cohort users' activity is collected, so most
  of what people receive is invisible.

## 3. pr_review and pr_review_comment

| Statistic | Value |
|---|---|
| P(pr_review in week \| pr_review_comment in week) | 0.75 |
| P(pr_review_comment \| pr_review) | 0.49 |
| Per-user P(comment \| review), users with ≥ 4 review-weeks (n = 1,974) | mean 0.47, std 0.28 |
| Review comments whose linked review is by the same user | 100% |
| Review comments linked to a review in PRReviews | 38% |
| Review comments that are replies (`in_reply_to_id` set) | 2% |
| PRReviews states | commented 49.9k, approved 36.9k, changes_requested 3.5k, dismissed 1.6k |

`pr_review_comment` is close to a subset of `pr_review`. Whether a reviewer also leaves line
comments is a personal habit that varies strongly between users (std 0.28). The 25% of
comment-weeks without a review come mostly from reviews missing from PRReviews (only 38% of
comments link to a review), not from replies.

## 4. Relationships with other tables and columns

Raw lifts on the label, from rows visible before each table's cutoff:

| Signal (source) | Share of user-weeks | open_pr | open_issue | issue_comment | pr_review | pr_review_comment |
|---|---|---|---|---|---|---|
| Pushed to a non-default branch, 7d (Pushes.branch) | 0.44 | 1.46 | 1.32 | 1.46 | 1.65 | 1.72 |
| Own PR opened < 30d and still open (PullRequests) | 0.46 | 1.57 | 1.30 | 1.44 | 1.60 | 1.64 |
| … from a fork (`is_cross_repo`) | 0.14 | 1.56 | 1.48 | 2.26 | 2.24 | 2.51 |
| … same repo | 0.36 | 1.65 | 1.35 | 1.29 | 1.48 | 1.46 |
| Merged someone else's PR, 30d (`merged_by_id`) | 0.38 | 1.66 | 1.41 | 1.44 | 1.60 | 1.61 |
| Assigned an open issue / PR seen < 60d (`assignee_id`) | 0.16 | 1.36 | 2.37 | 2.09 | 2.25 | 2.36 |
| Received comment / review on own items, 7d | 0.005 | 2.05 | 2.60 | 3.45 | 3.44 | 3.81 |

Repo demand, among users who reviewed in the last 90 days (40% of user-weeks): new PRs by
others in the repos they review, in the last visible 7 days.

| New PRs in their reviewed repos | Share | pr_review | pr_review_comment | issue_comment |
|---|---|---|---|---|
| ≥ 1  | 0.136 | 1.55 | 1.69 | 1.82 |
| ≥ 5  | 0.033 | 1.59 | 1.78 | 2.16 |
| ≥ 20 | 0.002 | 1.77 | 1.99 | 2.63 |
| none | 0.864 | 0.91 | 0.89 | 0.87 |

Other observations:

- **Issue comments target:** 28% own PRs, 27% others' PRs, 20% own issues, 24% others'
  issues; 3% are posted via an app.
- **Organisation repos:** 85% of reviews and review comments, 68% of issue comments, ~50% of
  opened PRs and issues happen in organisation repos.
- **Unusable columns:** `Pushes.is_new_ref` is always False; `PullRequests.draft` is null for
  95% of PRs; no cohort user is a bot.

These lifts are not controlled for overall activity: an active user is more likely to have an
open fork PR, an assignment and next week's activity at the same time. Section 6 measures what
is left after the baseline features.

## 5. Using the review / review-comment relationship in the GNN

`experiments/gnn_review/run.py` extends the heterogeneous GNN of `experiments/gnn` (which
already trains the six actions jointly) with:

1. **Habit features** on user nodes: weeks with review / review comment / both in 90 days,
   smoothed P(comment | review) and P(review | comment), comments per review, review state
   shares, reply and unlinked comment shares.
2. **Chain head:** two extra outputs P(comment | review) and P(comment | no review), giving
   P(comment) = P(review)·P(c | r) + (1 − P(review))·P(c | ¬r); validation chooses the direct
   head, the chain or their mean.
3. **Joint thresholds** tuned on the competition score instead of per-label F1.
4. **Consistency rule:** a predicted review comment also predicts a review.

Validation score, 3 seeds:

| Model | Seeds 0 / 1 / 2 | Mean |
|---|---|---|
| Original GNN | 48.05 / 48.21 / 47.28 | 47.85 |
| New script, without habit features and chain | 48.22 / 48.04 / 47.14 | 47.80 |
| + habit features only | 48.69 / 48.32 / 47.71 | 48.24 |
| + chain head only | 48.04 / 47.95 / 48.61 | 48.20 |
| + habit features + chain | 48.35 / 48.54 / 48.79 | **48.56** |
| … + joint thresholds | | 48.73 |
| … + consistency rule | | 48.74 |

- Habit features and the chain head add ~0.4 points each, ~0.7 together, and beat the
  original on every seed, but the gain is close to the seed-to-seed spread (0.3–0.5 points).
- `pr_review_comment` F1 itself barely moves (≈ 0.349 → 0.355); the gain spreads over
  `pr_review` and the other actions through the shared model.
- Joint thresholds are tuned and scored on the same two weeks, so their +0.17 is optimistic.
- The consistency rule changes nothing: the model rarely predicts a comment without a review.

## 6. Do cross-table features add information?

`experiments/relations/run.py` turns each relationship into a feature group and adds it to the
baseline's 60 features in the baseline LightGBM (2 seeds averaged). Baseline: score 49.56.

| Group | Δ score (points) | Δ F1 open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|---|
| org | +0.17 | +0.002 | +0.000 | +0.000 | +0.005 | +0.004 | +0.001 |
| push_branch | +0.14 | +0.001 | +0.005 | −0.003 | +0.005 | +0.001 | +0.000 |
| comment_mix | +0.13 | +0.001 | +0.004 | +0.002 | −0.003 | +0.003 | −0.000 |
| assigned | +0.08 | −0.002 | +0.002 | +0.003 | −0.001 | +0.008 | −0.002 |
| own_pr | −0.03 | −0.006 | +0.008 | +0.002 | −0.002 | +0.002 | −0.002 |
| demand | −0.08 | −0.000 | +0.003 | +0.005 | −0.002 | +0.006 | −0.001 |
| review_habit | −0.16 | −0.004 | +0.002 | −0.001 | +0.005 | −0.006 | +0.000 |
| maintainer | −0.21 | +0.002 | −0.001 | +0.001 | −0.004 | +0.002 | −0.002 |
| **all groups** | **+0.28** | +0.009 | +0.007 | −0.001 | +0.005 | +0.006 | +0.000 |

Every group alone stays within ±0.2 points, i.e. within noise. The strong raw lifts of
section 4 are mostly "this user is active", which the baseline's counts and recency features
already encode. Only the full set looks like a small real gain. The review habit features help
the GNN (+0.4) but not LightGBM, which can build such ratios from the raw counts itself.

## 7. Conclusions and next steps

- The actions form one collaboration cluster plus a weaker `open_pr` and an uninformative
  `push`. Sharing a model across actions (the GNN) and modelling `pr_review_comment` through
  `pr_review` uses this structure; it gives a small, consistent gain.
- Relationships with other tables are real but mostly redundant with activity volume; expect
  at most a few tenths of a point from them.
- Larger levers not yet tried:
  - **Outage handling in the GNN.** Its training weeks include June–early September, when
    non-push collection ran at ~5–20% of normal, plus push outages in August–October; the
    baseline's `--gap-mode` options exist for this, the GNN has none.
  - **Seed ensembling.** The seed spread (0.3–0.5 points) is as large as most gains above.
  - **All relation feature groups on the GNN user nodes** (untested).

## Reproduce

```bash
python experiments/gnn/run.py --name ref_default --seed 0          # original GNN
python experiments/gnn_review/run.py --seed 0                      # section 5, all changes
python experiments/gnn_review/run.py --seed 0 --no-habit --no-chain  # ablation
python experiments/relations/run.py                                 # section 6
```

Results are written to `experiments/<name>/outputs/` (`metrics.json`, `screen.json`). The
descriptive statistics of sections 2–4 came from a one-off analysis script that is not in the
repo; the same definitions are implemented in `experiments/relations/run.py`.
