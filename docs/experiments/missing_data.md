# Experiment: handling GH Archive's missing events (`experiments/missing_data/run.py`)

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
  validation scores elsewhere in these docs.
- Neither `relative` nor the push rule is in `experiments/baseline/run.py` yet. Both are flags of the
  GNN runner. For the GNN, the push rule helps (+6.0 on 11-07) and `relative` does not (see
  "The LightGBM missing-data fixes in the GNN").
- LightGBM runs with `--threads 8` (default). With all 28 cores it was ~170× slower under WSL
  (100 rounds: 187 s against 1.1 s).
