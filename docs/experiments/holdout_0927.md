# Holdout week 09-27 → 10-04 with push (`--holdout-week`)

Week 11-07 cannot score push (GH Archive has no PushEvents after 10-26). The week 09-27 → 10-04
(Taiwan time; 09-26 16:00 UTC) has labels for all 6 activities, so `--holdout-week` runs the same
protocol there and scores the 6-activity official score:
- training weeks: labels ending by 09-19;
- tuning week 09-19 → 09-26 (early stopping and the share of predicted positives);
- holdout week 09-26 16:00 UTC, no retrain, no `output.csv`.

```bash
python experiments/gnn/run.py --holdout --holdout-week "2026-09-26 16:00" --user-features   # -> outputs/all_ego_uf_ho0926/
bash runs/ho0927.sh   # het, het_notime, all_ego_uf, user_repo x seeds 0-2 -> outputs/<cfg>_ho0926_s*
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

## Chain-graph models on the same week (`runs/chain_ho0927.sh`)

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
bash runs/chain_ho0927.sh   # relgt_chain on GPU 0, gnn_chain on GPU 1, seeds 0-2; summary next to runs/ho0927.sh
SEQUENTIAL=1 bash runs/chain_ho0927.sh   # both on GPU 0, one after another (less RAM)
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

## Quiet users copy their previous week (`--quiet-rule prev-week`, `runs/qprev_ho0927.sh`)

```bash
bash runs/qprev_ho0927.sh                         # active = >= 5 events (old experiment's rule), seeds 0-2 -> outputs/ego_uf_qprev5_ho0926_s*
CONFIGS="ego_uf_qprev" bash runs/qprev_ho0927.sh  # active = >= 1 event -> outputs/ego_uf_qprev_ho0926_s*
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
old experiment (`archive/old_experiments/relgt_inspired/data.py`, `eligible_only`) kept a user as an
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

## Where the score is lost: quiet vs active users (`experiments/segments/run.py`)

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
archive drops a share of events at random (see [Handling GH Archive's missing events](missing_data.md)). The script measures them in two ways:
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
