# The LightGBM missing-data fixes in the GNN (`--relative-features`, `--push-share prev-week`)

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
GPU=0 HO_WEEK="2026-09-26 16:00" CONFIGS="all_ego_uf all_ego_uf_rel" bash runs/ho0927.sh
GPU=1 HO_WEEK="2026-11-07 16:00" CONFIGS="all_ego_uf all_ego_uf_rel" bash runs/ho0927.sh
```

The `*_pprev` configs of `runs/ho0927.sh` are not needed for these numbers: a run's
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
