# Drift-robust LightGBM

This experiment tests every follow-up idea from the 2026-11-04 review of the
leaderboard in one runner, under one protocol, on the official test
population. Each idea is a named **variant**; a backtest scores every variant
twice (validation and holdout) and writes a comparison table.

**Bottom line.** On the population the test is drawn from, LightGBM with the
relational features (`relational_lightgbm`, 54.08) is the best-supported model.
Graph models score far lower there (RelGT 46.11, GNN 44.66). None of the
drift fixes (masking ingestion anomalies, recency weights, refitting on recent
weeks, seed bagging, new feature sets) gives a gain that holds on both
validation and holdout; the best holdout variant is +0.5 points but wins only 3
of 5 weeks and loses on validation. The one robust effect is the **push
ingestion lag**: serving with a 6-day push lag costs 2.2 points, and training
with the same lag recovers 1.2 of them.

## Why: three problems in the data and the old comparison

**1. The graph leaderboard used a different, easier population.** RelGT, GNN,
the chunk/specialist models and relgt_db kept users with at least 5 events in
the **14-day** graph lookback (~1.7k users/week; base rate of `push` 0.71).
LightGBM and the official test list use at least 5 events in **90 days**
(~4.7k users/week, 3,960 test users; push base rate 0.44). On identical rows,
LightGBM was better than RelGT on every label (mean AUPRC 0.611 vs 0.550), and
re-running the graph models on the 90-day population (new `--eligibility-days 90`
flag in `experiments/relgt/run.py`) confirms it:

| Model, 90-day population, same 23,473 holdout rows | Score |
|---|---|
| LightGBM baseline (`base`) | 53.39 |
| Relational LightGBM (`relational`) | **54.08** |
| RelGT (`relgt --eligibility-days 90`) | 46.11 |
| GNN (`gnn --eligibility-days 90`) | 44.66 |

Blending RelGT or GNN probabilities into LightGBM adds at most ~0.4 points
(weight 0.8-0.9 chosen on the holdout itself, so an upper bound); not adopted.

**2. Activity levels drift strongly between weeks.** Distinct cohort users with
each action per weekly window:

| Week starting | Jul 8 - Aug 26 | Sep 2 | Sep 9 | Sep 16 | Oct 7 | Oct 21 |
|---|---|---|---|---|---|---|
| open_pr | ~100-200 | 1,168 | 1,304 | 610 | 388 | 296 |
| issue_comment | ~65-210 | 909 | 1,023 | 503 | 252 | 202 |
| push | 2,500-3,900 | 1,168 | 1,765 | 2,558 | 2,377 | 1,551 |

The default split trains on Mar 11 - Aug 5, calibrates on Aug 12 - Sep 9 (into
the surge) and holds out Sep 16 - Oct 14 (the decay). That is why per-label
threshold calibration chose 0.05-0.10 and fell from 53.4 to 43.4.

**3. Ingestion anomalies.** Pushes has no rows on Sep 2-4, Oct 7, Oct 15-17
and after Oct 26 (another ~6 days before the other tables end on Nov 1), and
almost none on Oct 23. Most of Sep 3-17 and Aug 7 are days where all five
non-push tables jump together. `data_quality.py` flags a day by comparing it
with the median of the same weekday over the previous 4 weeks: push outages
individually, non-push anomalies only when all five tables agree (single-table
swings are user behaviour). A (cutoff, label) pair is flagged when its target
week touches a flagged day of that label's table.

## Protocol

* Examples: the LightGBM baseline's (Wednesday 16:00 UTC cutoffs from
  2026-03-11, 5-actions-in-90-days eligibility, 7-day labels), cached under
  `cache/`.
* Split: 70/15/15 of the 32 weekly cutoffs. **Validation** = fit on the 22
  training weeks, score the 5 calibration weeks (Aug 12 - Sep 9). **Holdout** =
  score Sep 16 - Oct 14 with the variant's fitting scheme (`fixed`: training
  weeks only; `refit`: + calibration weeks; `rolling`: refit before each
  holdout week on every week whose label window has ended).
* Metrics: official `(macro-F1 + micro-F1) / 2` per week, averaged. The
  **masked** score drops, per week, labels flagged as anomalous (e.g. push in
  Sep 30 - Oct 14); it is only comparable between variants, not with the
  official score. mAP = mean average precision over labels (threshold-free).
* Thresholds: fixed 0.30 (push 0.45) unless the variant tunes them.
* Variants should be chosen on validation. Validation is dominated by the
  September surge, so holdout is also reported; a change is called an
  improvement only if it does not lose on either.

## Results (data release ending 2026-11-01)

Scores in %. Full tables: `outputs/grid1/RESULTS.md` (round 1) and
`outputs/grid2/RESULTS.md` (round 2).

| Variant | What it changes | Val | Val mAP | Holdout | Holdout mAP |
|---|---|---|---|---|---|
| `base` | baseline LightGBM features | 34.70 | 42.20 | 53.39 | 51.76 |
| `relational` | + relational features | **34.85** | **42.79** | **54.08** | **53.05** |
| `history` | + 4/12/26-week habit, 30/90-day counts, trend, streak, account age | 33.76 | 42.19 | 53.80 | 53.16 |
| `regime` | + cohort action rate last week / last 4 weeks | 32.09 | 41.18 | 53.73 | 52.54 |
| `all_features` | + push outage days in the feature windows | 32.09 | 41.18 | 53.73 | 52.54 |
| `mask` | all features, anomalous label weeks get weight 0 | 31.68 | 42.67 | 53.99 | 52.65 |
| `half_life_4` | mask + recency weights, half-life 4 weeks | 31.71 | 43.17 | 54.50 | 52.81 |
| `half_life_8` | mask + half-life 8 weeks | 31.67 | 43.29 | 54.20 | 52.89 |
| `last_8_weeks` | mask + train on the last 8 weeks only | 28.37 | 36.62 | 44.72 | 50.17 |
| `refit` | mask + refit on train + calibration weeks | 31.68 | 42.67 | 54.53 | 53.12 |
| `rolling` | mask + refit before every holdout week | 31.68 | 42.67 | 54.39 | 52.82 |
| `seeds_5` | rolling + half-life 8 + 5-seed bagging | 31.58 | 43.40 | **54.59** | 52.88 |
| `joint_thresholds` | rolling + half-life 8 + thresholds tuned on validation | 40.85* | 43.29 | 45.59 | 52.74 |
| `rel_mask` | relational features + masking | 34.85 | 42.38 | 53.93 | 52.94 |
| `rel_rolling_half_life_8` | relational + mask + rolling + half-life 8 | 34.16 | 42.52 | 54.12 | 52.85 |
| `rel_final` | the above + 5-seed bagging | 33.93 | 42.55 | 54.04 | 52.88 |

\* scored on the weeks it was tuned on.

Per-week holdout difference against `relational` (Sep 16, 23, 30, Oct 7, 14):

| Variant | Mean | Weekly difference (points) | Weeks won |
|---|---|---|---|
| `seeds_5` | 54.59 | -0.5, +1.7, +0.4, +1.0, -0.1 | 3/5 |
| `refit` | 54.53 | -1.2, +1.8, +1.1, +1.2, -0.5 | 3/5 |
| `half_life_4` | 54.50 | -0.4, +1.3, +1.1, +0.8, -0.8 | 3/5 |
| `rel_final` | 54.04 | -0.3, -0.1, 0.0, +0.3, 0.0 | 1/5 |

Push ingestion lag (6 days, as in this release):

| Variant | Holdout | Change |
|---|---|---|
| `rolling_half_life_8` (no lag) | 54.41 | |
| `lag6_naive` (trained without lag, served with lag) | 52.26 | -2.15 |
| `lag6_matched` (trained and served with lag) | 53.49 | -0.92 |
| `rel_final` → `rel_final_lag6` (matched lag) | 54.04 → 52.75 | -1.29 |

### What we learned

* **Features.** Relational features help on both splits (+0.7 holdout, +0.6
  val mAP). The history features raise holdout mAP slightly but lower F1 at the
  fixed thresholds (pr_review_comment 42.1 → 38.2) and lower validation. The
  regime features hurt both: with only 22-32 distinct values the trees learn
  week identity. The push coverage features are never used (identical scores).
* **Masking anomalous label weeks** is neutral on the relational model and +0.3
  on the all-features model.
* **Recency.** Short windows are harmful (`last_8_weeks` trains on the quiet
  summer and predicts almost no review comments). Half-life weights and
  refits gain +0.4-0.6 on the all-features model, i.e. they repair what the
  extra features cost, and are neutral (±0.5) on the relational model.
* **Thresholds.** Any tuning on these weeks fails because the positive rate
  moves 3-10x between weeks: tuned on validation → 45.6. An offline check with
  thresholds tuned on the holdout itself gained only +0.7, and predicting the
  previous week's positive count per label (top-k) scored below the fixed
  thresholds. Keep the fixed 0.30/0.45.
* **Bagging** 5 seeds: +0.1-0.2, within noise.
* **Push lag** is the biggest controllable risk for the real submission.

## Recommendation for the submission

1. Download the final release (11/11) and run
   `python -m experiments.drift_lightgbm.run predict` (variant `relational`). It
   trains on **every** week whose label window has ended (the
   `relational_lightgbm` predict command holds back the latest week for calibration)
   and uses the fixed thresholds.
2. Read the printed line `Push data ends ... other tables end ...`. If pushes
   lag by more than a day, predict with a lag-matched variant instead, e.g.
   add a `rel_*` variant with `train_push_lag` equal to the observed lag (the
   backtest shows matched training recovers about half of the loss).
3. `metadata.json` records the training weeks, data end dates and predicted
   positive rates; compare the rates with recent weeks as a sanity check.

## Run

```bash
python -m experiments.drift_lightgbm.run backtest                       # every variant
python -m experiments.drift_lightgbm.run backtest --variants relational,rel_final --headline rel_final
python -m experiments.drift_lightgbm.run predict                          # variant: relational
python -m experiments.drift_lightgbm.run predict --variant rel_final_lag6 --cutoff "2026-11-11 16:00:00"
python -m pytest experiments/drift_lightgbm
```

The first backtest builds the examples (a few minutes) into
`cache/examples_lag<N>_<hash>.parquet`; the hash covers the feature version,
start cutoff, push lag and the Parquet files' sizes and mtimes, so a new data
release rebuilds automatically. A backtest writes `RESULTS.md`,
`backtest_metrics.json` (headline variant plus every other variant under
`branch_metrics`, which `run_all_experiments.py` lists as branches) and
`variants/<name>_holdout.csv`. Prediction writes `predictions.csv`,
`probabilities.csv` and `metadata.json`.

## Files

| File | Content |
|---|---|
| `run.py` | example building and caching, variants, protocol, backtest and predict commands |
| `features.py` | history features and cohort-regime features |
| `data_quality.py` | outage/burst detection, label-week flags, push coverage features |
| `test_drift_lightgbm.py` | cutoff-safety, anomaly, protocol and scorer tests (the scorer is checked against `evaluate.py`) |

## Experiments removed in this round

Experiments that scored below 50 in `experiments/results/20261103_160623/RESULTS.md`
were removed: their registry entries in `run_all_experiments.py`, their code
paths and their commands in `cmd.md` and the READMEs.

| Experiment | Score | Removed |
|---|---|---|
| `s2p2`, `s2p2_user_emb`, `s2p2_frac50` | 38.18, 39.76, 36.39 | `--temporal-model s2p2` in `lightgbm_baseline/run.py` (`s2p2.py` stays for RelGT's hybrid modes) |
| `lookback_chunks`, `lookback_chunks_90_30_7` | 43.34, 45.35 | the `experiments/relgt_lookback_chunks/` package |
| `relgt_ensemble`, `relgt_ensemble_90_14` | 48.36, 44.79 | the `relgt_ensemble` mode (`gnn_ensemble` stays) |
| `gnn_ensemble_user_emb` | 49.79 | `--user-embedding` is rejected with `gnn_ensemble` |
| `lgbm_calibrated` | 43.36 | `--calibrated-threshold` in `lightgbm_baseline`, `relational_lightgbm` and `exam_fail_case.py` |
| `freq_nn_user_emb` | 45.37 | `--user-embedding` is rejected with `frequency_nn` |
| `relgt_db_frac50` | 49.63 | registry entry and `cmd.md` line (`--train-fraction` itself stays) |
