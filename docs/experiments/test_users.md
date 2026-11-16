# Experiment: train on the test users only (`runs/test_users_ho0927.ps1`)

Every runner trains on all eligible cohort users of each week (about 6,000 a week), while the
submission scores only the 4,465 users of `test_input.csv`. This experiment trains the best model on
the holdout week 09-27 → 10-04, the chain RelGT (`relgt_chain`; [holdout_0927](holdout_0927.md),
[ensemble](ensemble.md)), on the test users only, to see whether fitting the scored population beats
having more examples.

## Setup

- Model and protocol: `relgt_chain` of `runs/ensemble_ho0927.sh` (chain graph, 16 latest actions as
  seeds, 15 epochs; training labels end by 09-19, tune week 09-19 → 09-26, holdout week 09-26 16:00 UTC).
- Change: `--train-users test --val-users test --final-users test` (options of
  `experiments/relgt/run.py`, unused before). Training, early stopping, the decision shares and the
  final refit see only test users' examples; the graph and tokens are unchanged (other users' rows are
  still visible as context).
- One seed (s0). Run name `relgt_chain_tu_ens0926_s0`.
- Comparison: `experiments/ensemble/run.py relgt_chain_tu relgt_chain --suffix ens0926`. It keeps
  the users every run has, so the reference runs `relgt_chain_ens0926_s0-2` (trained on all users) are
  scored on the same test users of the holdout week, with the same decision rule (share predicted
  positive from the tune week, restricted to the same users). Compare `seed:relgt_chain_tu_ens0926_s0`
  with `seed:relgt_chain_ens0926_s*` and `bag:relgt_chain`. The `own:` rows are not comparable: the
  reference runs' own decisions were made and scored on all users.

```powershell
.\runs\test_users_ho0927.ps1              # 1 seed (~30 min on one GPU), then the comparison
.\runs\test_users_ho0927.ps1 -Smoke       # pipeline check: 1 epoch of 30 steps
.\runs\test_users_ho0927.ps1 -SkipRuns    # only the comparison
.\runs\test_users_ho0927.ps1 -Gpu 1       # force a GPU
```

Note: `data/db/` was rebuilt on 2026-11-16 with GH Archive up to 11-16 08:00 UTC, after the reference
runs. The rebuild renumbered PR ids but left user ids and the labels of weeks before 11-14 as they were;
`experiments/ensemble/run.py` stops if the two runs' labels differ.

## Results

Run on 2026-11-16 (`.\runs\test_users_ho0927.ps1`, seed 0, GPU 1, 12 min; log
`experiments/gnn/outputs/logs/relgt_chain_tu_ens0926_s0.log`, comparison in
`experiments/ensemble/outputs/relgt_chain_tu+relgt_chain_ens0926/metrics.json`).

| | Test users only (`relgt_chain_tu`) | All users (`relgt_chain`, reference) |
|---|---|---|
| Training examples | 135,490 | 237,683 |
| Tune-week users | 4,168 | 6,412 |
| Best epoch | 8 | 11, 8, 12 |
| Time per epoch | ~29 s | – |
| Run's own holdout score (its own users) | 56.42 final / 56.26 validation model, 4,152 test users | 53.04 / 53.01 / 51.72 final model, 6,064 users |

The run's own scores are on different users, so they cannot be compared. The comparison below uses the
4,152 test users eligible in the holdout week (and the 4,168 of the tune week) for every run. Shares come
from the tune week of these users, and RelGT's final-model probabilities are used:

| Variant | Score | Tune | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs `bag:relgt_chain_tu` [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| seed:relgt_chain_tu_s0 | 56.52 | 54.64 | 0.433 | 0.477 | 0.434 | 0.490 | 0.446 | 0.740 | 0 |
| seed:relgt_chain_s0 | 56.57 | 53.48 | 0.444 | 0.478 | 0.437 | 0.497 | 0.450 | 0.739 | |
| seed:relgt_chain_s1 | 56.09 | 54.23 | 0.392 | 0.476 | 0.428 | 0.484 | 0.441 | 0.741 | |
| seed:relgt_chain_s2 | 55.00 | 54.11 | 0.419 | 0.456 | 0.426 | 0.430 | 0.430 | 0.734 | |
| bag:relgt_chain (3 seeds) | **57.28** | 53.94 | 0.439 | 0.480 | 0.469 | 0.504 | 0.451 | 0.740 | +0.77 [−0.02, +1.56] |
| mean (tu + bag of all-user runs) | 56.76 | 54.61 | 0.436 | 0.493 | 0.441 | 0.487 | 0.444 | 0.736 | +0.25 [−0.33, +0.84] |
| select | 56.56 | 54.62 | 0.433 | 0.477 | 0.434 | 0.490 | 0.451 | 0.740 | +0.04 [−0.27, +0.35] |
| own:relgt_chain (all-user decisions, scored on these users) | 53.61 | – | 0.404 | 0.464 | 0.410 | 0.451 | 0.443 | 0.707 | |

### Reading

- **Training on test users only does not beat training on all users.** The single seed scores 56.52 on
  the test users. The three all-user seeds score 55.00–56.57 (mean 55.89) on the same users. It is inside
  that range, +0.6 over its mean, and one seed against a spread of 1.6 points does not separate them.
  Its tune-week score is the highest (54.64), and so is its validation mean AP (0.461 at epoch 8).
- **Seed bagging of the all-user model is still the best.** `bag:relgt_chain` scores 57.28, +0.77
  [−0.02, +1.56] over the test-only seed (the interval is over users, not seeds). As in
  [ensemble](ensemble.md), bagging RelGT seeds gives about +1.
- **The decision shares matter more than the training users.** The all-user runs' probabilities score
  53.61 with their own decisions (shares tuned on all 6,412 tune-week users) and 55.00–56.57 when the
  shares are tuned on the test users of the tune week. That is about +2.3 points, mostly on open_pr,
  pr_review and push (0.707 → 0.74). The eligible users outside the test set are almost all inactive:
  in the tune week 09-19, the 2,244 of them have positive rates of 0.000–0.003 on the five non-push
  activities and 0.034 on push, against 0.057–0.104 and 0.556 for the 4,168 test users (0.037–0.068 and
  0.373 over all 6,412). A share tuned on all eligible users is therefore too small for the test users.
- **Cost:** 57% of the training examples (135,490 against 237,683), 12 min for the whole run on one GPU.
- One holdout week, one seed of the new config.

### What this suggests

By default (`--val-users all`) the runners take the decision shares from all eligible users of the tune
week. Tuning them on the test users of the tune week, while still training on all users, is the change
this experiment supports. It was not run as its own config here: the comparison applied it to the saved
probabilities of the all-user runs. In `experiments/relgt/run.py`, `--val-users test` would also move
early stopping onto the test users. The push share of a submission under `--push-rule prev-week` would
need the same restriction.
