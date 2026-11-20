# Experiment: train on all 9,000 cohort users (`--train-pool cohort`, `runs/relgt_pool_ho0927.ps1`)

Every runner trains on the *eligible* users of each week: at least 3 records in the 90 days before the selection
time shifted to that week ([test_users](test_users.md) has the numbers), about 6,000 of the 9,000 cohort users. The
other users were never examples; their rows are only graph context. [test_users](test_users.md) tried the opposite
(test users only, 57% of the examples) and found no gain. This experiment adds the rest instead: every cohort user is
an example in the training weeks and in the final fit.

## Setup

- Model and protocol: `relgt_chain` of `runs/ensemble_ho0927.sh` (chain graph, 16 latest actions as seeds, 15
  epochs; training labels end by 09-19, tune week 09-19 → 09-26, holdout week 09-26 16:00 UTC), plus
  `--train-pool cohort` (new option of `experiments/relgt/run.py`, default `eligible`).
- What changes: in the weeks of the training and final fits each week's examples are its eligible users followed by
  every other cohort user. Labels are already computed for all users; a missing label stays masked in the loss.
- What does not change: early stopping, the decision shares (tune week) and the holdout week use the eligible users
  only (`ids[w]` in `run.py` covers the eligible part of a week's block), so scores are on the same users as the
  reference runs and compare directly.
- One seed (s0). Run name `relgt_chain_pool_ens0926_s0`.
- Comparison: `experiments/ensemble/run.py relgt_chain_pool relgt_chain --suffix ens0926` against the reference runs
  `relgt_chain_ens0926_s0-2`. Compare `seed:relgt_chain_pool_ens0926_s0` with `seed:relgt_chain_ens0926_s*` and
  `bag:relgt_chain`.

```powershell
.\runs\relgt_pool_ho0927.ps1              # 1 seed (~30-45 min on one GPU), then the comparison
.\runs\relgt_pool_ho0927.ps1 -Smoke       # pipeline check: 1 epoch of 30 steps
.\runs\relgt_pool_ho0927.ps1 -SkipRuns    # only the comparison
```

## What to expect, and what limits the comparison

- The added users are almost all inactive: in the tune week 09-19 the 2,244 eligible users outside the test set have
  positive rates of 0.000-0.003 on the five non-push activities ([test_users](test_users.md)), and the ~3,000
  ineligible users have even less history. The extra examples are mostly easy negatives.
- The token cache key changes, because the seed groups differ, so every week is sampled again (~10 min once). The
  random stream differs from the reference runs, so even the eligible users get different tokens. That adds a
  difference next to the training pool; the three reference seeds span 0.9-2.1 points.
- One holdout week, one seed: a difference below the reference seed range does not separate the two.

## Results

Run `.\runs\relgt_pool_ho0927.ps1` (seed 0, `relgt_chain_pool_ens0926_s0`, best epoch 13). 315,000 training examples
(237,683 eligible + 77,317 other cohort users) against 237,683 for the reference. Holdout week 09-26 16:00 UTC, 6,064
users; tune week 09-19, 6,412 users. Official 6-activity score.

### The run's own decisions (final model refit through the tune week; validation model trained to 09-19)

| | Score | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|---|
| `relgt_chain_pool`, validation model (1 seed) | 51.17 | 0.342 | 0.391 | 0.409 | 0.476 | 0.433 | 0.675 |
| `relgt_chain`, validation model (3 seeds) | 52.05 (52.00-52.14) | 0.404 | 0.426 | 0.391 | 0.451 | 0.446 | 0.675 |
| `relgt_chain_pool`, final model (1 seed) | 52.63 | 0.400 | 0.454 | 0.425 | 0.471 | 0.436 | 0.676 |
| `relgt_chain`, final model (3 seeds) | 52.59 (51.72-53.04) | 0.402 | 0.462 | 0.407 | 0.447 | 0.443 | 0.681 |

### Same decision rule on the same users (`experiments/ensemble/run.py relgt_chain_pool relgt_chain`)

Shares from the tune week of the combined probabilities, applied to the holdout week (final-model probabilities). The
reference for the bootstrap is `bag:relgt_chain_pool` (best tune-week score; it is the single run).

| Variant | Score | Tune | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs reference [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| seed:relgt_chain_pool_s0 (all users) | 52.50 | 51.94 | 0.441 | 0.454 | 0.425 | 0.471 | 0.425 | 0.673 | 0 |
| seed:relgt_chain_s0 | 53.85 | 51.01 | 0.444 | 0.469 | 0.437 | 0.495 | 0.456 | 0.682 | |
| seed:relgt_chain_s1 | 53.01 | 51.53 | 0.384 | 0.481 | 0.412 | 0.482 | 0.442 | 0.683 | |
| seed:relgt_chain_s2 | 51.72 | 51.42 | 0.407 | 0.434 | 0.421 | 0.411 | 0.433 | 0.679 | |
| bag:relgt_chain (3 seeds) | **54.42** | 51.63 | 0.434 | 0.481 | 0.461 | 0.502 | 0.454 | 0.684 | +1.92 [+1.10, +2.77] |
| mean (pool + bag of relgt_chain) | 53.59 | 52.53 | 0.428 | 0.474 | 0.433 | 0.483 | 0.448 | 0.679 | +1.09 [+0.41, +1.83] |
| select | 53.41 | 52.29 | 0.441 | 0.454 | 0.425 | 0.471 | 0.454 | 0.684 | +0.91 [+0.43, +1.37] |

The official scorer reproduces `mean` (53.59). `select` took open_pr, open_issue, issue_comment, pr_review from the
pool run and pr_review_comment, push from `relgt_chain`.

### Reading

- **Training on all users does not beat training on eligible users only.** The pool seed scores 52.50 on the common
  decision rule. The three reference seeds score 53.85, 53.01 and 51.72 (mean 52.86), so it is inside their range and
  0.36 below their mean. With its own decisions the final model is level (52.63 against 52.59), and the validation
  model is lower (51.17 against 52.05-52.14, below the reference range). One seed cannot separate these.
- **Per activity** (pool seed against the mean of the three reference seeds, same rule): open_pr 0.441 against 0.412
  and pr_review 0.471 against 0.463 are higher; open_issue 0.454 against 0.461, pr_review_comment 0.425 against 0.444
  and push 0.673 against 0.681 are lower; issue_comment is level (0.425, 0.423). Nothing is outside the spread of the
  reference seeds except possibly pr_review_comment and push, which are slightly below all three of them or at their
  lower edge.
- **The tune week again ranks the runs wrongly.** The pool run has the highest tune-week score of the single runs
  (51.94, against 51.01-51.53) and a below-average holdout score. This is the same pattern as in
  [ensemble](ensemble.md) (the lowest tune score, `bag:relgt_chain`, had the highest holdout score).
- **The +1.92 [+1.10, +2.77] of `bag:relgt_chain` is the seed bag, not the user set.** It compares a bag of three seeds
  with one run, so it repeats the bagging result of [ensemble](ensemble.md) (54.42 against single seeds 51.7-53.9).
  The interval is over users, not seeds. Mixing the pool run into the bag (`mean`, 53.59) lowers it from 54.42.
- **Cost:** 33% more training examples (315,000 against 237,683), 15 epochs run (best epoch 13, against 8-12), and
  token sampling redone for all weeks (~8 min).
- One seed, one holdout week, and the tokens of the eligible users are sampled with a different random stream than in
  the reference (the seed groups differ), so the comparison also contains sampling noise.

**Conclusion:** no evidence that adding the ~3,000 users without a history window helps. The extra examples are almost
all easy negatives, which fits the small differences. It is not adopted; the WRITEUP.md table and the submission
pipeline are unchanged. A second seed would be needed to say more. With padding tokens on top (3 seeds) the all-users
model reaches 53.82 as a bag, still below `bag:relgt_chain` (54.42): see [relgt_poolpad](relgt_poolpad.md).
