# Experiment: RelGT related rows (`--related-rows`, `runs/related_ho0927.ps1`)

The chain walk of `relgt_chain_recent` reaches an issue, comment or review, but not always the rows that give it
its context: the issue a comment is on, the other comments on that issue, a review's PR and line comments, the
comment a review comment replies to. These rows compete with repository neighbours and pushes for the K = 100
tokens. `--related-rows N` makes them come together. Every token that is an issue, issue comment, PR review or
PR review comment comes with up to N rows related to it through its foreign keys.

## Related rows (`RELATED` in `experiments/relgt/run.py`)

| Token | Related rows (FK path) |
|---|---|
| Issues | comments on it (`IssueComments.issue_id`, reverse); its PR (`Issues.pull_request_id`) |
| IssueComments | its issue (`issue_id`); the other comments on that issue (`issue_id`, then reverse) |
| PRReviews | its PR (`pr_id`); its line comments (`PRReviewComments.pull_request_review_id`, reverse); the PR's other reviews |
| PRReviewComments | its PR, its review, the comment it replies to (`pr_id`, `pull_request_review_id`, `in_reply_to_id`); replies to it; the PR's other review comments |

Comments on a PR's conversation are IssueComments, so they are covered by the IssueComments row.

## Sampling rule

- Only rows visible before the example's cutoff count, so nothing leaks.
- Up to N per token, ranked 1 FK step before 2 steps, then newest first. A step from one node keeps at most its
  1,000 newest rows.
- **Seeds.** The related rows of the 16 seed actions are taken before the walk. They can use at most half of the
  budget left after the seeds.
- **Walk rows.** A row's related rows (those not yet tokens and not in the same hop) count towards its cost. In
  the hop's newest-first order, a row is taken only if it and its related rows fit. Its related rows are taken
  right after the hop. Their own related rows come next, while budget is left.
- A related row is a token at its source's hop + its FK steps. It brings in its owner as walk rows do (another
  user's row costs one more token), and the walk goes on from it.
- With `--related-rows 0` (the default) the tokens are identical to before. This was checked against the
  committed code.

Check on 2,000 test users at the test week (`relgt_chain_recent` settings, N = 5). Of the related rows of each
issue, comment or review token (up to 5 each), the table gives the share that are tokens too:

| | Related rows sampled | 1 FK step | 2 FK steps | Users whose tokens change | Tokens changed per user |
|---|---|---|---|---|---|
| relgt_chain_recent | 84.3% | 82.6% | 90.8% | | |
| + `--related-rows 5` | 97.9% | 97.8% | 98.1% | 52% | 2.6 (of 100) |

The token mix shifts slightly from pushes (0.579 → 0.566) to issue comments (0.037 → 0.049) and PRs. Sampling
2,000 users takes 5 s, against about 3 s without the flag.

Applying the rule only to the 16 seed actions was tried first. It changed tokens for only 4.3% of users (0.05
tokens each), because the walk already reaches a seed's related rows. Since the data holds only cohort users'
activity, threads are short.

## Setup

| Config | Sampling |
|---|---|
| `relgt_chain` (reference) | chain, 16 latest actions, fk-order random |
| `relgt_chain_recent` (reference) | chain, 16 latest actions, fk-order recent |
| `relgt_chain_recent_related` | `relgt_chain_recent` + `--related-rows 5` |

Everything else is as in [sampling.md](sampling.md): K = 100, 15 epochs, training labels end by 09-19, tune
week 09-19 → 09-26, holdout week 09-26 16:00 UTC. Both the validation model and the final model are scored.
Seeds 0–2, runs `<cfg>_ens0926_s<seed>`. The reference runs already exist and are not rerun.

```powershell
.\runs\related_ho0927.ps1                 # seeds 0-2 on GPUs 0 and 1, then the summaries
.\runs\related_ho0927.ps1 -Seeds 0 -Gpus 1
.\runs\related_ho0927.ps1 -Smoke          # pipeline check: seed 0, 1 epoch of 30 steps
.\runs\related_ho0927.ps1 -SkipRuns       # only the summaries
```

## Results

Run 2026-11-17 (`runs/related_ho0927.ps1`), holdout week 09-26 16:00 UTC (6,064 users), tune week 09-19
(6,412 users), seeds 0-2 each. Official 6-activity score. The ensemble step (the rank-mean pipeline of
[ensemble.md](ensemble.md)):

```bash
python experiments/ensemble/run.py --suffix ens0926 relgt_chain relgt_chain_recent relgt_chain_recent_related
```

Output: `experiments/ensemble/outputs/relgt_chain+relgt_chain_recent+relgt_chain_recent_related_ens0926/metrics.json`.

### Single configs, mean of 3 seeds (the runs' own decisions)

| Config / model | Score (min-max) | Best epochs | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|---|---|
| relgt_chain, validation | 52.05 (52.00-52.14) | 11, 8, 12 | 0.404 | 0.426 | 0.391 | 0.451 | 0.446 | 0.675 |
| relgt_chain, final | 52.59 (51.72-53.04) | 11, 8, 12 | 0.402 | 0.462 | 0.407 | 0.447 | 0.443 | 0.681 |
| relgt_chain_recent, validation | 52.33 (51.95-52.73) | 7, 12, 10 | 0.380 | 0.432 | 0.405 | 0.455 | 0.441 | 0.679 |
| relgt_chain_recent, final | **53.41** (52.63-53.90) | 7, 12, 10 | 0.419 | 0.451 | 0.442 | 0.467 | 0.430 | 0.683 |
| relgt_chain_recent_related, validation | 52.03 (51.06-52.97) | 6, 5, 15 | 0.386 | 0.436 | 0.360 | 0.459 | 0.436 | 0.678 |
| relgt_chain_recent_related, final | 52.89 (52.64-53.06) | 6, 5, 15 | 0.420 | 0.460 | 0.376 | 0.460 | 0.427 | 0.687 |

### Seeds, bags and combinations (same decision rule, push share from the tune week, ranks averaged)

| Variant | Score | Tune | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs reference [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| seed:relgt_chain_recent_related s0 | 53.58 | 52.00 | 0.426 | 0.485 | 0.400 | 0.498 | 0.429 | 0.686 | |
| seed:relgt_chain_recent_related s1 | 53.20 | 52.18 | 0.395 | 0.462 | 0.420 | 0.473 | 0.434 | 0.689 | |
| seed:relgt_chain_recent_related s2 | 52.80 | 51.79 | 0.439 | 0.433 | 0.384 | 0.467 | 0.439 | 0.682 | |
| bag:relgt_chain | **54.42** | 51.63 | 0.434 | 0.481 | 0.461 | 0.502 | 0.454 | 0.684 | +0.36 [-0.44, +1.13] |
| bag:relgt_chain_recent (reference) | 54.06 | 52.56 | 0.447 | 0.459 | 0.408 | 0.510 | 0.460 | 0.683 | 0 |
| bag:relgt_chain_recent_related | 53.71 | 52.16 | 0.434 | 0.492 | 0.385 | 0.504 | 0.441 | 0.683 | -0.35 [-1.10, +0.41] |
| mean | 53.44 | 52.41 | 0.439 | 0.489 | 0.380 | 0.488 | 0.457 | 0.678 | -0.61 [-1.28, +0.04] |
| select | 54.57 | 52.62 | 0.447 | 0.459 | 0.461 | 0.504 | 0.460 | 0.683 | +0.51 [-0.01, +1.02] |

The official scorer on `mean` gives 53.44 (matches). `select` picked: open_pr, open_issue, pr_review_comment <-
relgt_chain_recent; issue_comment <- relgt_chain; pr_review, push <- relgt_chain_recent_related. The seed rows of
`relgt_chain` and `relgt_chain_recent` are in [ensemble.md](ensemble.md) (the same runs).

### Reading

- **Related rows do not help.** As single runs the final model scores 52.89 against 53.41 for `recent` alone
  (-0.5, inside the seed ranges of 0.4-1.3), the validation model 52.03 against 52.33. As a bag it is 53.71
  against 54.06 (-0.35 [-1.10, +0.41]). The per-seed mean through the ensemble pipeline is 53.19 (related)
  vs 53.36 (recent), so the difference is about -0.2 to -0.5 in every comparison, none of which excludes 0.
- **Where it moves:** issue_comment is lower (bag 0.385 vs 0.408; final model 0.376 vs 0.442), the only drop
  larger than the seed spread. push (0.687 vs 0.683 final model) and open_issue (0.492 vs 0.459 as a bag) are
  slightly higher; the other activities are level.
- **More seed spread in the validation model** (51.06-52.97) and best epochs of 6, 5, 15, so the extra
  tokens make the epoch choice less stable.
- **`mean`** of the three configs (53.44) is below all three bags (-0.61 [-1.28, +0.04] against the
  reference; the interval just includes 0). **`select`** gives the highest number of the
  table (54.57, +0.51 [-0.01, +1.02]) by choosing, per activity, the best bag on the tune week; the interval
  includes 0 and, as in the other ensemble tables, `select` has been below the best bag in earlier runs, so
  it is one week of evidence. Bagging the chain graph without `recent` (54.42) is still level with it.
- Conclusion: `--related-rows 5` is not adopted. It costs a longer token sequence per seed token and gives
  no measurable gain on this week. No change to the WRITEUP.md table.
