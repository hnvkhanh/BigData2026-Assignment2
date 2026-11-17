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

Pending.
