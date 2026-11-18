# Experiment: no actions of other users, except the user's own context (`runs/relgt_ctx_ho0927.ps1`)

On the chain graph the walk reaches rows of other users through shared nodes (a repository, an issue, a PR) and
through their chains. For the user in [recent_fill_real.png](../../figures/recent_fill_real.png) 51 of 79 activity
tokens belonged to other users, and with `--fk-order recent` the newest tokens were mostly those, not the user's own.

`--other-actions context` lets a row of another user be a token only in three cases (`context_filter` in
[run.py](../../experiments/relgt/run.py)), all judged at the cut-off:

| Allowed row of another user | Rule |
|---|---|
| comment on the user's PR or issue | an issue comment whose issue is the user's, or is the issue of the user's PR; a review (line) comment on the user's PR |
| review of the user's PR | a `PRReviews` row whose PR is the user's |
| issue opened in a repository of the user | an `Issues` row (any author) in a repository where the user has a visible push or an issue assigned to them |

Everything else of another user is dropped: pushes, PRs, other issues and other comments, whether the walk
would have reached it through a shared repository, a shared issue or the other user's chain. Users (the owner of an
allowed row still joins the tokens, one hop further), repositories and the user's own rows are not filtered. The filter
is applied to every candidate of every hop before the caps, so the budget goes to allowed rows. "The user's PR / issue"
means authored by the user (an issue assigned to the user only counts for the repository rule). `--related-rows` is not
allowed together with it (related rows are not filtered).

Check on 400 test users at the test cut-off, with the real sampler: an independent recomputation of the three rules with
Python sets found **0 violations** (every other-user activity token matches a rule). Other users' rows are 9% of all
activity tokens with `all` and 4% with `context` (the share drops by half, the user's own rows take the slots). Of the
other-user rows that remain, about 95% are issues in a pushed / assigned repository; comments on and reviews of the
user's own PRs and issues are rare (tens in 400 users).

| Config | Other users' actions | Fill | Compared with |
|---|---|---|---|
| `relgt_chain_ctx` | context only | random | `relgt_chain` ([ensemble.md](ensemble.md)) |
| `relgt_chain_ctx_recent` | context only | `--fk-order recent` | `relgt_chain_recent` ([sampling.md](sampling.md)) |

Everything else is as in [relgt_ablation.md](relgt_ablation.md): chain graph, `--chain-seed latest --num-seeds 16`,
K = 100, 15 epochs, training labels end by 09-19, tune week 09-19 → 09-26, holdout week 09-26 16:00 UTC; seeds 0-2, runs
`<cfg>_ens0926_s<seed>`. Only the 2 new configs are run (6 jobs).

```powershell
.\runs\relgt_ctx_ho0927.ps1             # the 2 configs x seeds 0-2 on GPUs 0 and 1, then the summaries
.\runs\relgt_ctx_ho0927.ps1 -Seeds 0    # one seed per config
.\runs\relgt_ctx_ho0927.ps1 -Smoke      # pipeline check
.\runs\relgt_ctx_ho0927.ps1 -SkipRuns   # only the summaries
```

One run by hand:

```powershell
python experiments/relgt/run.py --graph-structure chain --chain-seed latest --num-seeds 16 --other-actions context `
    --fk-order recent --end "2026-10-03 16:00" --holdout-weeks 1 --val-start "2026-09-19 16:00" `
    --val-end "2026-09-26 16:00" --epochs 15 --seed 0 --name relgt_chain_ctx_recent_ens0926_s0
```

## Results

Run 2026-11-18 (`runs/relgt_ctx_ho0927.ps1`, seeds 0-2, runs `<cfg>_ens0926_s<seed>`). Official 6-activity score, holdout
week 09-26 16:00 UTC (6,064 users), tune week 09-19 (6,412 users). Ensemble command (output
`experiments/ensemble/outputs/relgt_chain+relgt_chain_ctx+relgt_chain_recent+relgt_chain_ctx_recent_ens0926/metrics.json`):

```bash
python experiments/ensemble/run.py --suffix ens0926 relgt_chain relgt_chain_ctx relgt_chain_recent relgt_chain_ctx_recent
```

| Config | Mean of seeds, final model (min-max) | Mean of seeds, validation model | Best epochs | Bag of 3 seeds | Bag vs reference `bag:relgt_chain_ctx` (95% CI) |
|---|---|---|---|---|---|
| `relgt_chain` | 52.59 (51.72-53.04) | 52.05 | 11, 8, 12 | 54.42 | +0.36 [-0.42, +1.10] |
| `relgt_chain_ctx` | 52.58 (52.01-53.57) | 51.18 | 7, 8, 9 | 54.06 | reference |
| `relgt_chain_recent` | 53.41 (52.63-53.90) | 52.33 | 7, 12, 10 | 54.06 | -0.01 [-0.76, +0.75] |
| `relgt_chain_ctx_recent` | 52.91 (52.58-53.12) | 52.21 | 5, 4, 10 | 52.58 | -1.49 [-2.28, -0.65] |

The bootstrap reference is the bag with the best tune-week score, `relgt_chain_ctx` (52.60), not `relgt_chain_recent` as in the
other tables.

Per activity F1 (final model, mean of seeds / bag):

| Config | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| `relgt_chain` | 0.402 / 0.434 | 0.462 / 0.481 | 0.407 / 0.461 | 0.447 / 0.502 | 0.443 / 0.454 | 0.681 / 0.684 |
| `relgt_chain_ctx` | 0.415 / 0.435 | 0.462 / 0.472 | 0.378 / 0.415 | 0.462 / 0.497 | 0.431 / 0.439 | 0.683 / 0.691 |
| `relgt_chain_recent` | 0.419 / 0.447 | 0.451 / 0.459 | 0.442 / 0.408 | 0.467 / 0.510 | 0.430 / 0.460 | 0.683 / 0.683 |
| `relgt_chain_ctx_recent` | 0.415 / 0.445 | 0.466 / 0.475 | 0.413 / 0.377 | 0.463 / 0.495 | 0.409 / 0.410 | 0.682 / 0.670 |

Single seeds (score / tune-week score), seeds 0, 1, 2: chain_ctx 52.01/52.79, 53.68/51.82, 52.88/52.33;
chain_ctx_recent 52.99/51.99, 52.90/52.00, 53.25/51.34 (references: chain 53.85/51.01, 53.01/51.53, 51.72/51.42; chain_recent
52.91/52.40, 53.71/52.43, 53.47/51.79).

Mixing the four bags (same decision rule): `mean` 53.58 (-0.48 [-1.09, +0.09]), `select` 54.40 (+0.34 [-0.21, +0.88]); the
official scorer on `mean` gives 53.58 (matches). `select` picked open_pr, issue_comment, push <- chain_ctx, open_issue,
pr_review, pr_review_comment <- chain_recent.

Reading:
- **Leaving out other users' actions does not help.** Per seed, `ctx` is level with `relgt_chain` (52.58 vs 52.59) and
  `ctx_recent` is below `relgt_chain_recent` (52.91 vs 53.41, -0.50). As bags, `ctx` is 54.06 vs 54.42 (-0.36) and
  `ctx_recent` 52.58 vs 54.06 (-1.48). Only `bag:relgt_chain_ctx_recent` is significantly below the reference (-1.49
  [-2.28, -0.65]); the other bag differences include 0. The seed spread (1.1-1.6 for single seeds of one config) is larger
  than the per-seed differences.
- **Recent fill no longer adds anything on top of `ctx`.** `ctx_recent` against `ctx` is +0.33 per seed but -1.48 as a
  bag; with all rows `recent` was +0.82 per seed and -0.36 as a bag. The fresh rows that `recent` brought in were largely
  other users' rows (51 of 79 activity tokens in the figure's user), and with those removed there is less for it to
  select.
- **The validation models are worse with `ctx`** (51.18 against 52.05; `ctx_recent` 52.21 against 52.33, level), while the
  final models recover. The ctx configs stop at earlier best epochs (7, 8, 9 and 5, 4, 10).
- Per activity: `ctx` is slightly better on open_pr (+0.013 per seed) and pr_review (+0.015), and worse on issue_comment
  (0.378 vs 0.407; `ctx_recent` 0.413 vs 0.442) and pr_review_comment (0.431 vs 0.443; 0.409 vs 0.430). Issue comments and
  review comments depend on a thread that other users write, which `ctx` mostly keeps out (only comments on the user's own
  PRs and issues stay). Push is level (0.68), and the `ctx` bag has the best push (0.691).
- Mixing the bags is not better than the best bag (53.58 `mean` against 54.06-54.42; `select` 54.40 is level), as in the
  other ensembles ([ensemble.md](ensemble.md)).
- Conclusion: not adopted. `relgt_chain_recent` stays (best mean of seeds, 53.41). Other users' actions do carry some signal
  for the comment activities, and removing them is not a gain on this week. One week, three seeds each, as in the other
  RelGT comparisons.
