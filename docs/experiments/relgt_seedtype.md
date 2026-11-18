# Experiment: RelGT seeded at the latest action of each action type (`runs/relgt_seedtype_ho0927.ps1`)

`relgt_chain` / `relgt_chain_recent` seed the token walk at the user's 16 newest visible actions of any type
(`--chain-seed latest --num-seeds 16`). The newest actions are often one type: over the 4,465 test users at the test
cut-off, **73%** of those hop-0 seeds are pushes (51,832 of 71,000); a heavy pusher's 16 seeds are all pushes.

`--chain-seed latest-per-type` seeds at the latest visible action of **each action type** (PR, issue, issue comment,
PR review, PR review comment, push) instead. Every example has up to 6 hop-0 seeds, one per type the user has, newest
first; the user stays a hop-1 token; the ~10 token slots the 16-seed mode spent on seeds go to the walk. The model is
unchanged: the hop-0 tokens are mean-pooled for the readout and the codebook query, and `--history` / `--habit`
features join every seed token, exactly as with `--num-seeds > 1`. The seeds are the rows that the chain graph already
put at hop 1 (the latest row of each of the user's chains); this mode only moves them to hop 0.

Seeds per test user at the test cut-off (checked with the real sampler, 0 violations of: hop-0 set = newest visible row
of each table the user has, ordered newest first, user is a token, K = 100 tokens, a user without actions keeps the user
as seed):

| seeds | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| users | 74 | 747 | 494 | 851 | 547 | 1,752 |

Mean 4.4 seeds, against 16.0 with `--num-seeds 16`.

| Config | Seeds | Fill | Compared with |
|---|---|---|---|
| `relgt_chain_type` | latest action of each type (up to 6) | random | `relgt_chain` ([ensemble.md](ensemble.md)) |
| `relgt_chain_type_recent` | latest action of each type (up to 6) | `--fk-order recent` | `relgt_chain_recent` ([sampling.md](sampling.md)) |

Everything else is as in [relgt_ablation.md](relgt_ablation.md): chain graph, K = 100, 15 epochs, training labels end by
09-19, tune week 09-19 → 09-26, holdout week 09-26 16:00 UTC; the validation model and the final model (refit through
09-26) are both scored. Seeds 0-2, runs `<cfg>_ens0926_s<seed>`. Only the 2 new configs are run (6 jobs).

```powershell
.\runs\relgt_seedtype_ho0927.ps1             # the 2 configs x seeds 0-2 on GPUs 0 and 1, then the summaries
.\runs\relgt_seedtype_ho0927.ps1 -Seeds 0    # one seed per config
.\runs\relgt_seedtype_ho0927.ps1 -Smoke      # pipeline check
.\runs\relgt_seedtype_ho0927.ps1 -SkipRuns   # only the summaries
```

One run by hand:

```powershell
python experiments/relgt/run.py --graph-structure chain --chain-seed latest-per-type --fk-order recent `
    --end "2026-10-03 16:00" --holdout-weeks 1 --val-start "2026-09-19 16:00" --val-end "2026-09-26 16:00" `
    --epochs 15 --seed 0 --name relgt_chain_type_recent_ens0926_s0
```

## Results

Run 2026-11-17/18 (`runs/relgt_seedtype_ho0927.ps1`, seeds 0-2, runs `<cfg>_ens0926_s<seed>`). Official 6-activity score,
holdout week 09-26 16:00 UTC (6,064 users), tune week 09-19 (6,412 users). Ensemble command (output
`experiments/ensemble/outputs/relgt_chain+relgt_chain_type+relgt_chain_recent+relgt_chain_type_recent_ens0926/metrics.json`):

```bash
python experiments/ensemble/run.py --suffix ens0926 relgt_chain relgt_chain_type relgt_chain_recent relgt_chain_type_recent
```

| Config | Mean of seeds, final model (min-max) | Mean of seeds, validation model | Bag of 3 seeds | Bag vs reference `bag:relgt_chain_recent` (95% CI) |
|---|---|---|---|---|
| `relgt_chain` | 52.59 (51.72-53.04) | 52.05 | 54.42 | +0.36 [-0.44, +1.13] |
| `relgt_chain_type` | 52.48 (52.32-52.64) | 49.99 | 53.59 | -0.47 [-1.35, +0.43] |
| `relgt_chain_recent` | 53.41 (52.63-53.90) | 52.33 | 54.06 | reference |
| `relgt_chain_type_recent` | 52.53 (51.73-53.17) | 50.02 | 52.84 | -1.21 [-2.07, -0.35] |

Per activity F1 (final model, mean of seeds / bag):

| Config | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| `relgt_chain` | 0.402 / 0.434 | 0.462 / 0.481 | 0.407 / 0.461 | 0.447 / 0.502 | 0.443 / 0.454 | 0.681 / 0.684 |
| `relgt_chain_type` | 0.400 / 0.425 | 0.430 / 0.463 | 0.407 / 0.416 | 0.445 / 0.453 | 0.438 / 0.463 | 0.687 / 0.691 |
| `relgt_chain_recent` | 0.419 / 0.447 | 0.451 / 0.459 | 0.442 / 0.408 | 0.467 / 0.510 | 0.430 / 0.460 | 0.683 / 0.683 |
| `relgt_chain_type_recent` | 0.414 / 0.432 | 0.449 / 0.478 | 0.384 / 0.379 | 0.458 / 0.458 | 0.411 / 0.451 | 0.686 / 0.679 |

Single seeds (score / tune-week score), seeds 0, 1, 2: chain_type 52.89/51.37, 52.55/51.59, 53.08/50.58;
chain_type_recent 53.00/52.00, 52.65/51.66, 51.58/51.08. Best epochs: chain_type 7, 9, 13; chain_type_recent 13, 8, 13.

Mixing the four bags (same decision rule): `mean` 53.40 (-0.66 [-1.35, +0.04]), `select` 53.79 (-0.26 [-0.86, +0.34]);
the official scorer on `mean` gives 53.40 (matches). `select` picked open_pr, open_issue, pr_review_comment <-
chain_recent, issue_comment <- chain, pr_review, push <- chain_type_recent.

Reading:
- **No gain from seeding at the latest action of each type.** Per seed, `type` is level with its 16-seed pair
  (final model: 52.48 vs 52.59 without `recent`, 52.53 vs 53.41 with it), and as a bag it is lower (53.59 vs 54.42;
  52.84 vs 54.06). Only `bag:relgt_chain_type_recent` is significantly below the reference (-1.21 [-2.07, -0.35]);
  the other differences include 0.
- **The validation models are clearly worse with `type`** (49.99 / 50.02 against 52.05 / 52.33), while the final
  models (refit through the tune week) recover to 52.5. The validation score of `type_recent` varies most between seeds
  (47.86-51.70), so the type seeds are less stable.
- Per activity, `type` is level on push (0.68-0.69) and helps nothing consistently; `type_recent` loses on
  issue_comment (0.384 vs 0.442 as a mean, 0.379 vs 0.408 as a bag) and pr_review_comment.
- Mixing the bags is not better than the best bag (53.40 vs 54.06-54.42), as in the other ensembles
  ([ensemble.md](ensemble.md)).
- Conclusion: not adopted. `relgt_chain_recent` stays; the 16-seed mode is not hurting even though 73% of its hop-0
  seeds are pushes. One week, three seeds each, as in the other RelGT comparisons.
