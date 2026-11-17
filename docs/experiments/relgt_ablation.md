# Experiment: ablation of RelGT's token sampling components (`runs/relgt_ablation_ho0927.ps1`)

`relgt_chain_recent` differs from the plain fk-graph RelGT in three sampling choices. This run switches each on and
off to see which one carries the gain.

| Component | Flag | Off means |
|---|---|---|
| chain graph | `--graph-structure chain` | fk: every FK is an edge, the action links to its user |
| latest seed | `--chain-seed latest --num-seeds 16` | the user is the seed (needs chain) |
| recent fill | `--fk-order recent` | random: where nodes compete for the K = 100 tokens, the newest rows win when on |

The latest seed needs the chain graph, so 6 of the 8 combinations are valid:

| Config | chain | latest seed | recent | Runs |
|---|---|---|---|---|
| `relgt_fk` | off | off | off | existing ([sampling.md](sampling.md)) |
| `relgt_fk_recent` | off | off | on | existing ([sampling.md](sampling.md)) |
| `relgt_chain_user` | on | off | off | new |
| `relgt_chain_user_recent` | on | off | on | new |
| `relgt_chain` | on | on | off | existing ([ensemble.md](ensemble.md)) |
| `relgt_chain_recent` | on | on | on | existing ([sampling.md](sampling.md)) |

Everything else is as in [sampling.md](sampling.md): K = 100, 15 epochs, training labels end by 09-19, tune week
09-19 → 09-26, holdout week 09-26 16:00 UTC; the validation model and the final model (refit through 09-26) are
both scored. Seeds 0-2, runs `<cfg>_ens0926_s<seed>`. Only the 2 new configs were run (6 jobs); the others are
the existing runs.

```powershell
.\runs\relgt_ablation_ho0927.ps1             # the 2 new configs x seeds 0-2 on GPUs 0 and 1, then the summaries
.\runs\relgt_ablation_ho0927.ps1 -Seeds 0    # one seed per config
.\runs\relgt_ablation_ho0927.ps1 -Smoke      # pipeline check
.\runs\relgt_ablation_ho0927.ps1 -SkipRuns   # only the summaries
```

## Results

Run 2026-11-17, holdout week 09-26 16:00 UTC (6,064 users), tune week 09-19 (6,412 users), seeds 0-2 each.
Official 6-activity score. Ensemble step:

```bash
python experiments/ensemble/run.py --suffix ens0926 relgt_fk relgt_fk_recent relgt_chain_user relgt_chain_user_recent relgt_chain relgt_chain_recent
```

Output: `experiments/ensemble/outputs/relgt_fk+relgt_fk_recent+relgt_chain_user+relgt_chain_user_recent+relgt_chain+relgt_chain_recent_ens0926/metrics.json`.

### Single configs, mean of 3 seeds (the runs' own decisions)

| Config / model | Score (min-max) | Best epochs | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|---|---|
| relgt_fk, validation | 49.62 (49.05-50.62) | 15, 15, 15 | 0.338 | 0.383 | 0.366 | 0.431 | 0.408 | 0.671 |
| relgt_fk, final | 50.25 (50.15-50.31) | 15, 15, 15 | 0.367 | 0.410 | 0.372 | 0.433 | 0.398 | 0.672 |
| relgt_fk_recent, validation | 52.48 (52.18-53.04) | 5, 15, 8 | 0.409 | 0.443 | 0.398 | 0.463 | 0.422 | 0.677 |
| relgt_fk_recent, final | 52.44 (52.20-52.65) | 5, 15, 8 | 0.414 | 0.444 | 0.411 | 0.448 | 0.413 | 0.679 |
| relgt_chain_user, validation | 50.21 (49.88-50.63) | 13, 15, 9 | 0.321 | 0.424 | 0.358 | 0.448 | 0.401 | 0.677 |
| relgt_chain_user, final | 51.62 (51.01-52.09) | 13, 15, 9 | 0.373 | 0.454 | 0.394 | 0.449 | 0.412 | 0.681 |
| relgt_chain_user_recent, validation | 50.42 (49.76-51.21) | 14, 15, 14 | 0.307 | 0.425 | 0.327 | 0.462 | 0.440 | 0.672 |
| relgt_chain_user_recent, final | 52.04 (51.62-52.32) | 14, 15, 14 | 0.363 | 0.443 | 0.382 | 0.462 | 0.442 | 0.681 |
| relgt_chain, validation | 52.05 (52.00-52.14) | 11, 8, 12 | 0.404 | 0.426 | 0.391 | 0.451 | 0.446 | 0.675 |
| relgt_chain, final | 52.59 (51.72-53.04) | 11, 8, 12 | 0.402 | 0.462 | 0.407 | 0.447 | 0.443 | 0.681 |
| relgt_chain_recent, validation | 52.33 (51.95-52.73) | 7, 12, 10 | 0.380 | 0.432 | 0.405 | 0.455 | 0.441 | 0.679 |
| relgt_chain_recent, final | **53.41** (52.63-53.90) | 7, 12, 10 | 0.419 | 0.451 | 0.442 | 0.467 | 0.430 | 0.683 |

### Single seeds and bags (same decision rule, push share from the tune week, ranks averaged)

| Variant | Score | Tune | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs reference [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| seed:relgt_fk s0 / s1 / s2 | 49.83 / 50.61 / 49.86 | 49.13 / 48.94 / 48.89 | | | | | | | |
| seed:relgt_fk_recent s0 / s1 / s2 | 52.13 / 52.07 / 52.60 | 52.05 / 51.78 / 52.17 | | | | | | | |
| seed:relgt_chain_user s0 / s1 / s2 | 51.45 / 52.36 / 52.57 | 51.23 / 51.81 / 51.03 | | | | | | | |
| seed:relgt_chain_user_recent s0 / s1 / s2 | 52.60 / 52.67 / 53.11 | 51.10 / 51.63 / 51.23 | | | | | | | |
| seed:relgt_chain s0 / s1 / s2 | 53.85 / 53.01 / 51.72 | 51.01 / 51.53 / 51.42 | | | | | | | |
| seed:relgt_chain_recent s0 / s1 / s2 | 52.91 / 53.71 / 53.47 | 52.40 / 52.43 / 51.79 | | | | | | | |
| bag:relgt_fk | 52.66 | 50.67 | 0.426 | 0.446 | 0.441 | 0.472 | 0.393 | 0.681 | -0.51 [-1.67, +0.67] |
| bag:relgt_fk_recent | 54.10 | 52.45 | 0.422 | 0.464 | 0.433 | 0.487 | 0.461 | 0.685 | +0.93 [-0.11, +1.94] |
| bag:relgt_chain_user | 53.31 | 52.17 | 0.383 | 0.459 | 0.430 | 0.470 | 0.451 | 0.687 | +0.14 [-0.81, +1.05] |
| bag:relgt_chain_user_recent (reference) | 53.17 | 52.60 | 0.430 | 0.452 | 0.361 | 0.480 | 0.448 | 0.687 | 0 |
| bag:relgt_chain | **54.42** | 51.63 | 0.434 | 0.481 | 0.461 | 0.502 | 0.454 | 0.684 | +1.25 [+0.19, +2.26] |
| bag:relgt_chain_recent | 54.06 | 52.56 | 0.447 | 0.459 | 0.408 | 0.510 | 0.460 | 0.683 | +0.89 [-0.07, +1.84] |
| mean | 53.53 | 52.63 | 0.435 | 0.452 | 0.425 | 0.495 | 0.459 | 0.677 | +0.36 [-0.55, +1.24] |
| select | 52.95 | 53.23 | 0.422 | 0.459 | 0.361 | 0.480 | 0.460 | 0.684 | -0.22 [-0.99, +0.49] |

The reference of the bootstrap is the bag with the best tune-week score, `relgt_chain_user_recent` (tune 52.60
among the bags; the `mean` and `select` rows score higher on tune but are not bags). The official scorer on
`mean` gives 53.53 (matches). `select` picked: open_pr <- relgt_fk_recent, open_issue and pr_review_comment <-
relgt_chain_recent, issue_comment and pr_review <- relgt_chain_user_recent, push <- relgt_chain. The per-seed
activity columns are in `metrics.json`.

### Effect of each component (differences of the scores above)

| Effect | Comparison | Single (final model, 3-seed mean) | Bag |
|---|---|---|---|
| chain graph, random fill | chain_user - fk | +1.37 | +0.65 |
| chain graph, recent fill | chain_user_recent - fk_recent | -0.40 | -0.93 |
| latest seed, random fill | chain - chain_user | +0.97 | +1.11 |
| latest seed, recent fill | chain_recent - chain_user_recent | +1.37 | +0.89 |
| recent fill, fk | fk_recent - fk | +2.19 | +1.44 |
| recent fill, chain_user | chain_user_recent - chain_user | +0.42 | -0.14 |
| recent fill, chain | chain_recent - chain | +0.82 | -0.36 |

Only differences against the bootstrap reference have intervals (the last column of the bag table); these
differences between two other variants have none.

### Reading

- **The latest seed is the component that helps consistently.** It is positive in all four comparisons
  (+0.9 to +1.4), with or without recent fill, as single runs and as bags. The bag of `relgt_chain` is the only
  variant whose interval against the reference excludes 0 (+1.25 [+0.19, +2.26]).
- **Recent fill helps most when nothing else is changed.** On the fk graph it is +2.2 as single runs (+1.4 as
  bags), and it makes the runs fit faster (best epochs 5, 15, 8 against 15, 15, 15; `relgt_fk` is still
  improving at epoch 15). With the chain graph the gain shrinks to +0.4 / +0.8 as single runs and is
  -0.1 / -0.4 as bags: once the walk starts from the latest actions, the newest rows are already the ones that
  get picked.
- **The chain graph alone is not a clear gain.** With random fill it is +1.4 (single) / +0.7 (bag), with recent
  fill it is -0.4 / -0.9. The chain graph pays off through the latest seed, not by itself:
  `chain_user` and `chain_user_recent` (user seed) sit at 52.0-51.6, level with `fk_recent` (52.4).
- **The validation model lags the final model for the user-seed configs** (50.2 and 50.4 against 51.6 and 52.0).
  They also pick the last epochs (13, 15, 9 and 14, 15, 14), so 15 epochs may be too few for them and the
  user seed gives the validation model less to go on. `fk` and `fk_recent` have a small gap.
- **Everything is close at the bag level.** The best three bags (`chain` 54.42, `fk_recent` 54.10,
  `chain_recent` 54.06) are within 0.4 of each other, which is less than the seed spread (1.0-2.1 for single
  seeds of one config). `select` (52.95) and `mean` (53.53) are below the best bags again, as in the earlier
  ensemble tables.
- Conclusion: the components are ranked latest seed > recent fill > chain graph, but the config in use
  (`relgt_chain_recent`, which has all three) is not beaten by any ablation: its final-model score (53.41) is the
  highest of the six, its bag (54.06) is within noise of the best (54.42). No change to the WRITEUP.md table.
