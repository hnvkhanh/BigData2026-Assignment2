# Experiment: capping the token walk with `--max-hops` (`runs/relgt_hops_ho0927.ps1`)

`relgt_chain_recent` never sets `--max-hops`, so its walk goes up to the default 50 hops. With K = 100 tokens the walk
is full by hop 3 for about two thirds of the users (median 4 slots left after hop 2), so the cap rarely binds. Here it
is cut: a walk that stops at the cap leaves its remaining slots to repeated random tokens.

| Config | `--max-hops` | Compared with |
|---|---|---|
| `relgt_chain_recent_h3` | 3 | `relgt_chain_recent` (50), [sampling.md](sampling.md) |
| `relgt_chain_recent_h5` | 5 | |
| `relgt_chain_recent_h10` | 10 | |

Only `--max-hops` differs from `relgt_chain_recent` (chain graph, `--chain-seed latest --num-seeds 16`,
`--fk-order recent`, K = 100, 15 epochs, training labels end by 09-19, tune week 09-19 → 09-26, holdout week 09-26 16:00
UTC). The 3 new configs have **one seed (42)** each; the reference has seeds 0-2 (runs `<cfg>_ens0926_s<seed>`).

```powershell
.\runs\relgt_hops_ho0927.ps1             # the 3 configs, seed 42, on GPUs 0 and 1, then the summaries
.\runs\relgt_hops_ho0927.ps1 -Smoke      # pipeline check
.\runs\relgt_hops_ho0927.ps1 -SkipRuns   # only the summaries
```

## Results

Run 2026-11-18. Official 6-activity score, holdout week 09-26 16:00 UTC (6,064 users), tune week 09-19 (6,412 users).
Ensemble command (output
`experiments/ensemble/outputs/relgt_chain_recent+relgt_chain_recent_h3+relgt_chain_recent_h5+relgt_chain_recent_h10_ens0926/metrics.json`):

```bash
python experiments/ensemble/run.py --suffix ens0926 relgt_chain_recent relgt_chain_recent_h3 relgt_chain_recent_h5 relgt_chain_recent_h10
```

Each run's own decisions (final model = refit through 09-26; `relgt_chain_recent` is the mean of 3 seeds):

| Config | Seeds | Final model | Validation model | Best epochs |
|---|---|---|---|---|
| `relgt_chain_recent` (50) | 3 | 53.41 (52.63-53.90) | 52.33 | 7, 12, 10 |
| `relgt_chain_recent_h3` | 1 | 53.13 | 52.83 | 9 |
| `relgt_chain_recent_h5` | 1 | 52.84 | 52.28 | 9 |
| `relgt_chain_recent_h10` | 1 | 52.91 | 52.78 | 9 |

Per activity F1 (final model):

| Config | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| `relgt_chain_recent` | 0.419 | 0.451 | 0.442 | 0.467 | 0.430 | 0.683 |
| `relgt_chain_recent_h3` | 0.431 | 0.459 | 0.388 | 0.490 | 0.421 | 0.679 |
| `relgt_chain_recent_h5` | 0.396 | 0.464 | 0.425 | 0.446 | 0.429 | 0.681 |
| `relgt_chain_recent_h10` | 0.417 | 0.470 | 0.427 | 0.436 | 0.405 | 0.684 |

Single seeds and the seed bag under the same decision rule (the bag of a 1-seed config is that seed):

| Run / bag | Score | Tune | Bag vs reference `bag:relgt_chain_recent` (95% CI) |
|---|---|---|---|
| seed `relgt_chain_recent` s0 / s1 / s2 | 52.91 / 53.71 / 53.47 | 52.40 / 52.43 / 51.79 | |
| seed `_h3` s42 | 53.12 | 51.93 | |
| seed `_h5` s42 | 53.50 | 51.58 | |
| seed `_h10` s42 | 52.91 | 52.17 | |
| **bag `relgt_chain_recent`** (3 seeds) | **54.06** | 52.56 | reference (best tune-week score) |
| bag `_h3` (1 seed) | 53.12 | 51.93 | -0.94 [-1.64, -0.22] |
| bag `_h5` (1 seed) | 53.50 | 51.58 | -0.56 [-1.35, +0.20] |
| bag `_h10` (1 seed) | 52.91 | 52.17 | -1.15 [-2.02, -0.31] |
| mean of the 4 bags | 53.79 | 52.64 | -0.27 [-0.92, +0.41] |
| select per activity | 54.15 | 52.53 | +0.10 [-0.43, +0.63] |

The official scorer on the mean gives 53.79. `select` picked `open_pr` and `pr_review` from `_h3`, `issue_comment` from
`_h5`, and `open_issue`, `pr_review_comment`, `push` from `relgt_chain_recent`.

## Reading

- **No cap helps.** The three capped single seeds score 52.84-53.13 (final model), against 53.41 for the 3-seed mean
  at 50 hops, and each lies inside the range of the reference seeds (52.63-53.90). The seed noise is ~0.8 points, so
  the differences between caps mean nothing.
- The bag comparison is unfair to the caps: a bag of 3 seeds (54.06) against 1 seed. The significant gaps (`_h3`
  -0.94, `_h10` -1.15) are mostly the bag effect (~+0.6 to +1 point for 3 seeds), not the cap.
- Per activity, the caps move things in both directions (h3: open_pr / pr_review up, issue_comment down by 0.05);
  these are single-seed swings, and `select` over the 4 only gains +0.10 [-0.43, +0.63].
- Consistent with the walk being full by hop 3 for most users: the cap changes little because it rarely binds.

Not adopted; `relgt_chain_recent` stays at the default 50 hops. A fair comparison would need seeds 0-2 for each cap;
not worth it unless a reason to expect an effect appears.
