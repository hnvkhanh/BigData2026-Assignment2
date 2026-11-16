# Experiment: RelGT token sampling, graph structure x `--fk-order` (`runs/sampling_ho0927.ps1`)

RelGT sees each example as K = 100 sampled tokens, so which rows are sampled decides what it can use.
Two sampling choices have not been compared on the 09-27 holdout week:
- **Graph structure.** `chain` (the current best, `relgt_chain`): a user's actions of each type are
  linked in time order, the 16 latest actions are the seeds, and a breadth-first walk goes back along
  the chains ([experiments/relgt/README.md](../../experiments/relgt/README.md)). `fk`: every foreign key
  is an edge (each action links to its user), the user is the seed, and the tokens are a sample of its
  1- and 2-hop neighbours. The fk graph was compared only before June (`fk_exp1`, `fk_exp2_nopush`).
- **`--fk-order`.** Where nodes compete for the tokens (the cap of 1,000 neighbours per node, the budget
  of the hop where it runs out, the final pick), `random` (default, every run so far) keeps random
  rows and `recent` keeps the newest ones. No run has used `recent`.

## Setup

| Config | `--graph-structure` | Seeds of the sampling | `--fk-order` |
|---|---|---|---|
| `relgt_chain` (reference) | chain | 16 latest actions, BFS walk | random |
| `relgt_chain_recent` | chain | 16 latest actions, BFS walk | recent |
| `relgt_fk` | fk | the user, 1- and 2-hop sample | random |
| `relgt_fk_recent` | fk | the user, 1- and 2-hop sample | recent |

- Everything else as `relgt_chain` in `runs/ensemble_ho0927.sh`: K = 100, 15 epochs, training labels
  end by 09-19, tune week 09-19 → 09-26, holdout week 09-26 16:00 UTC; validation model and final
  model (refit through the tune week).
- The fk graph cannot take `--chain-seed latest` / `--num-seeds`, so the structure comparison also
  changes the seeds (user vs 16 latest actions). Within a structure, `random` vs `recent` changes only
  the sampling.
- Seeds 0–2, runs `<cfg>_ens0926_s<seed>`. The reference runs `relgt_chain_ens0926_s0-2` exist and are
  not rerun.
- Summaries: `scripts/summarize_runs.py` (each run's own decisions) and `experiments/ensemble/run.py`
  over the four configs (single seeds and seed bags with the same decision rule; paired bootstrap
  against the bag with the best tune-week score).

```powershell
.\runs\sampling_ho0927.ps1              # 3 configs x seeds 0-2 on GPUs 0 and 1, then the summaries
.\runs\sampling_ho0927.ps1 -Seeds 0     # one seed per config
.\runs\sampling_ho0927.ps1 -Smoke       # pipeline check: seed 0, 1 epoch of 30 steps
.\runs\sampling_ho0927.ps1 -SkipRuns    # only the summaries
```

Note: `data/db/` was rebuilt on 2026-11-16 after the reference runs. The labels of weeks before 11-14
are unchanged, and the ensemble stops if two runs' labels differ.

## Results

Run on 2026-11-17 with `.\runs\sampling_ho0927.ps1 -Configs relgt_chain_recent -SkipRuns` (summaries only, no
training). It read the existing runs `<cfg>_ens0926_s0-2` of all four configs (the `-Configs` filter only
limits which jobs would train) and printed the two summaries. Holdout week 09-26 16:00 UTC (6,064 users), tune
week 09-19 (6,412 users), official 6-activity score. The same numbers are in
[ensemble.md](ensemble.md) (RelGT-only ensemble section).

### Each run's own decisions (mean of 3 seeds; min-max of the final model)

| Config | Validation model | Final model | Best epochs (seeds 0-2) |
|---|---|---|---|
| relgt_chain | 52.05 | 52.59 (51.72-53.04) | 11, 8, 12 |
| relgt_chain_recent | 52.33 | 53.41 (52.63-53.90) | 7, 12, 10 |
| relgt_fk | 49.62 | 50.25 (50.15-50.31) | 15, 15, 15 |
| relgt_fk_recent | 52.48 | 52.44 (52.20-52.65) | 5, 15, 8 |

### Same decision rule, single seeds and seed bags

| Variant | Score | Tune | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs reference [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| bag:relgt_chain | **54.42** | 51.63 | 0.434 | 0.481 | 0.461 | 0.502 | 0.454 | 0.684 | +0.36 [-0.44, +1.13] |
| bag:relgt_chain_recent (reference) | 54.06 | 52.56 | 0.447 | 0.459 | 0.408 | 0.510 | 0.460 | 0.683 | 0 |
| bag:relgt_fk | 52.66 | 50.67 | 0.426 | 0.446 | 0.441 | 0.472 | 0.393 | 0.681 | -1.40 [-2.49, -0.32] |
| bag:relgt_fk_recent | 54.10 | 52.45 | 0.422 | 0.464 | 0.433 | 0.487 | 0.461 | 0.685 | +0.04 [-0.90, +0.97] |
| mean | 53.37 | 53.03 | 0.440 | 0.453 | 0.433 | 0.437 | 0.470 | 0.681 | -0.69 [-1.47, +0.12] |
| select | 54.32 | 52.74 | 0.422 | 0.459 | 0.461 | 0.510 | 0.460 | 0.684 | +0.26 [-0.32, +0.83] |

Single seeds (score / tune-week score, seeds 0, 1, 2): chain 53.85/51.01, 53.01/51.53, 51.72/51.42;
chain_recent 52.91/52.40, 53.71/52.43, 53.47/51.79; fk 49.83/49.13, 50.61/48.94, 49.86/48.89; fk_recent
52.13/52.05, 52.07/51.78, 52.60/52.17. `select` picked open_pr <- fk_recent; open_issue, pr_review,
pr_review_comment <- chain_recent; issue_comment, push <- chain. The official scorer on `mean` gives 53.37.
Ensemble output: `experiments/ensemble/outputs/relgt_chain+relgt_chain_recent+relgt_fk+relgt_fk_recent_ens0926/metrics.json`.

### Reading

- **Graph structure.** With random sampling, chain beats fk clearly: 52.59 vs 50.25 per run (final model), 54.42
  vs 52.66 as bags (the fk bag is the only interval that excludes 0, -1.40 [-2.49, -0.32] against
  chain_recent). Part of this is the seeds (16 latest actions vs the user), which the fk graph cannot take.
- **`--fk-order recent`.** It helps most where sampling is weakest: fk 50.25 -> 52.44 per run and 52.66 -> 54.10
  as bags. On chain it is a small gain per run (52.59 -> 53.41, inside the seed range) and a tie as bags
  (54.42 vs 54.06). Seed ranges are 1.3 points (chain_recent), 2.1 (chain), 0.5 (fk_recent) and 0.8 (fk).
- **Bags tie at the top.** chain, chain_recent and fk_recent give 54.1-54.4 with intervals that include 0
  against each other; the tune week ranks them in a different order (chain_recent 52.56 best, chain 51.63
  lowest), so it did not predict the holdout. `mean` (53.37) does not beat its best member.
- Conclusion: `recent` fixes most of the fk graph's deficit but adds nothing measurable on chain. One holdout
  week; no change to the WRITEUP.md table.

Smoke test (2026-11-16, `-Smoke`, seed 0, 1 epoch of 30 steps; its scores say nothing about the configs): all
three configs ran on GPUs 0 and 1, and both summaries read them with the reference smoke run. Token sampling
took 47 s (fk), 53 s (fk, recent) and 390 s (chain, recent); in a full run each new config pays it once per seed.
