# Experiment: ensemble of the tied graph models (`experiments/ensemble/run.py`)

On the holdout week 09-27 → 10-04 the graph models tie (52.3–53.1, seed ranges of 0.4–1.5 points;
[holdout_0927](holdout_0927.md)), but their per-activity F1 differs: RelGT on the chain graph is best on
open_issue (0.470), the ego GNN on open_pr (0.436), the temporal hetero GNN on pr_review (0.486),
issue_comment and pr_review_comment. Push F1 is the same everywhere. The error split shows that most of
the lost score is ranking active users on the five non-push activities. Averaging models that rank them
differently is the cheapest way to improve that ranking, so this experiment trains nothing new: it
combines the probabilities of the existing configs, and adds one new config, the chain model with the
baseline's history features on its seed tokens (`--history`, untested so far; the long-memory signal that
[holdout_0927](holdout_0927.md) suggests for the spring-heavy users hidden by capture loss).

## What was changed

- `experiments/gnn/run.py --holdout` now writes `holdout.npz`: the probabilities of every eligible user
  of the tune week and of the holdout week, their labels, the run's own decisions, and the share of users
  with a push in the previous week of push data (for `--push-rule prev-week`).
- `experiments/relgt/run.py --holdout-weeks` adds the probabilities of its final and validation models
  on the holdout week and of the validation model on the tune week to its `holdout.npz` (it saved only
  0/1 decisions before).
- `experiments/ensemble/run.py` combines runs; `runs/ensemble_ho0927.sh` makes the runs and calls it.

## Method

Members are configs with runs `<cfg>_<suffix>_s<seed>`. Per run and week, the probabilities of each
activity are turned into ranks in (0, 1], so models with different probability scales mix evenly
(`--normalize prob` averages the raw probabilities instead). Variants:

| Variant | Holdout probabilities |
|---|---|
| `seed:<run>` | one run, through the same pipeline (shows the effect of the decision rule alone) |
| `bag:<cfg>` | mean over the seeds of one config (seed bagging) |
| `mean` | mean over the configs' bags (`--weights` for an uneven mix) |
| `select` | per activity, the bag with the best tune-week F1 |

The decision rule is the one every runner uses: per activity, the share of users predicted positive
that maximises F1 on the tune week *of the same combination*, applied to the holdout week as a quantile
(`--rule joint`: the shares after a coordinate ascent of the score over the tune week). Push takes that
share (`--push-rule tune`) or the share of users with a push in the last 7 days of push data before the
week (`--push-rule prev-week`, the submission's rule). A RelGT member contributes its final model
(retrained through the tune week, `--relgt-variant final`) or its validation model; its tune-week
probabilities are always the validation model's.

The holdout week is scored with the official formula. A paired bootstrap over users (2,000 resamples,
the same draws for every variant) gives the 95% interval of each variant's difference to the reference,
the bag with the best tune-week score. The runs' own decisions (`own:<cfg>`, the numbers in the other
docs) are listed for comparison. The official scorer is run once on the main variant as a check of the
fast one.

```bash
bash runs/ensemble_ho0927.sh            # 4 configs x seeds 0-2 on two GPUs (~2.5 h), then the ensembles
SMOKE=1 bash runs/ensemble_ho0927.sh    # pipeline check in a few minutes (1 epoch, seed 0)
SKIP_RUNS=1 bash runs/ensemble_ho0927.sh
python experiments/ensemble/run.py --suffix ens0926 all_ego_uf het relgt_chain --push-rule prev-week
```

The runs are named `<cfg>_ens0926_s<seed>` so the documented `ho0926` runs stay as they are. The
ensembles write `experiments/ensemble/outputs/<members>_ens0926[_options]/metrics.json`.

## Results

Run on 2026-11-16 (`runs/ensemble_ho0927.sh`, console log `experiments/ensemble/outputs/ens0926_full.log`,
training ~2.5 h on two GPUs; ensembles and summary re-run on CPU into `ens0926_summary.log`). 4 configs x
seeds 0-2, holdout week 09-26 16:00 UTC (6,064 users). All numbers are the official 6-activity score.

Provenance note: a first attempt at the summary step started without `SKIP_RUNS` taking effect and retrained
`all_ego_uf_ens0926_s0` (holdout 53.18 -> 53.41, same config, GPU nondeterminism) before it was stopped.
The tables below use the retrained s0, so they differ slightly from the figures in `ens0926_full.log`
(e.g. `bag:all_ego_uf` 53.25 here). The `s0` and `s1` training logs of `all_ego_uf` were overwritten; the saved
results of `s1` were not touched.

### Single configs, mean of 3 seeds (the runs' own decisions)

| Config | Mean score (min-max) |
|---|---|
| all_ego_uf | 53.23 (52.97-53.41) |
| het | 52.71 (52.28-53.38) |
| relgt_chain, validation model | 52.05 (52.00-52.14) |
| relgt_chain, final model | 52.59 (51.72-53.04) |
| relgt_chain_hist, validation model | 51.80 (49.92-52.91) |
| relgt_chain_hist, final model | 53.09 (52.42-53.47) |

### Ensembles of all four configs (push share from the tune week, ranks averaged)

| Variant | Score | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs reference [95% CI] |
|---|---|---|---|---|---|---|---|---|
| own:all_ego_uf | 53.23 | 0.436 | 0.428 | 0.417 | 0.481 | 0.459 | 0.676 | |
| own:het | 52.71 | 0.413 | 0.440 | 0.423 | 0.483 | 0.450 | 0.671 | |
| own:relgt_chain | 52.59 | 0.402 | 0.462 | 0.407 | 0.447 | 0.443 | 0.681 | |
| own:relgt_chain_hist | 53.09 | 0.414 | 0.451 | 0.407 | 0.475 | 0.435 | 0.683 | |
| bag:all_ego_uf | 53.25 | 0.444 | 0.429 | 0.401 | 0.470 | 0.468 | 0.677 | -0.16 [-1.15, +0.77] |
| bag:het | 52.81 | 0.405 | 0.441 | 0.423 | 0.490 | 0.474 | 0.666 | -0.61 [-1.58, +0.43] |
| bag:relgt_chain | **54.42** | 0.434 | 0.481 | 0.461 | 0.502 | 0.454 | 0.684 | +1.00 [+0.12, +1.94] |
| bag:relgt_chain_hist (reference) | 53.42 | 0.445 | 0.447 | 0.430 | 0.453 | 0.451 | 0.683 | 0 |
| mean | 53.82 | 0.442 | 0.459 | 0.421 | 0.471 | 0.462 | 0.682 | +0.40 [-0.30, +1.16] |
| select | 52.89 | 0.444 | 0.429 | 0.401 | 0.453 | 0.451 | 0.684 | -0.53 [-1.24, +0.18] |

Tune-week scores of the same variants: bag:all_ego_uf 52.21, het 51.70, relgt_chain 51.63, relgt_chain_hist
52.39, mean 52.92, select 53.01.

### Other combinations (score; 95% CI against each table's reference)

| Ensemble | bag:all_ego_uf | bag:het | bag:relgt_chain | bag:relgt_chain_hist | mean | select |
|---|---|---|---|---|---|---|
| 4 configs, `--push-rule prev-week` (submission rule) | 51.94 | 52.27 | 53.73 | 52.81 | 53.08 (+0.27 [-0.47, +1.02]) | 52.16 |
| 4 configs, `--rule joint` | 53.05 | 53.51 | 53.06 | 53.88 | **54.07** (+0.20 [-0.54, +0.89]) | 52.93 |
| 3 documented configs (no hist) | 53.25 | 52.81 | 54.42 | - | 53.62 (+0.37 [-0.20, +1.01]) | 53.51 |
| relgt_chain + relgt_chain_hist, validation models | - | - | 53.19 | 52.92 | 53.11 (+0.18 [-0.41, +0.79]) | 52.80 |

References: 4 configs bag:relgt_chain_hist; 3 documented configs bag:all_ego_uf; chain pair
bag:relgt_chain_hist. Selected per activity (4 configs): open_pr, open_issue, issue_comment <- all_ego_uf;
pr_review, pr_review_comment <- relgt_chain_hist; push <- relgt_chain. The official scorer reproduces the
`mean` scores (53.82, 53.08, 54.07, 53.62, 53.11).

### Reading

- **Seed bagging** is worth most for RelGT on the chain graph: `bag:relgt_chain` 54.42 against 52.59
  (final model) / 52.05 (validation model) for the single runs, the only variant whose interval excludes 0
  against the reference (+1.00 [+0.12, +1.94]). For the ego GNN (53.25 vs 53.23) and het (52.81 vs 52.71)
  bagging adds nothing. The 54.42 is one week and one bag of three seeds, and its tune-week score (51.63) is
  the lowest, so the tune week did not predict it.
- **Mixing models**: `mean` is 53.6-54.1 across the settings, +0.2 to +0.4 over the reference bag, with
  intervals that include 0. On this week it does not beat `bag:relgt_chain` (54.42), and the one week does
  not separate them. The best number overall is `mean --rule joint` (54.07), also within noise of the rest.
- **`select`** is never better than `mean` (52.89 vs 53.82); it picks by tune week and loses on the holdout.
- **History features**: `bag:relgt_chain_hist` 53.42 against `bag:relgt_chain` 54.42, so no gain on this
  week; it is better on open_pr and push, worse on open_issue, issue_comment and pr_review. Per run the final
  model improves with history (53.09 vs 52.59) and the validation model does not (51.80 vs 52.05), so the
  effect is inside the seed range.
- **Submission push rule**: with `--push-rule prev-week` every variant drops 0.7-1.3 (`mean` 53.08 against
  53.82) because the previous-week push share (0.392) is below the tune-week share (about 0.46-0.50); the push
  F1 falls from about 0.68 to 0.65-0.67. Against single runs under the same rule (all_ego_uf 51.98, het 51.89
  by `summarize_runs.py`), the ensemble `mean` (53.08) and `bag:relgt_chain` (53.73) are about 1-2 points
  higher, in line with the tune-rule comparison.
- Conclusion: nothing here justifies changing the WRITEUP.md results table. The ensemble gain is at most about
  1 point, supported only for bagging RelGT seeds, and not measured on a second week.

### RelGT-only ensemble (chain / fk graph, with and without `recent`)

Same decision rule, same week (holdout 09-26: 6,064 users; tune week 09-19: 6,412 users), 4 RelGT configs x
seeds 0-2 (`<cfg>_ens0926_s<seed>`), combined without the ego GNN and het:

```bash
python experiments/ensemble/run.py --suffix ens0926 relgt_chain relgt_chain_recent relgt_fk relgt_fk_recent
```

Output: `experiments/ensemble/outputs/relgt_chain+relgt_chain_recent+relgt_fk+relgt_fk_recent_ens0926/metrics.json`.
Official 6-activity score. The reference for the bootstrap is `bag:relgt_chain_recent` (best tune-week score).

| Variant | Score | Tune | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs reference [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| own:relgt_chain | 52.59 | - | 0.402 | 0.462 | 0.407 | 0.447 | 0.443 | 0.681 | |
| own:relgt_chain_recent | 53.41 | - | 0.419 | 0.451 | 0.442 | 0.467 | 0.430 | 0.683 | |
| own:relgt_fk | 50.25 | - | 0.367 | 0.410 | 0.372 | 0.433 | 0.398 | 0.672 | |
| own:relgt_fk_recent | 52.44 | - | 0.414 | 0.444 | 0.411 | 0.448 | 0.413 | 0.679 | |
| bag:relgt_chain | **54.42** | 51.63 | 0.434 | 0.481 | 0.461 | 0.502 | 0.454 | 0.684 | +0.36 [-0.44, +1.13] |
| bag:relgt_chain_recent (reference) | 54.06 | 52.56 | 0.447 | 0.459 | 0.408 | 0.510 | 0.460 | 0.683 | 0 |
| bag:relgt_fk | 52.66 | 50.67 | 0.426 | 0.446 | 0.441 | 0.472 | 0.393 | 0.681 | -1.40 [-2.49, -0.32] |
| bag:relgt_fk_recent | 54.10 | 52.45 | 0.422 | 0.464 | 0.433 | 0.487 | 0.461 | 0.685 | +0.04 [-0.90, +0.97] |
| mean | 53.37 | 53.03 | 0.440 | 0.453 | 0.433 | 0.437 | 0.470 | 0.681 | -0.69 [-1.47, +0.12] |
| select | 54.32 | 52.74 | 0.422 | 0.459 | 0.461 | 0.510 | 0.460 | 0.684 | +0.26 [-0.32, +0.83] |

The official scorer on `mean` gives 53.37 (matches). `select` picked: open_pr <- relgt_fk_recent, open_issue,
pr_review, pr_review_comment <- relgt_chain_recent, issue_comment, push <- relgt_chain.

Single seeds (score / tune-week score): chain 53.85/51.01, 53.01/51.53, 51.72/51.42; chain_recent 52.91/52.40,
53.71/52.43, 53.47/51.79; fk 49.83/49.13, 50.61/48.94, 49.86/48.89; fk_recent 52.13/52.05, 52.07/51.78,
52.60/52.17 (seeds 0, 1, 2 in that order).

Reading:
- **Bagging** beats the mean of the three single seeds (same pipeline) for every config: chain 52.86 -> 54.42
  (+1.6), chain_recent 53.36 -> 54.06 (+0.7), fk 50.10 -> 52.66 (+2.6), fk_recent 52.27 -> 54.10 (+1.8). Seed
  ranges are 2.1 (chain), 0.8, 0.8 and 0.5 points, so the gain is clearest where the seeds disagree most.
- **`recent`** raises the weaker graph the most: fk 52.66 -> 54.10 as bags and 49.8-50.6 -> 52.1-52.6 per seed.
  On the chain graph the bags are 54.42 vs 54.06, a tie. The only interval that excludes 0 is `bag:relgt_fk`
  (worse, -1.40 [-2.49, -0.32]).
- **`mean` does not beat its best member** (53.37 vs 54.06-54.42; the interval to the reference, -0.69
  [-1.47, +0.12], includes 0) although it has the best tune-week score (53.03). `select` (54.32) is level with
  the best bag, not above it. As in the four-config table, picking or mixing by tune week does not carry to
  the holdout; the tune week ranks the bags wrongly (chain has the lowest tune score of the three good bags and
  the highest holdout score).
- Conclusion: among the RelGT variants only fk without `recent` is worse; chain, chain_recent and fk_recent
  tie at 54.1-54.4 as bags. No change to the WRITEUP.md table.

## Current stage (2026-11-16)

Done: the ensemble experiment on the holdout week 09-26 16:00 UTC (4 configs x 3 seeds, runs `*_ens0926_s*`,
ensembles in `experiments/ensemble/outputs/*_ens0926*/`), and the RelGT-only ensemble of chain / chain_recent /
fk / fk_recent. The results above are complete for that week.

Supported by this week: seed-bagging RelGT on the chain graph (+1.00 [+0.12, +1.94]); `select` overfits the
tune week; the push rule moves the score by about 1 point (`prev-week` is the submission-comparable one).
Not supported: that mixing models beats the best bag, and that history features help (intervals include 0).

Not done:
- No second holdout week. The older holdout runs saved only `metrics.json`, so a second week means
  retraining (`HO_WEEK="<UTC start of scored week>" bash runs/ensemble_ho0927.sh`, ~2.5 h on two GPUs; for the
  bagging check alone `CONFIGS="relgt_chain" QUEUE_A="relgt_chain" QUEUE_B=""`, ~30 min a seed). The week
  still has to be chosen and checked for enough labelled data and outage handling.
- The WRITEUP.md results table and the submission pipeline are unchanged: they do not use seed bagging or an
  ensemble.
- `all_ego_uf_ens0926_s0` was retrained by mistake (see the provenance note), so the original `s0` numbers
  are gone; the `ens0926_full.log` figures differ slightly from the tables.
