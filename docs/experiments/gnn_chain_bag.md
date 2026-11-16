# Experiment: seed bagging of the chain GNN, with a final stage (`runs/gnn_chain_bag_ho0927.ps1`)

On the holdout week 09-27 → 10-04 the chain RelGT gains most from seed bagging: `bag:relgt_chain` 54.42
against 52.59 for its single runs, +1.00 [+0.12, +1.94] over the reference ([ensemble](ensemble.md)). The
GNN on the same chain graph (`gnn_chain`, 52.48; [holdout_0927](holdout_0927.md)) was never bagged: its
runs predate `holdout.npz`. Its holdout runs also have one stage only (trained to 09-19, no refit), while
RelGT's final model is refit through the tune week and scores +1.1 over its validation model. This
experiment reruns `gnn_chain` with both stages and bags its seeds.

## What was changed

- `experiments/gnn/run.py --holdout-final` (with `--holdout`): after the validation model, a final model
  is refit from scratch on every labelled week through the tune week (`final_weeks`; the holdout week is
  in no split) and scored on the holdout week with the same decision rule (the tune week's shares).
  Saved as `holdout_final` in `metrics.json`, `prob_final` / `pred_final` in `holdout.npz`.
  `--final-epochs` sets its epochs (default: the validation stage's best epoch).
- `experiments/ensemble/run.py --gnn-variant validation|final`: which GNN model's holdout probabilities
  to use (as `--relgt-variant`). The tune-week probabilities are always the validation model's.

## Setup

- Model: `gnn_chain` of `runs/chain_ho0927.sh` (chain graph, 16 latest actions as seeds, K = 100,
  3 layers, 64 channels).
- Epochs per stage: the recorded best epoch of the same seed's documented run
  (`experiments/gnn/outputs/gnn_chain_ho0926_s<seed>/metrics.json`), 1 on seeds 0–2. The documented
  runs trained 6 epochs (best epoch 1, patience 5) and kept the first. Here the validation stage trains
  `--epochs 1` and the final stage `--final-epochs 1`.
- Seeds 0–2, runs `gnn_chain_ens0926_s<seed>`; the `_ens0926` suffix lets `experiments/ensemble/run.py`
  combine them with `relgt_chain_ens0926_s0-2`.
- Ensembles: the bag of the validation models, the bag of the final models, and the mean of
  `gnn_chain` and `relgt_chain` (both final models; push by the tune-week share and by `prev-week`).

```powershell
.\runs\gnn_chain_bag_ho0927.ps1              # seeds 0-2, then the ensembles
.\runs\gnn_chain_bag_ho0927.ps1 -Smoke       # pipeline check: seed 0, 1 epoch of 30 steps
.\runs\gnn_chain_bag_ho0927.ps1 -SkipRuns    # only the ensembles
```

Note: `data/db/` was rebuilt on 2026-11-16 after the documented `gnn_chain_ho0926` runs and the
`relgt_chain_ens0926` runs. The labels of weeks before 11-14 are unchanged, and the ensemble stops if
two runs' labels differ.

## Results

Run on 2026-11-16 (`.\runs\gnn_chain_bag_ho0927.ps1`, seeds 0–2 on GPU 1; logs
`experiments/gnn/outputs/logs/gnn_chain_ens0926_s*.log`, ensembles in
`experiments/ensemble/outputs/gnn_chain*_ens0926*/`). Holdout week 09-26 16:00 UTC, 6,064 users, official
6-activity score. Both stages trained 1 epoch (the recorded best epoch; the validation stage's best
epoch was again 1 on every seed).

### Per seed (each run's own decisions, push by the tune-week share)

| Seed | Validation model | Final model | Change |
|---|---|---|---|
| s0 | 52.67 | 52.73 | +0.06 |
| s1 | 52.79 | 53.52 | +0.73 |
| s2 | 53.02 | 53.42 | +0.40 |
| Mean | 52.83 | **53.22** | +0.39 |

The validation model is the documented `gnn_chain` (52.48, 51.90–52.94, [holdout_0927](holdout_0927.md))
trained 1 epoch instead of 6 with early stopping back to epoch 1, on the rebuilt `data/db/`; 52.83 is
inside the noise of that run. The final model is better on every seed (+0.06 to +0.73).

### Ensembles (same decision rule for every variant; final models of both)

Push by the tune-week share (`gnn_chain+relgt_chain_ens0926_gfinal`):

| Variant | Score | Tune | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs `bag:gnn_chain` [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| seed:gnn_chain s0 / s1 / s2 | 52.73 / 53.52 / 53.42 | 52.48 / 52.70 / 52.87 | | | | | | | |
| seed:relgt_chain s0 / s1 / s2 | 53.85 / 53.01 / 51.72 | 51.01 / 51.53 / 51.42 | | | | | | | |
| bag:gnn_chain (reference) | 53.15 | 52.87 | 0.445 | 0.427 | 0.372 | 0.486 | 0.478 | 0.679 | 0 |
| bag:relgt_chain | **54.42** | 51.63 | 0.434 | 0.481 | 0.461 | 0.502 | 0.454 | 0.684 | +1.27 [+0.29, +2.30] |
| mean (gnn_chain + relgt_chain) | 54.05 | 53.09 | 0.435 | 0.474 | 0.434 | 0.446 | 0.477 | 0.684 | +0.90 [+0.12, +1.69] |
| select | 53.24 | 53.00 | 0.445 | 0.427 | 0.372 | 0.486 | 0.478 | 0.684 | +0.09 [−0.26, +0.45] |

With the validation models instead of the final ones, `bag:gnn_chain` scores 52.83 (seeds 52.67–53.02).

Push by the submission's rule, `--push-rule prev-week` (`gnn_chain+relgt_chain_ens0926_pprev_gfinal`;
push share 0.392):

| Variant | Score | Tune | push F1 | vs `bag:gnn_chain` [95% CI] |
|---|---|---|---|---|
| seed:gnn_chain s0 / s1 / s2 | 51.95 / 52.68 / 52.33 | 51.17 / 51.25 / 51.41 | 0.662–0.666 | |
| seed:relgt_chain s0 / s1 / s2 | 53.00 / 52.12 / 50.89 | 49.75 / 50.42 / 50.35 | 0.663–0.666 | |
| bag:gnn_chain (reference) | 52.41 | 51.55 | 0.666 | 0 |
| bag:relgt_chain | **53.73** | 50.27 | 0.669 | +1.32 [+0.31, +2.42] |
| mean | 53.39 | 51.74 | 0.673 | +0.98 [+0.24, +1.78] |
| select | 52.52 | 51.62 | 0.669 | +0.10 [−0.27, +0.48] |

### Reading

- **The final stage helps the chain GNN a little:** +0.39 on average (52.83 → 53.22), positive on every
  seed. RelGT's final model gained more (+1.1 in [holdout_0927](holdout_0927.md)). With both stages at
  their best single-run settings, the two chain models are level: GNN 53.22 (52.73–53.52) and RelGT
  53.12 (52.72–53.59) in the documented runs.
- **Seed bagging does not help the GNN.** The bag of the final models (53.15) is at the mean of its seeds
  (53.22), as for the ego and hetero GNNs in [ensemble](ensemble.md). The GNN stops after 1 epoch, so its
  seeds differ little (52.73–53.52), while RelGT's seeds differ more (51.72–53.85) and their average
  gains about 1.4 over its seed mean.
- **Bagged RelGT beats bagged GNN on the chain graph:** 54.42 against 53.15, +1.27 [+0.29, +2.30]; with
  the prev-week push rule +1.32 [+0.31, +2.42]. RelGT is ahead on open_issue (0.481 vs 0.427),
  issue_comment (0.461 vs 0.372) and pr_review; the GNN on open_pr and pr_review_comment.
- **Mixing the two (`mean`, 54.05) does not beat the RelGT bag alone (54.42).** `select` picks the GNN for
  every non-push activity by the tune week and gains nothing.
- **The tune week again points the wrong way:** the GNN has the higher tune-week score (52.87 against
  51.63), RelGT the higher holdout score. Choosing by the tune week would pick the GNN bag.
- **Push rule:** `prev-week` costs every variant 0.7–0.8 (push F1 0.68 → 0.67) and leaves the order
  unchanged.
- One holdout week; the bootstrap intervals are over users, not seeds.

Conclusion: bagged RelGT on the chain graph (54.42; 53.73 with the submission's push rule) stays the
best result on this week. Bagging the chain GNN, or adding it to the RelGT bag, does not improve on it.
