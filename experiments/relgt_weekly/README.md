# RelGT, MLP and GNN on weekly activity vectors

The other RelGT/GNN experiments feed the model individual neighbours (PRs,
issues, repositories) with an event-level "days since last action" time. Here
the input is only **weekly totals**: each user-week sample is the user's last
`--weeks` (default 8) weekly vectors, concatenated into a sequence of tokens.

- Week `w` is `[cutoff - (w+1)*7d, cutoff - w*7d)`, aligned with the 7-day label
  window. No day-of-week or time-of-day information reaches the model.
- Each week vector has 18 values: for each of the six actions, `log1p` of the
  event count, distinct entities (PRs/issues/repos touched), and distinct
  repositories. Quiet weeks are all zeros but still present as tokens.
- Tokens: a user root token + one token per week (learned projection of the
  vector + embedding of the week's age). They go through the RelGT local
  transformer, centroid attention and global readout (`WeeklyRelGT` subclasses
  `RelGTInspired`), then an MLP head with six logits, trained with BCE.
- Same sample set, leakage rules (action, row and entity availability strictly
  before the cutoff), 14-day/5-action eligibility, 70/15/15 chronological
  split, per-label threshold calibration and per-week official scoring as the
  other RelGT experiments, so scores compare directly.

`--architecture` picks the model that reads the same weekly vectors:

| Value | Model |
|---|---|
| `relgt` (default) | as above: root + week tokens, local transformer, centroid attention |
| `nn` | MLP on the weekly vectors concatenated into one flat input (`weeks x 18`) |
| `gnn` | star-graph message passing (`RelationalMessageLayer`): the user root and one node per week, with a separate message weight per week age, then root + mean-pooled weeks into the head |

`--num-layers` is the number of transformer / message-passing / MLP layers.

```bash
python -m experiments.relgt_weekly.run                 # 8 weeks, RelGT
python -m experiments.relgt_weekly.run --architecture nn
python -m experiments.relgt_weekly.run --architecture gnn --weeks 4
python -m experiments.relgt_weekly.run --weeks 4 --user-embedding
python run_all_experiments.py --group weekly
```

Writes `backtest_metrics.json`, `holdout_probabilities.csv` (branch
`weekly_<architecture>`, usable with `exam_fail_case.py --model-branch weekly_gnn`)
and `weekly_<architecture>_model.pt`. Backtest only; there is no `predict` mode yet.
