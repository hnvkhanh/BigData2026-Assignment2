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

Pending: `runs/ensemble_ho0927.sh` was smoke-tested and launched in full on 2026-11-16 15:55 (console log
`experiments/ensemble/outputs/ens0926_full.log`); fill in the table from its summary
(`python scripts/summarize_runs.py --suffix ens0926 all_ego_uf het relgt_chain relgt_chain_hist` and the
`metrics.json` of each ensemble).

| Variant | Score | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs reference [95% CI] |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

What to read from it:
- `bag:<cfg>` against `own:<cfg>`: how much seed bagging alone is worth.
- `mean` against the best bag: whether mixing models adds to bagging. The bootstrap interval says
  whether the one week supports it; the seed ranges in the other docs do not resolve differences
  under about 1 point.
- `select` against `mean`: whether picking per activity beats averaging (it uses the tune week twice,
  so it can overfit that week).
- `bag:relgt_chain_hist` against `bag:relgt_chain`: whether the history features help the chain model.
- The `--push-rule prev-week` run is the number comparable with the submission pipeline.
