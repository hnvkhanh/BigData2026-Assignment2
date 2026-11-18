# Experiment: one RelGT per label (`runs/relgt_label_ho0927.ps1`)

Every RelGT run so far is one model with 6 outputs trained with a masked BCE over all labels, so the activities
share the token encoder and the attention layers. Only the LightGBM baseline ([baseline.md](baseline.md)) has one model
per activity. This experiment trains **one RelGT per label** and joins the six into one prediction, to see whether the
multi-task sharing helps or hurts.

`experiments/relgt/run.py --labels <label>` keeps only that label: the other labels are unknown to the loss (as
`--exclude-push-label` does for push), the epoch is chosen by the AP of that label, only its F1 threshold is fitted
and the others are 0 in the output. The 6-output head stays (the 5 unused outputs get no gradient). There are no joint
thresholds, as they need all 6 labels. The tokens do not depend on the labels, so the 6 runs of a seed use one token
cache.

`experiments/relgt/merge_labels.py` then takes column j of each probability, decision and rate array from the run of
label j and writes a normal run folder `<cfg>_perlabel_ens0926_s<seed>` (`holdout.npz`, `test.npz`, `output.csv`,
`metrics.json` rescored with the official scorer), so `scripts/summarize_runs.py` and `experiments/ensemble/run.py`
read it like any other config.

| Config | Models | Compared with |
|---|---|---|
| `relgt_chain_recent_perlabel` | 6 single-label `relgt_chain_recent` | `relgt_chain_recent` ([sampling.md](sampling.md)) |

Everything else is as in [relgt_ctx.md](relgt_ctx.md): holdout week 09-26 16:00 UTC, 15 epochs, runs
`<cfg>_only_<label>_ens0926_s<seed>`. One seed (42) for now, so 6 jobs. The existing `relgt_chain_recent` runs are seeds
0-2, so they are a reference by their mean, not a paired seed.

```powershell
.\runs\relgt_label_ho0927.ps1              # 6 labels, seed 42, GPUs 0 and 1, then merge and the summaries
.\runs\relgt_label_ho0927.ps1 -Smoke       # pipeline check
```

## Results

Pending: run the script, then add the table here (per-label F1 and the score of the merged run against
`relgt_chain_recent`).
