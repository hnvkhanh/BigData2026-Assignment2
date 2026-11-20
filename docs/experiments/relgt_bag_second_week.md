# Experiment: seed bagging of RelGT on a second holdout week (10-04 → 10-11)

On the holdout week 09-27 → 10-04, `bag:relgt_chain` scored 54.42 against 52.59 for the single final-model runs
(+1.00 [+0.12, +1.94] against the reference bag), from 3 seeds on one week, and the tune week ranked the bags the
wrong way round ([ensemble](ensemble.md)). This experiment repeats the bagging comparison on a different week.
Nothing in the submission or WRITEUP.md changes until it is read.

## Protocol

```powershell
.\runs\relgt_bag_ho1003.ps1            # relgt_chain and relgt_chain_recent x seeds 0-2 (GPUs 0 and 1) -> <cfg>_ens1003_s<seed>
.\runs\relgt_bag_ho1003.ps1 -Smoke     # pipeline check
```

- Holdout week: 10-03 16:00 → 10-10 16:00 UTC (10-04 → 10-11 Taiwan time). Tune week 09-26 → 10-03; training weeks
  have labels ending by 09-26. Everything else is the protocol of [ensemble](ensemble.md): 15 epochs, validation
  model and final model (refit through 10-03), chain graph with the 16 latest actions as seeds,
  `relgt_chain_recent` adds `--fk-order recent`.
- Analysis: `scripts/summarize_runs.py` (the runs' own decisions) and `experiments/ensemble/run.py` (`seed:<run>`,
  `bag:<cfg>`, `mean`, `select`, the same decision rule, paired bootstrap over users).

**Why this week** (read-only check of 2026-11-19; eligible users and the positive rate among them):

| Week (UTC start) | Eligible | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|---|
| 09-19 (tune of 09-26) | 6,412 | 0.068 | 0.041 | 0.052 | 0.048 | 0.037 | 0.373 |
| 09-26 (first holdout) | 6,064 | 0.093 | 0.055 | 0.074 | 0.069 | 0.048 | 0.418 |
| **10-03 (this one)** | 5,731 | 0.080 | 0.043 | 0.055 | 0.053 | 0.026 | 0.355 |
| 10-10 | 5,443 | 0.069 | 0.044 | 0.042 | 0.050 | 0.033 | 0.359 |
| 10-17 | 5,165 | 0.063 | 0.043 | 0.039 | 0.041 | 0.032 | 0.327 |

All 6 labels need push data, which ends 10-27 (`FEED_END`), so the candidates are the weeks starting 10-03, 10-10 and
10-17. Push events per day of the cohort: 10-07 is 0, and 10-08 → 10-10 are 1,261 / 932 / 1,044 against ~3,000 normally,
so the push labels of week 10-03 are understated (rate 0.355 against 0.418), more than the single gap day in
`GAPS` suggests. Weeks 10-10 and 10-17 lose 10-15 → 10-17 (0 events) and 10-23. Week 10-03 has the most users and the
smallest push loss; it is not a clean week.

## Pre-registered criterion (written before the results)

For each config: **`bag:<cfg>` beats the mean of its three `seed:<run>` scores on this week by more than 0.5 points.**
- Both configs → seed bagging is added to the submission pipeline as a follow-up task.
- One config → inconclusive, no change.
- Neither → the 09-26 gain is noise and is recorded as such.

Reported but not part of the criterion: the order of `bag:relgt_chain` and `bag:relgt_chain_recent` (54.42 against
54.06 on 09-26, a tie), `mean` and `select` against the best bag, and the paired-bootstrap intervals. The single week
with 3 seeds per config stays a limit: this is a second week, not a replication that settles the effect.

## Results

Pending: the run is launched by the user (~1.5 h on two GPUs, ~3 h on one).
