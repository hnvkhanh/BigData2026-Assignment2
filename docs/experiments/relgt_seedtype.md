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

Pending: the full runs have not been done yet. The table to fill, as in [relgt_ablation.md](relgt_ablation.md) (official
6-activity score, holdout week 09-26 16:00 UTC; paired bootstrap from `experiments/ensemble/run.py`):

| Config | Mean of seeds | Bag of 3 seeds | Difference to its 16-seed pair (95% CI) |
|---|---|---|---|
| `relgt_chain` | | | reference |
| `relgt_chain_type` | | | |
| `relgt_chain_recent` | | | reference |
| `relgt_chain_type_recent` | | | |
