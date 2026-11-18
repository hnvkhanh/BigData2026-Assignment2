# Experiment: no chain anchors, the graph grows from the seeds only (`runs/relgt_anchor_ho0927.ps1`)

On the chain graph a user has no edges to its own rows (the actor FK is replaced by chain edges), so a walk cannot
reach the user's other chains. `sample_walk` therefore adds the **anchors** to the hop-1 pool: the latest visible
row of each of the user's chains that is not a seed. With `--chain-seed latest --num-seeds 16` this guarantees
one more row per table next to the 16 seeds.

`--chain-anchors none` removes them. Tokens are the 16 latest actions (seeds), the user (hop 1) and what the walk
reaches from them through FK and chain edges (previous / next row of a seed's chain, its repository, parent PR or
issue, other users' rows, ...). Everything else is unchanged: seeds, K = 100, budget costs, caps, recency fill,
other users' rows, and the edges among the tokens.

| Config | Anchors | Fill | Compared with |
|---|---|---|---|
| `relgt_chain_noanchor` | none | random | `relgt_chain` ([ensemble.md](ensemble.md)) |
| `relgt_chain_recent_noanchor` | none | `--fk-order recent` | `relgt_chain_recent` ([sampling.md](sampling.md)) |

Everything else is as in [relgt_ctx.md](relgt_ctx.md): holdout week 09-26 16:00 UTC, 15 epochs, seeds 0-2, runs
`<cfg>_ens0926_s<seed>`; 6 jobs.

```powershell
.\runs\relgt_anchor_ho0927.ps1             # the 2 configs x seeds 0-2 on GPUs 0 and 1, then the summaries
.\runs\relgt_anchor_ho0927.ps1 -Seeds 0    # one seed per config
```

## Sampler check (CPU, 2,000 test users, test cut-off, K = 100, 16 seeds, recent fill)

| | short of K | mean random fill | own-action tokens | own tables covered |
|---|---|---|---|---|
| anchors (`relgt_chain_recent`) | 15.7% | 6.1% | 74.6 | 4.40 |
| none | 16.3% | 6.4% | 74.8 | 3.67 |

Tokens differ for 96.5% of the examples. Without anchors the user's own rows still fill the budget (the same
number), but they come from fewer tables (3.67 instead of 4.40 of 6 on average).

## Results

Pending: run the script, then add the table here (single seeds, bags, paired bootstrap against `relgt_chain` /
`relgt_chain_recent`).
