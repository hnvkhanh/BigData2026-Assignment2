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

Pending: only the smoke test has run (2026-11-16, `-Smoke`, seed 0, 1 epoch of 30 steps; its scores say
nothing about the configs). It checked the pipeline: all three configs ran on GPUs 0 and 1, and both
summaries read them with the reference smoke run. Token sampling took 47 s (fk), 53 s (fk, recent) and
390 s (chain, recent); in a full run each new config pays it once per seed.
