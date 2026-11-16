# Experiment: RelGNN on the chain and FK graphs (`runs/relgnn_ho0927.ps1`)

On the 09-27 holdout week the two chain-graph models are level as single runs. `gnn_chain` (`RowGNN`,
relation-typed GraphSAGE over the tokens) has a final model at 53.22, and `relgt_chain` at 53.12
([gnn_chain_bag](gnn_chain_bag.md), [holdout_0927](holdout_0927.md)). Both pass messages one foreign key
at a time, so a user meets a push and its repository in different layers. RelGNN (Chen et al., ICML 2025)
adds *atomic routes*: for a table with several keys, `dim_v -> fact -> dim_q` in one layer, with the fact
row fused with its `dim_v` row before `dim_q` attends over it (e.g. `Repositories -> Pushes -> Users`). This
experiment ports RelGNN and asks whether composite message passing beats `RowGNN` on the same tokens, and
how it does on the paper's plain FK graph.

## What was changed

- `experiments/relgnn/model.py`: RelGNN in plain PyTorch (atomic routes, `RelGNNConv`, the hetero conv,
  the model). Design and differences from the official code: [experiments/relgnn/README.md](../../experiments/relgnn/README.md).
- `experiments/gnn/run.py`:
  - `--token-model relgnn` (default `rowgnn`) swaps `RowGNN` for RelGNN on the same tokens, with
    `--relgnn-heads` (4) and `--relgnn-readout user|seeds` (user).
  - `--layers` / `--channels` now default to 3 / 64 for `rowgnn` (as before) and 2 / 128 for `relgnn`.
  - `--graph-structure fk` runs the token models on relgt's plain FK graph and its 1- / 2-hop tokens.
  - The validation stage's epoch lines now carry their time, and each stage prints its peak GPU memory.
  - The default `rowgnn` chain runs are unchanged (same arguments, tokens and cache key).
- `scripts/summarize_runs.py`: a second row, `<cfg> (final model)`, for GNN runs made with `--holdout-final`.

## Setup

| Config | Graph | Tokens | Readout |
|---|---|---|---|
| `relgnn_chain` | chain | 16 latest actions + BFS walk (`gnn_chain`'s cached tokens) | the user token (RelGNN's entity) |
| `relgnn_chain_seeds` | chain | the same | mean of the 16 seed actions (`RowGNN`'s) |
| `relgnn_fk` | fk | the user + 99 rows of its 1- / 2-hop neighbours (`relgt_fk`'s cached tokens, [sampling](sampling.md)) | the user token |
| `gnn_chain` (reference) | chain | as `relgnn_chain` | mean of the seeds, `RowGNN` |
| `relgt_chain` (reference) | chain | as `relgnn_chain` (sampled by the RelGT runner) | RelGT |

- Protocol of `gnn_chain` in `runs/gnn_chain_bag_ho0927.ps1` (`experiments/gnn/run.py --holdout
  --holdout-week "2026-09-26 16:00" --holdout-final`):
  - Training labels end by 09-19; the tune week 09-19 → 09-26 sets early stopping (validation loss,
    patience 5, at most 20 epochs) and the share of users predicted positive.
  - The holdout week 09-26 16:00 UTC is scored by the validation model and by the final model, which is
    refit through the tune week for the best epoch.
  - The 60 history features join the readout token, as in `gnn_chain`.
- RelGNN: 2 layers, 128 channels, 4 heads, K = 100. That is 39.6M parameters on the chain graph (54
  routes) and ~31.7M on the FK graph (42 routes). AdamW lr 3e-4, batch 512.
- Seeds 0–2, runs `<cfg>_ens0926_s<seed>` in `experiments/gnn/outputs`. The reference runs
  `gnn_chain_ens0926_s0-2` and `relgt_chain_ens0926_s0-2` exist and are not rerun.
- Summaries:
  - `scripts/summarize_runs.py`: each run's own decisions, validation and final models.
  - `experiments/ensemble/run.py --gnn-variant final`: all configs, single seeds and seed bags under the
    same decision rule, paired bootstrap against the bag with the best tune-week score; push by the tune
    week's share and by `prev-week`.
  - The mean of `relgnn_chain` and `relgt_chain`.

```powershell
.\runs\relgnn_ho0927.ps1                                 # 3 configs x seeds 0-2 on GPUs 0 and 1, then the summaries
.\runs\relgnn_ho0927.ps1 -Configs relgnn_chain -Gpus 1   # one config on one GPU
.\runs\relgnn_ho0927.ps1 -Smoke                          # pipeline check: seed 0, 1 epoch of 30 steps
.\runs\relgnn_ho0927.ps1 -SkipRuns                       # only the summaries
```

## Results

Pending: only the smoke test has run. It ran on 2026-11-16 (`-Smoke -Gpus 1`, seed 0, 1 epoch of 30 steps
per stage); its scores say nothing about the configs. What it checked:

- **The pipeline:** all three configs ran both stages and wrote `holdout.npz` with `prob_final` (1.3–2 min
  each). The summary listed their validation and final models, and the three ensemble calls read them
  together with `smoke_gnn_chain` and `smoke_relgt_chain`.
- **The tokens:** the chain configs loaded `gnn_chain`'s tokens and `relgnn_fk` loaded `relgt_fk`'s. No
  sampling was needed. The user token is in 100% of the 254,624 examples on both graphs.
- **Port correctness:** before the run, `RouteConv` matched a per-node reimplementation of PyG's
  `TransformerConv` + `final_proj` and `SAGEConv` to 1e-15 in float64, with identical gradients.
- **Cost:** 30 steps plus validation took 18–19 s on the chain graph and 12 s on fk. Peak GPU memory was
  4.7 GiB on the chain graph and 3.6 GiB on fk (RTX 3060). A full epoch (~450 steps) is ~4.5 min on the
  chain graph and ~3 min on fk. A full run trains best epoch + 5 epochs, then the final fit: about 35 min
  if the best epoch is 1, about 70 min if it is 5.
