# Archive

Code kept for reference only. It is not maintained and does not use `ghactivity/`.

## `old_experiments/relgt_inspired/`

The first RelGT-inspired model and its relation-aware GNN. The ego-graph GNN of
`experiments/gnn/run.py` (`--graph-structure ego`, the default) is a port of `gnn_model.py`; see
[docs/experiments/ego_gnn.md](../docs/experiments/ego_gnn.md), including the rerun of this code on
the current data.

The plain backtest still runs, from the repo root:

```bash
python -m archive.old_experiments.relgt_inspired.run backtest --temporal-model gnn    # or relgt
```

Outputs go to `archive/old_experiments/relgt_inspired/outputs/` (not committed). The options that
import `experiments.lightgbm_baseline` (LightGBM blending, the S2P2 model) no longer work: that
module was removed.
