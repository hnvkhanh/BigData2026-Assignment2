# Experiment: Relational Graph Transformer (`experiments/relgt/run.py`)

```bash
python experiments/relgt/run.py    # -> outputs/default/output.csv, ~35 min on an RTX 3060
```

A port of the official RelGT ([Dwivedi et al. 2025](https://arxiv.org/abs/2505.10960),
[code](https://github.com/snap-stanford/relgt)) in plain PyTorch. Same examples, labels,
eligibility rule, split, thresholds and final retrain as the baseline.

- *Graph*: every row of all 8 tables is a node, every foreign key an undirected edge (2.6M
  nodes, 5.0M edges). `PullRequests.merged_by_id` is left out (known only after the merge).
- *No leakage*: a node of table T is visible to week `s` only if its row time is before
  `s - lag_T`, the staleness that table has at the test week.
- *Tokens*: the user plus K-1 = 99 nodes sampled uniformly from its visible 1- and 2-hop
  neighbours. Each token combines node type, hop, time relative to `s`, the row's columns
  (per-table residual MLP; merged / closed / dismissed only as "before the cutoff" flags) and a
  GIN encoding of the sampled subgraph.
- *Model*: local transformer over the tokens with attention readout to the user, global
  attention to an EMA k-means codebook, feed-forward block, 6-output head. Channels 128,
  1 layer, 4 heads, dropout 0.3, Adam, batch 512. Tokens are cached in `experiments/relgt/cache/`.

Differences from the official code (torch_frame / GloVe column encoders replaced by
hand-picked columns, neighbour caps applied after the time filter, smaller model) are listed
in the script's docstring.

**Validation score** (earlier setup: test week 11-12, seed 0, weeks 10-07 / 10-14; not rerun on the revised assignment):

| Run | Val mean AP (best epoch) | 10-07 | 10-14 | Mean |
|---|---|---|---|---|
| Paper settings: lr 1e-4, 10 epochs (`outputs/lr1e-4_e10`) | 0.330 (10) | 46.48 | 42.32 | 44.40 |
| lr 3e-4, 20 epochs (default, `outputs/default`) | 0.356 (20) | 47.93 | 44.12 | **46.02** |

**Result: RelGT learns from the raw rows alone but stays below LightGBM (49.66) and the GNN
(48.03).** With lr 1e-4 it was still improving after 10 epochs; at 3e-4 validation AP levels
off after ~14 epochs. Push F1 is 0.613 (LightGBM 0.629). The model sees 99 tokens sampled
uniformly over the user's whole history, so recent activity, the strongest signal (see
Persistence in the EDA), is only partly visible. In the baseline it is counted exactly over
7/14/30 days. Only one seed was run, so differences of about 0.5 points are within noise.
