# RelGNN on the row graph (`experiments/relgnn/model.py`)

RelGNN (Chen, Kanatsoulis & Leskovec, *Composite Message Passing for Relational Deep Learning*, ICML 2025,
[arXiv 2502.06784](https://arxiv.org/abs/2502.06784)) predicting the 6 weekly activity labels for each
(cohort user, week). Plain PyTorch port of the official code
([snap-stanford/RelGNN](https://github.com/snap-stanford/RelGNN): `atomic_routes.py`, `relgnn_conv.py`,
`relgnn_hetero_conv.py`, `relgnn_nn.py`, `relgnn_model.py`). It runs on the token subgraphs of
[experiments/relgt](../relgt/README.md) and is trained by `experiments/gnn/run.py --token-model relgnn`, with
the same weeks, examples, early stopping, thresholds and outputs as the chain GNN (`RowGNN`). Only the
message passing over the tokens changes.

## 1. Atomic routes

RDL graphs have one node per row and one edge per foreign key. A row with several keys (a push: its user
and its repository) only connects the rows it references, so in a plain GNN the user needs two layers to
see a push's repository, and the push's own message arrives without it. RelGNN calls such a table a
*fact* and the referenced rows *dims*. For every ordered pair (q, v) of a fact's keys it adds the
**atomic route** `dim_v -> fact -> dim_q`: in one layer, `dim_q` attends over its fact rows, each already
fused with its `dim_v` row. A table with a single key, and every self-referencing key, gives two plain
`dim-dim` routes (the key and its reverse).

`atomic_routes()` derives them from `relgt.FKEYS` (`PullRequests.merged_by_id` is not a key there):

| Fact table | Keys | Routes |
|---|---|---|
| PullRequests | repo_id, author_id | 2 dim-fact-dim |
| Issues | repo_id, author_id, assignee_id, pull_request_id | 12 dim-fact-dim |
| IssueComments | issue_id, repo_id, author_id | 6 dim-fact-dim |
| PRReviews | pr_id, repo_id, reviewer_id | 6 dim-fact-dim |
| PRReviewComments | pr_id, repo_id, author_id, pull_request_review_id | 12 dim-fact-dim |
| PRReviewComments | in_reply_to_id (self) | 2 dim-dim |
| Pushes | user_id, repo_id | 2 dim-fact-dim |
| each activity table (chain graph only) | the actor's next row of the table (self) | 2 dim-dim each, 12 |

That is 42 routes on the FK graph and 54 on the chain graph. Users receive 15 routes (one per key of
another table that points at them), Repositories 12, PullRequests 8 (+2 chain routes). IssueComments and
Pushes are referenced by nothing, so they are updated only through their fused rows (and the chain).
Examples: `Repositories -> Pushes -> Users` (a user reads each push with its repository),
`Users -> PRReviews -> PullRequests` (a PR reads each review with its reviewer),
`PullRequests -> Issues -> Repositories`.

## 2. One layer

Per route, with the official modules:

```
dim-fact-dim  fused = lin_l(sum of the fact's dim_v rows) + lin_r(x_fact)        SAGEConv(aggr="sum")
              out_q = final_proj(TransformerConv((fused, x_q)))                  dim_q attends over its facts
              the fact nodes receive `fused`, the dim_q nodes `out_q`
dim-dim       out   = final_proj(TransformerConv((x_src, x_dst)))
```

`TransformerConv` (heads H, concatenated, root weight, no beta, no attention dropout): per head,
`softmax_j(q_i . k_j / sqrt(d))` over the edges into node i, the weighted sum of `v_j`, plus `lin_skip(x_i)`;
`final_proj` maps H·d back to d. Each node type sums the outputs of the routes that reach it, then a
LayerNorm per node type and ReLU (`RelGNN.forward` in `relgnn_nn.py`).

`final_proj` is linear, so it is applied to each edge's message before the sum and to the skip term as one
composed map. This is the same function with the same parameters, and it avoids a [nodes, H·d] tensor per
route. A check against a per-node reimplementation of PyG's `TransformerConv` and `SAGEConv` matches to
1e-15 in float64, with identical gradients.

## 3. Model

* **Rows:** relgt's `RowEncoder`, a residual MLP per table over hand-picked columns. It stands in for
  relbench's `HeteroEncoder` (a torch_frame ResNet per table) and uses the same 0.2 dropout.
* **Time:** relbench's `HeteroTemporalEncoder` is relgt's `PositionalEncoding` of the token's age in days
  before the week, plus a linear layer per node type. It is added to the row encoding. The user token's age
  is its time since first seen.
* **History features:** the 60 per-user features of the GNN runner pass through an MLP and are added to the
  readout token(s), as in `RowGNN`. RelGNN has none; in RDL terms they are extra columns of the user row.
* **Readout** (`--relgnn-readout`): `user` (default) reads the example's user token, the entity node, as in
  RelGNN. `seeds` reads the mean of the hop-0 tokens, as in `RowGNN`. The two differ only with
  `--chain-seed latest`, where the seeds are the user's latest actions.
* **Head:** one linear layer, as the official `MLP(num_layers=1)`.
* **Defaults:** 2 layers, 128 channels, 4 heads (`--layers`, `--channels`, `--relgnn-heads`). The official
  configs use 128 channels, 1–4 layers and 1–32 heads. At these defaults the model has 39.6M parameters on
  the chain graph and 31.7M on the FK graph, mostly the routes' attention projections. A batch of 512
  examples peaks at 4.7 GiB of GPU memory.

## 4. On the token subgraphs

Each example is K = 100 tokens and the FK edges among them, in both directions (`relgt.subgraph_edges`).
The relation id gives the key and the direction, so a route's edges are picked out by
(relation id, source node type) once per batch and shared by all layers.

* **FK graph** (`--graph-structure fk`, the paper's setting): token 0 is the user, and the other 99 are
  sampled from its 1- and 2-hop neighbours (`relgt.sample_tokens`). The user links to all its sampled rows,
  so `Repositories -> Pushes -> Users` and its siblings carry most of the signal in one layer.
* **Chain graph** (`--graph-structure chain --chain-seed latest --num-seeds 16`, the tokens of `gnn_chain`):
  an activity row loses its actor key and links to the actor's next row instead. The user links to the
  latest visible row of each table, and another user's row links to that user. So the actor-key routes
  carry the user's latest row of each table (and other users' rows to their owners). Older rows reach the
  user through the chain routes, one step per layer. With the user readout and 2 layers, the user sees
  about two rows of each chain plus whatever those rows fuse in. The `seeds` readout reads the 16 latest
  actions directly.

## 5. Differences from the official code

* **Sampler:** the official code uses `NeighborLoader` (128 then 64 neighbours per edge type, 2 hops,
  directional tree, uniform among earlier rows). Here the sample is relgt's K = 100 tokens per example with
  the induced edges, the same tokens as the RelGT and RowGNN runs. On the 09-27 holdout protocol the
  chain-graph tokens are the cached ones of `gnn_chain`, and the FK-graph tokens are those of `relgt_fk`
  (the same cache key).
* **Row features:** hand-picked columns (flags, log counts, one-hot categories, text as its length) instead
  of torch_frame column-type inference and GloVe text embeddings (the text is anonymised here).
* **Training:** the GNN runner's protocol. AdamW lr 3e-4, weight decay 1e-4, batch 512, early stopping on
  the tune week's validation loss (patience 5, at most `--epochs`), thresholds by the tune week's share of
  positives, and the final model refit through the tune week (`--holdout-final`). Relbench uses Adam
  lr 5e-3 and picks the epoch by validation metric.
* **Not ported:** `simplified_MP`, which only differs from the default when a batch has no edge of a route,
  and the shallow ID embeddings (`shallow_list`, empty for node tasks).

## 6. Run

```bash
# chain graph, gnn_chain's tokens; holdout week 09-27, validation and final models
python experiments/gnn/run.py --graph-structure chain --chain-seed latest --num-seeds 16 --token-model relgnn \
    --holdout --holdout-week "2026-09-26 16:00" --holdout-final --name relgnn_chain_ens0926_s0
# the paper's FK graph
python experiments/gnn/run.py --graph-structure fk --token-model relgnn \
    --holdout --holdout-week "2026-09-26 16:00" --holdout-final --name relgnn_fk_ens0926_s0
```

`runs/relgnn_ho0927.ps1` runs three configs × seeds 0–2 and their summaries; see
[docs/experiments/relgnn.md](../../docs/experiments/relgnn.md).
