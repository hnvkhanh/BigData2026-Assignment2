# Experiment: ego GNN with the hetero model's time encoding (`--ego-time func`)

The ego GNN encodes time inside each token: a fixed sinusoidal encoding of the log age, mixed
with the other token features by the MLP. `--ego-time func` moves time onto the edges, as in
the temporal hetero GNN:
- the token has no time part (5 parts instead of 6);
- every token ↔ user edge carries `φ = cos(w·log(1+age) + b)`, learned, of the token's age
  (`--ego-time-dim 16`);
- message token → user `W_r[h_token ‖ φ]`, attention `a_r·LeakyReLU(Q h_user + m)`, softmaxed
  over the tokens of each relation, relations summed;
- message user → token `W[h_user ‖ φ]`. A token has one incoming edge from the user, so there is
  no attention;
- the push repo ↔ member link, the self transform, the head and the training stay as before.

`--ego-time none` keeps that architecture with φ = 0, the ablation (like `--het-time none`).
The default `--ego-time sin` is the original model.

```bash
python experiments/gnn/run.py --user-features --ego-time func --holdout   # -> outputs/all_ego_uf_ftime_ho/
bash runs/ego_time_ho1107.sh   # ego_uf_ftime, ego_uf_notime x seeds 0-2 on one GPU (GPU=0); summary also lists all_ego_uf (sin)
```

**Results** (holdout protocol, 3 seeds): TODO.
