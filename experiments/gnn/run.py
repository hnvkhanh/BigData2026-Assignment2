"""Heterogeneous GNN over the relational data: cohort users <-> repositories.

Graph of one sample week s (built only from rows visible before the per-table cutoff s - lag,
exactly like the LightGBM baseline):
  * user nodes: the 9,000 cohort users, with the baseline's 60 history features;
  * repository nodes: every repo touched in the 90 days before the cutoff, with its activity
    volume per edge type and whether it belongs to an organisation;
  * 7 edge types user - repo: the 6 activities (PullRequests, Issues, IssueComments, PRReviews,
    PRReviewComments, Pushes) and "received" (comments / reviews other cohort users leave on the
    user's own issues and PRs, joined through Issues and PullRequests); each edge carries the
    number of events in the last 7 / 30 / 90 days and the days since the last one.

Model: per-type GraphSAGE-style message passing user -> repo -> user (2 rounds), then a head.
By default one shared model predicts the 6 activities (6 outputs, mean of the 6 BCE losses);
--actions picks a subset (e.g. --actions push, or --actions push pull_request), everything else
being the same so the runs can be compared. --users restricts the users of every stage.

--graph-structure chain replaces that graph by the relational entity graph of experiments/relgt:
every row a node, a user linked to its latest visible row of each table, a row to the previous and
next one of the same table, and another user's row bringing that user in. The graph and the tokens
are relgt's own (Graph, tokenize). Each example (eligible user, week) is its K tokens and the edges
among them; a token is its row features, table, hop and time before the week, RowGNN passes
relation-typed messages over those edges and reads out the seed token(s), which also hold the 60
history features. Weeks, labels, --actions, --users, thresholds and output are the same as above.
--layers should reach the tokens: a message travels one hop per layer.

Run from the repo root:
    python experiments/gnn/run.py
    python experiments/gnn/run.py --actions push
    python experiments/gnn/run.py --graph-structure chain --chain-seed latest --num-seeds 16 --exclude-push
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # after numpy: importing torch first trips a duplicate-OpenMP error on this machine
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("baseline_run", HERE.parent / "baseline" / "run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

EVENT_COLS, WEEK, DAY = base.EVENT_COLS, base.WEEK, base.DAY
EDGE_TYPES = EVENT_COLS + ["received"]
EDGE_WINDOWS = [7, 30, 90]
OUT_DIR = HERE / "outputs"


def week_graph(ev, s, lags, repo_is_org, keep):
    """User-repo edges of every type and the repo node features, from rows before each cutoff.

    Only users with keep[cohort index] are nodes; their ids are renumbered 0..keep.sum()-1.
    """
    new_id = np.cumsum(keep) - 1
    raw = {}
    for k in EDGE_TYPES:
        c = s - lags[k]
        e = ev[k]
        m = ((e.avail.values < np.datetime64(c)) & (e.t.values >= np.datetime64(c - 90 * DAY))
             & ~np.isnan(e.repo.values))
        m &= keep[e.uid.values.clip(0)] & (e.uid.values >= 0)
        u, r = new_id[e.uid.values[m]], e.repo.values[m].astype(np.int64)
        age = (np.datetime64(c) - e.t.values[m]) / np.timedelta64(1, "D")
        pair, inv = np.unique(u * len(repo_is_org) + r, return_inverse=True)
        cnt = np.stack([np.bincount(inv, weights=age < w, minlength=len(pair)) for w in EDGE_WINDOWS], 1)
        last = pd.Series(age).groupby(inv).min().values
        raw[k] = (pair // len(repo_is_org), pair % len(repo_is_org), cnt, last)

    repos = np.unique(np.concatenate([v[1] for v in raw.values()]))
    local = pd.Series(np.arange(len(repos)), index=repos)
    xr = [repo_is_org[repos].astype(np.float32)[:, None]]
    edges = []
    for k in EDGE_TYPES:
        u, r, cnt, last = raw[k]
        r = local.loc[r].values if len(r) else r
        # per repo: events of this type in 30 / 90 days and number of distinct cohort users
        xr.append(np.stack([np.bincount(r, weights=cnt[:, 1], minlength=len(repos)),
                            np.bincount(r, weights=cnt[:, 2], minlength=len(repos)),
                            np.bincount(r, minlength=len(repos))], 1))
        ef = np.concatenate([np.log1p(cnt), (last / 90)[:, None]], 1).astype(np.float32)
        edges.append((u.astype(np.int64), r.astype(np.int64), ef))
    xr = np.concatenate([xr[0]] + [np.log1p(x) for x in xr[1:]], 1).astype(np.float32)
    return xr, edges


def mean_agg(msg, idx, n):
    out = torch.zeros(n, msg.shape[1], device=msg.device).index_add_(0, idx, msg)
    deg = torch.zeros(n, device=msg.device).index_add_(0, idx, torch.ones_like(idx, dtype=msg.dtype))
    return out / deg.clamp(min=1)[:, None]


class HeteroSAGE(nn.Module):
    def __init__(self, du, dr, de, n_out, d=64, layers=2, dropout=0.1):
        super().__init__()
        nt = len(EDGE_TYPES)

        def enc(i):
            return nn.Sequential(nn.Linear(i, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))

        self.enc_u, self.enc_r = enc(du), enc(dr)
        self.u2r = nn.ModuleList([nn.ModuleList([nn.Linear(d + de, d) for _ in range(nt)])
                                  for _ in range(layers)])
        self.r2u = nn.ModuleList([nn.ModuleList([nn.Linear(d + de, d) for _ in range(nt)])
                                  for _ in range(layers)])
        self.upd_r = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
        self.upd_u = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
        self.norm_r = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.norm_u = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, g):
        hu, hr = self.enc_u(g["xu"]), self.enc_r(g["xr"])
        nu, nr = hu.shape[0], hr.shape[0]
        for layer in range(len(self.u2r)):
            mr = sum(mean_agg(torch.relu(lin(torch.cat([hu[u], e], 1))), r, nr)
                     for lin, (u, r, e) in zip(self.u2r[layer], g["edges"]))
            hr = self.norm_r[layer](hr + self.drop(torch.relu(self.upd_r[layer](torch.cat([hr, mr], 1)))))
            mu = sum(mean_agg(torch.relu(lin(torch.cat([hr[r], e], 1))), u, nu)
                     for lin, (u, r, e) in zip(self.r2u[layer], g["edges"]))
            hu = self.norm_u[layer](hu + self.drop(torch.relu(self.upd_u[layer](torch.cat([hu, mu], 1)))))
        return self.head(hu)


def masked_bce(logits, y, mask):
    """Mean over tasks of the BCE over that task's labelled, eligible users."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
    m = mask[:, None] & ~torch.isnan(y)
    per_task = (loss * m).sum(0) / m.sum(0).clamp(min=1)
    return per_task[m.any(0)].mean()


def fit(graphs, weeks, n_out, epochs, seed, device, val_weeks=None, patience=5, lr=2e-3):
    """Train on the given weeks; with val_weeks, early-stop on validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g0 = graphs[weeks[0]]
    model = HeteroSAGE(g0["xu"].shape[1], g0["xr"].shape[1], g0["edges"][0][2].shape[1], n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, best_epoch, bad = np.inf, None, epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for w in rng.permutation(weeks):
            g = graphs[w]
            opt.zero_grad()
            loss = masked_bce(model(g), g["y"], g["mask"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val_weeks is None:
            continue
        model.eval()
        with torch.no_grad():
            # weeks whose labels for this task are not known yet (e.g. push of the last week) are skipped
            vl = np.mean([masked_bce(model(graphs[w]), graphs[w]["y"], graphs[w]["mask"]).item()
                          for w in val_weeks if not torch.isnan(graphs[w]["y"]).all()])
        print(f"  epoch {epoch:2d}  val loss {vl:.4f}")
        if vl < best - 1e-4:
            best, best_epoch, bad = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def predict(model, g):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(g)).cpu().numpy()


class RowGNN(nn.Module):
    """Relation-typed GraphSAGE-style message passing over each example's K tokens (--graph-structure
    chain). rows and time_encoder are relgt's RowEncoder and TimeEncoder."""

    def __init__(self, rows, time_encoder, n_types, n_hops, n_rel, n_seed_feats, n_out, d=64, layers=3, dropout=0.1):
        super().__init__()
        self.rows, self.time_encoder = rows, time_encoder
        self.type_encoder = nn.Embedding(n_types, d)
        self.hop_encoder = nn.Embedding(n_hops, d)
        self.rel_encoder = nn.Embedding(n_rel, d)
        self.seed_encoder = nn.Sequential(nn.Linear(n_seed_feats, d), nn.ReLU(), nn.Linear(d, d))
        self.mix = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.msg = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, nodes, hops, paths, week, cidx, edge_index, edge_type, seed_x):
        B, K = nodes.shape
        tfs, types = self.rows(nodes, week)
        h = self.mix(torch.cat([tfs, self.type_encoder(types), self.hop_encoder(hops + 1),
                                self.time_encoder(self.rows.rel_time(nodes, week))], -1))
        seed = (hops == 0).unsqueeze(-1)
        h = (h + seed * self.seed_encoder(seed_x).unsqueeze(1)).view(B * K, -1)
        src, dst = edge_index
        rel = self.rel_encoder(edge_type)
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            m = mean_agg(torch.relu(msg(torch.cat([h[src], rel], 1))), dst, B * K)
            h = norm(h + self.drop(torch.relu(upd(torch.cat([h, m], 1)))))
        return self.head((h.view(B, K, -1) * seed).sum(1) / seed.sum(1))


def load_relgt():
    """experiments/relgt/run.py; loaded only when used, as relgt itself imports this file."""
    spec = importlib.util.spec_from_file_location("relgt_run", HERE.parent / "relgt" / "run.py")
    relgt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relgt)
    return relgt


def merge_repeated_tokens(nodes, edges, n_rel):
    """Tokens repeat when a walk does not fill K. Moves the edges to each node's first token, once, so
    a repeated row does not count twice in the mean aggregation; the repeats are left without edges."""
    K = nodes.shape[1]
    order = np.argsort(nodes, axis=1, kind="stable")
    ordered = np.take_along_axis(nodes, order, 1)
    starts = np.c_[np.ones(len(nodes), bool), ordered[:, 1:] != ordered[:, :-1]]
    first_sorted = np.maximum.accumulate(np.where(starts, np.arange(K)[None, :], 0), axis=1)
    first = np.empty_like(order)
    np.put_along_axis(first, order, np.take_along_axis(order, first_sorted, 1), 1)
    r, i, j, rel = edges
    r = r.astype(np.int64)
    key = np.unique(((r * K + first[r, i]) * K + first[r, j]) * n_rel + rel)
    rel, key = key % n_rel, key // n_rel
    j, key = key % K, key // K
    i, r = key % K, key // K
    return r.astype(np.int32), i.astype(np.int16), j.astype(np.int16), rel.astype(np.int8)


def token_stages(args, tables, feats, weeks, train_weeks, cohort, lags, test, labels, device):
    """--graph-structure chain: the tokens of every example, and the functions that train on weeks and
    predict one week (one value per row of its table; NaN for users without an example)."""
    relgt = load_relgt()
    all_weeks = weeks + [base.TEST_START]
    min_lag = min(lags.values())
    cutoffs = np.array([[((s - lags[relgt.LAG_KEY[t]] if t in relgt.LAG_KEY else s - min_lag) - base.DATA_START) / DAY
                         for t in relgt.TABLES] for s in all_weeks], np.float32)
    week_day = np.array([(s - base.DATA_START) / DAY for s in all_weeks], np.float32)

    # Examples: the eligible users of every labelled week, then the test users, as in relgt.
    rows = {w: np.flatnonzero(tables[w].eligible.values) for w in weeks}
    tt = tables[base.TEST_START]
    rows[base.TEST_START] = pd.Series(np.arange(len(tt)), index=cohort.github_user_id.values[tt.cidx.values]
                                      ).loc[test.UserID].values
    cidx = [tables[w].cidx.values[rows[w]] for w in all_weeks]
    G = relgt.Graph("chain")
    seeds = [G.offset[relgt.TABLES.index("Users")] + cohort.user_id.values[c] for c in cidx]
    nodes, hops, paths, edges = relgt.tokenize(G, seeds, cutoffs, args.num_neighbors, args.seed, "uniform", "random",
                                               "none", args.lookback_days, args.max_hops, args.chain_seed,
                                               args.exclude_push, args.num_seeds)
    edges = merge_repeated_tokens(nodes, edges, relgt.N_ALL_REL)

    label_cols = [f"y_{k}" for k in labels]
    y = np.concatenate([tables[w][label_cols].values[rows[w]] if w != base.TEST_START
                        else np.full((len(rows[w]), len(labels)), np.nan) for w in all_weeks])
    seed_x = np.log1p(np.concatenate([tables[w][feats].values[rows[w]] for w in all_weeks]))
    sizes = np.cumsum([0] + [len(c) for c in cidx])
    ids = {w: np.arange(sizes[i], sizes[i + 1]) for i, w in enumerate(all_weeks)}
    train_ids = np.concatenate([ids[w] for w in train_weeks])
    seed_x = (seed_x - seed_x[train_ids].mean(0)) / (seed_x[train_ids].std(0) + 1e-6)
    week = np.repeat(np.arange(len(all_weeks)), np.diff(sizes))
    S = relgt.Samples(nodes, hops, paths, week, np.concatenate(cidx), y, edges, device, seed_x)
    print(f"{sizes[-1]:,} examples x {args.num_neighbors} tokens, {len(edges[0]):,} subgraph edges; "
          f"share of tokens within {args.layers} hops of a seed: {(hops <= args.layers).mean():.0%}")

    def logits(model, examples):
        return torch.cat([model(*S.batch(examples[i:i + args.batch_size])[0])
                          for i in range(0, len(examples), args.batch_size)])

    def train(fit_weeks, epochs, val_weeks=None, patience=5):
        """Like fit: batches of examples; with val_weeks, early stop on the mean weekly validation loss."""
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = RowGNN(relgt.RowEncoder(G, cutoffs, week_day, args.channels, 0.1), relgt.TimeEncoder(args.channels),
                       len(relgt.TABLES) + 1, args.max_hops + 3, relgt.N_ALL_REL, seed_x.shape[1], len(labels),
                       d=args.channels, layers=args.layers).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best, best_state, best_epoch, bad = np.inf, None, epochs, 0
        examples = np.concatenate([ids[w] for w in fit_weeks])
        for epoch in range(1, epochs + 1):
            model.train()
            perm = rng.permutation(examples)
            for step, i in enumerate(range(0, len(perm), args.batch_size)):
                if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                    break
                inputs, yb = S.batch(perm[i:i + args.batch_size])
                opt.zero_grad()
                loss = masked_bce(model(*inputs), yb, torch.ones(len(yb), dtype=torch.bool, device=device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if val_weeks is None:
                continue
            model.eval()
            with torch.no_grad():
                vl = []
                for w in val_weeks:
                    yw = S.y[torch.tensor(ids[w], device=device)]
                    if not torch.isnan(yw).all():  # weeks whose labels are not known yet are skipped
                        vl.append(masked_bce(logits(model, ids[w]), yw, torch.ones(len(yw), dtype=torch.bool,
                                                                                   device=device)).item())
            vl = float(np.mean(vl))
            print(f"  epoch {epoch:2d}  val loss {vl:.4f}")
            if vl < best - 1e-4:
                best, best_epoch, bad = vl, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch

    def predict_week(model, w):
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(logits(model, ids[w])).cpu().numpy()
        out = np.full((len(tables[w]), len(labels)), np.nan)
        out[rows[w]] = p
        return out

    return train, predict_week


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", nargs="+", choices=EVENT_COLS, default=None, metavar="ACTION",
                    help=f"activities to train and predict, one shared model (default: all of "
                         f"{', '.join(EVENT_COLS)})")
    ap.add_argument("--task", choices=["all", "push"], default=None,
                    help="shortcut kept for old commands: all = every activity, push = --actions push")
    base.add_period_args(ap)
    ap.add_argument("--users", choices=["all", "test", "exclude_test"], default="all",
                    help="users used for the loss, early stopping and thresholds: all cohort users, "
                         "only the users of test_input.csv, or all cohort users except those. "
                         "Applies to the tables and graphs of every stage (train/val/final fit); "
                         "exclude_test still needs the test users in the test-week graph to predict them")
    ap.add_argument("--name", default=None,
                    help="name of the run: results are saved in outputs/<name>/ "
                         "(default: derived from the actions and --users)")
    ap.add_argument("--lr", type=float, default=None,
                    help="AdamW learning rate (default: 2e-3, with --graph-structure chain 3e-4)")
    ap.add_argument("--epochs", type=int, default=None, help="default: 40, with --graph-structure chain 20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph-structure", choices=["user-repo", "chain"], default="user-repo",
                    help="user-repo: one user <-> repository graph per week. chain: the row graph of "
                         "experiments/relgt, one token subgraph per example; the options below need it")
    g = ap.add_argument_group("--graph-structure chain (same meaning as in experiments/relgt)")
    g.add_argument("--max-hops", type=int, default=50, help="how far the walk may go from the seed")
    g.add_argument("--chain-seed", choices=["user", "latest"], default="user",
                   help="where the walk starts and the readout is taken: the user, or its latest visible action")
    g.add_argument("--num-seeds", type=int, default=1,
                   help="with --chain-seed latest: the N latest visible actions are the seeds")
    g.add_argument("--exclude-push", action="store_true", help="hide all Pushes rows from the graph")
    g.add_argument("--lookback-days", type=float, default=None, help="only activity rows of the last N days are visible")
    g.add_argument("--num-neighbors", type=int, default=100, help="K tokens per example")
    g.add_argument("--layers", type=int, default=3, help="message passing rounds")
    g.add_argument("--channels", type=int, default=64)
    g.add_argument("--batch-size", type=int, default=512)
    g.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full epoch")
    args = ap.parse_args()
    chain = args.graph_structure == "chain"
    args.lr = args.lr or (3e-4 if chain else 2e-3)
    args.epochs = args.epochs or (20 if chain else 40)
    if not chain and any(ap.get_default(a.dest) != getattr(args, a.dest) for a in g._group_actions):
        ap.error("these options need --graph-structure chain: " + ", ".join(
            a.option_strings[0] for a in g._group_actions if ap.get_default(a.dest) != getattr(args, a.dest)))
    if args.num_seeds != 1 and args.chain_seed != "latest":
        ap.error("--num-seeds needs --chain-seed latest")
    if not 1 <= args.num_seeds < args.num_neighbors:
        ap.error("--num-seeds must be in 1..num_neighbors - 1")
    if not 1 <= args.max_hops <= 120:
        ap.error("--max-hops must be in 1..120")
    if args.actions and args.task:
        ap.error("use either --actions or --task, not both")
    labels = args.actions or (["push"] if args.task == "push" else list(EVENT_COLS))
    labels = [k for k in EVENT_COLS if k in labels]  # canonical order, no duplicates
    args.task = "all" if labels == list(EVENT_COLS) else "+".join(labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_name = (args.task if args.users == "all" else f"{args.task}_{args.users}") + ("_chain" if chain else "")
    out_dir = OUT_DIR / (args.name or default_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, ev, own, data_end = base.load()
    repos = pd.read_parquet(base.DB / "Repositories.parquet", columns=["repo_id", "org_id"])
    repo_is_org = np.zeros(repos.repo_id.max() + 1, bool)
    repo_is_org[repos.repo_id.values] = repos.org_id.notna().values
    n = len(cohort)
    lags = {k: base.TEST_START - v for k, v in data_end.items()}

    # Same weeks and split as the baseline.
    weeks, train_weeks, val_weeks, final_weeks = base.split_weeks(args, data_end)
    print(f"task {args.task}, device {device}, {len(train_weeks)} train weeks, "
          f"validation {[str(w)[:10] for w in val_weeks]}")

    tables = {w: base.build_week(w, ev, own, lags, data_end, n) for w in weeks}
    tables[base.TEST_START] = base.build_week(base.TEST_START, ev, own, lags, data_end, n,
                                              with_labels=False)
    test = pd.read_csv(base.TEST_INPUT)
    is_test = np.isin(cohort.github_user_id.values, test.UserID.values)
    keep = {"all": np.ones(n, bool), "test": is_test, "exclude_test": ~is_test}[args.users]
    print(f"users {args.users}: {keep.sum()} of {n} cohort users in train/validation")
    # Users outside the selection are removed from the tables and from the graph. The test week
    # of 'exclude_test' is the one exception: the test users must be present to be predicted.
    sel = {w: keep for w in tables}
    if args.users == "exclude_test":
        sel[base.TEST_START] = np.ones(n, bool)
    tables = {w: t[sel[w]].reset_index(drop=True) for w, t in tables.items()}
    feats = [c for c in tables[weeks[0]].columns
             if c not in {"eligible", "cidx", "week"} and not c.startswith("y_")]
    if chain:
        train, predict_week = token_stages(args, tables, feats, weeks, train_weeks, cohort, lags, test, labels, device)
    else:
        raw_graphs = {w: week_graph(ev, w, lags, repo_is_org, sel[w]) for w in tables}

        # Standardise log1p features with statistics of the training weeks only.
        xu_tr = np.log1p(np.concatenate([tables[w][feats].values for w in train_weeks]))
        xr_tr = np.concatenate([raw_graphs[w][0] for w in train_weeks])
        mu_u, sd_u = xu_tr.mean(0), xu_tr.std(0) + 1e-6
        mu_r, sd_r = xr_tr.mean(0), xr_tr.std(0) + 1e-6

        graphs = {}
        for w, t in tables.items():
            xr, edges = raw_graphs[w]
            y = (t[[f"y_{k}" for k in labels]].values if w != base.TEST_START
                 else np.full((len(t), len(labels)), np.nan))
            graphs[w] = {
                "xu": torch.tensor((np.log1p(t[feats].values) - mu_u) / sd_u, dtype=torch.float32, device=device),
                "xr": torch.tensor((xr - mu_r) / sd_r, dtype=torch.float32, device=device),
                "edges": [(torch.tensor(u, device=device), torch.tensor(r, device=device),
                           torch.tensor(e, device=device)) for u, r, e in edges],
                "y": torch.tensor(y, dtype=torch.float32, device=device),
                "mask": torch.tensor(t.eligible.values, device=device),
            }
        print(f"graph of the test week: {n} users, {graphs[base.TEST_START]['xr'].shape[0]} repos, "
              f"{sum(len(u) for u, _, _ in graphs[base.TEST_START]['edges'])} edges")

        def train(fit_weeks, epochs, val_weeks=None):
            return fit(graphs, fit_weeks, len(labels), epochs, args.seed, device, val_weeks=val_weeks, lr=args.lr)

        def predict_week(model, w):
            return predict(model, graphs[w])

    model, best_epoch = train(train_weeks, args.epochs, val_weeks=val_weeks)
    print(f"best epoch {best_epoch}")

    # Thresholds on the pooled validation weeks (same rule as the baseline).
    metrics = {"task": args.task, "seed": args.seed, "lr": args.lr, "best_epoch": best_epoch, "per_label": {},
               "validation": []}
    if chain:
        metrics["graph"] = {a.dest: getattr(args, a.dest) for a in g._group_actions}
    val_p = {w: predict_week(model, w) for w in val_weeks}
    rates = {}
    for j, k in enumerate(labels):
        ys, ps = [], []
        for w in val_weeks:
            m = tables[w].eligible.values & ~np.isnan(tables[w][f"y_{k}"].values)
            ys.append(tables[w][f"y_{k}"].values[m])
            ps.append(val_p[w][m, j])
        y, p = np.concatenate(ys), np.concatenate(ps)
        thr, f1 = base.best_threshold(y, p)
        rates[k] = float((p >= thr).mean())
        metrics["per_label"][k] = {"threshold": thr, "val_f1": f1, "val_pred_rate": rates[k]}
        print(f"{k:18s} val F1 {f1:.4f}  threshold {thr:.3f}  predicted {rates[k]:.3f}  "
              f"positives {y.mean():.3f}")

    # Per-week scores on weeks where the labels this model predicts are all known.
    for w in val_weeks:
        t = tables[w]
        m = t.eligible.values
        if t.loc[m, [f"y_{k}" for k in labels]].isna().any().any():
            continue
        f1 = {}
        for j, k in enumerate(labels):
            y = t[f"y_{k}"].values[m]
            pred = val_p[w][m, j] >= metrics["per_label"][k]["threshold"]
            f1[k] = float(2 * (pred & (y == 1)).sum() / (pred.sum() + y.sum()))
        r = {"week": str(w), "users": int(m.sum()), "f1": f1}
        if args.task == "all":
            gids = cohort.github_user_id.values[t.cidx.values[m]]
            truth = base.to_output(gids, {k: t[f"y_{k}"].values[m] for k in EVENT_COLS})
            pred = base.to_output(gids, {k: val_p[w][m, j] >= metrics["per_label"][k]["threshold"]
                                         for j, k in enumerate(EVENT_COLS)})
            r.update(base.scores(pred, truth))
        metrics["validation"].append(r)
        print(f"validation {str(w)[:10]}: " + "  ".join(f"{k} F1 {v:.4f}" for k, v in f1.items())
              + (f"  score {100 * r['score']:.2f}" if "score" in r else ""))
    metrics["val_f1_mean"] = {k: float(np.mean([r["f1"][k] for r in metrics["validation"]]))
                              for k in labels}
    for k, v in metrics["val_f1_mean"].items():
        print(f"mean validation {k} F1 {v:.4f}")
    if args.task == "all":
        metrics["val_score_mean"] = float(np.mean([r["score"] for r in metrics["validation"]]))
        print(f"mean validation score {100 * metrics['val_score_mean']:.2f}")

    # Final model: all labelled weeks for the early-stopped number of epochs; the validation
    # share of predicted positives is reused as the decision rule (as in the baseline).
    final, _ = train(final_weeks, best_epoch)
    p = predict_week(final, base.TEST_START)
    tt = tables[base.TEST_START]
    cidx = pd.Series(np.arange(len(tt)), index=cohort.github_user_id.values[tt.cidx.values]).loc[test.UserID].values
    out = pd.DataFrame({"UserID": test.UserID.values})
    for j, k in enumerate(labels):
        q = p[cidx, j]
        out[k] = (q >= np.quantile(q, 1 - rates[k])).astype(int)
    name = "output.csv" if args.task == "all" else f"{args.task}.csv"
    out.to_csv(out_dir / name, index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {out_dir / name}")

    print(f"\nF1 per action (run {out_dir.name}, validation weeks):")
    print(f"  {'action':18s} {'pooled F1':>10s} {'mean weekly F1':>15s}")
    for k in labels:
        weekly = metrics["val_f1_mean"].get(k, float("nan"))
        print(f"  {k:18s} {metrics['per_label'][k]['val_f1']:10.4f} {weekly:15.4f}")
    print(f"  {'average':18s} {np.mean([metrics['per_label'][k]['val_f1'] for k in labels]):10.4f} "
          f"{np.mean([metrics['val_f1_mean'][k] for k in labels]):15.4f}")


if __name__ == "__main__":
    main()
