"""GNNs over the relational data, on the weeks, lags, eligibility, split and thresholds of the baseline.

--graph-structure ego (default) is the relation-aware ego-graph GNN of old_experiments/relgt_inspired
(gnn_model.py), ported to the current protocol. Each example (eligible user, week) is a star graph:
  * a learned user node (with --user-features, plus an MLP of the baseline's 60 history features);
  * 48 tokens from the last --ego-lookback days (14) before each table's cutoff s - lag: the 4 latest
    objects of each of the 6 activities (the PR / issue acted on; for a push, its repository; hop 1)
    and the 24 latest repositories of any activity (hop 2, relation "member");
  * a token is its entity id (learned embedding), node type, relation, hop, 4 counts (events, active
    days, events of the entity / repo by all cohort users, users per relation) and a sinusoidal
    encoding of its log age; the six are concatenated and mixed by an MLP.
Message passing (--ego-layers rounds): token -> user with one weight per relation, user -> token, and
push repository <-> member token of the same repo, mean-aggregated and added to a self transform. The
head reads the user node and the mean of all nodes. Mini-batches, AdamW lr 1e-3, early stopping.
Results go to outputs/<task>_ego[_uf][_ftime|_notime][_seg]/.
--ego-time func moves time to the edges as in --graph-structure hetero: the token has no time part;
each token <-> user edge carries the learned cos(w log(1 + age) + b) encoding phi of the token's age
(--ego-time-dim), the messages are W_r [h_token, phi] (token -> user) and W [h_user, phi] (user ->
token), and token -> user is attention-weighted, a_r . LeakyReLU(Q h_user + m) softmaxed over the
tokens of each relation, relations summed. --ego-time none is the same with phi = 0 (the ablation).

--segment activity (ego) trains two models: one on the quiet examples (no ego token, i.e. no visible
activity in the last --ego-lookback days before the cutoffs) and one on the active ones. Each is
early-stopped on its own segment's validation loss and retrained for its own best epoch. The quiet
model always gets the user features (its tokens are all padding); the active one follows
--user-features. Thresholds are tuned on the pooled predictions, as for one model; on the test week
the validation share of predicted positives is reused per segment. Every ego run also reports the
F1 per segment on validation, so single and split models can be compared segment by segment.
--active-min-events N (default 1) makes an example active only with >= N visible activity rows (all 6
activities) in that window; 5 reproduces the selection of the old experiment.

--graph-structure user-repo is the heterogeneous GraphSAGE over cohort users <-> repositories
(HeteroSAGE, the default before; experiments/gnn_review builds on it). Graph of one sample week s
(built only from rows visible before the per-table cutoff s - lag, exactly like the LightGBM baseline):
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

--graph-structure hetero is a temporal heterogeneous GNN, one full graph per week like user-repo, but
with the objects and the events kept (TemporalHeteroGNN, TGAT-style):
  * nodes: cohort users (the 60 history features), and the PRs, issues and repositories they act on
    in the 90 days (--history-days) before the per-table cutoffs, with counts of visible events;
  * edges, one per event: user -> PR / issue / pushed repo per activity (the latest --het-max-events
    per user and activity), repo -> user "received", PR / issue -> its repository; every relation
    also runs reversed with its own weights;
  * every edge's age (days before its cutoff; for PR / issue -> repo, since first seen) goes through
    a learned cos(w log(1 + age) + b) encoding (--het-time-dim), part of the message and of the
    attention score; messages are softmax-attention weighted per relation and node.
--het-time none feeds zeros instead (same architecture): the ablation of the time embedding.
Results go to outputs/<task>_het[_notime]/.

--graph-structure chain replaces that graph by the relational entity graph of experiments/relgt:
every row a node, a user linked to its latest visible row of each table, a row to the previous and
next one of the same table, and another user's row bringing that user in. The graph and the tokens
are relgt's own (Graph, tokenize). Each example (eligible user, week) is its K tokens and the edges
among them; a token is its row features, table, hop and time before the week, RowGNN passes
relation-typed messages over those edges and reads out the seed token(s), which also hold the 60
history features. Weeks, labels, --actions, --users, thresholds and output are the same as above.
--layers should reach the tokens: a message travels one hop per layer.

--time-enc edge (user-repo graph): the days since the last and the first event of each user-repo
edge (within the 90 days before the cutoff) go through relgt's sinusoidal PositionalEncoding and a
linear layer shared by the edge types (--time-dim each), and are appended to the edge features of
every message. Without it time is only seen through the 7 / 30 / 90-day counts and the linear
days-since-last / 90.

--history-days N: the model only sees rows of the last N days before each table's cutoff, in every
week (training, validation, final fit and test). The user features are recomputed from those rows
(the 30d / 90d / all-time counts, then copies of the shorter ones, are dropped; open PRs / issues
count only those opened in the window), the user-repo edges and repo features cover N days
(edge counts over 7 days and N days), with --graph-structure chain it sets --lookback-days N and with
ego --ego-lookback N.
Eligibility and labels are unchanged, so the run is scored on the same users as the others.

Run from the repo root:
    python experiments/gnn/run.py                       # ego, all 6 activities -> outputs/all_ego/
    python experiments/gnn/run.py --user-features       # -> outputs/all_ego_uf/
    python experiments/gnn/run.py --user-features --segment activity   # -> outputs/all_ego_uf_seg/
    python experiments/gnn/run.py --graph-structure user-repo
    python experiments/gnn/run.py --graph-structure user-repo --history-days 14
    python experiments/gnn/run.py --graph-structure user-repo --time-enc edge
    python experiments/gnn/run.py --graph-structure user-repo --actions push
    python experiments/gnn/run.py --graph-structure hetero --holdout              # -> outputs/all_het_ho/
    python experiments/gnn/run.py --graph-structure hetero --het-time none --holdout
    python experiments/gnn/run.py --graph-structure chain --chain-seed latest --num-seeds 16 --exclude-push
"""
import argparse
import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # after numpy: importing torch first trips a duplicate-OpenMP error on this machine
from torch import nn
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("baseline_run", HERE.parent / "baseline" / "run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

EVENT_COLS, WEEK, DAY = base.EVENT_COLS, base.WEEK, base.DAY
EDGE_TYPES = EVENT_COLS + ["received"]
EDGE_WINDOWS = [7, 30, 90]
OUT_DIR = HERE / "outputs"
NONPUSH = [k for k in EVENT_COLS if k != "push"]
# --holdout: train on weeks up to 10-24, tune on 10-31, score HOLDOUT (labels to the end of the data,
# ~2 h short of the full week; push has no data after 10-27, so only the 5 non-push activities count).
HOLDOUT_TUNE = pd.Timestamp("2026-10-31 16:00")
HOLDOUT = pd.Timestamp("2026-11-07 16:00")
PROGRESS = {"disable": False}  # --no-progress turns the tqdm bars off


def windowed_tables(tables, ev, own, lags, data_end, n, days):
    """--history-days: recompute the user features of every week from rows of the last `days`
    days before each table's cutoff only. Eligibility, labels, cidx and week are kept from the full
    tables; count features over longer windows (30d / 90d / all), now copies of the shorter ones,
    are dropped."""
    out = {}
    for s, t in tables.items():
        ev_w = {k: e[e.t.values >= np.datetime64(s - lags[k] - days * DAY)] for k, e in ev.items()}
        start = np.datetime64(s - lags["open_pr"] - days * DAY)
        own_w = {k: d[d.avail.values >= start] for k, d in own.items()}  # PRs / issues opened in the window
        f = base.build_week(s, ev_w, own_w, lags, data_end, n, with_labels=False)
        feats = [c for c in f.columns if c not in {"eligible", "cidx", "week"}
                 and not any(c.endswith(f"_cnt_{w}") for w in ["30d", "90d", "all"])]
        t = t.drop(columns=[c for c in t.columns if c not in {"eligible", "cidx", "week"}
                            and not c.startswith("y_")])
        out[s] = pd.concat([f[feats], t], axis=1)
    return out


def week_graph(ev, s, lags, repo_is_org, keep, times=False, window=90):
    """User-repo edges of every type and the repo node features, from rows before each cutoff.

    Only users with keep[cohort index] are nodes; their ids are renumbered 0..keep.sum()-1.
    Edges are the rows of the last `window` days; they count events in the EDGE_WINDOWS shorter
    than it and in the whole window.
    With times, also returns per edge type the days since the last and the first event of the
    pair in the window, (n_edges, 2), for --time-enc edge.
    """
    new_id = np.cumsum(keep) - 1
    windows = [w for w in EDGE_WINDOWS if w < window] + [window]
    raw = {}
    for k in EDGE_TYPES:
        c = s - lags[k]
        e = ev[k]
        m = ((e.avail.values < np.datetime64(c)) & (e.t.values >= np.datetime64(c - window * DAY))
             & ~np.isnan(e.repo.values))
        m &= keep[e.uid.values.clip(0)] & (e.uid.values >= 0)
        u, r = new_id[e.uid.values[m]], e.repo.values[m].astype(np.int64)
        age = (np.datetime64(c) - e.t.values[m]) / np.timedelta64(1, "D")
        pair, inv = np.unique(u * len(repo_is_org) + r, return_inverse=True)
        cnt = np.stack([np.bincount(inv, weights=age < w, minlength=len(pair)) for w in windows], 1)
        by_pair = pd.Series(age).groupby(inv)
        last, first = by_pair.min().values, by_pair.max().values
        raw[k] = (pair // len(repo_is_org), pair % len(repo_is_org), cnt, last, first)

    repos = np.unique(np.concatenate([v[1] for v in raw.values()]))
    local = pd.Series(np.arange(len(repos)), index=repos)
    xr = [repo_is_org[repos].astype(np.float32)[:, None]]
    edges, ages = [], []
    for k in EDGE_TYPES:
        u, r, cnt, last, first = raw[k]
        ages.append(np.stack([last, first], 1).astype(np.float32).reshape(-1, 2))
        r = local.loc[r].values if len(r) else r
        # per repo: events of this type in the two longest windows (30 / 90 days by default) and
        # number of distinct cohort users
        xr.append(np.stack([np.bincount(r, weights=cnt[:, -2], minlength=len(repos)),
                            np.bincount(r, weights=cnt[:, -1], minlength=len(repos)),
                            np.bincount(r, minlength=len(repos))], 1))
        ef = np.concatenate([np.log1p(cnt), (last / window)[:, None]], 1).astype(np.float32)
        edges.append((u.astype(np.int64), r.astype(np.int64), ef))
    xr = np.concatenate([xr[0]] + [np.log1p(x) for x in xr[1:]], 1).astype(np.float32)
    return (xr, edges, ages) if times else (xr, edges)


def mean_agg(msg, idx, n):
    out = torch.zeros(n, msg.shape[1], device=msg.device).index_add_(0, idx, msg)
    deg = torch.zeros(n, device=msg.device).index_add_(0, idx, torch.ones_like(idx, dtype=msg.dtype))
    return out / deg.clamp(min=1)[:, None]


class EdgeTimeEncoder(nn.Module):
    """--time-enc edge: relgt's sinusoidal PositionalEncoding + a linear layer, shared by the edge
    types, applied to the days since the last and the first event of each user-repo edge."""

    def __init__(self, d):
        super().__init__()
        self.pos = load_relgt().PositionalEncoding(d)
        self.linear = nn.Linear(d, d)

    def forward(self, ages):
        return self.linear(self.pos(ages.reshape(-1))).view(len(ages), -1)


class HeteroSAGE(nn.Module):
    def __init__(self, du, dr, de, n_out, d=64, layers=2, dropout=0.1, time_dim=0):
        super().__init__()
        nt = len(EDGE_TYPES)
        self.time_enc = EdgeTimeEncoder(time_dim) if time_dim else None
        de += 2 * time_dim

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
        edges = g["edges"]
        if self.time_enc is not None:
            edges = [(u, r, torch.cat([e, self.time_enc(t)], 1)) for (u, r, e), t in zip(edges, g["edge_t"])]
        for layer in range(len(self.u2r)):
            mr = sum(mean_agg(torch.relu(lin(torch.cat([hu[u], e], 1))), r, nr)
                     for lin, (u, r, e) in zip(self.u2r[layer], edges))
            hr = self.norm_r[layer](hr + self.drop(torch.relu(self.upd_r[layer](torch.cat([hr, mr], 1)))))
            mu = sum(mean_agg(torch.relu(lin(torch.cat([hr[r], e], 1))), u, nu)
                     for lin, (u, r, e) in zip(self.r2u[layer], edges))
            hu = self.norm_u[layer](hu + self.drop(torch.relu(self.upd_u[layer](torch.cat([hu, mu], 1)))))
        return self.head(hu)


def masked_bce(logits, y, mask):
    """Mean over tasks of the BCE over that task's labelled, eligible users."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
    m = mask[:, None] & ~torch.isnan(y)
    per_task = (loss * m).sum(0) / m.sum(0).clamp(min=1)
    return per_task[m.any(0)].mean()


def fit(graphs, weeks, make_model, epochs, seed, val_weeks=None, patience=5, lr=2e-3):
    """Full-graph training, one week per step (user-repo and hetero): make_model() builds the model,
    model(graph) gives the logits of the users. With val_weeks, early-stop on validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, best_epoch, bad = np.inf, None, epochs, 0
    bar = tqdm(range(1, epochs + 1), desc="epochs", unit="ep", dynamic_ncols=True, **PROGRESS)
    for epoch in bar:
        model.train()
        losses = []
        for w in rng.permutation(weeks):
            g = graphs[w]
            opt.zero_grad()
            loss = masked_bce(model(g), g["y"], g["mask"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        bar.set_postfix(train=f"{np.mean(losses):.4f}")
        if val_weeks is None:
            continue
        model.eval()
        with torch.no_grad():
            # weeks whose labels for this task are not known yet (e.g. push of the last week) are skipped
            vl = np.mean([masked_bce(model(graphs[w]), graphs[w]["y"], graphs[w]["mask"]).item()
                          for w in val_weeks if not torch.isnan(graphs[w]["y"]).all()])
        tqdm.write(f"  epoch {epoch:2d}  val loss {vl:.4f}")
        if vl < best - 1e-4:
            best, best_epoch, bad = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        bar.set_postfix(train=f"{np.mean(losses):.4f}", val=f"{vl:.4f}", best_ep=best_epoch)
        if bad >= patience:
            break
    bar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def predict(model, g):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(g)).cpu().numpy()


def train_batched(make_model, batch, ids, fit_weeks, epochs, val_weeks, lr, seed, batch_size, max_steps=0,
                  patience=5):
    """Mini-batch training over examples (ego and chain): batch(example ids) -> (model inputs, y). With
    val_weeks, early stop on the mean weekly validation loss; weeks whose labels are not known yet are
    skipped."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, best_epoch, bad = np.inf, None, epochs, 0
    examples = np.concatenate([ids[w] for w in fit_weeks])
    n_steps = -(-len(examples) // batch_size)
    if max_steps:
        n_steps = min(n_steps, max_steps)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(examples)
        bar = tqdm(range(n_steps), desc=f"epoch {epoch}/{epochs}", unit="batch", dynamic_ncols=True,
                   leave=False, **PROGRESS)
        run_loss = None
        for step in bar:
            i = step * batch_size
            inputs, yb = batch(perm[i:i + batch_size])
            opt.zero_grad()
            loss = masked_bce(model(*inputs), yb, torch.ones(len(yb), dtype=torch.bool, device=yb.device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:  # .item() syncs the GPU; do it only now and then
                lv = loss.item()
                run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
                bar.set_postfix(loss=f"{run_loss:.4f}")
        bar.close()
        if val_weeks is None:
            continue
        model.eval()
        with torch.no_grad():
            vl = []
            for w in val_weeks:
                if len(ids[w]) == 0:
                    continue
                lw, yw = batched_logits(model, batch, ids[w], batch_size)
                if not torch.isnan(yw).all():
                    vl.append(masked_bce(lw, yw, torch.ones(len(yw), dtype=torch.bool, device=yw.device)).item())
        vl = float(np.mean(vl))
        tqdm.write(f"  epoch {epoch:2d}  train loss {run_loss:.4f}  val loss {vl:.4f}")
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


def batched_logits(model, batch, examples, batch_size):
    """Logits and labels of the given examples, in chunks of batch_size."""
    out = [batch(examples[i:i + batch_size]) for i in range(0, len(examples), batch_size)]
    return torch.cat([model(*inputs) for inputs, _ in out]), torch.cat([y for _, y in out])


# ---------------------------------------------------------------------------------------------------
# --graph-structure ego (default): the relation-aware ego-graph GNN of
# old_experiments/relgt_inspired (gnn_model.py, model.py, data.py), on the current weeks and protocol.

EGO_OBJECTS_PER_REL = 4                                   # hop-1 objects kept per relation
EGO_OBJECT_SLOTS = len(EVENT_COLS) * EGO_OBJECTS_PER_REL  # 24
EGO_REPO_SLOTS = 24                                       # hop-2 repository membership tokens
EGO_K = EGO_OBJECT_SLOTS + EGO_REPO_SLOTS
EGO_REPO_REL = len(EVENT_COLS) + 1                        # relation 7; 1..6 are EVENT_COLS, 0 padding
EGO_N_REL = EGO_REPO_REL + 1
EGO_N_TYPES = 4                                           # padding, pull request, issue, repository
EGO_N_HOPS = 3                                            # padding, object, repository
EGO_N_FEATS = 4
SEGMENTS = ["quiet", "active"]                            # --segment activity: 0 = no ego token, 1 = some


def load_entity_events(cohort):
    """Activity rows of cohort users with the entity they act on, for the ego tokens.

    Per EVENT_COLS key: uid (cohort index), t, avail (as in base.load), repo, entity (global id:
    1 + PRs | issues | repositories, the old EntityVocabulary) and entity_avail (first_seen_at of that
    PR / issue / repository). Node type 1 = PR, 2 = issue, 3 = repository.
    """
    db = base.DB
    users = pd.read_parquet(db / "Users.parquet", columns=["user_id", "is_cohort"])
    to_cidx = np.full(len(users), -1, np.int64)
    to_cidx[cohort.user_id.values] = np.arange(len(cohort))
    prs = pd.read_parquet(db / "PullRequests.parquet",
                          columns=["pr_id", "repo_id", "author_id", "created_at", "first_seen_at"])
    issues = pd.read_parquet(db / "Issues.parquet", columns=["issue_id", "repo_id", "author_id", "is_pull_request",
                                                             "created_at", "first_seen_at"])
    comments = pd.read_parquet(db / "IssueComments.parquet",
                               columns=["issue_id", "repo_id", "author_id", "created_at"])
    reviews = pd.read_parquet(db / "PRReviews.parquet",
                              columns=["pr_id", "repo_id", "reviewer_id", "submitted_at", "first_seen_at"])
    rcomments = pd.read_parquet(db / "PRReviewComments.parquet",
                                columns=["pr_id", "repo_id", "author_id", "created_at"])
    pushes = pd.read_parquet(db / "Pushes.parquet", columns=["user_id", "repo_id", "created_at"])
    repos = pd.read_parquet(db / "Repositories.parquet", columns=["repo_id", "first_seen_at"])
    offset = {1: 1, 2: 1 + len(prs), 3: 1 + len(prs) + len(issues)}
    seen = {1: prs.set_index("pr_id").first_seen_at, 2: issues.set_index("issue_id").first_seen_at,
            3: repos.set_index("repo_id").first_seen_at}
    n_entities = offset[3] + len(repos)

    def later(*cols):
        return np.maximum.reduce([np.asarray(c, dtype="datetime64[ns]") for c in cols])

    def events(uid, t, avail, repo, entity, node_type):
        d = pd.DataFrame({"uid": uid.values, "t": t.values, "avail": avail, "repo": repo.values,
                          "entity": entity.values})
        d = d.dropna()
        d = d[d.t >= base.DATA_START]
        d["uid"] = to_cidx[d.uid.astype(np.int64).values]
        d = d[d.uid >= 0]
        ent = d.entity.astype(np.int64).values
        d["entity_avail"] = seen[node_type].reindex(ent).values
        d["entity"] = ent + offset[node_type]
        d["repo"] = d.repo.astype(np.int64)
        d["node_type"] = node_type
        return d.dropna(subset=["entity_avail"]).reset_index(drop=True)

    plain = issues[~issues.is_pull_request.fillna(False)]
    ev = {
        "open_pr": events(prs.author_id, prs.created_at, later(prs.created_at, prs.first_seen_at), prs.repo_id,
                          prs.pr_id, 1),
        "open_issue": events(plain.author_id, plain.created_at, later(plain.created_at, plain.first_seen_at),
                             plain.repo_id, plain.issue_id, 2),
        "issue_comment": events(comments.author_id, comments.created_at, comments.created_at.values,
                                comments.repo_id, comments.issue_id, 2),
        "pr_review": events(reviews.reviewer_id, reviews.submitted_at,
                            later(reviews.submitted_at, reviews.first_seen_at), reviews.repo_id, reviews.pr_id, 1),
        "pr_review_comment": events(rcomments.author_id, rcomments.created_at, rcomments.created_at.values,
                                    rcomments.repo_id, rcomments.pr_id, 1),
        "push": events(pushes.user_id, pushes.created_at, pushes.created_at.values, pushes.repo_id,
                       pushes.repo_id, 3),
    }
    return ev, n_entities, offset[3]


def ego_tokens(ent_ev, s, lags, keep, lookback, repo_offset):
    """The ego tokens of every cohort user for the week starting at s (old data.py:snapshot_tokens).

    Activity k is visible through its own cutoff c_k = s - lags[k]: a row is used if avail < c_k,
    entity_avail < c_k and c_k - lookback <= t < c_k. Per user, slots 0..23 are the 4 latest objects of
    each relation (hop 1; a push object is its repository) and slots 24..47 the 24 latest repositories
    over all relations (hop 2, relation 7). Token features: log events, log active days, log events of
    the entity (or repo) by all kept cohort users in the window, and its distinct users / relations.
    relative_days is the days from the latest such event to its cutoff. Only users with keep are used.
    Returns a dict of (n_cohort, EGO_K[, 4]) arrays; users without visible rows are all padding. Also
    "n_events", (n_cohort,): the number of visible rows of each user in the window.
    """
    parts = []
    for rel, k in enumerate(EVENT_COLS, start=1):
        c = np.datetime64(s - lags[k])
        e = ent_ev[k]
        m = ((e.avail.values < c) & (e.entity_avail.values < c) & (e.t.values < c)
             & (e.t.values >= c - np.timedelta64(int(lookback * 86400), "s")))
        m &= keep[e.uid.values]
        d = e.loc[m, ["uid", "repo", "entity", "node_type", "t"]].copy()
        d["rel"] = rel
        d["age"] = (c - d.t.values) / np.timedelta64(1, "D")
        parts.append(d)
    known = pd.concat(parts, ignore_index=True)
    n = len(keep)
    out = {"entity": np.zeros((n, EGO_K), np.int64), "type": np.zeros((n, EGO_K), np.int64),
           "rel": np.zeros((n, EGO_K), np.int64), "hop": np.zeros((n, EGO_K), np.int64),
           "feat": np.zeros((n, EGO_K, EGO_N_FEATS), np.float32), "days": np.zeros((n, EGO_K), np.float32),
           "n_events": np.zeros(n, np.int64)}
    if known.empty:
        return out
    out["n_events"] = np.bincount(known.uid.values, minlength=n)
    known["day"] = known.t.values.astype("datetime64[D]")

    def neighbours(by, structure_key):
        g = (known.groupby(by, sort=False)
             .agg(count=("t", "size"), active=("day", "nunique"), age=("age", "min")).reset_index())
        st = (known.groupby(structure_key, sort=False)
              .agg(degree=("t", "size"), users=("uid", "nunique"), rels=("rel", "nunique")).reset_index())
        return g.merge(st, on=structure_key, validate="many_to_one")

    def place(g, slot, entity, node_type, rel, hop):
        f = np.stack([np.log1p(g["count"].values), np.log1p(g.active.values), np.log1p(g.degree.values),
                      g.users.values / np.maximum(g.rels.values, 1)], 1).astype(np.float32)
        u = g.uid.values
        out["entity"][u, slot], out["type"][u, slot] = entity, node_type
        out["rel"][u, slot], out["hop"][u, slot] = rel, hop
        out["feat"][u, slot], out["days"][u, slot] = f, np.clip(g.age.values, 0, 3650)

    obj = neighbours(["uid", "rel", "entity", "node_type"], "entity")
    obj = obj.sort_values(["uid", "rel", "age", "count"], ascending=[True, True, True, False], kind="stable")
    obj["rank"] = obj.groupby(["uid", "rel"], sort=False).cumcount()
    obj = obj[obj["rank"] < EGO_OBJECTS_PER_REL]
    place(obj, (obj.rel.values - 1) * EGO_OBJECTS_PER_REL + obj["rank"].values, obj.entity.values,
          obj.node_type.values, obj.rel.values, 1)

    rep = neighbours(["uid", "repo"], "repo")
    rep = rep.sort_values(["uid", "age", "count"], ascending=[True, True, False], kind="stable")
    rep["rank"] = rep.groupby("uid", sort=False).cumcount()
    rep = rep[rep["rank"] < EGO_REPO_SLOTS]
    place(rep, EGO_OBJECT_SLOTS + rep["rank"].values, rep.repo.values + repo_offset, 3, EGO_REPO_REL, 2)
    real = out["type"] != 0
    assert (out["days"][real] > 0).all(), "an ego token is not strictly before its cutoff"
    return out


class EgoTimeEncoder(nn.Module):
    """Sinusoidal features of the log age in days, then a linear map (old model.py TimeEncoder)."""

    def __init__(self, d, n_freq=8):
        super().__init__()
        self.register_buffer("freq", torch.logspace(-1, 1, n_freq))
        self.proj = nn.Linear(2 * n_freq + 1, d)

    def forward(self, days):
        age = torch.log1p(days.clamp(min=0)).unsqueeze(-1)
        a = age * self.freq
        return self.proj(torch.cat([age, a.sin(), a.cos()], -1))


class EgoTokenEncoder(nn.Module):
    """Entity id, node type, relation, hop, row features and time, concatenated and mixed by an MLP
    (old model.py TokenEncoder)."""

    def __init__(self, n_entities, d, dropout, token_time=True):
        super().__init__()
        self.entity = nn.Embedding(n_entities, d, padding_idx=0)
        self.node_type = nn.Embedding(EGO_N_TYPES, d)
        self.rel = nn.Embedding(EGO_N_REL, d)
        self.hop = nn.Embedding(EGO_N_HOPS, d)
        self.feat = nn.Sequential(nn.Linear(EGO_N_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.time = EgoTimeEncoder(d) if token_time else None  # None: time goes on the edges instead
        self.mix = nn.Sequential(nn.Linear((6 if token_time else 5) * d, d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, entity, types, rel, hop, feat, days):
        parts = [self.entity(entity), self.node_type(types), self.rel(rel), self.hop(hop), self.feat(feat)]
        if self.time is not None:
            parts.append(self.time(days))
        return self.mix(torch.cat(parts, -1))


class EgoLayer(nn.Module):
    """One round of relation-aware message passing on the star graph (old gnn_model.py RelationalLayer):
    token -> user with one weight per relation, user -> token, and push repository <-> the repository
    membership token of the same repo, each mean-aggregated and added to a self transform.

    With time_dim (--ego-time func / none), the token <-> user edges carry phi (B, N, time_dim; row 0,
    the user, is zeros) as in TemporalHeteroLayer: messages W_r [h_token, phi] and W [h_user, phi], and
    token -> user is attention-weighted per relation (a_r . LeakyReLU(Q h_user + m), softmax over the
    tokens of relation r), relations summed. user -> token has one incoming edge, so no attention."""

    def __init__(self, d, dropout, time_dim=0):
        super().__init__()
        self.d, self.time_dim = d, time_dim
        self.self_loop = nn.Linear(d, d)
        self.token_to_user = nn.Linear(d + time_dim, (EGO_N_REL - 1) * d, bias=False)
        self.user_to_token = nn.Linear(d + time_dim, d, bias=False)
        self.same_entity = nn.Linear(d, d, bias=False)
        if time_dim:
            self.att = nn.Parameter(torch.randn(EGO_N_REL - 1, d) / np.sqrt(d))
            self.q = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _mean(adj, msg):
        return adj @ msg / adj.sum(-1, keepdim=True).clamp(min=1.0)

    def forward(self, x, rel, to_user, to_token, same, phi=None):
        B, N, _ = x.shape
        src = x if phi is None else torch.cat([x, phi], -1)
        per_rel = self.token_to_user(src).view(B, N, EGO_N_REL - 1, self.d)
        picked = per_rel.gather(2, rel.view(B, N, 1, 1).expand(-1, -1, 1, self.d)).squeeze(2)
        if phi is None:
            upd = (self.self_loop(x) + self._mean(to_user, picked) + self._mean(to_token, self.user_to_token(x))
                   + self._mean(same, self.same_entity(x)))
            return self.norm(x + self.drop(torch.relu(upd)))
        valid = to_user[:, 0] > 0                                         # (B, N): the user's in-edges
        score = (self.att[rel] * nn.functional.leaky_relu(self.q(x[:, :1]) + picked, 0.2)).sum(-1)
        group = nn.functional.one_hot(rel, EGO_N_REL - 1).bool() & valid.unsqueeze(-1)  # (B, N, R)
        mx = score.detach().unsqueeze(-1).masked_fill(~group, -torch.inf).amax(1)       # (B, R)
        mx = mx.masked_fill(torch.isinf(mx), 0.0)
        ex = (score - mx.gather(1, rel)).exp() * valid.to(x.dtype)
        den = (ex.unsqueeze(-1) * group.to(x.dtype)).sum(1)                             # (B, R)
        a = ex / den.gather(1, rel).clamp(min=1e-16)
        to_u = torch.cat([(a.unsqueeze(-1) * picked).sum(1, keepdim=True), x.new_zeros(B, N - 1, self.d)], 1)
        to_t = self.user_to_token(torch.cat([x[:, :1].expand(-1, N, -1), phi], -1)) * to_token[:, :, :1]
        upd = self.self_loop(x) + to_u + to_t + self._mean(same, self.same_entity(x))
        return self.norm(x + self.drop(torch.relu(upd)))


class EgoGNN(nn.Module):
    """Old gnn_model.py RelationalGNN: a learned user node (plus, with n_seed_feats, an MLP of the
    baseline user features) joined to its ego tokens, `layers` rounds of EgoLayer, then a head on the
    user node and the mean of all nodes."""

    def __init__(self, n_entities, n_out, d=64, layers=2, dropout=0.1, n_seed_feats=0, time="sin", time_dim=16):
        super().__init__()
        self.edge_time = time != "sin"   # func / none: phi on the token <-> user edges (none: phi = 0)
        self.tokens = EgoTokenEncoder(n_entities, d, dropout, token_time=not self.edge_time)
        self.time = FuncTimeEncoder(time_dim) if time == "func" else None
        self.time_dim = time_dim if self.edge_time else 0
        self.user_token = nn.Parameter(torch.randn(d) / np.sqrt(d))
        self.seed_encoder = (nn.Sequential(nn.Linear(n_seed_feats, d), nn.ReLU(), nn.Linear(d, d))
                             if n_seed_feats else None)
        self.layers = nn.ModuleList(EgoLayer(d, dropout, self.time_dim) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, entity, types, rel, hop, feat, days, seed_x=None):
        tok = self.tokens(entity, types, rel, hop, feat, days)
        B, K, _ = tok.shape
        user = self.user_token.expand(B, 1, -1)
        if self.seed_encoder is not None:
            user = user + self.seed_encoder(seed_x).unsqueeze(1)
        x = torch.cat([user, tok], 1)
        valid = types != 0
        # Node 0 is the user; padding tokens use relation 1's weights but have no edges.
        rels = torch.cat([rel.new_ones(B, 1), rel.clamp(min=1)], 1) - 1
        vf = valid.to(x.dtype)
        to_user = torch.zeros(B, K + 1, K + 1, dtype=x.dtype, device=x.device)
        to_user[:, 0, 1:] = vf
        to_token = torch.zeros_like(to_user)
        to_token[:, 1:, 0] = vf
        is_repo = types == 3
        link = ((entity.unsqueeze(2) == entity.unsqueeze(1)) & is_repo.unsqueeze(2) & is_repo.unsqueeze(1)
                & (hop.unsqueeze(2) != hop.unsqueeze(1)))
        same = torch.zeros_like(to_user)
        same[:, 1:, 1:] = link.to(x.dtype)
        phi = None
        if self.edge_time:
            phi = x.new_zeros(B, K + 1, self.time_dim)
            if self.time is not None:
                phi[:, 1:] = self.time(days) * vf.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, rels, to_user, to_token, same, phi)
        mask = torch.cat([valid.new_ones(B, 1), valid], 1).unsqueeze(-1).to(x.dtype)
        mean = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(torch.cat([x[:, 0], mean], -1))


def ego_stages(args, tables, feats, weeks, train_weeks, cohort, lags, test, labels, device, sel):
    """--graph-structure ego: the ego tokens of every example, and the functions that train on weeks,
    predict one week (one value per row of its table; NaN for users without an example) and give the
    segment of every row of a week (-1 no example, 0 quiet = no ego token, 1 active)."""
    all_weeks = weeks + [base.TEST_START]
    ent_ev, n_entities, repo_offset = load_entity_events(cohort)

    # Examples: the eligible users of every labelled week, then the test users, as in token_stages.
    rows = {w: np.flatnonzero(tables[w].eligible.values) for w in weeks}
    tt = tables[base.TEST_START]
    rows[base.TEST_START] = pd.Series(np.arange(len(tt)), index=cohort.github_user_id.values[tt.cidx.values]
                                      ).loc[test.UserID].values
    parts = {key: [] for key in ["entity", "type", "rel", "hop", "feat", "days"]}
    n_events = []
    for w in all_weeks:
        tok = ego_tokens(ent_ev, w, lags, sel[w], args.ego_lookback, repo_offset)
        c = tables[w].cidx.values[rows[w]]
        for key in parts:
            parts[key].append(tok[key][c])
        n_events.append(tok["n_events"][c])
    n_events = np.concatenate(n_events)
    label_cols = [f"y_{k}" for k in labels]
    y = np.concatenate([tables[w][label_cols].values[rows[w]] if w != base.TEST_START
                        else np.full((len(rows[w]), len(labels)), np.nan) for w in all_weeks])
    sizes = np.cumsum([0] + [len(rows[w]) for w in all_weeks])
    ids = {w: np.arange(sizes[i], sizes[i + 1]) for i, w in enumerate(all_weeks)}
    T = {key: torch.tensor(np.concatenate(v), device=device) for key, v in parts.items()}
    Y = torch.tensor(y, dtype=torch.float32, device=device)
    seed_x = None
    split = args.segment != "none"
    if args.user_features or split:  # the quiet model of --segment activity always uses them
        sx = np.log1p(np.concatenate([tables[w][feats].values[rows[w]] for w in all_weeks]))
        train_ids = np.concatenate([ids[w] for w in train_weeks])
        sx = (sx - sx[train_ids].mean(0)) / (sx[train_ids].std(0) + 1e-6)
        seed_x = torch.tensor(sx, dtype=torch.float32, device=device)
    real = T["type"] != 0
    seg = (n_events >= args.active_min_events).astype(np.int64)
    seg_rows = {}
    for w in all_weeks:
        seg_rows[w] = np.full(len(tables[w]), -1, np.int64)
        seg_rows[w][rows[w]] = seg[ids[w]]
    print(f"{sizes[-1]:,} examples x {EGO_K} ego tokens (last {args.ego_lookback:g} days), "
          f"{real.float().mean().item():.0%} of slots used, {(~real.any(1)).float().mean().item():.1%} "
          f"of examples without tokens, {n_entities:,} entities"
          + (f", {seed_x.shape[1]} user features" if seed_x is not None else ""))
    print(f"segments (>= {args.active_min_events} visible events = active): quiet {(seg == 0).mean():.1%}, "
          f"active {(seg == 1).mean():.1%} of examples")

    def batch(ex):
        i = torch.as_tensor(ex, device=device)
        inputs = [T[key][i] for key in ["entity", "type", "rel", "hop", "feat", "days"]]
        if seed_x is not None:
            inputs.append(seed_x[i])
        return inputs, Y[i]

    n_sx = 0 if seed_x is None else seed_x.shape[1]
    seg_feats = {"quiet": n_sx, "active": n_sx if args.user_features else 0}

    def make_model(n_seed_feats):
        return EgoGNN(n_entities, len(labels), d=args.ego_dim, layers=args.ego_layers, dropout=args.ego_dropout,
                      n_seed_feats=n_seed_feats, time=args.ego_time, time_dim=args.ego_time_dim).to(device)

    def train(fit_weeks, epochs, val_weeks=None):
        """One model, or with --segment activity a {segment: model} dict and a {segment: best epoch}
        dict; epochs may then be such a dict too (the final retrain)."""
        if not split:
            return train_batched(lambda: make_model(n_sx), batch, ids, fit_weeks, epochs, val_weeks, args.lr,
                                 args.seed, args.batch_size, args.max_steps_per_epoch)
        models, best = {}, {}
        for s, name in enumerate(SEGMENTS):
            sub = {w: ids[w][seg[ids[w]] == s] for w in ids}
            print(f"segment {name}: {sum(len(sub[w]) for w in fit_weeks):,} examples, "
                  f"{seg_feats[name]} user features")
            models[name], best[name] = train_batched(
                lambda n=seg_feats[name]: make_model(n), batch, sub, fit_weeks,
                epochs[name] if isinstance(epochs, dict) else epochs, val_weeks, args.lr, args.seed,
                args.batch_size, args.max_steps_per_epoch)
        return models, best

    def probs(model, ex):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(batched_logits(model, batch, ex, args.batch_size)[0]).cpu().numpy()

    def predict_week(model, w):
        if isinstance(model, dict):
            p = np.zeros((len(ids[w]), len(labels)))
            for s, name in enumerate(SEGMENTS):
                m = seg[ids[w]] == s
                if m.any():
                    p[m] = probs(model[name], ids[w][m])
        else:
            p = probs(model, ids[w])
        out = np.full((len(tables[w]), len(labels)), np.nan)
        out[rows[w]] = p
        return out

    return train, predict_week, seg_rows.__getitem__


# ---------------------------------------------------------------------------------------------------
# --graph-structure hetero: users, PRs, issues and repositories, one edge per event, with a functional
# time encoding of its age and time-aware attention (TGAT-style), one full graph per week.

HET_TYPES = ["user", "pr", "issue", "repo"]
HET_OBJ_TYPE = {"open_pr": 1, "open_issue": 2, "issue_comment": 2, "pr_review": 1, "pr_review_comment": 1,
                "push": 3}
# (name, source type, destination type, time-encoder family); every relation also runs reversed.
HET_RELS = ([(k, 0, HET_OBJ_TYPE[k], 0) for k in EVENT_COLS]
            + [("received", 3, 0, 1), ("pr_in", 1, 3, 2), ("issue_in", 2, 3, 2)])
HET_N_FAMILIES = 3  # user events, received events, object -> repo membership


def hetero_graph(ent_ev, ev, s, lags, keep, window, max_events, repo_is_org, max_seen=None):
    """Nodes, edges and edge ages of the --graph-structure hetero graph of week s.

    Activity k is visible through its own cutoff c_k = s - lags[k]: a row is used if avail < c_k,
    entity_avail < c_k and c_k - window <= t < c_k (as ego_tokens). Edges, one per event:
      * user -> object of activity k (PR, issue, or the repository of a push), the latest max_events
        per user and activity; age = c_k - t;
      * repo -> user "received" (rows of ev["received"], cutoff of "received"), latest max_events per
        user; age = c - t;
      * PR / issue -> its repository; age = days since the PR / issue was first seen.
    Nodes: the kept users (renumbered 0..keep.sum()-1 like week_graph), and the PRs, issues and
    repositories reached by an edge. Object features count all visible events (before the cap):
    PR / issue: log events per activity, log distinct cohort users, log days since first seen;
    repo: log events per activity and received, log distinct cohort users, is it an organisation's.
    With max_seen (--history-days), the days since first seen are clipped to it, so a PR / issue
    older than the window reads as "at least max_seen days old" and nothing before it is visible.
    Returns {"x": [None, x_pr, x_issue, x_repo], "ent": [None, pr ids, issue ids, repo ids],
    "edges": [(src, dst, age) per HET_RELS]}; PR / issue ids are load_entity_events' entity ids.
    """
    new_id = np.cumsum(keep) - 1
    win = np.timedelta64(int(window * 86400), "s")
    one_day = np.timedelta64(1, "D")

    def latest(d):
        d = d.sort_values(["uid", "age"], kind="stable")
        return d[d.groupby("uid", sort=False).cumcount().values < max_events]

    vis = []
    for rel, k in enumerate(EVENT_COLS):
        c = np.datetime64(s - lags[k])
        e = ent_ev[k]
        m = ((e.avail.values < c) & (e.entity_avail.values < c) & (e.t.values < c) & (e.t.values >= c - win)
             & keep[e.uid.values])
        vis.append(pd.DataFrame({"uid": e.uid.values[m], "entity": e.entity.values[m], "repo": e.repo.values[m],
                                 "age": (c - e.t.values[m]) / one_day,
                                 "seen": (c - e.entity_avail.values[m]) / one_day, "rel": rel,
                                 "otype": HET_OBJ_TYPE[k]}))
    allv = pd.concat(vis, ignore_index=True)
    used = pd.concat([latest(d) for d in vis], ignore_index=True)

    c = np.datetime64(s - lags["received"])
    e = ev["received"]
    m = ((e.avail.values < c) & (e.t.values < c) & (e.t.values >= c - win) & ~np.isnan(e.repo.values)
         & (e.uid.values >= 0))
    m &= keep[e.uid.values.clip(0)]
    rec_all = pd.DataFrame({"uid": e.uid.values[m], "repo": e.repo.values[m].astype(np.int64),
                            "age": (c - e.t.values[m]) / one_day})
    rec = latest(rec_all)

    ids = [None, np.unique(used.entity.values[used.otype.values == 1]),
           np.unique(used.entity.values[used.otype.values == 2]),
           np.unique(np.concatenate([used.repo.values, rec.repo.values]).astype(np.int64))]
    x = [None]
    obj_repo, obj_seen = [None], [None]
    nk = len(EVENT_COLS)
    for t in (1, 2):
        a = allv[(allv.otype.values == t) & np.isin(allv.entity.values, ids[t])]
        idx, n_t = np.searchsorted(ids[t], a.entity.values), len(ids[t])
        cnt = np.bincount(idx * nk + a.rel.values, minlength=n_t * nk).reshape(n_t, nk)
        by = pd.DataFrame({"idx": idx, "uid": a.uid.values, "seen": a.seen.values, "repo": a.repo.values}
                          ).groupby("idx")
        users = by.uid.nunique().reindex(range(n_t), fill_value=0).values
        seen = by.seen.max().reindex(range(n_t)).values
        if max_seen is not None:
            seen = np.minimum(seen, max_seen)
        obj_repo.append(np.searchsorted(ids[3], by.repo.first().reindex(range(n_t)).values.astype(np.int64)))
        obj_seen.append(seen)
        x.append(np.concatenate([np.log1p(cnt), np.log1p(users)[:, None], np.log1p(seen)[:, None]], 1))
    a = allv[np.isin(allv.repo.values, ids[3])]
    r = rec_all[np.isin(rec_all.repo.values, ids[3])]
    n_r = len(ids[3])
    idx = np.concatenate([np.searchsorted(ids[3], a.repo.values), np.searchsorted(ids[3], r.repo.values)])
    col = np.concatenate([a.rel.values, np.full(len(r), nk)])
    cnt = np.bincount(idx * (nk + 1) + col, minlength=n_r * (nk + 1)).reshape(n_r, nk + 1)
    users = (pd.Series(np.concatenate([a.uid.values, r.uid.values])).groupby(idx).nunique()
             .reindex(range(n_r), fill_value=0).values)
    x.append(np.concatenate([np.log1p(cnt), np.log1p(users)[:, None], repo_is_org[ids[3]][:, None]], 1))
    x = [None] + [v.astype(np.float32) for v in x[1:]]

    edges = []
    for rel, k in enumerate(EVENT_COLS):
        d = used[used.rel.values == rel]
        t = HET_OBJ_TYPE[k]
        dst = np.searchsorted(ids[t], d.repo.values if t == 3 else d.entity.values)
        edges.append((new_id[d.uid.values], dst, d.age.values))
    edges.append((np.searchsorted(ids[3], rec.repo.values), new_id[rec.uid.values], rec.age.values))
    for t in (1, 2):
        edges.append((np.arange(len(ids[t])), obj_repo[t], obj_seen[t]))
    edges = [(src.astype(np.int64), dst.astype(np.int64), age.astype(np.float32)) for src, dst, age in edges]
    assert all((age > 0).all() for _, _, age in edges), "a hetero edge is not strictly before its cutoff"
    return {"x": x, "ent": ids, "edges": edges}


class FuncTimeEncoder(nn.Module):
    """Learned functional time encoding cos(w log(1 + days) + b) (TGAT's Bochner encoding, on the log
    age); w starts at log-spaced frequencies as EgoTimeEncoder."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.logspace(-1, 1, d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, days):
        return torch.cos(torch.log1p(days.clamp(min=0)).unsqueeze(-1) * self.w + self.b)


def scatter_softmax(score, idx, n):
    """Softmax of score over the entries sharing the same idx (torch_geometric.utils.softmax)."""
    mx = torch.full((n,), -torch.inf, device=score.device, dtype=score.dtype)
    mx = mx.scatter_reduce(0, idx, score, "amax", include_self=True)
    ex = (score - mx[idx]).exp()
    den = torch.zeros(n, device=score.device, dtype=score.dtype).index_add(0, idx, ex)
    return ex / den[idx].clamp(min=1e-16)


class TemporalHeteroLayer(nn.Module):
    """One round over every directed relation r (src -> dst): message m = W_r [h_src, phi(age)],
    attention a_r . LeakyReLU(Q_dst h_dst + m) softmaxed over the incoming edges of r of each node; the
    relations are summed per node type, then h = LN(h + drop(relu(U [h, agg]))) per type."""

    def __init__(self, d, time_dim, n_rel, dropout):
        super().__init__()
        nt = len(HET_TYPES)
        self.msg = nn.ModuleList(nn.Linear(d + time_dim, d) for _ in range(n_rel))
        self.att = nn.Parameter(torch.randn(n_rel, d) / np.sqrt(d))
        self.q = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(nt))
        self.upd = nn.ModuleList(nn.Linear(2 * d, d) for _ in range(nt))
        self.norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(nt))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edges):
        q = [lin(x) for lin, x in zip(self.q, h)]
        agg = [torch.zeros_like(x) for x in h]
        for r, (st, dt, src, dst, phi) in enumerate(edges):
            if len(src) == 0:
                continue
            m = self.msg[r](torch.cat([h[st][src], phi], 1))
            score = (self.att[r] * nn.functional.leaky_relu(q[dt][dst] + m, 0.2)).sum(-1)
            a = scatter_softmax(score, dst, len(h[dt]))
            agg[dt] = agg[dt].index_add(0, dst, a[:, None] * m)
        return [norm(x + self.drop(torch.relu(upd(torch.cat([x, ag], 1)))))
                for norm, upd, x, ag in zip(self.norm, self.upd, h, agg)]


class TemporalHeteroGNN(nn.Module):
    """--graph-structure hetero: an MLP encoder per node type (plus, with n_entities, a learned entity
    embedding of PRs / issues / repos), `layers` rounds of TemporalHeteroLayer over HET_RELS in both
    directions, then a head on the user nodes. use_time=False feeds zeros instead of the time encoding."""

    def __init__(self, dims, n_out, d=64, layers=2, dropout=0.1, time_dim=16, use_time=True, n_entities=0):
        super().__init__()

        def enc(i):
            return nn.Sequential(nn.Linear(i, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))

        self.enc = nn.ModuleList(enc(i) for i in dims)
        self.entity = nn.Embedding(n_entities, d) if n_entities else None
        self.time = nn.ModuleList(FuncTimeEncoder(time_dim) for _ in range(HET_N_FAMILIES))
        self.time_dim, self.use_time = time_dim, use_time
        self.layers = nn.ModuleList(TemporalHeteroLayer(d, time_dim, 2 * len(HET_RELS), dropout)
                                    for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, g):
        h = [enc(x) for enc, x in zip(self.enc, g["x"])]
        if self.entity is not None:
            h = [h[0]] + [x + self.entity(e) for x, e in zip(h[1:], g["ent"][1:])]
        edges = []
        for (_, st, dt, fam), (src, dst, age) in zip(HET_RELS, g["edges"]):
            phi = self.time[fam](age) if self.use_time else age.new_zeros(len(age), self.time_dim)
            edges += [(st, dt, src, dst, phi), (dt, st, dst, src, phi)]
        for layer in self.layers:
            h = layer(h, edges)
        return self.head(h[0])


def hetero_stages(args, tables, feats, train_weeks, cohort, ev, lags, sel, repo_is_org, labels, device):
    """--graph-structure hetero: the graph of every week (hetero_graph), and the functions that train on
    weeks (full-graph fit, as user-repo) and predict one week (one value per row of its table)."""
    ent_ev, n_entities, repo_offset = load_entity_events(cohort)
    window = args.history_days or 90
    raw = {w: hetero_graph(ent_ev, ev, w, lags, sel[w], window, args.het_max_events, repo_is_org,
                           max_seen=args.history_days)
           for w in tqdm(list(tables), desc="hetero graphs", unit="week", dynamic_ncols=True, **PROGRESS)}
    for w, t in tables.items():
        raw[w]["x"][0] = np.log1p(t[feats].values).astype(np.float32)
        raw[w]["ent"][3] = raw[w]["ent"][3] + repo_offset  # repo ids -> entity ids, for --het-entity-emb
    # Standardise every node type with statistics of the training weeks only.
    stats = []
    for i in range(len(HET_TYPES)):
        xt = np.concatenate([raw[w]["x"][i] for w in train_weeks])
        stats.append((xt.mean(0), xt.std(0) + 1e-6))

    graphs = {}
    for w, t in tables.items():
        g = raw[w]
        y = (t[[f"y_{k}" for k in labels]].values if w != base.TEST_START
             else np.full((len(t), len(labels)), np.nan))
        graphs[w] = {
            "x": [torch.tensor((x - mu) / sd, dtype=torch.float32, device=device) for x, (mu, sd) in zip(g["x"], stats)],
            "ent": [None] + [torch.tensor(e, device=device) for e in g["ent"][1:]],
            "edges": [tuple(torch.tensor(a, device=device) for a in e) for e in g["edges"]],
            "y": torch.tensor(y, dtype=torch.float32, device=device),
            "mask": torch.tensor(t.eligible.values, device=device),
        }
    gt = raw[base.TEST_START]
    print(f"hetero graph of the test week ({window:g}-day window, <= {args.het_max_events} events per user "
          f"and relation): " + ", ".join(f"{len(x):,} {name}" for name, x in zip(HET_TYPES, gt["x"])) + "; edges "
          + ", ".join(f"{r[0]} {len(e[0]):,}" for r, e in zip(HET_RELS, gt["edges"])))
    print(f"edges per week: mean {np.mean([sum(len(e[0]) for e in g['edges']) for g in raw.values()]):,.0f}"
          + (f", GPU memory {torch.cuda.memory_allocated() / 2**30:.2f} GiB" if device == "cuda" else ""))
    dims = [x.shape[1] for x in gt["x"]]

    def make_model():
        return TemporalHeteroGNN(dims, len(labels), d=args.het_dim, layers=args.het_layers, dropout=args.het_dropout,
                                 time_dim=args.het_time_dim, use_time=args.het_time == "func",
                                 n_entities=n_entities if args.het_entity_emb else 0).to(device)

    def train(fit_weeks, epochs, val_weeks=None):
        return fit(graphs, fit_weeks, make_model, epochs, args.seed, val_weeks=val_weeks, lr=args.lr)

    def predict_week(model, w):
        return predict(model, graphs[w])

    return train, predict_week


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


@functools.cache
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

    def make_model():
        return RowGNN(relgt.RowEncoder(G, cutoffs, week_day, args.channels, 0.1), relgt.TimeEncoder(args.channels),
                      len(relgt.TABLES) + 1, args.max_hops + 3, relgt.N_ALL_REL, seed_x.shape[1], len(labels),
                      d=args.channels, layers=args.layers).to(device)

    def train(fit_weeks, epochs, val_weeks=None):
        return train_batched(make_model, S.batch, ids, fit_weeks, epochs, val_weeks, args.lr, args.seed,
                             args.batch_size, args.max_steps_per_epoch)

    def predict_week(model, w):
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(batched_logits(model, S.batch, ids[w], args.batch_size)[0]).cpu().numpy()
        out = np.full((len(tables[w]), len(labels)), np.nan)
        out[rows[w]] = p
        return out

    return train, predict_week


def main():
    global HOLDOUT, HOLDOUT_TUNE
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
                    help="AdamW learning rate (default: ego 1e-3, user-repo 2e-3, chain 3e-4)")
    ap.add_argument("--epochs", type=int, default=None, help="default: ego / chain 20, user-repo 40")
    ap.add_argument("--batch-size", type=int, default=None, help="ego / chain examples per batch "
                    "(default: ego 256, chain 512)")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="ego / chain; 0 = full epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", action="store_true",
                    help="train on weeks up to 10-24, tune (early stopping, decision rule) on 10-31 and "
                         "score the held-out week 11-07 on the 5 non-push activities; no final model / "
                         "output.csv. Overrides --val-start / --val-end; run name gets _ho")
    ap.add_argument("--holdout-week", default=None, metavar="UTC",
                    help="with --holdout: score this week instead of 11-07 (UTC start, Saturday 16:00, e.g. "
                         "'2026-09-26 16:00'); the tune week is the week before. A week that is fully "
                         "labelled (push included) is scored on all the activities; run name gets _ho<MMDD>")
    ap.add_argument("--no-progress", action="store_true", help="turn off the tqdm progress bars")
    ap.add_argument("--time-enc", choices=["none", "edge"], default="none",
                    help="user-repo graph: edge = add a learned sinusoidal encoding of the days since the "
                         "last and the first event of each edge to its features")
    ap.add_argument("--time-dim", type=int, default=16, help="size of each time encoding (--time-enc edge)")
    ap.add_argument("--history-days", type=int, default=None,
                    help="the model only sees rows of the last N days before each cutoff (user features, "
                         "graph edges and repo features; with --graph-structure chain, the tokens), in "
                         "training, validation and test alike (default: all, edges 90 days)")
    ap.add_argument("--graph-structure", choices=["ego", "user-repo", "hetero", "chain"], default="ego",
                    help="ego: one star graph per example, the user and its recent PRs / issues / repos "
                         "(old_experiments/relgt_inspired GNN). user-repo: one user <-> repository graph "
                         "per week (HeteroSAGE). hetero: one user / PR / issue / repo graph per week, one "
                         "edge per event with a time encoding and time-aware attention. chain: the row "
                         "graph of experiments/relgt, one token subgraph per example")
    hg = ap.add_argument_group("--graph-structure hetero")
    hg.add_argument("--het-dim", type=int, default=64)
    hg.add_argument("--het-layers", type=int, default=2, help="message passing rounds")
    hg.add_argument("--het-dropout", type=float, default=0.1)
    hg.add_argument("--het-max-events", type=int, default=32,
                    help="latest event edges kept per user and relation (default 32)")
    hg.add_argument("--het-time", choices=["func", "none"], default="func",
                    help="func: learned cos(w log(1 + age) + b) encoding of every edge's age; none: zeros "
                         "(ablation, same architecture)")
    hg.add_argument("--het-time-dim", type=int, default=16)
    hg.add_argument("--het-entity-emb", action="store_true",
                    help="add a learned embedding of each PR / issue / repo id to its node")
    eg = ap.add_argument_group("--graph-structure ego")
    eg.add_argument("--ego-lookback", type=float, default=14,
                    help="only activity rows of the last N days are tokens (default 14, as the old model)")
    eg.add_argument("--user-features", action="store_true",
                    help="add an MLP of the baseline user features to the learned user node")
    eg.add_argument("--ego-time", choices=["sin", "func", "none"], default="sin",
                    help="sin: sinusoidal log-age encoding inside each token (the original); func: the "
                         "hetero model's learned cos(w log(1+age)+b) on the token <-> user edges, in the "
                         "messages and the attention; none: the func architecture with zeros (ablation)")
    eg.add_argument("--ego-time-dim", type=int, default=16, help="size of phi (--ego-time func)")
    eg.add_argument("--ego-dim", type=int, default=64)
    eg.add_argument("--ego-layers", type=int, default=2, help="message passing rounds")
    eg.add_argument("--ego-dropout", type=float, default=0.1)
    eg.add_argument("--segment", choices=["none", "activity"], default="none",
                    help="activity: one model for quiet examples (no ego token) and one for active ones")
    eg.add_argument("--active-min-events", type=int, default=1,
                    help="an example is active with >= N visible activity rows in the ego window "
                         "(default 1 = any token; 5 = the old experiment's selection)")
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
    args = ap.parse_args()
    chain, ego = args.graph_structure == "chain", args.graph_structure == "ego"
    het = args.graph_structure == "hetero"
    args.lr = args.lr or {"ego": 1e-3, "user-repo": 2e-3, "hetero": 2e-3, "chain": 3e-4}[args.graph_structure]
    args.epochs = args.epochs or (40 if args.graph_structure in ("user-repo", "hetero") else 20)
    args.batch_size = args.batch_size or (256 if ego else 512)
    for group, needed in [(g, "chain"), (eg, "ego"), (hg, "hetero")]:
        changed = [a.option_strings[0] for a in group._group_actions if ap.get_default(a.dest) != getattr(args, a.dest)]
        if args.graph_structure != needed and changed:
            ap.error(f"these options need --graph-structure {needed}: " + ", ".join(changed))
    if args.graph_structure != "user-repo" and args.time_enc != "none":
        ap.error("--time-enc is for the user-repo graph (the ego and chain tokens already encode their time)")
    if args.time_enc != "none" and (args.time_dim < 2 or args.time_dim % 2):
        ap.error("--time-dim must be an even number >= 2")
    if het and (args.het_max_events < 1 or args.het_time_dim < 1 or args.het_layers < 1):
        ap.error("--het-max-events, --het-time-dim and --het-layers must be >= 1")
    time_dim = args.time_dim if args.time_enc == "edge" else 0
    hist = args.history_days
    if hist is not None and not 8 <= hist <= 90:
        ap.error("--history-days must be in 8..90")
    if hist and chain:
        if args.lookback_days not in (None, hist):
            ap.error("--history-days and --lookback-days disagree")
        args.lookback_days = float(hist)
    if hist and ego:
        if args.ego_lookback not in (ap.get_default("ego_lookback"), hist):
            ap.error("--history-days and --ego-lookback disagree")
        args.ego_lookback = float(hist)
    if ego and args.active_min_events < 1:
        ap.error("--active-min-events must be >= 1")
    if ego and args.ego_lookback <= 0:
        ap.error("--ego-lookback must be positive")
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
    default_name = ((args.task if args.users == "all" else f"{args.task}_{args.users}") + ("_chain" if chain else "")
                    + ("_ego" if ego else "") + ("_uf" if ego and args.user_features else "")
                    + ({"func": "_ftime", "none": "_notime"}.get(args.ego_time, "") if ego else "")
                    + ("_het" if het else "") + ("_notime" if het and args.het_time == "none" else "")
                    + (("_seg" + (str(args.active_min_events) if args.active_min_events != 1 else ""))
                       if ego and args.segment != "none" else "")
                    + (f"_lb{args.ego_lookback:g}" if ego and not hist and args.ego_lookback != 14 else "")
                    + ("_time" if time_dim else "") + (f"_h{hist}" if hist else "")
                    + (("_ho" + (args.holdout_week[5:10].replace("-", "") if args.holdout_week else ""))
                       if args.holdout else ""))
    PROGRESS["disable"] = args.no_progress
    if args.holdout_week and not args.holdout:
        ap.error("--holdout-week needs --holdout")
    if args.holdout:
        if args.holdout_week:
            HOLDOUT = pd.Timestamp(args.holdout_week)
            HOLDOUT_TUNE = HOLDOUT - WEEK
            if HOLDOUT.dayofweek != 5 or HOLDOUT.hour != 16:
                ap.error("--holdout-week must be a Saturday 16:00 UTC, like every week start")
            if not args.end:
                args.end = str(HOLDOUT)  # weeks that end after the holdout week start are not built as weeks
        if args.val_start or args.val_end:
            ap.error("--holdout sets the validation week itself (the week before the holdout); "
                     "drop --val-start / --val-end")
        args.val_start, args.val_end = str(HOLDOUT_TUNE), str(HOLDOUT_TUNE + WEEK)
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
    if args.holdout:
        assert val_weeks == [HOLDOUT_TUNE] and train_weeks[-1] + WEEK <= HOLDOUT_TUNE, (train_weeks, val_weeks)
        full = HOLDOUT + WEEK <= min(data_end.values())  # every activity is labelled for the whole week
        assert HOLDOUT not in weeks and (args.holdout_week or not full), \
            "the holdout week is now fully labelled: move HOLDOUT_TUNE / HOLDOUT forward"
        if full:
            tables[HOLDOUT] = base.build_week(HOLDOUT, ev, own, lags, data_end, n)
        else:
            # Labels to the end of the data (only data_end of the labels is lifted; features use the lags).
            tables[HOLDOUT] = base.build_week(HOLDOUT, ev, own, lags, {k: pd.Timestamp.max for k in data_end}, n)
            tables[HOLDOUT]["y_push"] = np.nan  # no push data after 10-27
        weeks = weeks + [HOLDOUT]  # built (graphs / tokens) like any week, but in none of the splits
        assert HOLDOUT not in train_weeks + val_weeks + final_weeks
        lag_short = max(HOLDOUT + WEEK - data_end[k] for k in (EVENT_COLS if full else NONPUSH))
        print(f"holdout week {str(HOLDOUT)[:10]}: {int(tables[HOLDOUT].eligible.sum()):,} eligible users, "
              f"labels end {lag_short} before the full week, tune week {str(HOLDOUT_TUNE)[:10]}, "
              f"last train week {str(train_weeks[-1])[:10]}")
    tables[base.TEST_START] = base.build_week(base.TEST_START, ev, own, lags, data_end, n,
                                              with_labels=False)
    if hist:
        tables = windowed_tables(tables, ev, own, lags, data_end, n, hist)
        print(f"history: only the last {hist} days before each cutoff are visible")
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
    segment_of = None  # ego only: segment of every row of a week (-1 no example, 0 quiet, 1 active)
    split = ego and args.segment != "none"
    if ego:
        train, predict_week, segment_of = ego_stages(args, tables, feats, weeks, train_weeks, cohort, lags, test, labels, device,
                                         sel)
    elif het:
        train, predict_week = hetero_stages(args, tables, feats, train_weeks, cohort, ev, lags, sel, repo_is_org,
                                            labels, device)
    elif chain:
        train, predict_week = token_stages(args, tables, feats, weeks, train_weeks, cohort, lags, test, labels, device)
    else:
        raw_graphs = {w: week_graph(ev, w, lags, repo_is_org, sel[w], times=True, window=hist or 90) for w in tables}

        # Standardise log1p features with statistics of the training weeks only.
        xu_tr = np.log1p(np.concatenate([tables[w][feats].values for w in train_weeks]))
        xr_tr = np.concatenate([raw_graphs[w][0] for w in train_weeks])
        mu_u, sd_u = xu_tr.mean(0), xu_tr.std(0) + 1e-6
        mu_r, sd_r = xr_tr.mean(0), xr_tr.std(0) + 1e-6

        graphs = {}
        for w, t in tables.items():
            xr, edges, ages = raw_graphs[w]
            y = (t[[f"y_{k}" for k in labels]].values if w != base.TEST_START
                 else np.full((len(t), len(labels)), np.nan))
            graphs[w] = {
                "xu": torch.tensor((np.log1p(t[feats].values) - mu_u) / sd_u, dtype=torch.float32, device=device),
                "xr": torch.tensor((xr - mu_r) / sd_r, dtype=torch.float32, device=device),
                "edges": [(torch.tensor(u, device=device), torch.tensor(r, device=device),
                           torch.tensor(e, device=device)) for u, r, e in edges],
                "edge_t": [torch.tensor(a, device=device) for a in ages] if time_dim else None,
                "y": torch.tensor(y, dtype=torch.float32, device=device),
                "mask": torch.tensor(t.eligible.values, device=device),
            }
        print(f"graph of the test week: {n} users, {graphs[base.TEST_START]['xr'].shape[0]} repos, "
              f"{sum(len(u) for u, _, _ in graphs[base.TEST_START]['edges'])} edges")

        g0 = graphs[train_weeks[0]]

        def make_model():
            return HeteroSAGE(g0["xu"].shape[1], g0["xr"].shape[1], g0["edges"][0][2].shape[1], len(labels),
                              time_dim=time_dim).to(device)

        def train(fit_weeks, epochs, val_weeks=None):
            return fit(graphs, fit_weeks, make_model, epochs, args.seed, val_weeks=val_weeks, lr=args.lr)

        def predict_week(model, w):
            return predict(model, graphs[w])

    model, best_epoch = train(train_weeks, args.epochs, val_weeks=val_weeks)
    print(f"best epoch {best_epoch}")

    # Thresholds on the pooled validation weeks (same rule as the baseline).
    metrics = {"task": args.task, "seed": args.seed, "lr": args.lr, "best_epoch": best_epoch,
               "history_days": hist, "n_user_features": len(feats), "per_label": {}, "validation": []}
    metrics["graph_structure"] = args.graph_structure
    if args.graph_structure == "user-repo":
        metrics.update(time_enc=args.time_enc, time_dim=time_dim)
    if ego:
        metrics["segment"] = args.segment
        metrics["active_min_events"] = args.active_min_events
    if het:
        metrics["graph"] = {a.dest: getattr(args, a.dest) for a in hg._group_actions}
    if chain or ego:
        metrics["graph"] = {a.dest: getattr(args, a.dest) for a in (g if chain else eg)._group_actions}
        metrics["graph"].update(batch_size=args.batch_size, max_steps_per_epoch=args.max_steps_per_epoch)
    val_p = {w: predict_week(model, w) for w in val_weeks}
    rates, seg_rates = {}, {}
    for j, k in enumerate(labels):
        ys, ps, ss = [], [], []
        for w in val_weeks:
            m = tables[w].eligible.values & ~np.isnan(tables[w][f"y_{k}"].values)
            ys.append(tables[w][f"y_{k}"].values[m])
            ps.append(val_p[w][m, j])
            if segment_of is not None:
                ss.append(segment_of(w)[m])
        y, p = np.concatenate(ys), np.concatenate(ps)
        if not len(y):  # --holdout: push has no labels in the tune week; it is not scored either
            metrics["per_label"][k] = {"threshold": None, "val_f1": None, "val_pred_rate": None}
            print(f"{k:18s} no labels in the validation weeks: not tuned, not scored")
            continue
        thr, f1 = base.best_threshold(y, p)
        rates[k] = float((p >= thr).mean())
        metrics["per_label"][k] = {"threshold": thr, "val_f1": f1, "val_pred_rate": rates[k]}
        if split:
            sv = np.concatenate(ss)
            seg_rates[k] = {name: float((p[sv == s] >= thr).mean()) for s, name in enumerate(SEGMENTS)
                            if (sv == s).any()}
            metrics["per_label"][k]["val_pred_rate_by_segment"] = seg_rates[k]
        print(f"{k:18s} val F1 {f1:.4f}  threshold {thr:.3f}  predicted {rates[k]:.3f}  "
              f"positives {y.mean():.3f}")
    # Labels scored per week: all of them, or with --holdout the 5 non-push ones (push has no labels).
    scored = [k for k in labels if k in rates]
    if args.holdout and not scored:
        ap.error("--holdout scores the non-push activities: add at least one to --actions")

    def decide(q, seg, k):
        """Decision rule carried to a new week: the validation share of predicted positives (one
        share per segment with --segment, since each segment model has its own probability scale)."""
        pred = np.zeros(len(q), bool)
        parts = [(seg == s, seg_rates[k].get(name, rates[k])) for s, name in enumerate(SEGMENTS)] if split \
            else [(np.ones(len(q), bool), rates[k])]
        for ms, share in parts:
            if ms.any() and share > 0:
                pred[ms] = q[ms] >= np.quantile(q[ms], 1 - share)
        return pred

    def week_result(w, pred, tag):
        """F1 per scored label, official score over them (6-activity runs) and per-segment F1 of week
        w; pred = {label: bool array over the eligible rows of the week}."""
        t = tables[w]
        m = t.eligible.values
        f1 = {}
        for k in scored:
            y = t[f"y_{k}"].values[m]
            f1[k] = float(2 * (pred[k] & (y == 1)).sum() / max(pred[k].sum() + y.sum(), 1))
        r = {"week": str(w), "users": int(m.sum()), "f1": f1}
        if segment_of is not None:
            sw = segment_of(w)[m]
            r["per_segment"] = {}
            for s, name in enumerate(SEGMENTS):
                ms = sw == s
                d = {"users": int(ms.sum()), "f1": {}, "pos_rate": {}}
                for k in scored:
                    y = t[f"y_{k}"].values[m][ms] == 1
                    pk = pred[k][ms]
                    d["f1"][k] = float(2 * (pk & y).sum() / max(pk.sum() + y.sum(), 1))
                    d["pos_rate"][k] = float(y.mean()) if len(y) else float("nan")
                d["f1_avg"] = float(np.mean(list(d["f1"].values())))
                r["per_segment"][name] = d
                print(f"  {tag} {str(w)[:10]} {name:6s} ({d['users']:,} users): "
                      + "  ".join(f"{k} {v:.3f}" for k, v in d["f1"].items()) + f"  avg {d['f1_avg']:.4f}")
        if args.task == "all":
            gids = cohort.github_user_id.values[t.cidx.values[m]]
            zero = np.zeros(int(m.sum()), bool)
            truth = base.to_output(gids, {k: np.nan_to_num(t[f"y_{k}"].values[m]) for k in EVENT_COLS})
            out = base.to_output(gids, {k: pred.get(k, zero) for k in EVENT_COLS})
            r.update(base.scores(out, truth, events=scored))
        print(f"{tag} {str(w)[:10]}: " + "  ".join(f"{k} F1 {v:.4f}" for k, v in f1.items())
              + (f"  score {100 * r['score']:.2f}" if "score" in r else "")
              + (f" ({len(scored)} activities)" if len(scored) != len(EVENT_COLS) and "score" in r else ""))
        return r

    # Per-week scores on weeks where the labels scored are all known.
    for w in val_weeks:
        m = tables[w].eligible.values
        if tables[w].loc[m, [f"y_{k}" for k in scored]].isna().any().any():
            continue
        metrics["validation"].append(week_result(
            w, {k: val_p[w][m, labels.index(k)] >= metrics["per_label"][k]["threshold"] for k in scored},
            "validation"))
    metrics["val_f1_mean"] = {k: float(np.mean([r["f1"][k] for r in metrics["validation"]]))
                              for k in scored}
    for k, v in metrics["val_f1_mean"].items():
        print(f"mean validation {k} F1 {v:.4f}")
    if args.task == "all":
        metrics["val_score_mean"] = float(np.mean([r["score"] for r in metrics["validation"]]))
        print(f"mean validation score {100 * metrics['val_score_mean']:.2f}")
    if segment_of is not None and metrics["validation"]:
        metrics["per_segment"] = {
            name: {"users": float(np.mean([r["per_segment"][name]["users"] for r in metrics["validation"]])),
                   "f1": {k: float(np.mean([r["per_segment"][name]["f1"][k] for r in metrics["validation"]]))
                          for k in scored}}
            for name in SEGMENTS}
        for name, d in metrics["per_segment"].items():
            d["f1_avg"] = float(np.mean(list(d["f1"].values())))
            print(f"mean validation F1, {name} segment: {d['f1_avg']:.4f}")

    if args.holdout:
        # The model trained up to 10-24 (no retrain), the tune week's share of predicted positives.
        m = tables[HOLDOUT].eligible.values
        q = predict_week(model, HOLDOUT)[m]
        seg = segment_of(HOLDOUT)[m] if segment_of is not None else None
        metrics["holdout_protocol"] = {"train_last": str(train_weeks[-1]), "tune": str(HOLDOUT_TUNE),
                                       "holdout": str(HOLDOUT), "scored": scored,
                                       "label_end": str(min(data_end[k] for k in scored))}
        metrics["holdout"] = week_result(HOLDOUT, {k: decide(q[:, labels.index(k)], seg, k) for k in scored},
                                         "HOLDOUT")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        print(f"wrote {out_dir / 'metrics.json'} (--holdout: no final model, no output csv)")
    else:
        # Final model: all labelled weeks for the early-stopped number of epochs; the validation
        # share of predicted positives is reused as the decision rule (as in the baseline).
        final, _ = train(final_weeks, best_epoch)
        p = predict_week(final, base.TEST_START)
        tt = tables[base.TEST_START]
        cidx = pd.Series(np.arange(len(tt)), index=cohort.github_user_id.values[tt.cidx.values]).loc[test.UserID].values
        out = pd.DataFrame({"UserID": test.UserID.values})
        test_seg = segment_of(base.TEST_START)[cidx] if segment_of is not None else None
        if test_seg is not None:
            print("test users per segment: " + ", ".join(f"{name} {(test_seg == s).sum()}"
                                                          for s, name in enumerate(SEGMENTS)))
        for j, k in enumerate(labels):
            out[k] = decide(p[cidx, j], test_seg, k).astype(int)
        name = "output.csv" if args.task == "all" else f"{args.task}.csv"
        out.to_csv(out_dir / name, index=False)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        print(f"wrote {out_dir / name}")

    print(f"\nF1 per action (run {out_dir.name}, validation weeks):")
    print(f"  {'action':18s} {'pooled F1':>10s} {'mean weekly F1':>15s}")
    for k in scored:
        weekly = metrics["val_f1_mean"].get(k, float("nan"))
        print(f"  {k:18s} {metrics['per_label'][k]['val_f1']:10.4f} {weekly:15.4f}")
    print(f"  {'average':18s} {np.mean([metrics['per_label'][k]['val_f1'] for k in scored]):10.4f} "
          f"{np.mean([metrics['val_f1_mean'][k] for k in scored]):15.4f}")
    if args.holdout:
        h = metrics["holdout"]
        print(f"\nHOLDOUT {str(HOLDOUT)[:10]} ({h['users']:,} users, {len(scored)} activities): "
              + (f"score {100 * h['score']:.2f}  " if "score" in h else "")
              + "  ".join(f"{k} {v:.4f}" for k, v in h["f1"].items()))


if __name__ == "__main__":
    main()
