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

Run from the repo root:
    python experiments/gnn/run.py
    python experiments/gnn/run.py --actions push
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
    ap.add_argument("--lr", type=float, default=2e-3, help="AdamW learning rate")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.actions and args.task:
        ap.error("use either --actions or --task, not both")
    labels = args.actions or (["push"] if args.task == "push" else list(EVENT_COLS))
    labels = [k for k in EVENT_COLS if k in labels]  # canonical order, no duplicates
    args.task = "all" if labels == list(EVENT_COLS) else "+".join(labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = OUT_DIR / (args.name or (args.task if args.users == "all" else f"{args.task}_{args.users}"))
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
    test = pd.read_csv(base.EVAL_DIR / "test_input.csv")
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

    model, best_epoch = fit(graphs, train_weeks, len(labels), args.epochs, args.seed, device,
                            val_weeks=val_weeks, lr=args.lr)
    print(f"best epoch {best_epoch}")

    # Thresholds on the pooled validation weeks (same rule as the baseline).
    metrics = {"task": args.task, "seed": args.seed, "lr": args.lr, "best_epoch": best_epoch, "per_label": {},
               "validation": []}
    val_p = {w: predict(model, graphs[w]) for w in val_weeks}
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
    final, _ = fit(graphs, final_weeks, len(labels), best_epoch, args.seed, device, lr=args.lr)
    p = predict(final, graphs[base.TEST_START])
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
