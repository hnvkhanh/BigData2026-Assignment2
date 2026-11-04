"""The heterogeneous GNN of experiments/gnn, using that pr_review_comment ~ subset of pr_review.

On test-aligned user-weeks, P(review | review comment) = 0.75 and P(review comment | review) =
0.49, and the second one varies a lot between users (std 0.28). Four changes over experiments/gnn:

  1. Habit features on the user nodes (--no-habit drops them): over the 90 days before the
     cutoff, weeks with a review / a review comment / both, the smoothed P(comment | review) and
     P(review | comment), comments per review, the share of reviews per state, the share of
     comments that are replies or have no review in PRReviews.
  2. Chain head (--no-chain drops it): two extra outputs P(comment | review) and
     P(comment | no review), trained on the rows where the review label is 1 / 0, compose
     P(comment) = P(review) P(c | r) + (1 - P(review)) P(c | not r). The validation picks the
     direct output, the chain or their mean for pr_review_comment.
  3. Thresholds tuned jointly on the score (macro-F1 + micro-F1) / 2 over the validation weeks
     (coordinate ascent starting from the per-label F1 thresholds) instead of per-label F1.
  4. Consistency: a predicted review comment also predicts a review, kept only if it raises the
     validation score.

Run from the repo root:
    python experiments/gnn_review/run.py
    python experiments/gnn_review/run.py --no-habit --no-chain   # only 3 and 4
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gnn_run", HERE.parent / "gnn" / "run.py")
gnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gnn)
base = gnn.base

import torch  # noqa: E402  (after numpy, see experiments/gnn)
from torch import nn  # noqa: E402

EVENT_COLS, DAY = base.EVENT_COLS, base.DAY
IR, IC = EVENT_COLS.index("pr_review"), EVENT_COLS.index("pr_review_comment")
STATES = ["approved", "commented", "changes_requested"]
PRIOR_C_GIVEN_R, PRIOR_R_GIVEN_C, PRIOR_N = 0.47, 0.75, 2.0
OUT_DIR = HERE / "outputs"


def load_review_detail(cohort):
    """Reviews (with state) and review comments (reply / linked to a review) of cohort users."""
    users = pd.read_parquet(base.DB / "Users.parquet", columns=["user_id"])
    to_cidx = np.full(users.user_id.max() + 1, -1, np.int64)
    to_cidx[cohort.user_id.values] = np.arange(len(cohort))
    r = pd.read_parquet(base.DB / "PRReviews.parquet",
                        columns=["reviewer_id", "submitted_at", "first_seen_at", "state"])
    c = pd.read_parquet(base.DB / "PRReviewComments.parquet",
                        columns=["author_id", "created_at", "in_reply_to_id", "pull_request_review_id"])
    r = r[r.reviewer_id.notna() & r.submitted_at.notna()]
    c = c[c.author_id.notna() & c.created_at.notna()]
    R = pd.DataFrame({"uid": to_cidx[r.reviewer_id.astype(np.int64).values], "t": r.submitted_at.values,
                      "avail": np.maximum(r.submitted_at.values, r.first_seen_at.values),
                      "state": r.state.values})
    C = pd.DataFrame({"uid": to_cidx[c.author_id.astype(np.int64).values], "t": c.created_at.values,
                      "avail": c.created_at.values, "reply": c.in_reply_to_id.notna().values,
                      "unlinked": c.pull_request_review_id.isna().values})
    return R[R.uid >= 0].reset_index(drop=True), C[C.uid >= 0].reset_index(drop=True)


def habit_features(s, R, C, lags, n):
    """Review-commenting habit of every cohort user, from rows visible before both tables' cutoffs."""
    c = np.datetime64(min(s - lags["pr_review"], s - lags["pr_review_comment"]))
    lo = c - np.timedelta64(90, "D")
    r = R[(R.avail.values < c) & (R.t.values >= lo)]
    m = C[(C.avail.values < c) & (C.t.values >= lo)]

    def week_keys(d):  # distinct (user, week before the cutoff) pairs
        return np.unique(d.uid.values * 16 + (c - d.t.values) // np.timedelta64(7, "D"))

    kr, kc = week_keys(r), week_keys(m)
    rw = np.bincount(kr // 16, minlength=n)
    cw = np.bincount(kc // 16, minlength=n)
    bw = np.bincount(np.intersect1d(kr, kc) // 16, minlength=n)
    nr = np.bincount(r.uid.values, minlength=n)
    nc = np.bincount(m.uid.values, minlength=n)
    f = {"rc_review_weeks_90d": rw, "rc_comment_weeks_90d": cw, "rc_both_weeks_90d": bw,
         "rc_p_comment_given_review": (bw + PRIOR_N * PRIOR_C_GIVEN_R) / (rw + PRIOR_N),
         "rc_p_review_given_comment": (bw + PRIOR_N * PRIOR_R_GIVEN_C) / (cw + PRIOR_N),
         "rc_comments_per_review": nc / (nr + 1)}
    for st in STATES:
        f[f"rc_share_{st}"] = np.bincount(r.uid.values, weights=r.state.values == st, minlength=n) / np.maximum(nr, 1)
    for col in ["reply", "unlinked"]:
        f[f"rc_share_{col}"] = np.bincount(m.uid.values, weights=m[col].values, minlength=n) / np.maximum(nc, 1)
    return pd.DataFrame(f)


def chain_loss(logits, y, mask, chain):
    """Masked BCE of the 6 activities; with the chain, plus the two conditional comment heads."""
    loss = gnn.masked_bce(logits[:, :len(EVENT_COLS)], y, mask)
    if not chain:
        return loss
    yr, yc = y[:, IR], y[:, IC]
    known = mask & ~torch.isnan(yr) & ~torch.isnan(yc)
    parts = []
    for j, cond in [(len(EVENT_COLS), yr == 1), (len(EVENT_COLS) + 1, yr == 0)]:
        sel = known & cond
        if sel.any():
            parts.append(nn.functional.binary_cross_entropy_with_logits(logits[sel, j], yc[sel]))
    # every head weighs the same as one activity
    return (len(EVENT_COLS) * loss + sum(parts)) / (len(EVENT_COLS) + len(parts))


def fit(graphs, weeks, chain, epochs, seed, device, val_weeks=None, patience=5, lr=2e-3):
    """experiments/gnn's fit with the chain outputs and loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g0 = graphs[weeks[0]]
    n_out = len(EVENT_COLS) + (2 if chain else 0)
    model = gnn.HeteroSAGE(g0["xu"].shape[1], g0["xr"].shape[1], g0["edges"][0][2].shape[1], n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, best_epoch, bad = np.inf, None, epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for w in rng.permutation(weeks):
            g = graphs[w]
            opt.zero_grad()
            loss = chain_loss(model(g), g["y"], g["mask"], chain)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val_weeks is None:
            continue
        model.eval()
        with torch.no_grad():
            vl = np.mean([chain_loss(model(graphs[w]), graphs[w]["y"], graphs[w]["mask"], chain).item()
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


def activity_probs(p, comment_mode):
    """The 6 activity probabilities, pr_review_comment taken from the direct head, the chain or both."""
    out = p[:, :len(EVENT_COLS)].copy()
    if comment_mode != "direct":
        chain = p[:, IR] * p[:, -2] + (1 - p[:, IR]) * p[:, -1]
        out[:, IC] = chain if comment_mode == "chain" else (out[:, IC] + chain) / 2
    return out


def score(y, pred):
    """(macro-F1 + micro-F1) / 2 of 0/1 matrices (rows x 6), as in evaluate.py."""
    tp = (pred & y).sum(0)
    denom = pred.sum(0) + y.sum(0)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0)
    return (f1.mean() + 2 * tp.sum() / max(denom.sum(), 1)) / 2


def apply(p, thr, consistent):
    pred = p >= thr
    if consistent:
        pred[:, IR] |= pred[:, IC]
    return pred


def joint_thresholds(y, p, thr, consistent, passes=3, grid=200):
    """Coordinate ascent of the score over one threshold at a time."""
    thr = thr.copy()
    best = score(y, apply(p, thr, consistent))
    for _ in range(passes):
        improved = False
        for j in range(p.shape[1]):
            for t in np.unique(np.quantile(p[:, j], np.linspace(0, 1, grid + 1)[1:-1])):
                cand = thr.copy()
                cand[j] = t
                sc = score(y, apply(p, cand, consistent))
                if sc > best + 1e-9:
                    best, thr, improved = sc, cand, True
        if not improved:
            break
    return thr, best


def main():
    ap = argparse.ArgumentParser()
    base.add_period_args(ap)
    ap.add_argument("--no-habit", action="store_true", help="leave out the review-commenting habit features")
    ap.add_argument("--no-chain", action="store_true", help="leave out the conditional comment heads")
    ap.add_argument("--name", default=None, help="results go to outputs/<name>/ (default: from the flags)")
    ap.add_argument("--lr", type=float, default=2e-3, help="AdamW learning rate")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    chain = not args.no_chain
    name = args.name or "_".join(["full"] + [x for x, off in [("nohabit", args.no_habit),
                                                               ("nochain", args.no_chain)] if off])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, ev, own, data_end = base.load()
    R, C = load_review_detail(cohort)
    repos = pd.read_parquet(base.DB / "Repositories.parquet", columns=["repo_id", "org_id"])
    repo_is_org = np.zeros(repos.repo_id.max() + 1, bool)
    repo_is_org[repos.repo_id.values] = repos.org_id.notna().values
    n = len(cohort)
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    weeks, train_weeks, val_weeks, final_weeks = base.split_weeks(args, data_end)
    print(f"run {name}, device {device}, {len(train_weeks)} train weeks, "
          f"validation {[str(w)[:10] for w in val_weeks]}")

    tables = {w: base.build_week(w, ev, own, lags, data_end, n) for w in weeks}
    tables[base.TEST_START] = base.build_week(base.TEST_START, ev, own, lags, data_end, n, with_labels=False)
    if not args.no_habit:
        tables = {w: pd.concat([t, habit_features(w, R, C, lags, n)], axis=1) for w, t in tables.items()}
    feats = [c for c in tables[weeks[0]].columns if c not in {"eligible", "cidx", "week"} and not c.startswith("y_")]
    print(f"{len(feats)} user features")
    keep = np.ones(n, bool)
    raw_graphs = {w: gnn.week_graph(ev, w, lags, repo_is_org, keep) for w in tables}

    xu_tr = np.log1p(np.concatenate([tables[w][feats].values for w in train_weeks]))
    xr_tr = np.concatenate([raw_graphs[w][0] for w in train_weeks])
    mu_u, sd_u = xu_tr.mean(0), xu_tr.std(0) + 1e-6
    mu_r, sd_r = xr_tr.mean(0), xr_tr.std(0) + 1e-6
    graphs = {}
    for w, t in tables.items():
        xr, edges = raw_graphs[w]
        y = (t[[f"y_{k}" for k in EVENT_COLS]].values if w != base.TEST_START
             else np.full((len(t), len(EVENT_COLS)), np.nan))
        graphs[w] = {
            "xu": torch.tensor((np.log1p(t[feats].values) - mu_u) / sd_u, dtype=torch.float32, device=device),
            "xr": torch.tensor((xr - mu_r) / sd_r, dtype=torch.float32, device=device),
            "edges": [(torch.tensor(u, device=device), torch.tensor(r, device=device),
                       torch.tensor(e, device=device)) for u, r, e in edges],
            "y": torch.tensor(y, dtype=torch.float32, device=device),
            "mask": torch.tensor(t.eligible.values, device=device),
        }

    model, best_epoch = fit(graphs, train_weeks, chain, args.epochs, args.seed, device,
                            val_weeks=val_weeks, lr=args.lr)
    print(f"best epoch {best_epoch}")
    val_raw = {w: gnn.predict(model, graphs[w]) for w in val_weeks}

    def pooled(k, mode):  # labels and probabilities of activity k over the validation weeks
        ys, ps = [], []
        for w in val_weeks:
            m = tables[w].eligible.values & ~np.isnan(tables[w][f"y_{k}"].values)
            ys.append(tables[w][f"y_{k}"].values[m])
            ps.append(activity_probs(val_raw[w][m], mode)[:, EVENT_COLS.index(k)])
        return np.concatenate(ys), np.concatenate(ps)

    metrics = {"run": name, "seed": args.seed, "lr": args.lr, "best_epoch": best_epoch,
               "n_features": len(feats), "comment_modes": {}, "per_label": {}}
    mode = "direct"
    if chain:
        f1s = {md: base.best_threshold(*pooled("pr_review_comment", md))[1] for md in ["direct", "chain", "mean"]}
        mode = max(f1s, key=f1s.get)
        metrics["comment_modes"] = f1s
        print("pr_review_comment val F1 by head: " + "  ".join(f"{k} {v:.4f}" for k, v in f1s.items())
              + f"  -> {mode}")
    metrics["comment_mode"] = mode

    thr = np.zeros(len(EVENT_COLS))
    for j, k in enumerate(EVENT_COLS):
        thr[j], f1 = base.best_threshold(*pooled(k, mode))
        metrics["per_label"][k] = {"threshold": thr[j], "val_f1": f1}

    # Joint tuning on the validation weeks whose 6 labels are all known.
    full = [w for w in val_weeks if not tables[w].loc[tables[w].eligible, [f"y_{k}" for k in EVENT_COLS]].isna().any().any()]
    rows = {w: tables[w].eligible.values for w in full}
    Y = np.concatenate([tables[w][[f"y_{k}" for k in EVENT_COLS]].values[rows[w]] for w in full]).astype(bool)
    P = np.concatenate([activity_probs(val_raw[w][rows[w]], mode) for w in full])
    variants = {"per-label F1 thresholds": (thr, False)}
    for consistent in [False, True]:
        t, _ = joint_thresholds(Y, P, thr, consistent)
        variants[f"joint thresholds{' + consistency' if consistent else ''}"] = (t, consistent)
    variants["per-label F1 thresholds + consistency"] = (thr, True)

    def weekly(t, consistent):
        out = []
        for w in full:
            y = tables[w][[f"y_{k}" for k in EVENT_COLS]].values[rows[w]].astype(bool)
            out.append(score(y, apply(activity_probs(val_raw[w][rows[w]], mode), t, consistent)))
        return out

    metrics["variants"] = {}
    print(f"\nvalidation score (weeks {[str(w)[:10] for w in full]}):")
    for v, (t, consistent) in variants.items():
        ws = weekly(t, consistent)
        metrics["variants"][v] = {"weekly": ws, "mean": float(np.mean(ws)), "thresholds": t.tolist(),
                                  "consistent": consistent}
        print(f"  {v:40s} " + "  ".join(f"{100 * s:.2f}" for s in ws) + f"   mean {100 * np.mean(ws):.2f}")
    chosen = max(metrics["variants"], key=lambda v: metrics["variants"][v]["mean"])
    thr, consistent = variants[chosen]
    metrics["chosen"] = chosen
    print(f"chosen: {chosen}")
    pred = apply(P, thr, consistent)
    for j, k in enumerate(EVENT_COLS):
        metrics["per_label"][k]["final_threshold"] = float(thr[j])
        metrics["per_label"][k]["val_pred_rate"] = float(pred[:, j].mean())
        metrics["per_label"][k]["val_f1_final"] = float(2 * (pred[:, j] & Y[:, j]).sum() / (pred[:, j].sum() + Y[:, j].sum()))
        print(f"  {k:18s} F1 {metrics['per_label'][k]['val_f1_final']:.4f}  "
              f"predicted {pred[:, j].mean():.3f}  positives {Y[:, j].mean():.3f}")

    # Final model on all labelled weeks; the validation share of predicted positives (before the
    # consistency rule) is reused as the decision rule, as in the baseline.
    final, _ = fit(graphs, final_weeks, chain, best_epoch, args.seed, device, lr=args.lr)
    test = pd.read_csv(base.EVAL_DIR / "test_input.csv")
    tt = tables[base.TEST_START]
    cidx = pd.Series(np.arange(len(tt)), index=cohort.github_user_id.values[tt.cidx.values]).loc[test.UserID].values
    q = activity_probs(gnn.predict(final, graphs[base.TEST_START])[cidx], mode)
    rates = (P >= thr).mean(0)
    out_pred = apply(q, np.array([np.quantile(q[:, j], 1 - rates[j]) for j in range(len(EVENT_COLS))]), consistent)
    out = pd.DataFrame({"UserID": test.UserID.values})
    for j, k in enumerate(EVENT_COLS):
        out[k] = out_pred[:, j].astype(int)
    out.to_csv(out_dir / "output.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {out_dir / 'output.csv'}")


if __name__ == "__main__":
    main()
