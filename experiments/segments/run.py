"""Quiet vs active users: where the holdout score is lost, and which users the capture drops hit.

Quiet = an example (eligible user, week) with no visible row in the 14 days before the per-table
cutoffs, the split of experiments/gnn/run.py (`--segment activity`, `holdout.per_segment`).

1. Error split. The runs analysed saved no predictions (holdout runs since 2026-11-16 also write
   holdout.npz with per-user probabilities), so the per-segment confusion counts are rebuilt from a
   run's metrics.json: per activity, the overall F1, the F1 and positive rate of each segment, and
   the number of users predicted positive (validation share x users; push: holdout_push.tune.share).
   With TP = F1 * (predicted + positives) / 2 for each segment and for both together, the
   predictions of each segment follow. The rebuilt counts give back the reported score. Then the
   score is recomputed with one kind of error removed (e.g. no misses on quiet users).
2. Capture drops by user type, per period (spring Feb 7 - Apr 25, summer, the Sep burst, recent):
   a. the model segment above, over the eligible examples of the weeks;
   b. a fixed type from spring: "heavy" if the user has an event in >= half of the spring weeks
      (all 9,000 cohort users). Spring stops before May because the cohort was sampled among users
      active in May.
3. The quiet users of one week split by their spring type (default: holdout week 09-26).

Run from the repo root:
    python experiments/segments/run.py                        # runs all_ego_uf_ho0926_s0-2
    python experiments/segments/run.py --runs het_ho0926_s0   # any run with holdout.per_segment
Writes experiments/segments/outputs/{error_split.csv, periods_segment.csv, periods_type.csv,
weekly_segment.csv, quiet_week.csv}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GNN_OUT = ROOT / "experiments" / "gnn" / "outputs"
OUT = Path(__file__).resolve().parent / "outputs"

ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
NP = ACTS[:-1]
SPRING = ("2026-02-07", "2026-04-26")
PERIODS = {
    "spring": SPRING,
    "summer": ("2026-07-04", "2026-08-23"),     # non-push ~3-25% of normal, push thinning from 08-18
    "sep_burst": ("2026-08-29", "2026-09-13"),  # non-push back at 100-250%
    "recent": ("2026-09-19", "2026-10-18"),     # non-push 10-30%, push 30-60%
    "nov": ("2026-10-24", "2026-11-01"),        # non-push ~12%, push 0-4% (no push labels)
}


# --- 1. Error split -------------------------------------------------------------------------

def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def score(c):
    """Official score from per-activity (tp, fp, fn): (macro-F1 + micro-F1) / 2, in points."""
    mac = np.mean([f1(*c[a]) for a in ACTS])
    mic = f1(*np.sum([c[a] for a in ACTS], 0))
    return 100 * (mac + mic) / 2


def rebuild(m):
    """Per-segment (tp, fp, fn) of every activity from a holdout metrics.json."""
    h = m["holdout"]
    seg = h["per_segment"]
    n = h["users"]
    nq, na = seg["quiet"]["users"], seg["active"]["users"]
    quiet, active = {}, {}
    for a in ACTS:
        share = m["holdout_push"]["tune"]["share"] if a == "push" else m["per_label"][a]["val_pred_rate"]
        pred = round(share * n)
        pq = round(seg["quiet"]["pos_rate"][a] * nq)
        pa = round(seg["active"]["pos_rate"][a] * na)
        f, fq, fa = h["f1"][a], seg["quiet"]["f1"][a], seg["active"]["f1"][a]
        # fq (pred_q + pq) + fa (pred - pred_q + pa) = f (pred + pq + pa)
        pred_q = (f * (pred + pq + pa) - fq * pq - fa * (pred + pa)) / (fq - fa)
        tq, ta = fq * (pred_q + pq) / 2, fa * (pred - pred_q + pa) / 2
        quiet[a] = (tq, pred_q - tq, pq - tq)
        active[a] = (ta, pred - pred_q - ta, pa - ta)
    return quiet, active


def scenarios(q, a):
    """Score with one kind of error removed. A removed miss becomes a hit, a removed false alarm a
    true negative."""
    def join(qq, aa):
        return {k: tuple(np.add(qq[k], aa[k])) for k in ACTS}

    def perfect(s):
        return {k: (s[k][0] + s[k][2], 0, 0) for k in ACTS}

    def no_fp(s):
        return {k: (s[k][0], 0, s[k][2]) for k in ACTS}

    def no_fn(s):
        return {k: (s[k][0] + s[k][2], s[k][1], 0) for k in ACTS}

    return {
        "now": score(join(q, a)),
        "quiet perfect": score(join(perfect(q), a)),
        "quiet no false alarms": score(join(no_fp(q), a)),
        "quiet no misses": score(join(no_fn(q), a)),
        "active perfect": score(join(q, perfect(a))),
        "active no false alarms": score(join(q, no_fp(a))),
        "active no misses": score(join(q, no_fn(a))),
    }


def error_split(runs):
    rows = []
    for r in runs:
        m = json.load(open(GNN_OUT / r / "metrics.json"))
        if "per_segment" not in m.get("holdout", {}):
            print(f"{r}: no holdout.per_segment (only ego runs write it), skipped")
            continue
        q, a = rebuild(m)
        sc = scenarios(q, a)
        print(f"\n== {r}: reported {100 * m['holdout']['score']:.2f}, rebuilt {sc['now']:.2f}")
        print(f"{'activity':18s} {'pos Q':>6s} {'FP Q':>6s} {'FN Q':>6s} | {'pos A':>6s} {'TP A':>6s} "
              f"{'FP A':>6s} {'FN A':>6s}")
        for k in ACTS:
            (tq, fpq, fnq), (ta, fpa, fna) = q[k], a[k]
            fpq = max(fpq, 0.0)  # the rebuild leaves -0.x where the model predicts nobody
            print(f"{k:18s} {tq + fnq:6.0f} {fpq:6.0f} {fnq:6.0f} | {ta + fna:6.0f} {ta:6.0f} {fpa:6.0f} {fna:6.0f}")
            rows.append(dict(run=r, activity=k, tp_q=tq, fp_q=fpq, fn_q=fnq, tp_a=ta, fp_a=fpa, fn_a=fna))
        for name, v in sc.items():
            print(f"  {name:24s} {v:6.2f}  ({v - sc['now']:+.2f})")
            rows.append(dict(run=r, activity=f"score: {name}", tp_q=v))
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "error_split.csv", index=False)
    s = df[df.activity.str.startswith("score: ")].groupby("activity", sort=False).tp_q.mean()
    c = df[~df.activity.str.startswith("score: ")]
    print(f"\nMean over {df.run.nunique()} runs:")
    for name, v in s.items():
        print(f"  {name[7:]:24s} {v:6.2f}  ({v - s.iloc[0]:+.2f})")
    per_run = c.groupby("run")[["fp_q", "fn_q", "fp_a", "fn_a"]].sum().mean()
    print(f"  errors per run (6 activities): quiet {per_run.fp_q + per_run.fn_q:.0f} "
          f"(false alarms {per_run.fp_q:.0f}, misses {per_run.fn_q:.0f}), active "
          f"{per_run.fp_a + per_run.fn_a:.0f} (false alarms {per_run.fp_a:.0f}, misses {per_run.fn_a:.0f})")


# --- 2-3. Capture drops by user type ---------------------------------------------------------

def drops(quiet_week):
    from analysis import eda  # loads the data once (~5 s): the models' examples, cutoffs and labels

    D = eda.Data()
    ws, w = D.week_starts, D.n_weeks
    sp = np.asarray((ws >= SPRING[0]) & (ws < SPRING[1]))
    heavy = (sum(D.counts[k][:, sp] for k in ACTS) > 0).mean(1) >= 0.5
    print(f"\nspring type (event in >= half of the weeks {SPRING[0]} .. {SPRING[1]}): "
          f"heavy {heavy.sum():,}, light {(~heavy).sum():,}")

    ex = D.ex[D.ex.j < w].copy()
    ex["quiet"] = ex.vis_14d == 0
    ex["heavy"] = heavy[ex.cidx.values]
    ex["n_np"] = sum(D.counts[k][ex.cidx.values, ex.j.values] for k in NP)
    ex["n_push"] = D.counts["push"][ex.cidx.values, ex.j.values]
    ex["y_np"] = ex[[f"y_{k}" for k in NP]].max(axis=1, skipna=False)  # any non-push activity

    def seg_stats(g):
        q, a = g[g.quiet], g[~g.quiet]
        return {"examples (all weeks)": len(g), "quiet share": g.quiet.mean(), "quiet who were heavy": q.heavy.mean(),
                "non-push rate active": a.y_np.mean(), "non-push rate quiet": q.y_np.mean(),
                "push rate active": a.y_push.mean(), "push rate quiet": q.y_push.mean(),
                "non-push events/user active": a.n_np.mean(), "non-push events/user quiet": q.n_np.mean(),
                "push events/user active": a.n_push.mean(), "push events/user quiet": q.n_push.mean()}

    weekly = pd.DataFrame([{"week": ws[j], **seg_stats(g)} for j, g in ex.groupby("j")])
    weekly.to_csv(OUT / "weekly_segment.csv", index=False)

    seg_rows, type_rows = {}, {}
    for name, (a, b) in PERIODS.items():
        js = np.flatnonzero((ws >= a) & (ws < b))
        seg_rows[name] = seg_stats(ex[ex.j.isin(js)])
        r = {}
        for t, m in [("heavy", heavy), ("light", ~heavy)]:
            npc = sum(D.counts[k][m][:, js] for k in NP)
            pc = D.counts["push"][m][:, js]
            r[f"{t} non-push events/user"] = npc.mean()
            r[f"{t} non-push active share"] = (npc > 0).mean()
            r[f"{t} push events/user"] = pc.mean()
            r[f"{t} push active share"] = (pc > 0).mean()
        type_rows[name] = r
    pd.set_option("display.width", 250)
    for title, rows, fname in [("model segment, eligible examples, mean per week", seg_rows, "periods_segment.csv"),
                               ("fixed spring type, all cohort users, mean per week", type_rows, "periods_type.csv")]:
        t = pd.DataFrame(rows)
        t.to_csv(OUT / fname)
        print(f"\n== {title} ==")
        print(t.round(3).to_string())
        print("as a fraction of spring:")
        print(t.div(t.spring, axis=0).round(2).to_string())

    j = int(np.flatnonzero(ws == pd.Timestamp(quiet_week))[0])
    g = ex[ex.j == j]
    rows = []
    for (quiet, h), s in g.groupby(["quiet", "heavy"]):
        rows.append({"segment": "quiet" if quiet else "active", "spring type": "heavy" if h else "light",
                     "users": len(s), "push rate": s.y_push.mean(), "pushers": int(s.y_push.sum()),
                     "any non-push rate": s.y_np.mean()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "quiet_week.csv", index=False)
    print(f"\n== week {quiet_week}: segment x spring type ==")
    print(t.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[f"all_ego_uf_ho0926_s{s}" for s in range(3)],
                    help="folders of experiments/gnn/outputs with a holdout metrics.json")
    ap.add_argument("--week", default="2026-09-26 16:00", help="week of part 3 (UTC start)")
    ap.add_argument("--skip-drops", action="store_true", help="only the error split (no data loading)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    error_split(args.runs)
    if not args.skip_drops:
        drops(args.week)


if __name__ == "__main__":
    main()
