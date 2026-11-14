"""Collect every experiment's validation / holdout score from its metrics files into one JSON.

    python scripts/collect_scores.py results/scores_new.json
    python scripts/collect_scores.py results/scores_old.json --root results/archive_old_data
"""
import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mean(x):
    return sum(x) / len(x) if x else None


def collect(root):
    out = {}

    def load(pattern):
        for f in sorted(glob.glob(str(root / pattern), recursive=True)):
            yield Path(f), json.loads(Path(f).read_text())

    for f, m in load("experiments/baseline/outputs/*/metrics.json"):
        for method in ["lightgbm", "active_last_30d"]:
            rows = [r for r in m["validation"] if r["method"] == method]
            out[f"baseline/{f.parent.name}/{method}"] = {
                "score": mean([r["score"] for r in rows]), "weeks": [r["week"][:10] for r in rows],
                "weekly": [r["score"] for r in rows], "users": [r["users"] for r in rows]}
    for f, m in load("experiments/gnn/outputs/*/metrics.json"):
        out[f"gnn/{f.parent.name}"] = {
            "score": m.get("val_score_mean"), "f1": m.get("val_f1_mean") or m.get("val_push_f1_mean"),
            "weeks": [r["week"][:10] for r in m["validation"]], "weekly": [r.get("score") for r in m["validation"]],
            "users": [r["users"] for r in m["validation"]]}
    for f, m in load("experiments/gnn_review/outputs/*/metrics.json"):
        out[f"gnn_review/{f.parent.name}"] = {"score": m["variants"][m["chosen"]]["mean"], "chosen": m["chosen"],
                                              "weekly": m["variants"][m["chosen"]]["weekly"]}
    for f, m in load("experiments/relations/outputs/screen.json"):
        for k, r in m["results"].items():
            out[f"relations/{k}"] = {"score": r["score"], "seeds": m["seeds"]}
    for f, m in load("experiments/relgt/outputs/*/metrics.json"):
        out[f"relgt/{f.parent.name}"] = {
            "score": m["val_score_mean"], "holdout": m.get("holdout_score_mean"),
            "tuned": m["variants"][m["chosen"]]["mean"], "val_map": max(h["val_ap"] for h in m["history"])
            if m["history"] and "val_ap" in m["history"][0] else None,
            "weeks": [r["week"][:10] for r in m["validation"]], "holdout_weeks": [r["week"][:10] for r in m["holdout"]],
            "users": [r["users"] for r in m["validation"]]}
    for f, m in load("experiments/two_model/outputs/*/metrics.json"):
        out[f"two_model/{f.parent.name}"] = {"holdout": m.get("holdout_score_mean")}
    for pattern, tag in [("old_experiments/relgt_inspired/outputs/*/backtest_metrics.json", "relgt_inspired"),
                         ("experiments/relgt/outputs/*/backtest_metrics.json", "relgt_backtest")]:
        for f, m in load(pattern):
            out[f"{tag}/{f.parent.name}"] = {"score": m["score"], "model": m["temporal_model"], "split": m["split"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores = collect(args.root)
    args.out.write_text(json.dumps(scores, indent=2))
    print(f"{len(scores)} results -> {args.out}")
