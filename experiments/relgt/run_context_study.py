"""Run relgt_db under every relational-context mode and tabulate the results.

    python -m experiments.relgt.run_context_study
    python -m experiments.relgt.run_context_study --seeds 42 43 --epochs 5
    python -m experiments.relgt.run_context_study --aggregate-only

Each (mode, seed) run is a normal ``run backtest --temporal-model relgt_db``
subprocess writing to ``<study-dir>/<mode>_seed<seed>``. A run whose
``backtest_metrics.json`` already exists is skipped (``--force`` reruns), so an
interrupted study resumes. Unrecognized arguments are forwarded to the backtest,
e.g. ``--num-neighbors 256 --hidden-dim 128``. After the runs, ``results.csv``,
``results.json`` and ``summary.md`` are written to the study directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .db_sampler import CONTEXT_MODES

DEFAULT_STUDY_DIR = Path(__file__).parent / "outputs" / "context_study"
DESCRIPTIONS = {
    "r0": "own rows only",
    "r1": "own rows + directly touching rows",
    "r2": "full 1/2-hop incl. repositories",
    "r3": "random rows, count-matched to r2",
}


def run_dir(study_dir: Path, mode: str, seed: int) -> Path:
    return study_dir / f"{mode}_seed{seed}"


def run_one(
    study_dir: Path, mode: str, seed: int, extra: list[str], force: bool
) -> bool:
    out = run_dir(study_dir, mode, seed)
    if (out / "backtest_metrics.json").exists() and not force:
        print(f"[skip] {mode} seed {seed}: already done ({out})")
        return True
    command = [
        sys.executable, "-m", "experiments.relgt.run", "backtest",
        "--temporal-model", "relgt_db",
        "--context-mode", mode,
        "--seed", str(seed),
        "--output-dir", str(out),
        *extra,
    ]  # fmt: skip
    print(f"[run] {mode} seed {seed}: {' '.join(command)}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "stdout.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )  # fmt: skip
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    if code:
        print(f"[fail] {mode} seed {seed}: exit code {code}")
    return code == 0


def collect(study_dir: Path) -> pd.DataFrame:
    rows = []
    for mode in CONTEXT_MODES:
        for path in sorted(study_dir.glob(f"{mode}_seed*/backtest_metrics.json")):
            m = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "mode": mode,
                "description": DESCRIPTIONS[mode],
                "seed": m.get("seed", int(path.parent.name.rsplit("seed", 1)[1])),
                "score": m["score"],
                "macro_f1": m["macro_f1"],
                "micro_f1": m["micro_f1"],
                "selected_epoch": m.get("selected_epoch"),
                "num_neighbors": m.get("num_neighbors"),
                "epochs": m.get("epochs"),
                "run_dir": str(path.parent),
            }
            row.update({f"f1_{k}": v for k, v in m["per_action_f1"].items()})
            rows.append(row)
    return pd.DataFrame(rows)


def write_reports(study_dir: Path, extra: list[str]) -> None:
    frame = collect(study_dir)
    if frame.empty:
        print("No finished runs to report.")
        return
    frame.to_csv(study_dir / "results.csv", index=False)
    frame.to_json(study_dir / "results.json", orient="records", indent=2)

    metrics = [c for c in frame.columns if c in ("score", "macro_f1", "micro_f1") or c.startswith("f1_")]
    mean = frame.groupby("mode")[metrics].mean().reindex(
        [m for m in CONTEXT_MODES if m in set(frame["mode"])]
    )
    runs = frame.groupby("mode").size().reindex(mean.index)
    baseline = mean.loc["r2", "score"] if "r2" in mean.index else None

    lines = [
        "# Relational-context study (relgt_db)",
        "",
        f"Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}. "
        f"Extra backtest args: `{' '.join(extra) or '(none)'}`.",
        "",
        "Holdout metrics, mean over seeds. `d score vs r2` is the score difference to the full context.",
        "",
        "| mode | context | runs | score | macro F1 | micro F1 | d score vs r2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, row in mean.iterrows():
        delta = "" if baseline is None else f"{row['score'] - baseline:+.4f}"
        lines.append(
            f"| {mode} | {DESCRIPTIONS[mode]} | {runs[mode]} | {row['score']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['micro_f1']:.4f} | {delta} |"
        )
    action_cols = [c for c in mean.columns if c.startswith("f1_")]
    lines += [
        "",
        "Per-action F1 (mean over seeds):",
        "",
        "| mode | " + " | ".join(c[3:] for c in action_cols) + " |",
        "| --- |" + " --- |" * len(action_cols),
    ]
    for mode, row in mean.iterrows():
        lines.append(f"| {mode} | " + " | ".join(f"{row[c]:.4f}" for c in action_cols) + " |")
    lines += ["", "Individual runs are in `results.csv`.", ""]
    (study_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {study_dir / 'results.csv'}, results.json, summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--modes", nargs="+", choices=CONTEXT_MODES, default=list(CONTEXT_MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--force", action="store_true", help="rerun finished runs")
    parser.add_argument("--aggregate-only", action="store_true")
    args, extra = parser.parse_known_args()

    args.study_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    if not args.aggregate_only:
        for seed in args.seeds:
            for mode in args.modes:
                if not run_one(args.study_dir, mode, seed, extra, args.force):
                    failed.append(f"{mode}/seed{seed}")
    write_reports(args.study_dir, extra)
    if failed:
        sys.exit(f"Failed runs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
