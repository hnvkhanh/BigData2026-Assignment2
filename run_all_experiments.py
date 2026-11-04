"""Run every registered backtest experiment and write a comparison report.

Each experiment is launched as a subprocess with its own output folder under
``experiments/results/<tag>/runs/<name>/``. When all runs finish (or fail), the
holdout metrics are collected into ``results.csv``, ``results.json`` and a
ranked ``RESULTS.md``. All backtests use the same weekly cutoffs and the
official per-week scoring (mean over holdout weeks of ``evaluate.py``'s Score).

Examples (from the repository root):

    python run_all_experiments.py --list
    python run_all_experiments.py --dry-run
    python run_all_experiments.py --quick --tag smoke          # 1 epoch each
    python run_all_experiments.py --group relgt --group chunks
    python run_all_experiments.py --only gnn --skip frac50
    python run_all_experiments.py --tag full                   # everything
    python run_all_experiments.py --tag full --report-only     # rebuild report

With two or more GPUs the experiments run in parallel, one at a time per GPU
(or --jobs-per-gpu N), each pinned to its GPU with CUDA_VISIBLE_DEVICES. Use
--gpus 0,2 to pick GPUs or --no-parallel to run sequentially.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "experiments" / "results"
LABELS = (
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
)

RELGT = ["-m", "experiments.relgt.run", "backtest"]
CHUNKS = "experiments.relgt_activity_chunks.run"
GRU = "experiments.gru_relational_ple.run"
WEEKLY = "experiments.relgt_weekly.run"
RELATIONAL = ["-m", "experiments.relational_lightgbm.run", "backtest"]
RELGT_RELATIONAL = ["-m", "experiments.relgt_relational.run", "backtest"]
RELGRAPH = ["-m", "experiments.relgt_relgraph.run", "backtest", "--select-epoch"]
DB_RELATIONAL = ["-m", "experiments.relgt_db_relational.run", "backtest"]
DRIFT = ["-m", "experiments.drift_lightgbm.run", "backtest"]
NO_SUMMER = ["-m", "experiments.gnn_no_summer.run", "backtest"]
SEQUENCE = ["-m", "experiments.sequence_forecasters.run", "backtest"]
RT = ["-m", "experiments.relational_transformer.run", "backtest"]
ELIG90 = ["--eligibility-days", "90"]
THRESHOLDS = ["--chunk-thresholds", "20,200"]
# Never train on less than half the data; the unused half is abandoned, not
# moved into calibration, so every run calibrates on the same weeks.
MIN_TRAIN_FRACTION = 0.5
HALF = ["--train-fraction", str(MIN_TRAIN_FRACTION), "--discard-unused-train"]


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    command: list[str]  # arguments after the Python executable
    quick: list[str] = field(default_factory=list)  # extra args for --quick
    needs: str | None = None  # importable module required to run
    description: str = ""


def _e(name, group, command, quick=(), needs=None, description=""):
    return Experiment(name, group, list(command), list(quick), needs, description)


FAST_S2P2 = ["--s2p2-epochs", "1"]
FAST_EPOCH = ["--epochs", "1", "--batch-size", "64"]
FAST_DB = ["--epochs", "1", "--max-steps-per-epoch", "30"]
FAST_SEQUENCE = ["--epochs", "1"]

EXPERIMENTS: list[Experiment] = [
    # --- GRU ------------------------------------------------------------------
    _e("gru", "gru", ["-m", GRU], FAST_EPOCH,
       description="GRU + relational attention + PLE"),
    # --- RelGT / GNN (graph snapshots) ---------------------------------------
    _e("relgt", "relgt", RELGT, FAST_EPOCH, description="RelGT"),
    _e("relgt_user_emb", "relgt", [*RELGT, "--user-embedding"], FAST_EPOCH,
       description="RelGT + user embedding"),
    _e("relgt_frac50", "relgt", [*RELGT, *HALF], FAST_EPOCH,
       description="RelGT on 50% of examples"),
    _e("gnn", "relgt", [*RELGT, "--temporal-model", "gnn"], FAST_EPOCH,
       description="Relational GNN"),
    _e("gnn_user_emb", "relgt", [*RELGT, "--temporal-model", "gnn", "--user-embedding"],
       FAST_EPOCH, description="GNN + user embedding"),
    _e("gnn_frac50", "relgt", [*RELGT, "--temporal-model", "gnn", *HALF],
       FAST_EPOCH, description="GNN on 50% of examples"),
    _e("gnn_no_summer", "relgt", [*NO_SUMMER], FAST_EPOCH,
       description="GNN trained without Jun-Aug label weeks"),
    _e("gnn_no_summer_jul_aug", "relgt", [*NO_SUMMER, "--summer-months", "7,8"],
       FAST_EPOCH, description="GNN trained without Jul-Aug label weeks"),
    _e("gnn_no_summer_cal_sep", "relgt", [*NO_SUMMER, "--calibration-months", "9"],
       FAST_EPOCH, description="GNN without Jun-Aug, thresholds calibrated on September"),
    _e("gnn_september", "relgt", ["-m", "experiments.gnn_september.run", "backtest"],
       FAST_EPOCH, description="GNN trained only on September weeks"),
    _e("gnn_september_ensemble", "relgt",
       ["-m", "experiments.gnn_september_ensemble.run", "backtest"], FAST_EPOCH,
       description="Ensemble: GNN on all weeks + GNN on September only"),
    _e("relgt_elig90", "relgt", [*RELGT, *ELIG90], FAST_EPOCH,
       description="RelGT on the official test population (5 events in 90 days)"),
    _e("gnn_elig90", "relgt", [*RELGT, "--temporal-model", "gnn", *ELIG90], FAST_EPOCH,
       description="GNN on the official test population (5 events in 90 days)"),
    _e("hybrid_relgt_s2p2", "relgt", [*RELGT, "--temporal-model", "hybrid"],
       [*FAST_EPOCH, *FAST_S2P2], needs="easy_tpp", description="RelGT + S2P2 fusion"),
    _e("relgt_lightgbm", "relgt", [*RELGT, "--temporal-model", "relgt_lightgbm"],
       FAST_EPOCH, description="RelGT + LightGBM fusion"),
    _e("relgt_lightgbm_s2p2", "relgt",
       [*RELGT, "--temporal-model", "relgt_lightgbm_s2p2"],
       [*FAST_EPOCH, *FAST_S2P2], needs="easy_tpp",
       description="RelGT + LightGBM + S2P2 fusion"),
    # --- RelGT on the relational database ------------------------------------
    _e("relgt_db", "relgt_db", [*RELGT, "--temporal-model", "relgt_db"], FAST_DB,
       description="RelGT on the full relational database"),
    _e("relgt_db_local", "relgt_db",
       [*RELGT, "--temporal-model", "relgt_db", "--conv-type", "local"], FAST_DB,
       description="relgt_db, local attention"),
    _e("relgt_db_no_gnn", "relgt_db",
       [*RELGT, "--temporal-model", "relgt_db", "--ablate", "gnn"], FAST_DB,
       description="relgt_db ablation: no subgraph positional encoding"),
    _e("relgt_db_no_tfs", "relgt_db",
       [*RELGT, "--temporal-model", "relgt_db", "--ablate", "tfs"], FAST_DB,
       description="relgt_db ablation: no column features"),
    # --- Relational Transformer (stanford-star) on the relational database --
    _e("relational_transformer", "relational_transformer", RT, FAST_DB,
       description="Relational Transformer: cell tokens, column/feature/neighbor attention"),
    _e("relational_transformer_no_history", "relational_transformer",
       [*RT, "--history-weeks", "0"], FAST_DB,
       description="Relational Transformer ablation: no in-context history weeks"),
    _e("relational_transformer_mtp", "relational_transformer",
       [*RT, "--mtp-weight", "0.1"], FAST_DB,
       description="Relational Transformer + masked-token auxiliary loss"),
    # --- DM2026 sequence forecasters (LSTM / Informer / FEDformer) -------------
    _e("sequence_forecasters", "sequence", [*SEQUENCE], FAST_SEQUENCE,
       description="LSTM + Informer + FEDformer on 91-day activity, calibrated blend"),
    _e("sequence_forecasters_dm2026_blend", "sequence",
       [*SEQUENCE, "--fusion-weights", "lstm=0.4,informer=0.2,fedformer=0.4"], FAST_SEQUENCE,
       description="Sequence forecasters with DM2026's fixed 0.4/0.2/0.4 blend"),
    # --- Activity chunks ------------------------------------------------------
    _e("chunks", "chunks", ["-m", CHUNKS, *THRESHOLDS], FAST_EPOCH,
       description="GNN per prior-activity chunk + global"),
    _e("chunks_no_global", "chunks", ["-m", CHUNKS, "--no-global-baseline"], FAST_EPOCH,
       description="Chunk GNNs, tertile thresholds, no global model"),
    _e("chunks_balanced", "chunks", ["-m", CHUNKS, *THRESHOLDS, "--balance-chunks"],
       FAST_EPOCH, description="Chunk GNNs with equal training size"),
    _e("chunks_user_emb", "chunks", ["-m", CHUNKS, *THRESHOLDS, "--user-embedding"],
       FAST_EPOCH, description="Chunk GNNs + user embedding"),
    _e("chunks_frac50", "chunks", ["-m", CHUNKS, *THRESHOLDS, *HALF],
       FAST_EPOCH, description="Chunk GNNs on 50% of training"),
    # --- RelGT on weekly activity vectors ------------------------------------
    _e("relgt_weekly", "weekly", ["-m", WEEKLY], FAST_EPOCH,
       description="RelGT over the last 8 weekly vectors"),
    _e("relgt_weekly_4w", "weekly", ["-m", WEEKLY, "--weeks", "4"], FAST_EPOCH,
       description="RelGT over the last 4 weekly vectors"),
    _e("relgt_weekly_13w", "weekly", ["-m", WEEKLY, "--weeks", "13"], FAST_EPOCH,
       description="RelGT over the last 13 weekly vectors"),
    _e("relgt_weekly_user_emb", "weekly", ["-m", WEEKLY, "--user-embedding"],
       FAST_EPOCH, description="Weekly RelGT + user embedding"),
    _e("relgt_weekly_frac50", "weekly", ["-m", WEEKLY, *HALF], FAST_EPOCH,
       description="Weekly RelGT on 50% of examples"),
    _e("nn_weekly", "weekly", ["-m", WEEKLY, "--architecture", "nn"], FAST_EPOCH,
       description="MLP over the last 8 concatenated weekly vectors"),
    _e("nn_weekly_4w", "weekly", ["-m", WEEKLY, "--architecture", "nn", "--weeks", "4"],
       FAST_EPOCH, description="MLP over the last 4 weekly vectors"),
    _e("nn_weekly_13w", "weekly", ["-m", WEEKLY, "--architecture", "nn", "--weeks", "13"],
       FAST_EPOCH, description="MLP over the last 13 weekly vectors"),
    _e("nn_weekly_user_emb", "weekly",
       ["-m", WEEKLY, "--architecture", "nn", "--user-embedding"], FAST_EPOCH,
       description="Weekly MLP + user embedding"),
    _e("nn_weekly_frac50", "weekly", ["-m", WEEKLY, "--architecture", "nn", *HALF],
       FAST_EPOCH, description="Weekly MLP on 50% of examples"),
    _e("gnn_weekly", "weekly", ["-m", WEEKLY, "--architecture", "gnn"], FAST_EPOCH,
       description="GNN over the last 8 weekly vectors"),
    _e("gnn_weekly_4w", "weekly", ["-m", WEEKLY, "--architecture", "gnn", "--weeks", "4"],
       FAST_EPOCH, description="GNN over the last 4 weekly vectors"),
    _e("gnn_weekly_13w", "weekly", ["-m", WEEKLY, "--architecture", "gnn", "--weeks", "13"],
       FAST_EPOCH, description="GNN over the last 13 weekly vectors"),
    _e("gnn_weekly_user_emb", "weekly",
       ["-m", WEEKLY, "--architecture", "gnn", "--user-embedding"], FAST_EPOCH,
       description="Weekly GNN + user embedding"),
    _e("gnn_weekly_frac50", "weekly", ["-m", WEEKLY, "--architecture", "gnn", *HALF],
       FAST_EPOCH, description="Weekly GNN on 50% of examples"),
    # --- LightGBM + cross-action relational features -------------------------
    _e("relational_lgbm", "relational", RELATIONAL,
       description="LightGBM + owner/author/reviewer/assignee/sequence features"),
    # --- Drift-robust LightGBM (every variant is a branch) -------------------
    _e("drift_lgbm", "drift", DRIFT,
       description="LightGBM + history/regime/coverage features, anomaly masking, recency, refits"),
    _e("relgt_relational", "relational", RELGT_RELATIONAL, FAST_EPOCH,
       description="RelGT + relational features on the root token"),
    _e("gnn_relational", "relational", [*RELGT_RELATIONAL, "--architecture", "gnn"],
       FAST_EPOCH, description="GNN + relational features on the root token"),
    _e("relgt_relgraph", "relational", RELGRAPH, FAST_EPOCH,
       description="RelGT on the graph + typed relationship edges (owner, assignee, people, ...)"),
    _e("gnn_relgraph", "relational", [*RELGRAPH, "--architecture", "gnn"], FAST_EPOCH,
       description="GNN on the graph + typed relationship edges"),
    _e("relgt_db_relational", "relational", [*DB_RELATIONAL, "--epochs", "10"], FAST_DB,
       description="relgt_db on the database + relationship tables (owners, notifications, collaborations)"),
    _e("relgt_db_relational_reserve", "relational",
       [*DB_RELATIONAL, "--epochs", "10", "--reserve", "32"], FAST_DB,
       description="relgt_db_relational with 32 tokens reserved for relationship rows"),
]


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------
def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    chosen = []
    for experiment in EXPERIMENTS:
        if args.group and experiment.group not in args.group:
            continue
        if args.only and not any(token in experiment.name for token in args.only):
            continue
        if args.skip and any(token in experiment.name for token in args.skip):
            continue
        chosen.append(experiment)
    return chosen


def build_command(experiment: Experiment, run_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, *experiment.command, "--output-dir", str(run_dir)]
    if args.quick:
        command += experiment.quick
    command += args.extra
    return command


def find_metrics(run_dir: Path) -> Path | None:
    path = run_dir / "backtest_metrics.json"
    return path if path.is_file() else None


PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def detect_gpus() -> list[str]:
    """GPU ids this script may use: the visible ones, honouring CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [item.strip() for item in visible.split(",") if item.strip()]
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.run([nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=15, check=True)
            return [line.strip() for line in output.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


def run_experiment(
    experiment: Experiment, tag_dir: Path, args: argparse.Namespace, gpu: str | None = None
) -> dict:
    run_dir = tag_dir / "runs" / experiment.name
    log_path = tag_dir / "logs" / f"{experiment.name}.log"
    record = {"name": experiment.name, "status": "pending", "seconds": 0.0}

    if experiment.needs and importlib.util.find_spec(experiment.needs) is None:
        record["status"] = f"skipped (missing {experiment.needs})"
        return record
    if find_metrics(run_dir) and not args.rerun:
        record["status"] = "cached"
        return record

    command = build_command(experiment, run_dir, args)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    where = f" [GPU {gpu}]" if gpu is not None else ""
    log(f"\n=== {experiment.name}{where}: {' '.join(command[1:])}")
    if args.dry_run:
        record["status"] = "dry-run"
        return record

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                timeout=args.timeout, env=env,
            )
        record["status"] = "ok" if completed.returncode == 0 and find_metrics(run_dir) else (
            f"failed (exit {completed.returncode})"
        )
    except subprocess.TimeoutExpired:
        record["status"] = f"timeout ({args.timeout}s)"
    record["seconds"] = round(time.time() - started, 1)
    log(f"    {experiment.name}: {record['status']} in {record['seconds']}s (log: {log_path})")
    return record


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _metric_blocks(metrics: dict) -> list[tuple[str, dict]]:
    """Return (branch, metric dict) pairs: the headline plus any named branches."""
    if "score" in metrics:
        headline = metrics
    else:
        headline = metrics.get("holdout_metrics", {})
    blocks = [("", headline)] if "score" in headline else []
    for key in ("branch_holdout", "branch_metrics", "metrics_by_branch"):
        branches = metrics.get(key)
        if isinstance(branches, dict):
            for branch, value in branches.items():
                if isinstance(value, dict) and "score" in value and value is not headline:
                    blocks.append((branch, value))
    return blocks


def collect_rows(tag_dir: Path, records: dict[str, dict]) -> list[dict]:
    rows = []
    for experiment in EXPERIMENTS:
        record = records.get(experiment.name)
        run_dir = tag_dir / "runs" / experiment.name
        metrics_path = find_metrics(run_dir)
        if metrics_path is None:
            if record:
                rows.append({
                    "experiment": experiment.name, "group": experiment.group,
                    "branch": "", "status": record["status"],
                    "description": experiment.description,
                })
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        status = (record or {}).get("status", "ok")
        for branch, block in _metric_blocks(metrics):
            row = {
                "experiment": experiment.name, "group": experiment.group,
                "branch": branch, "status": status,
                "score": block["score"], "macro_f1": block["macro_f1"],
                "micro_f1": block["micro_f1"],
                "seconds": (record or {}).get("seconds", ""),
                "description": experiment.description,
            }
            for label in LABELS:
                row[f"f1_{label}"] = block.get("per_action_f1", {}).get(label)
            rows.append(row)
    return rows


def write_report(tag_dir: Path, rows: list[dict], args: argparse.Namespace) -> None:
    columns = ["experiment", "branch", "group", "status", "score", "macro_f1", "micro_f1",
               *[f"f1_{label}" for label in LABELS], "seconds", "description"]
    with (tag_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (tag_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    scored = sorted((r for r in rows if r.get("score") is not None),
                    key=lambda r: r["score"], reverse=True)
    unscored = [r for r in rows if r.get("score") is None]
    fmt = lambda v: "-" if v in (None, "") else f"{100 * v:.2f}"
    lines = [
        f"# Experiment comparison ({args.tag})",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}"
        + (" in **quick** mode (1 epoch; numbers are for smoke-testing only)" if args.quick else "")
        + ".",
        "",
        "All numbers are percentages on the chronological holdout weeks. Each holdout week "
        "is scored as its own submission with `assignment2/evaluation/evaluate.py` "
        "(Score = (macro-F1 + micro-F1) / 2) and averaged over weeks. Rows with a "
        "branch are sub-models reported by multi-model experiments (chunks, drift).",
        "",
        "## Ranking",
        "",
        "| # | Experiment | Branch | Score | Macro-F1 | Micro-F1 | " +
        " | ".join(LABELS) + " | Time (s) |",
        "|---|---|---|---|---|---|" + "---|" * len(LABELS) + "---|",
    ]
    for rank, row in enumerate(scored, start=1):
        lines.append(
            f"| {rank} | {row['experiment']} | {row['branch'] or '-'} | "
            f"**{fmt(row['score'])}** | {fmt(row['macro_f1'])} | {fmt(row['micro_f1'])} | "
            + " | ".join(fmt(row.get(f"f1_{label}")) for label in LABELS)
            + f" | {row.get('seconds') or '-'} |"
        )
    if scored:
        best = {label: max(scored, key=lambda r: r.get(f"f1_{label}") or 0) for label in LABELS}
        lines += ["", "## Best per activity", "", "| Activity | F1 | Experiment (branch) |", "|---|---|---|"]
        for label, row in best.items():
            lines.append(f"| {label} | {fmt(row.get(f'f1_{label}'))} | "
                         f"{row['experiment']}{' / ' + row['branch'] if row['branch'] else ''} |")
    if unscored:
        lines += ["", "## Not scored", "", "| Experiment | Status |", "|---|---|"]
        lines += [f"| {r['experiment']} | {r['status']} |" for r in unscored]
    lines += ["", "## Experiments", "", "| Name | Group | Description |", "|---|---|---|"]
    lines += [f"| {e.name} | {e.group} | {e.description} |" for e in EXPERIMENTS]
    (tag_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {tag_dir / 'RESULTS.md'} and results.csv/results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="results folder name under experiments/results/ "
                             "(reuse a tag to resume: finished runs are skipped)")
    parser.add_argument("--group", action="append", choices=sorted({e.group for e in EXPERIMENTS}),
                        help="only this group (repeatable)")
    parser.add_argument("--only", action="append", help="only names containing this text")
    parser.add_argument("--skip", action="append", help="skip names containing this text")
    parser.add_argument("--quick", action="store_true", help="1-epoch smoke settings")
    parser.add_argument("--rerun", action="store_true", help="rerun even if results exist")
    parser.add_argument("--timeout", type=int, default=None, help="seconds per experiment")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--gpus", help="comma-separated GPU ids to use (default: all visible)")
    parser.add_argument("--jobs-per-gpu", type=int, default=1,
                        help="experiments to run at once on each GPU (default 1)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="run one experiment at a time even with several GPUs")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild the report from existing runs")
    parser.add_argument("extra", nargs="*",
                        help="extra args appended to every command (after --)")
    args = parser.parse_args()

    for flag_index, token in enumerate(args.extra):
        if token == "--train-fraction" and flag_index + 1 < len(args.extra):
            if float(args.extra[flag_index + 1]) < MIN_TRAIN_FRACTION:
                parser.error(f"--train-fraction below {MIN_TRAIN_FRACTION} is not allowed")
    chosen = selected_experiments(args)
    if args.list:
        for e in chosen:
            print(f"{e.name:28s} {e.group:12s} {e.description}")
        return

    tag_dir = RESULTS_ROOT / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    if not args.report_only:
        gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else detect_gpus()
        parallel = len(gpus) > 1 and not args.no_parallel and not args.dry_run
        print(f"GPUs: {', '.join(gpus) if gpus else 'none'}; "
              + (f"running up to {len(gpus) * args.jobs_per_gpu} experiments in parallel"
                 if parallel else "running experiments sequentially"))
        if parallel:
            slots: queue.Queue[str] = queue.Queue()
            for _ in range(args.jobs_per_gpu):
                for gpu in gpus:
                    slots.put(gpu)
            stop = threading.Event()

            def worker(experiment: Experiment) -> dict:
                if stop.is_set():
                    return {"name": experiment.name, "status": "not run (stopped)", "seconds": 0.0}
                gpu = slots.get()
                try:
                    return run_experiment(experiment, tag_dir, args, gpu)
                finally:
                    slots.put(gpu)

            with ThreadPoolExecutor(max_workers=slots.qsize()) as pool:
                futures = [pool.submit(worker, e) for e in chosen]
                for future in futures:
                    record = future.result()
                    records[record["name"]] = record
                    if args.stop_on_error and record["status"].startswith(("failed", "timeout")):
                        stop.set()
        else:
            single_gpu = gpus[0] if len(gpus) == 1 else None
            for experiment in chosen:
                record = run_experiment(experiment, tag_dir, args, single_gpu)
                records[experiment.name] = record
                if args.stop_on_error and record["status"].startswith(("failed", "timeout")):
                    break
        if args.dry_run:
            return
    else:
        records = {e.name: {"status": "ok", "seconds": ""} for e in chosen}
    write_report(tag_dir, collect_rows(tag_dir, records), args)

    failed = [n for n, r in records.items() if r["status"].startswith(("failed", "timeout"))]
    if failed:
        print("Failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
