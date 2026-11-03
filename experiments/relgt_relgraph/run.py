"""RelGT / GNN on a relationship-augmented graph.

Each user-week sample is the ``experiments.relgt`` 14-day neighbourhood (the
user's own actions on PRs, issues and repositories) plus typed relationship
edges from ``graph.py``: owned repositories, assigned issues, @mentions, review
candidates, open authored PRs/issues, active threads, reviewed repositories,
unpaired pushed branches and interaction partners (other users). The model
attends over / passes messages along all of them. ``--relations none`` keeps
only the base neighbourhood and is the control.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.relational_lightgbm.features import load_relational_tables
from experiments.relgt.data import (
    LABELS,
    LOOKBACK_DAYS,
    GraphTokens,
    TrainingExamples,
    load_entity_vocabulary,
    load_events,
    snapshot_tokens,
)
from experiments.relgt.gnn_model import RelationalGNN
from experiments.relgt.model import RelGTInspired
from experiments.relgt.run import (
    _data_signature,
    calculate_metrics,
    choose_thresholds,
    chronological_splits,
    load_or_build_examples,
    prediction_example_indices,
    select_device,
)
from experiments.relgt_relational.run import (
    DAY_NS,
    cutoff_timestamps,
    fit_model,
    predict_probabilities,
)
from experiments.relgt_relgraph.graph import (
    NUM_NODE_TYPES,
    NUM_RELATIONS,
    RELATION_IDS,
    RELATIONS,
    GlobalIds,
    Snapshot,
    concat_tokens,
    num_relation_slots,
    relation_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_TEST_INPUT = ROOT / "assignment2" / "evaluation" / "test_input.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_START_CUTOFF = "2026-03-11 16:00:00"
DEFAULT_PREDICTION_CUTOFF = "2026-11-11 16:00:00"
ARCHITECTURES = ("relgt", "gnn")
GRAPH_VERSION = 1
TOKEN_FIELDS = ("entity_ids", "node_type_ids", "relation_ids", "hop_ids", "node_features", "relative_days")


def parse_relations(value: str) -> list[str]:
    keyword = value.strip().lower()
    if keyword == "none":
        return []
    if keyword == "all":
        return list(RELATIONS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(names) - set(RELATIONS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown relation(s): {', '.join(unknown)}. Choose from {', '.join(RELATIONS)}, 'all' or 'none'."
        )
    return [name for name in RELATIONS if name in names]


class RelationContext:
    """Tables and id spaces needed to build relationship tokens at any cutoff."""

    def __init__(self, db_dir: Path) -> None:
        users = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
        self.cohort = pd.Index(users.loc[users["is_cohort"], "user_id"].astype(np.int64))
        self.ids = GlobalIds(load_entity_vocabulary(db_dir), num_users=len(users))
        self.tables = load_relational_tables(db_dir)

    def tokens(self, cutoff: pd.Timestamp, user_ids: np.ndarray, relations: list[str]) -> GraphTokens:
        return relation_tokens(
            Snapshot.at(self.tables, cutoff), user_ids, relations, self.ids, self.tables.repo_owner, self.cohort
        )


def build_relation_tokens(
    context: RelationContext,
    examples: TrainingExamples,
    start_cutoff: pd.Timestamp,
    relations: list[str],
) -> GraphTokens:
    """Relationship tokens for every example, aligned with ``examples`` rows."""
    width, count = num_relation_slots(relations), len(examples.labels)
    out = GraphTokens(
        user_ids=examples.tokens.user_ids,
        entity_ids=np.zeros((count, width), np.int32),
        node_type_ids=np.zeros((count, width), np.uint8),
        relation_ids=np.zeros((count, width), np.uint8),
        hop_ids=np.zeros((count, width), np.uint8),
        node_features=np.zeros((count, width, 4), np.float32),
        relative_days=np.zeros((count, width), np.float32),
    )
    cutoffs = cutoff_timestamps(examples.cutoff_days, start_cutoff)
    for index, (day, cutoff) in enumerate(sorted(cutoffs.items()), start=1):
        rows = np.flatnonzero(examples.cutoff_days == day)
        part = context.tokens(cutoff, examples.tokens.user_ids[rows], relations)
        for name in TOKEN_FIELDS:
            getattr(out, name)[rows] = getattr(part, name)
        filled = (part.entity_ids != 0).sum(axis=1)
        print(f"Built relationship edges {index}/{len(cutoffs)}: {cutoff} (mean {filled.mean():.1f} edges/user)")
    return out


def load_or_build_relation_tokens(
    db_dir: Path,
    output_dir: Path,
    examples: TrainingExamples,
    start_cutoff: pd.Timestamp,
    relations: list[str],
    context: RelationContext | None,
) -> GraphTokens:
    cache_path = output_dir / "cache" / "relation_tokens.npz"
    metadata_path = cache_path.with_suffix(".json")
    signature = {
        **_data_signature(db_dir, start_cutoff, LOOKBACK_DAYS, LOOKBACK_DAYS),
        "graph_version": GRAPH_VERSION,
        "relations": relations,
        "examples": int(len(examples.labels)),
    }
    if cache_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) == signature:
            with np.load(cache_path, allow_pickle=False) as stored:
                print(f"Loaded cached relationship edges from {cache_path}")
                return GraphTokens(user_ids=examples.tokens.user_ids, **{name: stored[name] for name in TOKEN_FIELDS})
        print(f"Ignoring stale relationship-edge cache {cache_path}")
    tokens = build_relation_tokens(context or RelationContext(db_dir), examples, start_cutoff, relations)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **{name: getattr(tokens, name) for name in TOKEN_FIELDS})
    metadata_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    return tokens


def model_factory(args: argparse.Namespace, num_entities: int):
    def build() -> torch.nn.Module:
        if args.architecture == "gnn":
            return RelationalGNN(
                num_entities=num_entities,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                num_node_types=NUM_NODE_TYPES,
                num_relations=NUM_RELATIONS,
            )
        return RelGTInspired(
            num_entities=num_entities,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_local_layers=args.num_layers,
            dropout=args.dropout,
            num_node_types=NUM_NODE_TYPES,
            num_relations=NUM_RELATIONS,
        )

    return build


def augmented_examples(args: argparse.Namespace, output_dir: Path) -> tuple[TrainingExamples, int]:
    """Base examples with relationship tokens appended; returns them and the vocabulary size."""
    examples = load_or_build_examples(args.db_dir, output_dir, args.start_cutoff)
    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id"])
    num_entities = GlobalIds(load_entity_vocabulary(args.db_dir), len(users)).num_entities
    if not args.relations:
        return examples, num_entities
    extra = load_or_build_relation_tokens(
        args.db_dir, output_dir, examples, args.start_cutoff, args.relations, None
    )
    return (
        TrainingExamples(
            tokens=concat_tokens(examples.tokens, extra),
            labels=examples.labels,
            cutoff_days=examples.cutoff_days,
            num_entities=num_entities,
        ),
        num_entities,
    )


def _output_dir(command: str, requested: Path | None) -> Path:
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DEFAULT_OUTPUT_ROOT / f"{command}_{timestamp}_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output_dir}")
    return output_dir


def _branch(args: argparse.Namespace) -> str:
    return f"{args.architecture}_relgraph" if args.relations else args.architecture


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "architecture": args.architecture,
        "relations": args.relations,
        "relation_ids": {name: RELATION_IDS[name] for name in args.relations},
        "relation_slots": {name: RELATIONS[name][1] for name in args.relations},
        "graph_version": GRAPH_VERSION,
        "graph_lookback_days": LOOKBACK_DAYS,
        "eligibility": f"5 actions in the {LOOKBACK_DAYS} days before the cutoff (as experiments.relgt)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "select_epoch": args.select_epoch,
    }


def _no_root_features(count: int) -> np.ndarray:
    return np.zeros((count, 0), dtype=np.float32)


def run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _output_dir("backtest", args.output_dir)
    examples, num_entities = augmented_examples(args, output_dir)
    root = _no_root_features(len(examples.labels))
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    train = np.flatnonzero(examples.cutoff_days < train_end)
    calibration = np.flatnonzero((examples.cutoff_days >= train_end) & (examples.cutoff_days < calibration_end))
    holdout = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not len(train) or not len(calibration) or not len(holdout):
        raise ValueError("The chronological split produced an empty train, calibration or holdout set.")

    device = select_device(args.device)
    width = examples.tokens.entity_ids.shape[1]
    extra_width = num_relation_slots(args.relations)
    slot_fill = float((examples.tokens.entity_ids[:, width - extra_width:] != 0).mean()) if extra_width else 0.0
    print(f"Device: {device}; {args.architecture} over {width} neighbour slots ({len(args.relations)} relation types)")
    model, history = fit_model(
        examples, root, train, device, args, calibration, model_factory(args, num_entities)
    )
    calibration_probabilities = predict_probabilities(model, examples.tokens, root, calibration, device, args.batch_size)
    thresholds = choose_thresholds(examples.labels[calibration], calibration_probabilities)
    holdout_probabilities = predict_probabilities(model, examples.tokens, root, holdout, device, args.batch_size)
    metrics = calculate_metrics(examples.labels[holdout], holdout_probabilities, thresholds, examples.cutoff_days[holdout])

    edge_counts = {}
    if args.relations:
        relation_ids = examples.tokens.relation_ids
        for name in args.relations:
            per_user = (relation_ids == RELATION_IDS[name]).sum(axis=1)
            edge_counts[name] = {"mean_edges": float(per_user.mean()), "users_with_edge": float((per_user > 0).mean())}
    branch = _branch(args)
    results = {
        **metrics,
        "experiment": "relgt_relgraph",
        "temporal_model": branch,
        **_settings(args),
        "device": str(device),
        "threshold_selection": "per label, on calibration weeks only",
        "epoch_history": history,
        "relation_edge_stats": edge_counts,
        "relation_slot_fill": slot_fill,
        "split": {
            "train_examples": int(len(train)),
            "calibration_examples": int(len(calibration)),
            "holdout_examples": int(len(holdout)),
            "train_weeks": int(len(np.unique(examples.cutoff_days[train]))),
            "calibration_weeks": int(len(np.unique(examples.cutoff_days[calibration]))),
            "holdout_weeks": int(len(np.unique(examples.cutoff_days[holdout]))),
            "holdout_start": str(pd.Timestamp(int(calibration_end) * DAY_NS)),
        },
    }
    torch.save(model.state_dict(), output_dir / f"{branch}_model.pt")
    labels = examples.labels[holdout]
    pd.DataFrame(
        {
            "user_id": examples.tokens.user_ids[holdout],
            "cutoff": pd.to_datetime(examples.cutoff_days[holdout].astype(np.int64) * DAY_NS),
            **{f"actual_{label}": labels[:, column] for column, label in enumerate(LABELS)},
            **{f"{branch}_{label}": holdout_probabilities[:, column] for column, label in enumerate(LABELS)},
        }
    ).to_csv(output_dir / "holdout_probabilities.csv", index=False)
    (output_dir / "backtest_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nHOLDOUT {branch}: score {metrics['score']:.4f} (macro {metrics['macro_f1']:.4f}, micro {metrics['micro_f1']:.4f})")
    for label in LABELS:
        print(f"  {label:<18} F1 {metrics['per_action_f1'][label]:.4f}")
    print(f"Saved metrics to {output_dir / 'backtest_metrics.json'}")
    return results


def run_prediction(args: argparse.Namespace) -> None:
    output_dir = _output_dir("predict", args.output_dir)
    examples, num_entities = augmented_examples(args, output_dir)
    root = _no_root_features(len(examples.labels))
    usable = prediction_example_indices(examples.cutoff_days, args.cutoff)
    train_end, _ = chronological_splits(examples.cutoff_days[usable])
    train = usable[examples.cutoff_days[usable] < train_end]
    calibration = usable[examples.cutoff_days[usable] >= train_end]
    if not len(train) or not len(calibration):
        raise ValueError("No chronological training or calibration examples precede the prediction cutoff.")
    device = select_device(args.device)
    model, _ = fit_model(examples, root, train, device, args, calibration, model_factory(args, num_entities))
    thresholds = choose_thresholds(
        examples.labels[calibration],
        predict_probabilities(model, examples.tokens, root, calibration, device, args.batch_size),
    )

    target_users = pd.read_csv(args.test_input, dtype={"UserID": "int64"})
    if target_users["UserID"].duplicated().any():
        raise ValueError(f"{args.test_input} contains duplicate UserIDs.")
    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id", "github_user_id", "is_cohort"])
    cohort = users.loc[users["is_cohort"]]
    mapped = target_users["UserID"].map(cohort.set_index("github_user_id")["user_id"])
    if mapped.isna().any():
        raise ValueError(f"{int(mapped.isna().sum())} test UserID(s) are not in this release's cohort.")
    internal_ids = mapped.astype(np.int64).to_numpy()
    vocabulary = load_entity_vocabulary(args.db_dir)
    events = load_events(args.db_dir, cohort["user_id"].astype(int).tolist(), vocabulary)
    tokens = snapshot_tokens(events, args.cutoff, internal_ids, vocabulary)
    if args.relations:
        tokens = concat_tokens(tokens, RelationContext(args.db_dir).tokens(args.cutoff, internal_ids, args.relations))
    probabilities = predict_probabilities(
        model, tokens, _no_root_features(len(internal_ids)), np.arange(len(internal_ids)), device, args.batch_size
    )

    branch = _branch(args)
    predictions = pd.DataFrame({"UserID": target_users["UserID"]})
    probability_frame = predictions.copy()
    for column, label in enumerate(LABELS):
        predictions[label] = (probabilities[:, column] >= thresholds[label]).astype(np.int8)
        probability_frame[label] = probabilities[:, column]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    probability_frame.to_csv(output_dir / "probabilities.csv", index=False)
    torch.save(model.state_dict(), output_dir / f"{branch}_model.pt")
    metadata = {
        "experiment": "relgt_relgraph",
        "temporal_model": branch,
        **_settings(args),
        "prediction_cutoff_utc": str(args.cutoff),
        "training_examples": int(len(train)),
        "calibration_examples": int(len(calibration)),
        "thresholds": thresholds,
        "labels": list(LABELS),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions for {len(predictions):,} users to {output_dir / 'predictions.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backtest", "predict"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
        sub.add_argument("--output-dir", type=Path, default=None,
                         help="output folder (default: a new timestamped folder under outputs/)")
        sub.add_argument("--start-cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_START_CUTOFF))
        sub.add_argument("--architecture", choices=ARCHITECTURES, default="relgt")
        sub.add_argument(
            "--relations",
            type=parse_relations,
            default=list(RELATIONS),
            help=f"relationship edge types ({', '.join(RELATIONS)}), comma-separated, "
                 "'all' (default) or 'none' for the plain 14-day graph",
        )
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        sub.add_argument("--epochs", type=int, default=5)
        sub.add_argument("--batch-size", type=int, default=256)
        sub.add_argument("--hidden-dim", type=int, default=64)
        sub.add_argument("--num-heads", type=int, default=4)
        sub.add_argument("--num-layers", type=int, default=2)
        sub.add_argument("--dropout", type=float, default=0.1)
        sub.add_argument("--learning-rate", type=float, default=1e-3)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--select-epoch", action="store_true",
                         help="keep the epoch with the best calibration-week score instead of the last one")
    predict = subparsers.choices["predict"]
    predict.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    predict.add_argument("--cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF))
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "hidden_dim", "num_heads", "num_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.architecture == "relgt" and args.hidden_dim % args.num_heads:
        parser.error("--hidden-dim must be divisible by --num-heads.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1).")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "backtest":
        run_backtest(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
