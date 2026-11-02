"""Backtest push and non-push relational GNN specialists."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from experiments.relgt.data import (
    EntityVocabulary,
    LABELS,
    GraphTokens,
    TrainingExamples,
    load_entity_vocabulary,
    load_events,
    snapshot_tokens,
)
from experiments.relgt.gnn_model import RelationalGNN
from experiments.relgt.run import (
    calculate_metrics,
    chronological_splits,
    load_or_build_examples,
    select_device,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = ROOT / "assignment2" / "data" / "db"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_START_CUTOFF = "2026-03-01 00:00:00"
LOOKBACK_DAYS = 14
PUSH_INDEX = LABELS.index("push")
PUSH_RELATION_ID = PUSH_INDEX + 1
PUSH_OUTPUTS = (PUSH_INDEX,)
NON_PUSH_OUTPUTS = tuple(index for index in range(len(LABELS)) if index != PUSH_INDEX)
PULL_REQUEST_OUTPUTS = tuple(
    LABELS.index(label)
    for label in ("open_pr", "pr_review", "pr_review_comment")
)
ISSUE_OUTPUTS = tuple(LABELS.index(label) for label in ("open_issue", "issue_comment"))
PULL_REQUEST_RELATIONS = tuple(index + 1 for index in PULL_REQUEST_OUTPUTS)
ISSUE_RELATIONS = tuple(index + 1 for index in ISSUE_OUTPUTS)


def build_tokens_for_relations(
    examples: TrainingExamples,
    events: pd.DataFrame,
    vocabulary: EntityVocabulary,
    relation_ids: tuple[int, ...],
) -> GraphTokens:
    """Rebuild snapshots using only selected actions and their graph context."""
    selected_events = events.loc[events["relation_id"].isin(relation_ids)]
    reference = examples.tokens
    fields = (
        "entity_ids",
        "node_type_ids",
        "relation_ids",
        "hop_ids",
        "node_features",
        "relative_days",
    )
    arrays = {
        field: np.empty_like(getattr(reference, field))
        for field in fields
    }

    for cutoff_day in np.unique(examples.cutoff_days):
        indices = np.flatnonzero(examples.cutoff_days == cutoff_day)
        snapshot = snapshot_tokens(
            selected_events,
            pd.Timestamp(int(cutoff_day), unit="D"),
            reference.user_ids[indices],
            vocabulary,
            lookback_days=LOOKBACK_DAYS,
        )
        for field in fields:
            arrays[field][indices] = getattr(snapshot, field)

    return GraphTokens(user_ids=reference.user_ids.copy(), **arrays)


def build_non_push_tokens(
    examples: TrainingExamples,
    events: pd.DataFrame,
    vocabulary: EntityVocabulary,
) -> GraphTokens:
    """Build snapshots without push events or push-derived repository context."""
    relation_ids = tuple(
        relation_id
        for relation_id in range(1, len(LABELS) + 1)
        if relation_id != PUSH_RELATION_ID
    )
    return build_tokens_for_relations(examples, events, vocabulary, relation_ids)


def fit_specialist(
    examples: TrainingExamples,
    tokens: GraphTokens,
    train_indices: np.ndarray,
    output_indices: tuple[int, ...],
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> RelationalGNN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = RelationalGNN(
        num_entities=examples.num_entities,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_indices=output_indices,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        shuffled = np.random.permutation(train_indices)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            inputs = _batch_inputs(tokens, batch_indices, device)
            targets = torch.as_tensor(
                examples.labels[np.ix_(batch_indices, output_indices)],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"BCE={total_loss / max(batches, 1):.5f}"
        )
    return model


def _batch_inputs(
    tokens: GraphTokens, indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(tokens.entity_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(
            tokens.node_type_ids[indices], dtype=torch.long, device=device
        ),
        torch.as_tensor(tokens.relation_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(tokens.hop_ids[indices], dtype=torch.long, device=device),
        torch.as_tensor(
            tokens.node_features[indices], dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            tokens.relative_days[indices], dtype=torch.float32, device=device
        ),
    )


def predict_specialist(
    model: RelationalGNN,
    tokens: GraphTokens,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            predictions.append(
                torch.sigmoid(model(*_batch_inputs(tokens, batch_indices, device)))
                .cpu()
                .numpy()
            )
    if not predictions:
        return np.empty((0, len(model.output_indices)), dtype=np.float32)
    return np.concatenate(predictions, axis=0)


def thresholds_for_outputs(
    labels: np.ndarray, probabilities: np.ndarray, output_indices: tuple[int, ...]
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    candidates = np.linspace(0.01, 0.99, 99)
    for column, label_index in enumerate(output_indices):
        scores = [
            f1_score(
                labels[:, label_index],
                probabilities[:, column] >= threshold,
                zero_division=0,
            )
            for threshold in candidates
        ]
        thresholds[LABELS[label_index]] = float(candidates[int(np.argmax(scores))])
    return thresholds


def metrics_for_outputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    output_indices: tuple[int, ...],
) -> dict[str, Any]:
    predictions = np.column_stack(
        [
            probabilities[:, column] >= thresholds[LABELS[label_index]]
            for column, label_index in enumerate(output_indices)
        ]
    ).astype(np.uint8)
    selected_labels = labels[:, output_indices]
    per_action = {
        LABELS[label_index]: float(
            f1_score(
                selected_labels[:, column],
                predictions[:, column],
                zero_division=0,
            )
        )
        for column, label_index in enumerate(output_indices)
    }
    macro = float(np.mean(list(per_action.values())))
    micro = float(
        f1_score(selected_labels.ravel(), predictions.ravel(), zero_division=0)
    )
    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "score": (macro + micro) / 2,
        "per_action_f1": per_action,
        "thresholds": thresholds,
    }


def combine_specialist_outputs(
    specialist_probabilities: dict[str, np.ndarray],
    specialist_thresholds: dict[str, dict[str, float]],
    output_labels: dict[str, tuple[int, ...]],
    selected_specialists: tuple[str, ...],
    row_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    probabilities = np.zeros((row_count, len(LABELS)), dtype=np.float32)
    thresholds: dict[str, float] = {}
    assigned_labels: set[int] = set()
    for specialist in selected_specialists:
        indices = output_labels[specialist]
        branch_probabilities = specialist_probabilities[specialist]
        if branch_probabilities.shape != (row_count, len(indices)):
            raise ValueError(
                f"{specialist} probabilities do not match its output labels."
            )
        for column, label_index in enumerate(indices):
            if label_index in assigned_labels:
                raise ValueError(
                    f"Multiple specialists predict label {LABELS[label_index]}."
                )
            assigned_labels.add(label_index)
            probabilities[:, label_index] = branch_probabilities[:, column]
            thresholds[LABELS[label_index]] = specialist_thresholds[specialist][
                LABELS[label_index]
            ]
    if assigned_labels != set(range(len(LABELS))):
        missing = [LABELS[i] for i in range(len(LABELS)) if i not in assigned_labels]
        raise ValueError(
            "Combined specialists must cover all labels exactly once; missing: "
            + ", ".join(missing)
        )
    return probabilities, thresholds


def _create_output_dir(requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        requested_dir.mkdir(parents=True, exist_ok=True)
        return requested_dir
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DEFAULT_OUTPUT_ROOT / f"backtest_{timestamp}_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output directory: {output_dir}")
    return output_dir


def run_backtest(args: argparse.Namespace) -> None:
    args.output_dir = _create_output_dir(args.output_dir)
    examples = load_or_build_examples(
        args.db_dir,
        args.output_dir,
        args.start_cutoff,
        lookback_days=LOOKBACK_DAYS,
    )
    train_end, calibration_end = chronological_splits(examples.cutoff_days)
    train_indices = np.flatnonzero(examples.cutoff_days < train_end)
    calibration_indices = np.flatnonzero(
        (examples.cutoff_days >= train_end)
        & (examples.cutoff_days < calibration_end)
    )
    holdout_indices = np.flatnonzero(examples.cutoff_days >= calibration_end)
    if not train_indices.size or not calibration_indices.size or not holdout_indices.size:
        raise ValueError(
            "The chronological split produced an empty train, calibration, "
            "or holdout set."
        )

    users = pd.read_parquet(args.db_dir / "Users.parquet", columns=["user_id", "is_cohort"])
    cohort_user_ids = (
        users.loc[users["is_cohort"], "user_id"].astype(np.int32).tolist()
    )
    vocabulary = load_entity_vocabulary(args.db_dir)
    events = load_events(args.db_dir, cohort_user_ids, vocabulary)
    device = select_device(args.device)

    specialist_specs: list[
        tuple[str, GraphTokens, tuple[int, ...], str, int]
    ] = [
        (
            "push_specialist",
            examples.tokens,
            PUSH_OUTPUTS,
            "all six action relations",
            args.seed,
        )
    ]
    if args.variant in {"all_non_push", "both"}:
        specialist_specs.append(
            (
                "non_push_specialist",
                build_non_push_tokens(examples, events, vocabulary),
                NON_PUSH_OUTPUTS,
                "all non-push action relations",
                args.seed + 1,
            )
        )
    if args.variant in {"grouped", "both"}:
        specialist_specs.extend(
            [
                (
                    "pr_specialist",
                    build_tokens_for_relations(
                        examples, events, vocabulary, PULL_REQUEST_RELATIONS
                    ),
                    PULL_REQUEST_OUTPUTS,
                    "pull-request-related actions only",
                    args.seed + 2,
                ),
                (
                    "issue_specialist",
                    build_tokens_for_relations(
                        examples, events, vocabulary, ISSUE_RELATIONS
                    ),
                    ISSUE_OUTPUTS,
                    "issue-related actions only",
                    args.seed + 3,
                ),
            ]
        )

    holdout_probabilities: dict[str, np.ndarray] = {}
    holdout_metrics: dict[str, dict[str, Any]] = {}
    specialist_thresholds: dict[str, dict[str, float]] = {}
    output_labels: dict[str, tuple[int, ...]] = {}
    input_descriptions: dict[str, str] = {}
    for name, tokens, output_indices, input_description, seed in specialist_specs:
        print(f"Training {name} for: {', '.join(LABELS[i] for i in output_indices)}")
        model = fit_specialist(
            examples, tokens, train_indices, output_indices, device, args, seed
        )
        calibration = predict_specialist(
            model, tokens, calibration_indices, device, args.batch_size
        )
        holdout = predict_specialist(
            model, tokens, holdout_indices, device, args.batch_size
        )
        branch_thresholds = thresholds_for_outputs(
            examples.labels[calibration_indices], calibration, output_indices
        )
        branch_metrics = metrics_for_outputs(
            examples.labels[holdout_indices],
            holdout,
            branch_thresholds,
            output_indices,
        )
        specialist_thresholds[name] = branch_thresholds
        output_labels[name] = output_indices
        input_descriptions[name] = input_description
        holdout_metrics[name] = branch_metrics
        holdout_probabilities[name] = holdout
        torch.save(model.state_dict(), args.output_dir / f"{name}_model.pt")

    combined_probability_sets: dict[str, np.ndarray] = {}
    combined_threshold_sets: dict[str, dict[str, float]] = {}
    combination_specs: list[tuple[str, tuple[str, ...]]] = []
    if args.variant in {"all_non_push", "both"}:
        combination_specs.append(
            ("combined_specialists", ("push_specialist", "non_push_specialist"))
        )
    if args.variant in {"grouped", "both"}:
        combination_specs.append(
            (
                "combined_grouped",
                ("push_specialist", "pr_specialist", "issue_specialist"),
            )
        )
    for combined_name, component_names in combination_specs:
        probabilities, branch_thresholds = combine_specialist_outputs(
            holdout_probabilities,
            specialist_thresholds,
            output_labels,
            component_names,
            len(holdout_indices),
        )
        combined_probability_sets[combined_name] = probabilities
        combined_threshold_sets[combined_name] = branch_thresholds
        holdout_metrics[combined_name] = calculate_metrics(
            examples.labels[holdout_indices], probabilities, branch_thresholds
        )

    primary_combination = (
        "combined_specialists"
        if "combined_specialists" in combined_probability_sets
        else "combined_grouped"
    )
    combined_probabilities = combined_probability_sets[primary_combination]
    thresholds = combined_threshold_sets[primary_combination]

    holdout_cutoffs = pd.to_datetime(
        examples.cutoff_days[holdout_indices], unit="D"
    )
    probability_columns: dict[str, np.ndarray] = {
        "user_id": examples.tokens.user_ids[holdout_indices],
        "cutoff": holdout_cutoffs,
        **{
            f"actual_{label}": examples.labels[holdout_indices, column]
            for column, label in enumerate(LABELS)
        },
    }
    for combined_name, probabilities in combined_probability_sets.items():
        probability_columns.update(
            {
                f"{combined_name}_{label}": probabilities[:, column]
                for column, label in enumerate(LABELS)
            }
        )
    for specialist_name, probabilities in holdout_probabilities.items():
        probability_columns.update(
            {
                f"{specialist_name}_{LABELS[label_index]}": probabilities[:, column]
                for column, label_index in enumerate(output_labels[specialist_name])
            }
        )
    probability_path = args.output_dir / "holdout_probabilities.csv"
    pd.DataFrame(probability_columns).to_csv(probability_path, index=False)

    results: dict[str, Any] = {
        "experiment": "Relational GNN action specialists",
        "variant": args.variant,
        "lookback_days": LOOKBACK_DAYS,
        "specialists": {
            specialist_name: {
                "input": input_descriptions[specialist_name],
                "output_labels": [
                    LABELS[index] for index in output_labels[specialist_name]
                ],
            }
            for specialist_name in output_labels
        },
        "combined_variants": {
            name: list(components) for name, components in combination_specs
        },
        "train_examples": int(len(train_indices)),
        "calibration_examples": int(len(calibration_indices)),
        "holdout_examples": int(len(holdout_indices)),
        "train_end_exclusive": str(pd.Timestamp(train_end, unit="D")),
        "holdout_start": str(pd.Timestamp(calibration_end, unit="D")),
        "device": str(device),
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "thresholds": thresholds,
        "holdout_metrics": holdout_metrics[primary_combination],
        "branch_holdout": holdout_metrics,
    }
    metrics_path = args.output_dir / "backtest_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved holdout metrics to {metrics_path}")
    print(f"Saved holdout probabilities to {probability_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--start-cutoff",
        type=pd.Timestamp,
        default=pd.Timestamp(DEFAULT_START_CUTOFF),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        choices=("all_non_push", "grouped", "both"),
        default="both",
        help=(
            "train the all-non-push baseline, PR/issue grouped specialists, "
            "or both variants"
        ),
    )
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "hidden_dim", "num_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1).")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    run_backtest(args)


if __name__ == "__main__":
    main()
