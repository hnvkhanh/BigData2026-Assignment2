"""Backtest or predict with ``relgt_db`` on the database plus relationship tables.

Everything except the database and the token sampler is ``relgt_db``
(``experiments/relgt/db_run.py``): samples, labels, eligibility, the
chronological split, model, training, epoch selection and thresholds.
``--tables none --reserve 0`` is therefore plain ``relgt_db``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from experiments.relgt.db_run import load_database_context, run_db_backtest, run_db_prediction  # noqa: E402
from experiments.relgt.run import (  # noqa: E402
    DEFAULT_PREDICTION_CUTOFF,
    DEFAULT_TEST_INPUT,
    add_common_arguments,
    create_run_output_dir,
)
from experiments.relgt_db_relational.sampler import reserved_sampler  # noqa: E402
from experiments.relgt_db_relational.tables import DERIVED_TABLES, load_relational_database  # noqa: E402

TABLES_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def parse_tables(value: str) -> list[str]:
    if value == "none":
        return []
    if value == "all":
        return list(DERIVED_TABLES)
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(DERIVED_TABLES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown table(s) {', '.join(unknown)}; choose from {', '.join(DERIVED_TABLES)}"
        )
    return [name for name in DERIVED_TABLES if name in names]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backtest", "predict"):
        sub = subparsers.add_parser(command)
        add_common_arguments(sub)
        sub.add_argument(
            "--tables",
            type=parse_tables,
            default=list(DERIVED_TABLES),
            help=f"relationship tables to add: none, all (default) or a list of {', '.join(DERIVED_TABLES)}",
        )
        sub.add_argument(
            "--reserve",
            type=int,
            default=0,
            help="tokens per seed kept for relationship rows and the rows they point to (default: 0 = upstream sampling)",
        )
        if command == "predict":
            sub.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
            sub.add_argument("--cutoff", type=pd.Timestamp, default=pd.Timestamp(DEFAULT_PREDICTION_CUTOFF))
    args = parser.parse_args(argv)
    args.temporal_model = "relgt_db"
    if args.epochs < 1 or args.batch_size < 1 or args.num_layers < 1:
        parser.error("--epochs, --batch-size and --num-layers must be positive.")
    if args.hidden_dim % args.num_heads:
        parser.error("--hidden-dim must be divisible by --num-heads.")
    if not 2 <= args.num_neighbors <= 32_767:
        parser.error("--num-neighbors must be between 2 and 32767.")
    if not 0 <= args.reserve <= args.num_neighbors - 2:
        parser.error("--reserve must be between 0 and --num-neighbors - 2.")
    if args.reserve and not args.tables:
        parser.error("--reserve needs at least one relationship table.")
    if args.output_dir is None:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"{args.command}_{stamp}_{os.getpid()}"
    return args


def build_context(args: argparse.Namespace):
    database = load_relational_database(args.db_dir, args.tables)
    for name in args.tables:
        print(f"Relationship table {name}: {len(database.tables[name]):,} rows")
    context = load_database_context(args.db_dir, args.text_buckets, database)
    context.seed_sampler = reserved_sampler(context.graph, args.tables, args.reserve)
    context.signature = {
        "relationship_tables": list(args.tables),
        "reserve": args.reserve,
        "tables_version": TABLES_VERSION,
    }
    return context


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {args.db_dir}")
    args.output_dir = create_run_output_dir(args.command, args.output_dir)
    context = build_context(args)
    if args.command == "backtest":
        run_db_backtest(args, context)
    else:
        run_db_prediction(args, context)


if __name__ == "__main__":
    main()
