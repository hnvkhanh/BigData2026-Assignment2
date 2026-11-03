"""Load the assignment's relational database as a temporal pkey/fkey graph.

This is the RelBench ``make_pkey_fkey_graph`` step that upstream RelGT relies
on, reimplemented without PyTorch Geometric, torch_frame, or RelBench:

- every table is a node type and every row is a node;
- every foreign key is an undirected edge between the referencing row and the
  referenced row;
- every remaining column is typed (numerical, categorical, text,
  multicategorical, timestamp) and materialized into arrays that the model
  encodes per table.

Rows, edges, and late columns carry the time they become known, so a sampler
and encoder can restrict them to what was known strictly before a cutoff.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# Missing or never-known times compare as later than every cutoff.
NEVER = np.iinfo(np.int64).max
SECONDS_PER_DAY = 86_400
MAX_CATEGORIES = 512
MIN_TEXT_TOKENS = 2
MAX_TEXT_TOKENS = 32
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STYPES = ("numerical", "categorical", "text", "multicategorical", "timestamp")


@dataclass(frozen=True)
class TableSchema:
    """Keys, row time, and leakage rules of one table.

    ``late_columns`` maps a column to the timestamp column that tells when its
    value became known, e.g. ``merged_by_id`` is known only at ``merged_at``.
    """

    name: str
    primary_key: str
    time_column: str
    foreign_keys: dict[str, str] = field(default_factory=dict)
    late_columns: dict[str, str] = field(default_factory=dict)
    drop_columns: tuple[str, ...] = ()
    stypes: dict[str, str] = field(default_factory=dict)
    multicategorical_separator: str = "|"


# Declared from assignment2/data/db/SCHEMA.md. Dropped columns are provenance
# ids, commit hashes, GitHub ids that are not row indices, or exact duplicates
# of kept columns (Pushes.ref is ref_type + branch).
GITHUB_SCHEMA: tuple[TableSchema, ...] = (
    TableSchema(
        "Users",
        "user_id",
        "first_seen_at",
        drop_columns=("github_user_id",),
    ),
    TableSchema(
        "Repositories",
        "repo_id",
        "first_seen_at",
        drop_columns=("github_repo_id", "org_id"),
    ),
    TableSchema(
        "PullRequests",
        "pr_id",
        "first_seen_at",
        foreign_keys={
            "repo_id": "Repositories",
            "author_id": "Users",
            "merged_by_id": "Users",
        },
        late_columns={
            "merged_at": "merged_at",
            "merged_by_id": "merged_at",
            "closed_at": "closed_at",
        },
        drop_columns=("github_pr_id", "head_sha", "base_sha", "head_repo_id"),
    ),
    TableSchema(
        "Issues",
        "issue_id",
        "first_seen_at",
        foreign_keys={
            "repo_id": "Repositories",
            "author_id": "Users",
            "pull_request_id": "PullRequests",
            "assignee_id": "Users",
        },
        late_columns={"closed_at": "closed_at", "state_reason": "closed_at"},
        drop_columns=("github_issue_id",),
        stypes={"labels": "multicategorical"},
    ),
    TableSchema(
        "IssueComments",
        "comment_id",
        "created_at",
        foreign_keys={
            "issue_id": "Issues",
            "repo_id": "Repositories",
            "author_id": "Users",
        },
        drop_columns=("github_comment_id", "event_id"),
    ),
    TableSchema(
        "PRReviews",
        "review_id",
        "first_seen_at",
        foreign_keys={
            "pr_id": "PullRequests",
            "repo_id": "Repositories",
            "reviewer_id": "Users",
        },
        late_columns={"dismissed_at": "dismissed_at"},
        drop_columns=("github_review_id", "commit_id", "event_id"),
    ),
    TableSchema(
        "PRReviewComments",
        "comment_id",
        "created_at",
        foreign_keys={
            "pr_id": "PullRequests",
            "repo_id": "Repositories",
            "author_id": "Users",
            "pull_request_review_id": "PRReviews",
            "in_reply_to_id": "PRReviewComments",
        },
        drop_columns=("github_comment_id", "commit_id", "event_id"),
    ),
    TableSchema(
        "Pushes",
        "push_id",
        "created_at",
        foreign_keys={"user_id": "Users", "repo_id": "Repositories"},
        drop_columns=("github_push_id", "event_id", "ref"),
    ),
)


@dataclass
class TableFeatures:
    """Materialized column features of one table, grouped by semantic type.

    Every ``*_known_at`` array names, per feature column, an index into
    ``known_at`` (or -1 when the value is known as soon as the row is).
    """

    name: str
    num_rows: int
    numerical_columns: list[str]
    numerical: np.ndarray  # float32 [rows, columns], NaN when missing
    numerical_known_at: np.ndarray
    categorical_columns: list[str]
    categorical: np.ndarray  # int32 [rows, columns], 0 = missing / rare
    categorical_cardinalities: list[int]  # including the missing code
    categorical_known_at: np.ndarray
    text_columns: list[str]  # text and multicategorical, both token-hashed
    text: list[np.ndarray]  # int32 [rows, tokens] per column, 0 = padding
    text_known_at: np.ndarray
    timestamp_columns: list[str]
    timestamps: np.ndarray  # int64 seconds [rows, columns], NEVER if missing
    timestamp_known_at: np.ndarray
    known_at: np.ndarray  # int64 seconds [rows, late-time columns]


@dataclass
class TemporalGraph:
    """Undirected pkey/fkey graph in CSR form, sorted by edge time per node."""

    table_names: list[str]
    table_offsets: np.ndarray  # first global node id of each table, plus total
    node_time: np.ndarray  # int64 seconds the row becomes known
    indptr: np.ndarray  # int64 [nodes + 1]
    indices: np.ndarray  # int32 neighbor global ids
    edge_time: np.ndarray  # int64 seconds both rows and the key are known
    edge_types: list[tuple[str, str, str]]  # (table, column, referenced table)

    @property
    def num_nodes(self) -> int:
        return int(self.table_offsets[-1])

    def node_type(self, table: str) -> int:
        return self.table_names.index(table)

    def global_ids(self, table: str, rows: np.ndarray) -> np.ndarray:
        return np.asarray(rows, dtype=np.int64) + self.table_offsets[self.node_type(table)]


@dataclass
class RelationalDatabase:
    schema: tuple[TableSchema, ...]
    tables: dict[str, pd.DataFrame]

    def table_schema(self, name: str) -> TableSchema:
        for table in self.schema:
            if table.name == name:
                return table
        raise KeyError(name)


def to_seconds(values: pd.Series) -> np.ndarray:
    """Convert a datetime column to int64 seconds; NaT becomes ``NEVER``."""
    times = pd.to_datetime(values)
    seconds = times.to_numpy(dtype="datetime64[s]").astype(np.int64)
    seconds[times.isna().to_numpy()] = NEVER
    return seconds


def timestamp_seconds(timestamp: pd.Timestamp) -> int:
    return int(pd.Timestamp(timestamp).value // 1_000_000_000)


def load_database(
    db_dir: Path, schema: tuple[TableSchema, ...] = GITHUB_SCHEMA
) -> RelationalDatabase:
    """Read every declared table and validate keys against the schema."""
    tables: dict[str, pd.DataFrame] = {}
    for table in schema:
        path = db_dir / f"{table.name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Table {table.name} is missing: {path}")
        frame = pd.read_parquet(path)
        frame = frame.drop(columns=[c for c in table.drop_columns if c in frame])
        keys = frame[table.primary_key].to_numpy()
        if not np.array_equal(keys, np.arange(len(frame))):
            raise ValueError(
                f"{table.name}.{table.primary_key} must be the row index 0..n-1."
            )
        missing = {table.time_column, *table.foreign_keys, *table.late_columns.values()}
        missing -= set(frame.columns)
        if missing:
            raise ValueError(f"{table.name} is missing columns: {sorted(missing)}")
        # The sampler finds the rows known at a time as a prefix of the table.
        if (np.diff(to_seconds(frame[table.time_column])) < 0).any():
            raise ValueError(f"{table.name} must be sorted by {table.time_column}.")
        tables[table.name] = frame
    for table in schema:
        for column, target in table.foreign_keys.items():
            if target not in tables:
                raise ValueError(f"{table.name}.{column} references unknown {target}.")
    return RelationalDatabase(schema=schema, tables=tables)


def infer_stype(series: pd.Series) -> str | None:
    """Propose a semantic type like RelBench's ``get_stype_proposal``."""
    if series.nunique(dropna=True) <= 1:
        return None
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if series.nunique(dropna=True) <= MAX_CATEGORIES:
        return "categorical"
    return "text"


def stable_hash(token: str, buckets: int) -> int:
    """Map a token to ``1..buckets-1``; 0 is reserved for padding."""
    return zlib.crc32(token.encode("utf-8")) % (buckets - 1) + 1


def _hash_token_lists(
    token_lists: pd.Series, buckets: int, max_tokens: int | None = None
) -> np.ndarray:
    lengths = token_lists.str.len().fillna(0).to_numpy(dtype=np.int64)
    if max_tokens is None:
        positive = lengths[lengths > 0]
        typical = int(np.ceil(np.percentile(positive, 95))) if len(positive) else 0
        max_tokens = int(np.clip(typical, MIN_TEXT_TOKENS, MAX_TEXT_TOKENS))
    hashed = np.zeros((len(token_lists), max_tokens), dtype=np.int32)
    exploded = token_lists.explode()
    exploded = exploded[exploded.notna()]
    if exploded.empty:
        return hashed
    rows = np.repeat(np.arange(len(token_lists)), lengths)
    positions = np.arange(len(rows)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    keep = positions < max_tokens
    codes, uniques = pd.factorize(exploded.to_numpy(dtype=object))
    bucket_of = np.fromiter(
        (stable_hash(str(token), buckets) for token in uniques),
        dtype=np.int32,
        count=len(uniques),
    )
    hashed[rows[keep], positions[keep]] = bucket_of[codes[keep]]
    return hashed


def hash_text(series: pd.Series, buckets: int) -> np.ndarray:
    """Hash lower-cased alphanumeric word pieces of each cell."""
    # Cells are truncated before tokenizing; only the first tokens are kept.
    text = series.astype("string").fillna("").str.slice(0, 8 * MAX_TEXT_TOKENS)
    return _hash_token_lists(text.str.lower().str.findall(TOKEN_PATTERN), buckets)


def hash_multicategorical(
    series: pd.Series, column: str, separator: str, buckets: int
) -> np.ndarray:
    """Hash each separated value as one whole token, namespaced by column."""
    values = series.astype("string").fillna("").str.lower().str.split(separator)
    tokens = values.map(
        lambda items: [f"{column}={item.strip()}" for item in items if item.strip()]
    )
    return _hash_token_lists(tokens, buckets)


def materialize_table(
    frame: pd.DataFrame, table: TableSchema, text_buckets: int
) -> TableFeatures:
    """Type every non-key column and convert it to model-ready arrays."""
    key_columns = {table.primary_key, *table.foreign_keys}
    late_time_columns = sorted(set(table.late_columns.values()))
    known_at = np.stack(
        [to_seconds(frame[column]) for column in late_time_columns], axis=1
    ) if late_time_columns else np.zeros((len(frame), 0), dtype=np.int64)

    def known_index(column: str) -> int:
        source = table.late_columns.get(column)
        return late_time_columns.index(source) if source is not None else -1

    groups: dict[str, list[str]] = {stype: [] for stype in STYPES}
    for column in frame.columns:
        if column in key_columns:
            continue
        stype = table.stypes.get(column) or infer_stype(frame[column])
        if stype is None:
            continue
        if stype not in STYPES:
            raise ValueError(f"Unknown stype {stype!r} for {table.name}.{column}.")
        groups[stype].append(column)

    numerical = np.zeros((len(frame), len(groups["numerical"])), dtype=np.float32)
    for index, column in enumerate(groups["numerical"]):
        values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
        values = np.sign(values) * np.log1p(np.abs(values))
        finite = np.isfinite(values)
        if finite.any():
            mean = values[finite].mean()
            std = values[finite].std() or 1.0
            values = (values - mean) / std
        numerical[:, index] = values

    categorical = np.zeros((len(frame), len(groups["categorical"])), dtype=np.int32)
    cardinalities: list[int] = []
    for index, column in enumerate(groups["categorical"]):
        values = frame[column].astype("string")
        top = values.value_counts(dropna=True).index[: MAX_CATEGORIES - 1]
        lookup = {value: code for code, value in enumerate(top, start=1)}
        categorical[:, index] = (
            values.map(lookup).fillna(0).to_numpy(dtype=np.int32)
        )
        cardinalities.append(len(lookup) + 1)

    text_columns = groups["text"] + groups["multicategorical"]
    text = [hash_text(frame[column], text_buckets) for column in groups["text"]]
    text += [
        hash_multicategorical(
            frame[column], column, table.multicategorical_separator, text_buckets
        )
        for column in groups["multicategorical"]
    ]

    timestamps = (
        np.stack([to_seconds(frame[column]) for column in groups["timestamp"]], axis=1)
        if groups["timestamp"]
        else np.zeros((len(frame), 0), dtype=np.int64)
    )
    return TableFeatures(
        name=table.name,
        num_rows=len(frame),
        numerical_columns=groups["numerical"],
        numerical=numerical,
        numerical_known_at=np.array(
            [known_index(c) for c in groups["numerical"]], dtype=np.int64
        ),
        categorical_columns=groups["categorical"],
        categorical=categorical,
        categorical_cardinalities=cardinalities,
        categorical_known_at=np.array(
            [known_index(c) for c in groups["categorical"]], dtype=np.int64
        ),
        text_columns=text_columns,
        text=text,
        text_known_at=np.array([known_index(c) for c in text_columns], dtype=np.int64),
        timestamp_columns=groups["timestamp"],
        timestamps=timestamps,
        timestamp_known_at=np.array(
            [known_index(c) for c in groups["timestamp"]], dtype=np.int64
        ),
        known_at=known_at,
    )


def materialize_features(
    database: RelationalDatabase, text_buckets: int
) -> dict[str, TableFeatures]:
    return {
        table.name: materialize_table(database.tables[table.name], table, text_buckets)
        for table in database.schema
    }


def build_temporal_graph(database: RelationalDatabase) -> TemporalGraph:
    """Connect each foreign key to its referenced row in both directions.

    An edge is known once both rows are known and, for a late key such as
    ``merged_by_id``, once its value is known. Keys never known are dropped.
    """
    table_names = [table.name for table in database.schema]
    sizes = [len(database.tables[name]) for name in table_names]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    node_time = np.concatenate(
        [
            to_seconds(database.tables[table.name][table.time_column])
            for table in database.schema
        ]
    )

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    times: list[np.ndarray] = []
    edge_types: list[tuple[str, str, str]] = []
    for table_index, table in enumerate(database.schema):
        frame = database.tables[table.name]
        for column, target in table.foreign_keys.items():
            keys = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
            present = np.isfinite(keys)
            target_size = len(database.tables[target])
            if (keys[present] < 0).any() or (keys[present] >= target_size).any():
                raise ValueError(f"{table.name}.{column} has keys outside {target}.")
            rows = np.flatnonzero(present)
            child = rows + offsets[table_index]
            parent = keys[present].astype(np.int64) + offsets[table_names.index(target)]
            known = np.maximum(node_time[child], node_time[parent])
            late_source = table.late_columns.get(column)
            if late_source is not None:
                late_known = to_seconds(frame[late_source])[rows]
                known = np.maximum(known, late_known)
            keep = known != NEVER
            sources.append(child[keep])
            targets.append(parent[keep])
            times.append(known[keep])
            edge_types.append((table.name, column, target))

    source = np.concatenate(sources + targets)
    target = np.concatenate(targets + sources)
    edge_time = np.concatenate(times + times)
    order = np.lexsort((edge_time, source))
    counts = np.bincount(source, minlength=int(offsets[-1]))
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return TemporalGraph(
        table_names=table_names,
        table_offsets=offsets,
        node_time=node_time,
        indptr=indptr,
        indices=target[order].astype(np.int32),
        edge_time=edge_time[order],
        edge_types=edge_types,
    )
