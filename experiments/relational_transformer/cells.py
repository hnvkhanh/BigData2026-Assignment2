"""Turn database rows into Relational Transformer cell tokens.

Upstream RT (stanford-star/relational-transformer) tokenizes a relational
context as *cells*: every non-key, non-null value of every sampled row is one
token carrying its value, its semantic type, and its column. Rows are tied
together only through their foreign-key parents (``f2p``), which the
attention masks use. Keys are never tokens.

The prediction target is a row of a task table. Here that table is
``UserWeeks``: one row per (user, weekly cutoff) with a foreign key to the
user, the cutoff as a datetime cell, and one number cell per label. The six
label cells of the row being predicted are masked targets; earlier weeks of
the same user whose label window closed before the cutoff are in-context rows
with their labels visible, as RT puts labelled task rows in its context.

Every value must be known strictly before the cutoff: rows come from the
``relgt_db`` sampler (rows and edges known before the seed time), late columns
are hidden until their ``known_at`` time, timestamps after the cutoff are
dropped, a foreign-key parent is linked only once the edge is known, and a
history label counts an action only once it was ingested before the cutoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from experiments.relgt.data import HORIZON_DAYS, LABELS
from experiments.relgt.database import (
    MAX_TEXT_TOKENS,
    NEVER,
    SECONDS_PER_DAY,
    RelationalDatabase,
    TableFeatures,
    TemporalGraph,
    to_seconds,
)


SEM_NUMBER, SEM_CATEGORY, SEM_TEXT, SEM_DATETIME = range(4)
SEM_TYPES = ("number", "category", "text", "datetime")
TASK_TABLE = "UserWeeks"
DATETIME_FEATURES = 5
WEEK_SECONDS = HORIZON_DAYS * SECONDS_PER_DAY
# Local row ids: 0 is the task row being predicted, 1..H its history rows,
# and context rows follow. Padding cells use PAD_ROW.
PAD_ROW = -1


def datetime_features(times: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Age before the cutoff plus time-of-day and day-of-week phase."""
    times = np.asarray(times, dtype=np.int64)
    age_days = (np.asarray(cutoffs, dtype=np.int64) - times) / SECONDS_PER_DAY
    day = (times % SECONDS_PER_DAY) / SECONDS_PER_DAY * 2 * math.pi
    # 1970-01-01 was a Thursday; the offset only shifts the phase.
    week = (times % WEEK_SECONDS) / WEEK_SECONDS * 2 * math.pi
    return np.stack(
        [np.log1p(np.maximum(age_days, 0.0)), np.sin(day), np.cos(day), np.sin(week), np.cos(week)],
        axis=-1,
    ).astype(np.float32)


@dataclass
class Cells:
    """A flat list of cells; ``owner`` indexes the row entry each belongs to."""

    owner: np.ndarray  # int64 [n]
    column: np.ndarray  # int64 [n], global column id
    sem: np.ndarray  # int8 [n]
    number: np.ndarray  # float32 [n]
    category: np.ndarray  # int64 [n], 0 = none
    text: np.ndarray  # int32 [n, MAX_TEXT_TOKENS], 0 = padding
    datetime: np.ndarray  # float32 [n, DATETIME_FEATURES]
    target_label: np.ndarray  # int8 [n], label index of a masked target, else -1

    @classmethod
    def make(
        cls,
        owner: np.ndarray,
        column: np.ndarray,
        sem: int,
        number: np.ndarray | None = None,
        category: np.ndarray | None = None,
        text: np.ndarray | None = None,
        datetime: np.ndarray | None = None,
        target_label: np.ndarray | None = None,
    ) -> Cells:
        count = len(owner)
        column = np.broadcast_to(np.asarray(column, dtype=np.int64), (count,))
        return cls(
            owner=np.asarray(owner, dtype=np.int64),
            column=column.copy(),
            sem=np.full(count, sem, dtype=np.int8),
            number=(
                np.zeros(count, np.float32) if number is None else np.asarray(number, np.float32)
            ),
            category=(
                np.zeros(count, np.int64) if category is None else np.asarray(category, np.int64)
            ),
            text=np.zeros((count, MAX_TEXT_TOKENS), np.int32) if text is None else text,
            datetime=(
                np.zeros((count, DATETIME_FEATURES), np.float32) if datetime is None else datetime
            ),
            target_label=(
                np.full(count, -1, np.int8)
                if target_label is None
                else np.asarray(target_label, np.int8)
            ),
        )

    @classmethod
    def concat(cls, parts: list[Cells]) -> Cells:
        return cls(
            **{
                field.name: np.concatenate([getattr(part, field.name) for part in parts])
                for field in fields(cls)
            }
        )

    def __len__(self) -> int:
        return len(self.owner)


@dataclass
class TableColumns:
    """Global column ids of one table's feature columns."""

    numerical: np.ndarray
    categorical: np.ndarray
    category_offsets: np.ndarray  # first shared category id of each column
    text: np.ndarray
    timestamps: np.ndarray
    marker: int  # cell for a row whose values are all missing or unknown


def _known(
    features: TableFeatures, rows: np.ndarray, cutoffs: np.ndarray, known_index: np.ndarray
) -> np.ndarray:
    """Whether each late column is known before the cutoff (index -1 = always)."""
    known_at = np.concatenate(
        [features.known_at[rows], np.full((len(rows), 1), np.iinfo(np.int64).min)], axis=1
    )
    return known_at[:, known_index] < cutoffs[:, None]


@dataclass
class ActivityHistory:
    """The earliest time each (user, week, label) action became known.

    Weeks are the 7-day windows ``[week_starts[w], week_starts[w] + 7 days)``.
    An action is known once both its row and its target entity are available.
    A history label at cutoff ``c`` is 1 only if such an action is known
    before ``c``, so ingestion lag can never reveal a late action.
    """

    user_ids: np.ndarray  # sorted int32 Users row indices
    week_starts: np.ndarray  # int64 seconds
    first_known: np.ndarray  # int64 [users, weeks, labels], NEVER without an action

    @classmethod
    def build(
        cls,
        events: pd.DataFrame,
        user_ids: np.ndarray,
        first_week_start: int,
        last_cutoff: int,
    ) -> ActivityHistory:
        user_ids = np.unique(np.asarray(user_ids, dtype=np.int32))
        num_weeks = (last_cutoff - first_week_start) // WEEK_SECONDS
        if num_weeks < 1:
            raise ValueError("The history must cover at least one complete week.")
        week_starts = first_week_start + WEEK_SECONDS * np.arange(num_weeks, dtype=np.int64)
        first_known = np.full((len(user_ids), num_weeks, len(LABELS)), NEVER, dtype=np.int64)
        action = to_seconds(events["action_at"])
        known = np.maximum(
            to_seconds(events["available_at"]), to_seconds(events["entity_available_at"])
        )
        week = np.floor_divide(action - first_week_start, WEEK_SECONDS)
        event_users = events["user_id"].to_numpy(dtype=np.int32)
        user = np.searchsorted(user_ids, event_users)
        in_users = user < len(user_ids)
        in_users[in_users] = user_ids[user[in_users]] == event_users[in_users]
        keep = in_users & (week >= 0) & (week < num_weeks) & (action != NEVER)
        label = events["relation_id"].to_numpy(dtype=np.int64) - 1
        np.minimum.at(first_known, (user[keep], week[keep], label[keep]), known[keep])
        return cls(user_ids, week_starts, first_known)

    def rows(
        self, user_ids: np.ndarray, cutoffs: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ``count`` most recent weeks whose label window closed by each cutoff.

        Returns labels uint8 [n, count, labels], week starts int64 [n, count]
        and a presence mask bool [n, count]; most recent week first.
        """
        user = np.minimum(np.searchsorted(self.user_ids, user_ids), len(self.user_ids) - 1)
        if not np.array_equal(self.user_ids[user], user_ids):
            raise ValueError("History was not built for every requested user.")
        closed = np.searchsorted(self.week_starts + WEEK_SECONDS, cutoffs, side="right")
        weeks = closed[:, None] - 1 - np.arange(count)[None, :]
        present = weeks >= 0
        safe = np.maximum(weeks, 0)
        known = self.first_known[user[:, None], safe] < np.asarray(cutoffs)[:, None, None]
        labels = (known & present[..., None]).astype(np.uint8)
        starts = np.where(present, self.week_starts[safe], 0).astype(np.int64)
        return labels, starts, present


class CellBuilder:
    """Build padded RT cell sequences for batches of (context rows, cutoff)."""

    def __init__(
        self,
        database: RelationalDatabase,
        graph: TemporalGraph,
        features: dict[str, TableFeatures],
    ) -> None:
        self.graph = graph
        self.features = [features[name] for name in graph.table_names]
        self.column_names: list[str] = []
        self.column_tables: list[str] = []
        self.table_columns: list[TableColumns] = []
        next_category = 1
        for name, table_features in zip(graph.table_names, self.features):
            cardinalities = np.asarray(table_features.categorical_cardinalities, dtype=np.int64)
            self.table_columns.append(
                TableColumns(
                    numerical=self._add_columns(name, table_features.numerical_columns),
                    categorical=self._add_columns(name, table_features.categorical_columns),
                    category_offsets=(
                        next_category
                        + np.concatenate([[0], np.cumsum(cardinalities)[:-1]]).astype(np.int64)
                        if len(cardinalities)
                        else np.zeros(0, np.int64)
                    ),
                    text=self._add_columns(name, table_features.text_columns),
                    timestamps=self._add_columns(name, table_features.timestamp_columns),
                    marker=int(self._add_columns(name, ["<row>"])[0]),
                )
            )
            next_category += int(cardinalities.sum())
        self.num_categories = next_category
        self.cutoff_column = int(self._add_columns(TASK_TABLE, ["cutoff"])[0])
        self.label_columns = self._add_columns(TASK_TABLE, list(LABELS))

        # Foreign-key parents of every row and when each link becomes known,
        # computed as ``build_temporal_graph`` computes edge times.
        offsets = graph.table_offsets
        self.parents: list[np.ndarray] = []
        self.parent_known: list[np.ndarray] = []
        for index, name in enumerate(graph.table_names):
            schema = database.table_schema(name)
            frame = database.tables[name]
            child_time = graph.node_time[offsets[index] : offsets[index + 1]]
            parents = np.full((len(frame), len(schema.foreign_keys)), -1, dtype=np.int64)
            known = np.full(parents.shape, NEVER, dtype=np.int64)
            for column_index, (column, target) in enumerate(schema.foreign_keys.items()):
                keys = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
                present = np.isfinite(keys)
                parent = keys[present].astype(np.int64) + offsets[graph.table_names.index(target)]
                edge_time = np.maximum(child_time[present], graph.node_time[parent])
                late = schema.late_columns.get(column)
                if late is not None:
                    edge_time = np.maximum(edge_time, to_seconds(frame[late])[present])
                parents[present, column_index] = parent
                known[present, column_index] = edge_time
            self.parents.append(parents)
            self.parent_known.append(known)
        self.max_parents = max([1, *(p.shape[1] for p in self.parents)])

    def _add_columns(self, table: str, columns: list[str]) -> np.ndarray:
        start = len(self.column_names)
        self.column_names += [f"{column} of {table}" for column in columns]
        self.column_tables += [table] * len(columns)
        return np.arange(start, len(self.column_names), dtype=np.int64)

    @property
    def num_columns(self) -> int:
        return len(self.column_names)

    def table_cells(
        self, table: int, rows: np.ndarray, cutoffs: np.ndarray, owners: np.ndarray
    ) -> Cells:
        """Cells of ``rows`` of one table as known at their cutoffs."""
        features = self.features[table]
        columns = self.table_columns[table]
        parts: list[Cells] = []
        if len(columns.numerical):
            values = features.numerical[rows]
            present = np.isfinite(values) & _known(
                features, rows, cutoffs, features.numerical_known_at
            )
            i, j = np.nonzero(present)
            parts.append(
                Cells.make(owners[i], columns.numerical[j], SEM_NUMBER, number=values[i, j])
            )
        if len(columns.categorical):
            codes = features.categorical[rows].astype(np.int64)
            present = (codes > 0) & _known(features, rows, cutoffs, features.categorical_known_at)
            i, j = np.nonzero(present)
            parts.append(
                Cells.make(
                    owners[i],
                    columns.categorical[j],
                    SEM_CATEGORY,
                    category=columns.category_offsets[j] + codes[i, j],
                )
            )
        if len(columns.text):
            known = _known(features, rows, cutoffs, features.text_known_at)
            for index, all_tokens in enumerate(features.text):
                tokens = all_tokens[rows]
                (i,) = np.nonzero(tokens.any(axis=1) & known[:, index])
                padded = np.zeros((len(i), MAX_TEXT_TOKENS), dtype=np.int32)
                width = min(tokens.shape[1], MAX_TEXT_TOKENS)
                padded[:, :width] = tokens[i, :width]
                parts.append(Cells.make(owners[i], columns.text[index], SEM_TEXT, text=padded))
        if len(columns.timestamps):
            values = features.timestamps[rows]
            present = (values != NEVER) & (values < cutoffs[:, None])
            present &= _known(features, rows, cutoffs, features.timestamp_known_at)
            i, j = np.nonzero(present)
            parts.append(
                Cells.make(
                    owners[i],
                    columns.timestamps[j],
                    SEM_DATETIME,
                    datetime=datetime_features(values[i, j], cutoffs[i]),
                )
            )
        if not parts:
            return Cells.make(np.zeros(0, np.int64), np.zeros(0, np.int64), SEM_NUMBER)
        return Cells.concat(parts)

    def build(
        self,
        context_nodes: np.ndarray,
        cutoffs: np.ndarray,
        history_labels: np.ndarray,
        history_starts: np.ndarray,
        history_present: np.ndarray,
        max_cells: int,
    ) -> dict[str, np.ndarray]:
        """Pad the cells of a batch to ``[batch, cells]`` arrays.

        ``context_nodes`` int64 [batch, K] are global row ids with the seed
        user first; repeated rows are kept once. Cells are ordered by local
        row (task row, history rows, then context rows in sampler order) and
        truncated to ``max_cells``, so the six targets always survive.
        """
        batch_size = len(context_nodes)
        num_history = history_labels.shape[1]
        first_context = 1 + num_history
        graph = self.graph

        entry_batch, entry_local, entry_node = [], [], []
        for b, nodes in enumerate(np.asarray(context_nodes, dtype=np.int64)):
            _, first = np.unique(nodes, return_index=True)
            unique = nodes[np.sort(first)]
            if unique[0] != nodes[0]:
                raise AssertionError("The seed row must stay first.")
            entry_batch.append(np.full(len(unique), b, np.int64))
            entry_local.append(first_context + np.arange(len(unique), dtype=np.int64))
            entry_node.append(unique)
        entry_batch = np.concatenate(entry_batch)
        entry_local = np.concatenate(entry_local)
        entry_node = np.concatenate(entry_node)
        entry_cutoff = np.asarray(cutoffs, dtype=np.int64)[entry_batch]
        entry_table = np.searchsorted(graph.table_offsets, entry_node, side="right") - 1
        entry_row = entry_node - graph.table_offsets[entry_table]
        num_entries = len(entry_node)

        # Foreign-key parents as local row ids, when the parent is in context.
        entry_parents = np.full((num_entries, self.max_parents), PAD_ROW, dtype=np.int64)
        lookup = entry_batch * graph.num_nodes + entry_node
        order = np.argsort(lookup)
        parts: list[Cells] = []
        for table in np.unique(entry_table):
            (entries,) = np.nonzero(entry_table == table)
            rows, row_cutoffs = entry_row[entries], entry_cutoff[entries]
            parts.append(self.table_cells(int(table), rows, row_cutoffs, entries))
            parents = self.parents[table][rows]
            if not parents.shape[1]:
                continue
            known = (parents >= 0) & (self.parent_known[table][rows] < row_cutoffs[:, None])
            keys = entry_batch[entries, None] * graph.num_nodes + parents
            position = np.searchsorted(lookup, keys, sorter=order)
            position = order[np.minimum(position, num_entries - 1)]
            found = known & (lookup[position] == keys)
            entry_parents[entries, : parents.shape[1]] = np.where(
                found, entry_local[position], PAD_ROW
            )
        context_cells = Cells.concat(parts)
        empty = np.flatnonzero(np.bincount(context_cells.owner, minlength=num_entries) == 0)
        markers = np.array(
            [self.table_columns[t].marker for t in entry_table[empty]], dtype=np.int64
        )
        cells = [context_cells, Cells.make(empty, markers, SEM_NUMBER)]

        # Task rows: one target row and the history rows per example; each has
        # the seed user (local row ``first_context``) as its only parent.
        batch_index = np.arange(batch_size, dtype=np.int64)
        task_batch = np.repeat(batch_index, first_context)
        task_local = np.tile(np.arange(first_context, dtype=np.int64), batch_size)
        task_entries = num_entries + np.arange(len(task_batch), dtype=np.int64)
        task_parents = np.full((len(task_batch), self.max_parents), PAD_ROW, dtype=np.int64)
        task_parents[:, 0] = first_context
        cutoffs = np.asarray(cutoffs, dtype=np.int64)
        target_owner = task_entries[task_local == 0]
        cells.append(
            Cells.make(
                target_owner,
                self.cutoff_column,
                SEM_DATETIME,
                datetime=datetime_features(cutoffs, cutoffs),
            )
        )
        cells.append(
            Cells.make(
                np.repeat(target_owner, len(LABELS)),
                np.tile(self.label_columns, batch_size),
                SEM_NUMBER,
                target_label=np.tile(np.arange(len(LABELS)), batch_size),
            )
        )
        history_b, history_h = np.nonzero(history_present)
        history_owner = num_entries + history_b * first_context + 1 + history_h
        cells.append(
            Cells.make(
                history_owner,
                self.cutoff_column,
                SEM_DATETIME,
                datetime=datetime_features(
                    history_starts[history_b, history_h], cutoffs[history_b]
                ),
            )
        )
        cells.append(
            Cells.make(
                np.repeat(history_owner, len(LABELS)),
                np.tile(self.label_columns, len(history_owner)),
                SEM_NUMBER,
                number=history_labels[history_b, history_h].reshape(-1),
            )
        )
        cells = Cells.concat(cells)
        all_batch = np.concatenate([entry_batch, task_batch])
        all_local = np.concatenate([entry_local, task_local])
        all_parents = np.concatenate([entry_parents, task_parents])

        cell_batch = all_batch[cells.owner]
        cell_local = all_local[cells.owner]
        order = np.lexsort((cells.column, cell_local, cell_batch))
        cell_batch = cell_batch[order]
        starts = np.searchsorted(cell_batch, batch_index)
        rank = np.arange(len(order)) - starts[cell_batch]
        keep = rank < max_cells
        order, cell_batch, rank = order[keep], cell_batch[keep], rank[keep]
        length = int(rank.max()) + 1

        def padded(values: np.ndarray, fill: float | int) -> np.ndarray:
            out = np.full((batch_size, length, *values.shape[1:]), fill, dtype=values.dtype)
            out[cell_batch, rank] = values[order]
            return out

        target_label = padded(cells.target_label, -1)
        target_b, target_s = np.nonzero(target_label >= 0)
        target_pos = np.zeros((batch_size, len(LABELS)), dtype=np.int64)
        target_pos[target_b, target_label[target_b, target_s]] = target_s
        if len(target_b) != batch_size * len(LABELS):
            raise AssertionError("Every example needs all six target cells.")

        num_rows = int(all_local.max()) + 1
        row_parents = np.full((batch_size, num_rows, self.max_parents), PAD_ROW, dtype=np.int64)
        row_parents[all_batch, all_local] = all_parents
        return {
            "row": padded(cell_local, PAD_ROW),
            "row_parents": row_parents,
            "column": padded(cells.column, 0),
            "sem": padded(cells.sem.astype(np.int64), SEM_NUMBER),
            "number": padded(cells.number, 0.0),
            "category": padded(cells.category, 0),
            "text": padded(cells.text, 0),
            "datetime": padded(cells.datetime, 0.0),
            "is_target": target_label >= 0,
            "padding": padded(np.zeros(len(cells), dtype=bool), True),
            "target_pos": target_pos,
        }
