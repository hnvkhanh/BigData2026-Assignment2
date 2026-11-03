"""RelGT's temporal 1/2-hop token sampler over the pkey/fkey graph.

Follows upstream ``gather_1_and_2_hop_with_seed_time`` / ``_process_one_seed``
(snap-stanford/relgt ``utils.py``):

1. collect 1-hop neighbors of the seed (at most 5,000, sampled at random) and
   their neighbors as 2-hop (at most 1,000 per 1-hop node), excluding the seed
   and nodes already at hop 1;
2. sample ``K - 1`` tokens uniformly from the union, without replacement when
   there are enough and with replacement otherwise;
3. if there are none, sample random nodes and mark them hop 3;
4. shuffle all tokens except the seed and keep the edges among them for the
   GNN positional encoding.

``context`` selects the relational-context configuration (see ``CONTEXT_MODES``).

Unlike upstream, every node, edge, and fallback node must be known strictly
before the seed time, so future rows and edges never enter a token set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .database import SECONDS_PER_DAY, TemporalGraph


MAX_ONE_HOP = 5_000
MAX_TWO_HOP_PER_NODE = 1_000
FALLBACK_HOP = 3
# r0: the user's own rows only (hop 1).
# r1: r0 plus other rows that directly touch them (hop 2, excluding Repositories).
# r2: full upstream 1/2-hop context, including Repositories.
# r3: control; random known rows, as many as r2 would have candidates.
CONTEXT_MODES = ("r0", "r1", "r2", "r3")


@dataclass
class TokenSets:
    """``K`` sampled tokens per seed plus the edges among them."""

    node_types: np.ndarray  # int8 [samples, K]
    node_rows: np.ndarray  # int32 [samples, K], row within its table
    hops: np.ndarray  # int8 [samples, K]
    relative_days: np.ndarray  # float32 [samples, K], seed time - node time
    edge_offsets: np.ndarray  # int64 [samples + 1]
    edges: np.ndarray  # int16 [2, total edges], local token positions

    def __len__(self) -> int:
        return len(self.node_types)


def _ranges(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened ``arange(start, end)`` positions and their owners."""
    lengths = ends - starts
    owners = np.repeat(np.arange(len(starts)), lengths)
    positions = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return starts[owners] + positions, owners


def _neighbors_before(
    graph: TemporalGraph, nodes: np.ndarray, seed_time: int
) -> tuple[np.ndarray, np.ndarray]:
    positions, owners = _ranges(graph.indptr[nodes], graph.indptr[nodes + 1])
    known = graph.edge_time[positions] < seed_time
    return graph.indices[positions[known]].astype(np.int64), owners[known]


def _cap_per_owner(
    owners: np.ndarray, cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Keep at most ``cap`` random entries for each owner."""
    if not len(owners) or np.bincount(owners).max() <= cap:
        return np.arange(len(owners))
    order = np.lexsort((rng.random(len(owners)), owners))
    sorted_owners = owners[order]
    first = np.searchsorted(sorted_owners, sorted_owners, side="left")
    return np.sort(order[np.arange(len(order)) - first < cap])


def _known_nodes(graph: TemporalGraph, seed_time: int) -> np.ndarray:
    """Count, per table, the rows known before the seed time.

    Rows are sorted by their time column, so the known rows of a table are a
    prefix; the result holds each prefix length.
    """
    counts = np.zeros(len(graph.table_names), dtype=np.int64)
    for table in range(len(graph.table_names)):
        start, end = graph.table_offsets[table], graph.table_offsets[table + 1]
        counts[table] = np.searchsorted(graph.node_time[start:end], seed_time, "left")
    return counts


def _fallback_nodes(
    graph: TemporalGraph, seed_time: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    known = _known_nodes(graph, seed_time)
    if not known.sum():
        raise ValueError("No database rows are known before the seed time.")
    draws = rng.integers(0, known.sum(), size=count)
    cumulative = np.cumsum(known)
    tables = np.searchsorted(cumulative, draws, side="right")
    rows = draws - (cumulative[tables] - known[tables])
    return graph.table_offsets[tables] + rows


def sample_seed(
    graph: TemporalGraph,
    seed_node: int,
    seed_time: int,
    num_tokens: int,
    rng: np.random.Generator,
    context: str = "r2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global node ids, hops, and local edges for one seed."""
    if context not in CONTEXT_MODES:
        raise ValueError(f"context must be one of {CONTEXT_MODES}.")
    one_hop, _ = _neighbors_before(graph, np.array([seed_node]), seed_time)
    one_hop = np.unique(one_hop)
    if len(one_hop) > MAX_ONE_HOP:
        one_hop = np.sort(rng.choice(one_hop, MAX_ONE_HOP, replace=False))

    two_hop, owners = _neighbors_before(graph, one_hop, seed_time)
    two_hop = np.unique(two_hop[_cap_per_owner(owners, MAX_TWO_HOP_PER_NODE, rng)])
    two_hop = two_hop[(two_hop != seed_node) & ~np.isin(two_hop, one_hop)]

    needed = num_tokens - 1
    full_count = len(one_hop) + len(two_hop)  # r2 candidate count
    if context == "r0":
        two_hop = two_hop[:0]
    elif context == "r1":
        repositories = graph.table_names.index("Repositories")
        types = np.searchsorted(graph.table_offsets, two_hop, side="right") - 1
        two_hop = two_hop[types != repositories]

    candidates = np.concatenate([one_hop, two_hop])
    candidate_hops = np.concatenate(
        [np.ones(len(one_hop), np.int8), np.full(len(two_hop), 2, np.int8)]
    )
    if context == "r3":
        # Same number of candidates as r2 would have (capped at what is drawn).
        candidates = _fallback_nodes(graph, seed_time, min(full_count, needed), rng)
        candidates = candidates[candidates != seed_node]
        candidate_hops = np.full(len(candidates), FALLBACK_HOP, np.int8)
    if len(candidates) >= needed:
        chosen = rng.choice(len(candidates), needed, replace=False)
    elif len(candidates):
        chosen = rng.choice(len(candidates), needed, replace=True)
    else:
        chosen = None
    if chosen is None:
        neighbors = _fallback_nodes(graph, seed_time, needed, rng)
        neighbor_hops = np.full(needed, FALLBACK_HOP, np.int8)
    else:
        order = rng.permutation(needed)
        neighbors = candidates[chosen][order]
        neighbor_hops = candidate_hops[chosen][order]
    nodes = np.concatenate([[seed_node], neighbors]).astype(np.int64)
    hops = np.concatenate([[0], neighbor_hops]).astype(np.int8)

    # A repeated node maps to its last token, as in upstream's local_map.
    unique_nodes, last_reversed = np.unique(nodes[::-1], return_index=True)
    position_of = len(nodes) - 1 - last_reversed
    targets, sources = _neighbors_before(graph, unique_nodes, seed_time)
    inside = np.isin(targets, unique_nodes)
    edges = np.stack(
        [
            position_of[sources[inside]],
            position_of[np.searchsorted(unique_nodes, targets[inside])],
        ]
    ).astype(np.int16)
    return nodes, hops, edges


def sample_tokens(
    graph: TemporalGraph,
    seed_nodes: np.ndarray,
    seed_times: np.ndarray,
    num_tokens: int,
    seed: int,
    context: str = "r2",
) -> TokenSets:
    """Sample a token set for every (seed node, seed time) pair."""
    if num_tokens < 2:
        raise ValueError("num_tokens must be at least 2.")
    if num_tokens > np.iinfo(np.int16).max:
        raise ValueError("num_tokens must fit in int16 edge positions.")
    count = len(seed_nodes)
    node_types = np.zeros((count, num_tokens), dtype=np.int8)
    node_rows = np.zeros((count, num_tokens), dtype=np.int32)
    hops = np.zeros((count, num_tokens), dtype=np.int8)
    relative_days = np.zeros((count, num_tokens), dtype=np.float32)
    edge_offsets = np.zeros(count + 1, dtype=np.int64)
    edge_batches: list[np.ndarray] = []
    for index, (seed_node, seed_time) in enumerate(zip(seed_nodes, seed_times)):
        if index and index % 20_000 == 0:
            print(f"Sampled token sets for {index:,}/{count:,} seeds")
        # One generator per (seed node, seed time) keeps samples independent of
        # batching and worker count.
        rng = np.random.default_rng([seed, int(seed_node), int(seed_time)])
        nodes, sample_hops, edges = sample_seed(
            graph, int(seed_node), int(seed_time), num_tokens, rng, context
        )
        types = np.searchsorted(graph.table_offsets, nodes, side="right") - 1
        node_types[index] = types
        node_rows[index] = nodes - graph.table_offsets[types]
        hops[index] = sample_hops
        relative_days[index] = (seed_time - graph.node_time[nodes]) / SECONDS_PER_DAY
        relative_days[index, 0] = 0.0
        edge_offsets[index + 1] = edge_offsets[index] + edges.shape[1]
        edge_batches.append(edges)
    return TokenSets(
        node_types=node_types,
        node_rows=node_rows,
        hops=hops,
        relative_days=relative_days,
        edge_offsets=edge_offsets,
        edges=(
            np.concatenate(edge_batches, axis=1)
            if edge_batches
            else np.zeros((2, 0), dtype=np.int16)
        ),
    )


def batch_edge_index(tokens: TokenSets, indices: np.ndarray) -> np.ndarray:
    """Concatenate the local edges of ``indices``, offset to batch positions."""
    positions, owners = _ranges(
        tokens.edge_offsets[indices], tokens.edge_offsets[indices + 1]
    )
    num_tokens = tokens.node_types.shape[1]
    return tokens.edges[:, positions].astype(np.int64) + owners * num_tokens
