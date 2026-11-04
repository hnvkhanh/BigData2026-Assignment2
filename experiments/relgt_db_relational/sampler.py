"""RelGT token sampling that keeps room for relationship rows.

Upstream RelGT samples ``K - 1`` tokens uniformly from the seed's 1- and 2-hop
neighbourhood. An active user has thousands of push and PR rows there, so the
few relationship rows (an owned repo, a mention, a new PR on a reviewed repo)
are rarely drawn. ``sample_seed_reserved`` first fills up to ``reserve``
tokens with relationship rows linked to the seed, the most recent first and
alternating between relationship tables, each followed by the rows it points
to (the PR, issue, repository or partner user, at hop 2). The remaining tokens
are sampled as upstream does, without repeating the reserved rows.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from experiments.relgt.database import TemporalGraph
from experiments.relgt.db_sampler import _neighbors_before, local_edges, sample_seed


def node_tables(graph: TemporalGraph, nodes: np.ndarray) -> np.ndarray:
    return np.searchsorted(graph.table_offsets, nodes, side="right") - 1


def relationship_order(
    graph: TemporalGraph, rows: np.ndarray, tables: np.ndarray
) -> np.ndarray:
    """Most recent first within each table; tables take turns."""
    if not len(rows):
        return rows
    recency = np.lexsort((-graph.node_time[rows], tables))
    rows, tables = rows[recency], tables[recency]
    first = np.searchsorted(tables, tables, side="left")
    rank = np.arange(len(rows)) - first
    return rows[np.lexsort((-graph.node_time[rows], rank))]


def reserved_tokens(
    graph: TemporalGraph,
    seed_node: int,
    seed_time: int,
    reserve: int,
    relationship_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``reserve`` relationship rows (hop 1) and the rows they point to (hop 2)."""
    one_hop, _ = _neighbors_before(graph, np.array([seed_node]), seed_time)
    one_hop = np.unique(one_hop)
    tables = node_tables(graph, one_hop)
    keep = np.isin(tables, relationship_types)
    ordered = relationship_order(graph, one_hop[keep], tables[keep])
    direct = set(one_hop.tolist())

    nodes: list[int] = []
    hops: list[int] = []
    taken = {seed_node}
    for row in ordered:
        if len(nodes) >= reserve:
            break
        nodes.append(int(row))
        hops.append(1)
        taken.add(int(row))
        targets, _ = _neighbors_before(graph, np.array([row]), seed_time)
        for target in np.unique(targets):
            if len(nodes) >= reserve:
                break
            if int(target) not in taken:
                nodes.append(int(target))
                # A target can also be a direct neighbour of the seed (the user's own PR).
                hops.append(1 if target in direct else 2)
                taken.add(int(target))
    return np.asarray(nodes, dtype=np.int64), np.asarray(hops, dtype=np.int8)


def sample_seed_reserved(
    graph: TemporalGraph,
    seed_node: int,
    seed_time: int,
    num_tokens: int,
    rng: np.random.Generator,
    context: str = "r2",
    *,
    reserve: int,
    relationship_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``sample_seed`` with up to ``reserve`` tokens kept for relationship rows."""
    if not 0 <= reserve <= num_tokens - 2:
        raise ValueError("reserve must leave at least one uniformly sampled token.")
    kept, kept_hops = reserved_tokens(graph, seed_node, seed_time, reserve, relationship_types)
    rest, rest_hops, _ = sample_seed(
        graph, seed_node, seed_time, num_tokens - len(kept), rng, context, exclude=kept
    )
    nodes = np.concatenate([rest[:1], kept, rest[1:]])
    hops = np.concatenate([rest_hops[:1], kept_hops, rest_hops[1:]])
    return nodes, hops, local_edges(graph, nodes, seed_time)


def reserved_sampler(graph: TemporalGraph, tables: list[str], reserve: int):
    """A ``sample_tokens`` seed sampler, or upstream's when nothing is reserved."""
    if not reserve or not tables:
        return sample_seed
    types = np.asarray([graph.node_type(name) for name in tables], dtype=np.int64)
    return partial(sample_seed_reserved, reserve=reserve, relationship_types=types)
