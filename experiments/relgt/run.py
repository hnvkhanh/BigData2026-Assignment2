"""Relational Graph Transformer (RelGT, Dwivedi et al. 2025) for the 6 weekly activity labels.

Port of the official implementation (https://github.com/snap-stanford/relgt: model.py,
local_module.py, codebook.py, encoders.py, utils.py) to this data, in plain PyTorch (no PyTorch
Geometric, torch_frame or relbench).

Relational entity graph: one node per row of each of the 8 tables and one undirected edge per
foreign key (Pushes.user_id -> Users, IssueComments.issue_id -> Issues, ...).
PullRequests.merged_by_id is left out because it is only known after the merge.

Examples are the baseline's: (cohort user, week start s) for users passing the selection rule,
6 labels, the same weeks and time split (experiments/baseline/run.py). A node of table T is
visible to an example only if its row time is before s - lag_T, the staleness each table has at
the test week, so nothing after the cutoff leaks in.

Tokens (utils.py): the seed user plus K-1 nodes sampled uniformly from its visible 1- and 2-hop
neighbours (without replacement if there are enough, with replacement otherwise, random visible
nodes as hop 3 if there are none). Each token is encoded by five elements, concatenated and
mixed by an MLP: node type, hop distance, time relative to s, the row's own columns, and a GIN
positional encoding of the sampled subgraph (random inputs). Then a local transformer over the
K tokens with an attention readout to the seed, plus global attention from the seed to an EMA
vector-quantised codebook of centroids, a feed-forward block and a 2-layer head with 6 outputs.

Differences from the official code:
  * Row features are hand-picked columns per table (flags, counts, one-hot categories, text as
    its length) encoded by a per-table residual MLP, instead of torch_frame column-type
    inference, GloVe text embeddings and torch_frame's ResNet. Columns known only later
    (merged_at, closed_at, dismissed_at) enter as "happened before the cutoff" flags.
  * The neighbour caps (5,000 1-hop nodes, 1,000 per 1-hop node) are applied after the time
    filter rather than before it.
  * The codebook assignment buffer covers the cohort users (the only seeds), not every node.
  * Attention dropout is applied only in training.
  * Defaults are smaller than the paper's (channels 128 vs 512, K 100 vs 300) to fit one 12 GB
    GPU, and train longer at a higher learning rate (20 epochs at 3e-4 vs 10 at 1e-4, which was
    still improving when it stopped); see --help.
  * As in the other experiments, the epoch is chosen on validation (mean average precision over
    the 6 labels, the official multilabel metric), thresholds maximise F1 there, and the final
    model is retrained on all labelled weeks.

Using that pr_review_comment ~ subset of pr_review (see experiments/gnn_review), both off by default:
  * --habit: the user's review-commenting habit over the 90 days before the cutoff (the 11
    features of experiments/gnn_review) is encoded by an MLP and added to the seed token.
  * --review-aux: a 7th training output, "pr_review or pr_review_comment" this week.
Every run also tunes the thresholds jointly on the score over the validation weeks and keeps
them if they beat the per-label F1 thresholds.

--history (off by default) adds the baseline's per-user history features (counts over 7 / 14 /
30 / 90 days and overall, days since the last event, active days, repositories, open PRs and
issues; experiments/baseline/run.py build_week) to the seed token the same way as --habit. Each
source uses its own staleness cutoff, the same rule as the graph's visibility.

--path-budget path (off by default) replaces the uniform token sample by per-path budgets. A
path is the table of a 1-hop node, or the tables of (1-hop node, 2-hop node), e.g. User ->
Pushes or User -> PRReviews -> PullRequests. The K-1 tokens are split evenly over the paths the
seed has, a path with fewer candidates passing its spare tokens to the others, so a frequent
path (pushes) no longer crowds out a rare one (reviews). --path-order recent fills each path
with its latest rows instead of random ones.

Relation-aware extensions, each off by default:
  * --path-token: the token's path (above; the seed and the random fallback have their own ids)
    is embedded as a sixth token element. Type and hop alone cannot tell a repository reached
    through pushes from one reached through comments.
  * --rel-pe: the GIN positional encoding passes GINE messages ReLU(x_src + e_rel), e_rel a
    learned embedding of the edge's FK relation (e.g. PRReviews.pr_id) and direction.
  * --attn-bias: a learned per-head bias on the local attention logits, one term for the hop
    pair of the two tokens and one for the relation of each FK edge between them (as in
    Graphormer's spatial / edge encodings); zero-initialised.
  * --action-edges sequence|repo|both: the FK graph links a user's actions only through the user
    or a shared PR / issue / repository node, which is rarely sampled with them. This adds token
    edges between a seed's sampled activity rows of the same actor, each to the next one in time
    (sequence) or the next one in the same repository (repo), as new relations for the GIN, and
    for --rel-pe / --attn-bias. Built among the visible tokens only, so they leak nothing.

--graph-structure chain (default fk) changes the graph itself: each activity row is linked to its
actor's previous and next row of the same table, and keeps its other FKs (repo, PR, issue, review,
reply) but not its actor FK. For each example the user is linked instead to the latest visible row
of each of its chains, and any other user's row among the tokens to that user, who joins the tokens
with it (sample_walk, subgraph_edges). Tokens come from a breadth-first walk from the user (nearest
hops first, at most --max-hops), so the newest actions are nearest and the walk goes back in time.
--chain-seed latest makes the user's latest visible action token 0 instead of the user: the walk
starts there and at the latest action of each other type, the user is a hop 1 token,
and the readout is taken from that action. Its time is kept (a user seed has time 0), so the
seed token carries how long ago the user was last active. --num-seeds N makes the N latest actions
the seeds (all hop 0, part of the K tokens, fewer if the user has fewer): the walk starts from all of
them, the readout and the codebook query use the mean of the seed tokens, the other tokens are
attention-weighted on that mean, and --history / --habit features join every seed token.
--chain-seed latest-per-type seeds at the latest visible action of each action type instead (PR, issue,
issue comment, review, review comment, push: up to 6 seeds, one per type the user has, newest first), so
a heavy pusher's seeds are not all pushes; it is handled like several seeds (mean readout), --num-seeds
is not used with it.
--related-rows N (chain only, off by default) samples the context of every token that is an issue,
issue comment, PR review or PR review comment: up to N rows per such token reached through its foreign
keys (RELATED: an issue's comments and PR, a comment's issue and the other comments on it, a review's
PR, line comments and the PR's other reviews, a review comment's PR, review, the comment it replies
to, its replies and the PR's other review comments), nearest first, newest first. The seeds' related
rows are taken before the walk, within half of the budget left after the seeds. In the walk a row's
related rows add to its cost, so it is taken only if they fit too, and they come right after its hop.
Related rows are tokens at their source's hop + their FK steps, bring in their owners as walk rows do,
have their own related rows taken while budget is left, and the walk goes on from them.
--recency-attn (off by default) makes the model attend more to recent actions without changing the
tokens: the attention weight of an action D days older than the example's newest action token is
multiplied by (1 + D / --recency-days) ^ -decay, in the local attention and in the readout. Decay is
learned, starts at 1 and is at least 0, one per head and one for the readout; Users and Repositories
are not actions and keep their weight. The learned values are saved in metrics.json.

--exclude-push hides every Pushes row from the graph, so tokens are never pushes and a repository
reached only through pushes is not reached. The 6 labels (push included), the eligibility rule and
the --history features do not change. A user with only pushes has no neighbours left and gets the
random fallback tokens.

--lookback-days N (default: all history) hides activity rows older than N days before their
table's cutoff, so the tokens come from recent activity only; Users and Repositories stay visible.
The --history / --habit seed features keep their own windows.

--holdout-weeks N keeps the N newest labelled weeks out of training, validation and the final fit and
scores them with the validation model (its thresholds) and the final model (the validation share of
positives). holdout.npz then holds, per user, the 0/1 decisions and the probabilities of both models on
each holdout week and the validation model's probabilities on the validation (tune) weeks, so
experiments/ensemble/run.py can combine runs.

Run from the repo root (needs PyTorch, a GPU is used if available):
    python experiments/relgt/run.py
    python experiments/relgt/run.py --habit --review-aux --name habit_aux
    python experiments/relgt/run.py --history --name history
    python experiments/relgt/run.py --test-pre-june --history --name test_pre_june_history
"""
import argparse
import os
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch  # after numpy: importing torch first trips a duplicate-OpenMP error on this machine
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ghactivity as base  # noqa: E402
from experiments.gnn_review import run as review  # noqa: E402  (habit features, joint thresholds)

EVENT_COLS, WEEK, DAY = base.EVENT_COLS, base.WEEK, base.DAY
IR, IC = review.IR, review.IC
OUT_DIR = HERE / "outputs"
CACHE_DIR = HERE / "cache"

TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments", "PRReviews",
          "PRReviewComments", "Pushes"]
TIME_COL = {"Users": "first_seen_at", "Repositories": "first_seen_at", "PullRequests": "first_seen_at",
            "Issues": "first_seen_at", "IssueComments": "created_at", "PRReviews": "first_seen_at",
            "PRReviewComments": "created_at", "Pushes": "created_at"}
# Which source's data end sets the staleness of each table (Users / Repositories: the latest).
LAG_KEY = {"PullRequests": "open_pr", "Issues": "open_issue", "IssueComments": "issue_comment",
           "PRReviews": "pr_review", "PRReviewComments": "pr_review_comment", "Pushes": "push"}
FKEYS = {
    "PullRequests": [("repo_id", "Repositories"), ("author_id", "Users")],
    "Issues": [("repo_id", "Repositories"), ("author_id", "Users"), ("assignee_id", "Users"),
               ("pull_request_id", "PullRequests")],
    "IssueComments": [("issue_id", "Issues"), ("repo_id", "Repositories"), ("author_id", "Users")],
    "PRReviews": [("pr_id", "PullRequests"), ("repo_id", "Repositories"), ("reviewer_id", "Users")],
    "PRReviewComments": [("pr_id", "PullRequests"), ("repo_id", "Repositories"), ("author_id", "Users"),
                         ("pull_request_review_id", "PRReviews"), ("in_reply_to_id", "PRReviewComments")],
    "Pushes": [("user_id", "Users"), ("repo_id", "Repositories")],
}
COLUMNS = {
    "Users": ["is_bot", "is_cohort"],
    "Repositories": ["org_id"],
    "PullRequests": ["number", "is_cross_repo", "base_ref", "created_at", "merged_at", "closed_at"],
    "Issues": ["is_pull_request", "title", "body", "num_labels", "milestone_title", "issue_type",
               "locked", "created_via_app", "created_at", "closed_at"],
    "IssueComments": ["is_pull_request", "body", "via_app"],
    "PRReviews": ["state", "body", "submitted_at", "dismissed_at"],
    "PRReviewComments": ["file_extension", "subject_type", "position", "body"],
    "Pushes": ["branch", "is_new_ref"],
}
# The tables are anonymized: letters become "a", so main / master are aaaa / aaaaaa (other names of
# those lengths look the same) and a file extension only keeps its length.
DEFAULT_BRANCHES = ["aaaa", "aaaaaa"]
FILE_EXTENSIONS = ["a", "aa", "aaa", "aaaa"]
USER_SETS = ["all", "test", "exclude-test"]
MAX_HOP = 3  # 1- and 2-hop neighbours, 3 = random fallback node
RELATIONS = [(t, col) for t in TABLES for col, _ in FKEYS.get(t, [])]  # edge types, x2 for direction
N_REL = len(RELATIONS)
# Who performed each activity row; --action-edges chains a user's actions through it.
ACTOR_COL = {"PullRequests": "author_id", "Issues": "author_id", "IssueComments": "author_id",
             "PRReviews": "reviewer_id", "PRReviewComments": "author_id", "Pushes": "user_id"}
# --action-edges: which chains to add, as (relation offset, chain within a repository only).
ACTION_CHAINS = {"none": [], "sequence": [(0, False)], "repo": [(1, True)], "both": [(0, False), (1, True)]}
N_ACTION_REL = 2  # next action, next action in the same repository
# --graph-structure chain: "next action of the same type by the same actor", then its reverse.
CHAIN_REL = 2 * N_REL + 2 * N_ACTION_REL
N_ALL_REL = CHAIN_REL + 2  # relation ids with every option on
# --related-rows: the rows that give a seed action its context, as FK paths of 1-2 steps
# (table, column, +1 = follow the FK to the referenced row, -1 = to the rows referencing it).
RELATED = {
    "Issues": [[("IssueComments", "issue_id", -1)],                       # comments on the issue
               [("Issues", "pull_request_id", 1)]],                       # its PR
    "IssueComments": [[("IssueComments", "issue_id", 1)],                 # the issue
                      [("IssueComments", "issue_id", 1), ("IssueComments", "issue_id", -1)]],  # its thread
    "PRReviews": [[("PRReviews", "pr_id", 1)],                            # the PR
                  [("PRReviewComments", "pull_request_review_id", -1)],   # the review's line comments
                  [("PRReviews", "pr_id", 1), ("PRReviews", "pr_id", -1)]],  # other reviews of the PR
    "PRReviewComments": [[("PRReviewComments", "pr_id", 1)],              # the PR
                         [("PRReviewComments", "pull_request_review_id", 1)],  # its review
                         [("PRReviewComments", "in_reply_to_id", 1)],     # the comment it replies to
                         [("PRReviewComments", "in_reply_to_id", -1)],    # replies to it
                         [("PRReviewComments", "pr_id", 1), ("PRReviewComments", "pr_id", -1)]],  # PR's others
}


# --- Relational entity graph -----------------------------------------------------------------

def days(t):
    """Days since the data start as float32; missing times become +inf (never visible)."""
    d = ((pd.Series(t) - base.DATA_START) / DAY).to_numpy(dtype="float64", na_value=np.inf)
    return d.astype(np.float32)


def row_features(name, d):
    """Static columns of a table as floats, and the columns that are known only later."""
    def flag(s):
        return s.fillna(False).astype(bool)

    def text_len(s):
        return np.log1p(s.fillna("").str.len())

    def lag(late, early):
        return np.log1p(((late - early) / DAY).clip(lower=0).fillna(0))

    f, later = {}, {}
    if name == "Users":
        f = {"bot": d.is_bot, "cohort": d.is_cohort}
    elif name == "Repositories":
        f = {"org": d.org_id.notna()}
    elif name == "PullRequests":
        f = {"cross_repo": flag(d.is_cross_repo), "has_author": d.author_id.notna(),
             "opened_seen": d.created_at.notna(), "seen_lag": lag(d.first_seen_at, d.created_at),
             "number": np.log1p(d.number.astype(float).fillna(0)),
             "default_base": d.base_ref.isin(DEFAULT_BRANCHES)}
        later = {"merged": d.merged_at, "closed": d.closed_at}
    elif name == "Issues":
        f = {"is_pr": d.is_pull_request, "pr_known": d.pull_request_id.notna(),
             "labels": np.log1p(d.num_labels.astype(float).fillna(0)), "assignee": d.assignee_id.notna(),
             "milestone": d.milestone_title.notna(), "typed": d.issue_type.notna(), "locked": flag(d.locked),
             "via_app": d.created_via_app.notna(), "title_len": text_len(d.title), "body_len": text_len(d.body),
             "seen_lag": lag(d.first_seen_at, d.created_at)}
        later = {"closed": d.closed_at}
    elif name == "IssueComments":
        f = {"is_pr": d.is_pull_request, "via_app": d.via_app.notna(), "body_len": text_len(d.body)}
    elif name == "PRReviews":
        # "dismissed" is a later state, so it is not one-hot encoded; dismissed_at covers it.
        f = {s: d.state == s for s in ["approved", "commented", "changes_requested"]}
        f.update({"body_len": text_len(d.body), "seen_lag": lag(d.first_seen_at, d.submitted_at)})
        later = {"dismissed": d.dismissed_at}
    elif name == "PRReviewComments":
        f = {"line": d.subject_type == "line", "reply": d.in_reply_to_id.notna(),
             "in_review": d.pull_request_review_id.notna(), "outdated": d.position.isna(),
             "body_len": text_len(d.body)}
        f.update({f"ext_{e}": d.file_extension == e for e in FILE_EXTENSIONS})
    elif name == "Pushes":
        f = {"new_ref": flag(d.is_new_ref), "default_branch": d.branch.isin(DEFAULT_BRANCHES),
             "branch_len": text_len(d.branch)}
    x = np.nan_to_num(pd.DataFrame(f).astype("float64").values).astype(np.float32)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
    lat = np.stack([days(v) for v in later.values()], 1) if later else np.zeros((len(d), 0), np.float32)
    return x.astype(np.float32), lat


def csr(src, dst, n):
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(src, minlength=n))
    return ptr, dst[order]


class Graph:
    """All rows of all tables as one node set: type, row time, features and FK adjacency.

    structure="fk": every FK is an edge. "chain": the actor FK of an activity row (ACTOR_COL)
    is replaced by edges to the actor's previous and next row of the same table (CHAIN_REL). All
    other FKs stay. A table's rows become visible in row-time order, so the visible part of a chain
    is a prefix and a chain edge never points at a row after the cutoff. Which row is a user's
    latest depends on the cutoff, so user links are made per example (subgraph_edges)."""

    def __init__(self, structure="fk"):
        self.structure = structure
        t0 = time.time()
        sizes, times, self.feats, self.later, fsrc, fdst, frel = [], [], [], [], [], [], []
        frames = {}
        for name in TABLES:
            cols = [TIME_COL[name]] + [c for c, _ in FKEYS.get(name, [])] + COLUMNS[name]
            frames[name] = pd.read_parquet(base.DB / f"{name}.parquet", columns=list(dict.fromkeys(cols)))
            sizes.append(len(frames[name]))
        self.offset = np.concatenate([[0], np.cumsum(sizes)])
        self.n = int(self.offset[-1])
        self.ntype = np.repeat(np.arange(len(TABLES), dtype=np.int64), sizes)
        self.actor = np.full(self.n, -1, np.int64)  # user node of an activity row, -1 otherwise
        self.repo = np.full(self.n, -1, np.int64)  # repository node of a row, -1 if none
        self.next_row = np.full(self.n, -1, np.int64)  # chain: the actor's next row of the table
        self.actor_rel = np.full(len(TABLES), -1, np.int64)  # per table, the relation of its actor FK
        # Parent rows, for --other-actions context: the issue of a comment, the PR of an issue / review /
        # review comment, the assignee of an issue (-1 if none).
        self.issue = np.full(self.n, -1, np.int64)
        self.pr = np.full(self.n, -1, np.int64)
        self.assignee = np.full(self.n, -1, np.int64)
        parent = {("IssueComments", "issue_id"): self.issue, ("Issues", "pull_request_id"): self.pr,
                  ("PRReviews", "pr_id"): self.pr, ("PRReviewComments", "pr_id"): self.pr,
                  ("Issues", "assignee_id"): self.assignee}
        for t, name in enumerate(TABLES):
            d = frames.pop(name)
            times.append(days(d[TIME_COL[name]]))
            for col, ref in FKEYS.get(name, []):
                v = d[col].to_numpy(dtype="float64", na_value=np.nan)
                m = ~np.isnan(v)
                src = self.offset[t] + np.flatnonzero(m)
                dst = self.offset[TABLES.index(ref)] + v[m].astype(np.int64)
                if (name, col) in parent:
                    parent[(name, col)][src] = dst
                rel = np.full(m.sum(), RELATIONS.index((name, col)), np.int64)
                if col == ACTOR_COL.get(name):
                    self.actor[src] = dst
                    self.actor_rel[t] = rel[0] if len(rel) else -1
                    if structure == "chain":
                        o = np.lexsort((src, times[-1][src - self.offset[t]], dst))  # by actor, then time
                        src, dst = src[o], dst[o]
                        same = dst[1:] == dst[:-1]
                        fsrc.append(src[:-1][same])  # earlier row -> next row
                        fdst.append(src[1:][same])
                        frel.append(np.full(same.sum(), CHAIN_REL, np.int64))
                        self.next_row[src[:-1][same]] = src[1:][same]
                        continue
                elif col == "repo_id":
                    self.repo[src] = dst
                fsrc.append(src)
                fdst.append(dst)
                frel.append(rel)
            x, lat = row_features(name, d)
            self.feats.append(x)
            self.later.append(lat)
        self.time = np.concatenate(times)
        fsrc, fdst = np.concatenate(fsrc), np.concatenate(fdst)
        self.fptr, self.fadj = csr(fsrc, fdst, self.n)  # forward FK edges, for subgraph adjacency
        self.frel = csr(fsrc, np.concatenate(frel), self.n)[1]  # their relation, in the same order
        self.ptr, self.adj = csr(np.concatenate([fsrc, fdst]), np.concatenate([fdst, fsrc]), self.n)
        # Each user's activity rows by table, then time: finds the latest visible row of a chain.
        rows = np.flatnonzero(self.actor >= 0)
        rows = rows[np.lexsort((self.time[rows], self.ntype[rows]))]
        self.aptr, self.aadj = csr(self.actor[rows], rows, self.n)
        print(f"graph ({structure}): {self.n:,} nodes, {len(fsrc):,} edges ({time.time() - t0:.0f}s)")

    def reverse(self):
        """Reverse FK adjacency (referenced row -> referencing rows) and its relations, built once."""
        if not hasattr(self, "rptr"):
            src = np.repeat(np.arange(self.n), np.diff(self.fptr))
            self.rptr, self.radj = csr(self.fadj, src, self.n)
            self.rrel = csr(self.fadj, self.frel, self.n)[1]
        return self.rptr, self.radj, self.rrel


# --- Tokenisation (utils.py: gather_1_and_2_hop_with_seed_time, _process_one_seed) -----------

def expand(ptr, adj, ids, nodes):
    """(id, neighbour) for every neighbour of every node."""
    start, cnt = ptr[nodes], ptr[nodes + 1] - ptr[nodes]
    within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return np.repeat(ids, cnt), adj[np.repeat(start, cnt) + within]


def cap(group, value, k, rng, time=None):
    """Keep at most k values per group: random ones, or with time (row time per node) the newest."""
    if not len(group) or np.bincount(group).max() <= k:
        return group, value
    tie = rng.random(len(group))
    order = np.lexsort((tie, group) if time is None else (tie, -time[value], group))
    group, value = group[order], value[order]
    first = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    rank = np.arange(len(group)) - np.repeat(first, np.diff(np.r_[first, len(group)]))
    return group[rank < k], value[rank < k]


N_PATHS = len(TABLES) * (len(TABLES) + 1)  # 1-hop: table of the node; 2-hop: (1-hop table, table)


def path_budget(c, k, rng):
    """Tokens per (seed, path) from candidate counts c [seeds, paths], k per seed (water-filling).

    Every path gets min(count, level) with the level as high as k allows; the few tokens left
    go to random paths that still have unused candidates.
    """
    lo, hi = np.zeros(len(c), np.int64), np.full(len(c), k, np.int64)
    while (lo < hi).any():
        mid = (lo + hi + 1) // 2
        ok = np.minimum(c, mid[:, None]).sum(1) <= k
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid - 1)
    alloc = np.minimum(c, lo[:, None])
    rem = k - alloc.sum(1)
    room = c > alloc
    rank = np.argsort(np.argsort(-np.where(room, rng.random(c.shape), -1.0), 1), 1)
    return alloc + (room & (rank < rem[:, None]))


def visible_nodes(G, cutoff, lookback=None, exclude_push=False):
    """Rows before their table's cutoff; with lookback (days), activity rows older than
    cutoff - lookback are hidden too (Users and Repositories stay visible). exclude_push hides
    every Pushes row."""
    visible = G.time < cutoff[G.ntype]
    if exclude_push:
        visible &= G.ntype != TABLES.index("Pushes")
    if lookback:
        activity = np.array([t in LAG_KEY for t in TABLES])[G.ntype]
        visible &= ~activity | (G.time >= cutoff[G.ntype] - lookback)
    return visible


def sample_tokens(G, seeds, cutoff, K, rng, cap1=5000, cap2=1000, budget="uniform", order="random",
                  lookback=None, exclude_push=False, fk_order="random"):
    """K tokens (node, hop) per seed from its 1- and 2-hop neighbours visible before the cutoff.

    fk_order="recent" prefers newer rows: the caps keep the newest neighbours and the K-1 tokens
    are the newest of the union (random otherwise).

    budget="uniform" samples the union of both hops uniformly; "path" splits the K-1 tokens
    evenly over the seed's paths (path_budget), taking a random subset (order="random") or the
    latest rows (order="recent") of each path. lookback and exclude_push limit the visible rows
    (visible_nodes).
    """
    visible = visible_nodes(G, cutoff, lookback, exclude_push)
    recent = G.time if fk_order == "recent" else None
    b1, v1 = expand(G.ptr, G.adj, np.arange(len(seeds)), seeds)
    m = visible[v1]
    b1, v1 = cap(b1[m], v1[m], cap1, rng, recent)
    i2, v2 = expand(G.ptr, G.adj, np.arange(len(v1)), v1)
    m = visible[v2] & (v2 != seeds[b1[i2]])
    i2, v2 = cap(i2[m], v2[m], cap2, rng, recent)

    # Union of both hops, each node once at its smallest hop, in random order per seed.
    b = np.concatenate([b1, b1[i2]])
    v = np.concatenate([v1, v2])
    hop = np.concatenate([np.ones(len(v1), np.int8), np.full(len(v2), 2, np.int8)])
    path = np.concatenate([G.ntype[v1], len(TABLES) * (1 + G.ntype[v1[i2]]) + G.ntype[v2]])
    key = b * G.n + v
    order_ = np.lexsort((hop, key))
    first = np.r_[True, key[order_][1:] != key[order_][:-1]]
    b, v, hop, path = b[order_][first], v[order_][first], hop[order_][first], path[order_][first]

    if budget == "path":  # keep each (seed, path)'s budget, the rest of the code then picks them all
        c = np.bincount(b * N_PATHS + path, minlength=len(seeds) * N_PATHS).reshape(len(seeds), N_PATHS)
        alloc = path_budget(c, K - 1, rng)
        prio = -G.time[v] if order == "recent" else np.zeros(len(b))
        o = np.lexsort((rng.random(len(b)), prio, path, b))
        b, v, hop, path = b[o], v[o], hop[o], path[o]
        g = b * N_PATHS + path
        start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
        rank = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
        keep = rank < alloc.ravel()[g]
        b, v, hop, path = b[keep], v[keep], hop[keep], path[keep]
    tie = rng.random(len(b))
    order = np.lexsort((tie, b) if recent is None else (tie, -recent[v], b))
    v, hop, path = v[order], hop[order], path[order]

    cnt = np.bincount(b, minlength=len(seeds))
    start = np.cumsum(cnt) - cnt
    pick = np.where(cnt[:, None] >= K - 1, np.arange(K - 1)[None, :],
                    (rng.random((len(seeds), K - 1)) * cnt[:, None]).astype(np.int64))
    nodes = np.empty((len(seeds), K), np.int64)
    hops = np.zeros((len(seeds), K), np.int8)
    paths = np.full((len(seeds), K), N_PATHS, np.int8)  # N_PATHS = seed, N_PATHS + 1 = fallback
    nodes[:, 0] = seeds
    has = cnt > 0
    pos = (start[:, None] + pick)[has]
    nodes[has, 1:], hops[has, 1:], paths[has, 1:] = v[pos], hop[pos], path[pos]
    if (~has).any():
        nodes[~has, 1:] = rng.choice(np.flatnonzero(visible), size=((~has).sum(), K - 1))
        hops[~has, 1:] = MAX_HOP
        paths[~has, 1:] = N_PATHS + 1
    return nodes, hops, paths


def related_rows(G, src_b, src_v, src_r, visible, per_row, rng, exclude, cap1=1000):
    """--related-rows: the visible rows RELATED to each source row (example src_b, node src_v, rank src_r
    among its example's sources, 0 first), at most per_row per source, leaving out the keys
    (example * G.n + node) in exclude (sorted), in the order they should get the budget: the source of
    lowest rank first, then 1 FK step before 2, then the newest row. A row related to several sources of
    an example counts once, for its fewest steps, then lowest rank. A step reaching more than cap1 rows
    from one node keeps the newest.
    Returns (example, node, steps, path, source index); path as in sample_walk."""
    rptr, radj, rrel = G.reverse()
    out = []
    for name, paths in RELATED.items():
        s = np.flatnonzero(G.ntype[src_v] == TABLES.index(name))
        if not len(s):
            continue
        for steps in paths:
            idx, par, cur = s, src_v[s], src_v[s]
            for table, col, d in steps:
                ptr, adj, rel = (G.fptr, G.fadj, G.frel) if d > 0 else (rptr, radj, rrel)
                i, nb = expand(ptr, adj, np.arange(len(cur)), cur)
                r = expand(ptr, rel, np.arange(len(cur)), cur)[1]
                m = (r == RELATIONS.index((table, col))) & visible[nb]
                i, nb = cap(i[m], nb[m], cap1, rng, G.time)
                idx, par, cur = idx[i], cur[i], nb
            out.append((src_b[idx], cur, np.full(len(idx), len(steps), np.int64), par, idx))
    if not out:
        z = np.zeros(0, np.int64)
        return z, z, z, z, z
    b, v, step, par, src = (np.concatenate(x) for x in zip(*out))
    key = b * G.n + v
    keep = ~np.isin(key, exclude)
    b, v, step, par, src, key = b[keep], v[keep], step[keep], par[keep], src[keep], key[keep]
    rank = src_r[src]
    o = np.lexsort((rank, step, key))
    first = np.r_[True, key[o][1:] != key[o][:-1]] if len(o) else np.zeros(0, bool)
    b, v, step, par, src, rank = (x[o][first] for x in (b, v, step, par, src, rank))
    o = np.lexsort((rng.random(len(b)), -G.time[v], step, rank, b))
    b, v, step, par, src = b[o], v[o], step[o], par[o], src[o]
    g = src  # cap per source row (src is unique per (example, source))
    o2 = np.argsort(g, kind="stable")
    gs = g[o2]
    start = np.r_[0, np.flatnonzero(np.diff(gs)) + 1] if len(gs) else np.zeros(0, np.int64)
    within = np.empty(len(g), np.int64)
    within[o2] = np.arange(len(gs)) - np.repeat(start, np.diff(np.r_[start, len(gs)]))
    keep = within < per_row
    b, v, step, par, src = b[keep], v[keep], step[keep], par[keep], src[keep]
    path = np.where(step == 1, G.ntype[v], len(TABLES) * (1 + G.ntype[par]) + G.ntype[v])
    return b, v, step, path, src


def context_filter(G, seeds, own_b, own_v, visible):
    """--other-actions context: returns allowed(b, v), which says for each (example b, node v) whether v may be
    a token. Rows with no actor (users, repositories) and rows of the example's own user always may. A row of
    another user may only if it is
      - an issue comment on an issue of the user, or on the issue of a PR of the user,
      - a review or a review (line) comment on a PR of the user,
      - an issue (any author) in a repository where the user has a visible push or an issue assigned to them.
    own_b, own_v are the example's own visible activity rows (all of its chains)."""
    n, B = G.n, len(seeds)
    ntype = G.ntype[own_v]
    key = lambda m: np.sort(own_b[m] * n + own_v[m])
    own_pr = key(ntype == TABLES.index("PullRequests"))
    own_issue = key(ntype == TABLES.index("Issues"))
    rptr, radj, _ = G.reverse()  # rows that reference the user through a FK: the issues assigned to them
    ab, av = expand(rptr, radj, np.arange(B), seeds)
    m = (G.ntype[av] == TABLES.index("Issues")) & (G.assignee[av] == seeds[ab]) & visible[av]
    ab, av = ab[m], av[m]
    push = ntype == TABLES.index("Pushes")
    rb = np.r_[own_b[push], ab]
    rv = np.r_[G.repo[own_v[push]], G.repo[av]]
    ok = rv >= 0
    ctx_repo = np.unique(rb[ok] * n + rv[ok])
    t_ic, t_is = TABLES.index("IssueComments"), TABLES.index("Issues")
    t_rv, t_rc = TABLES.index("PRReviews"), TABLES.index("PRReviewComments")

    def member(b, x, keys):
        return (x >= 0) & np.isin(b * n + np.maximum(x, 0), keys)

    def allowed(b, v):
        a = G.actor[v]
        ok = (a < 0) | (a == seeds[b])  # not an activity row, or the user's own
        t = G.ntype[v]
        m = ~ok & (t == t_ic)
        if m.any():
            issue = G.issue[v[m]]
            pr = np.where(issue >= 0, G.pr[np.maximum(issue, 0)], -1)  # the issue's PR, if it is one
            ok[m] = member(b[m], issue, own_issue) | member(b[m], pr, own_pr)
        m = ~ok & ((t == t_rv) | (t == t_rc))
        if m.any():
            ok[m] = member(b[m], G.pr[v[m]], own_pr)
        m = ~ok & (t == t_is)
        if m.any():
            ok[m] = member(b[m], G.repo[v[m]], ctx_repo)
        return ok

    return allowed


def sample_walk(G, seeds, cutoff, K, rng, max_hops, cap1=1000, lookback=None, start="user", exclude_push=False,
                num_seeds=1, fk_order="random", related=0, others="all", anchors="chain"):
    """--graph-structure chain: K tokens per example by a breadth-first walk over the visible graph,
    nearest hops first, up to max_hops; a hop with more new nodes than the example's remaining budget
    gives a random subset. Every node reached is a token once; if they do not fill K the rest repeat
    random ones. An example with nothing reachable gets random rows, hop max_hops + 1.

    start="user": token 0 is the user. start="latest": token 0 is the user's latest visible action
    (the user if it has none), with num_seeds > 1 the first tokens, hop 0, are the num_seeds latest
    (fewer if it has fewer), newest first, and the user is a hop 1 token. Either way the first hop
    holds the latest visible row of each of the user's chains that is not a seed (the anchors; with
    anchors="none" they are not added: tokens come only from the seeds and what the walk reaches from them).
    start="latest-per-type": the hop-0 seeds are the latest visible row of each of the user's chains,
    that is one seed per activity table the user has (up to 6), newest first; num_seeds is not used.

    An activity row of another user brings that user in as a token, one hop further, so a sampled
    row always shows whose it is (subgraph_edges links them); the two share the budget. The walk does
    not go on from another user: their other rows are reached only through chains and shared nodes.

    fk_order="recent": where nodes compete (the cap on a node's neighbours, the budget of a hop, the
    final tokens), the newer rows win instead of random ones.

    others="context": an activity row of another user can be a token only if it is part of the example
    user's own context (other_allowed): a comment (issue comment or review comment) on the user's PR or issue,
    a review of the user's PR, or an issue opened in a repository where the user pushed or was assigned an
    issue. Every other row of another user (pushes, PRs, other issues and comments) is left out, the walk
    does not reach it through shared repositories or the other user's chains. Users, repositories and the
    user's own rows are not filtered. "all": no filter.

    related > 0: every token that is an issue, issue comment, PR review or PR review comment (seed, walk
    row or related row) comes with up to related rows RELATED to it (related_rows: an issue's comments,
    a comment's issue and thread, a review's PR and line comments, ...) that are not tokens yet, at its
    hop + their FK steps (1 or 2). The same cost rule applies (a new owner costs one more token).
    The seeds' related rows get up to half of the budget left after the seeds. A walk row's related rows
    count towards its cost, so in a hop's priority order a row is taken only if it and its related rows
    fit; they are then taken right after the hop, and theirs after them, while budget is left. The walk
    goes on from a related row as if it had reached it at that hop."""
    visible = visible_nodes(G, cutoff, lookback, exclude_push)
    recent = G.time if fk_order == "recent" else None
    B = len(seeds)
    users = TABLES.index("Users")

    def owner(v):
        """The visible actor of each node, -1 if it has none."""
        a = G.actor[v]
        return np.where((a >= 0) & visible[a], a, -1)

    def charge(b, v, room, extra_cost=None):
        """Which rows (in priority order, grouped by example) fit in room[b]: a row costs 1, or 2 if
        it is the first to bring in a new owner, plus extra_cost. Also returns their owners."""
        a = owner(v)
        with_owner = np.flatnonzero(a >= 0)
        owner_key = b[with_owner] * G.n + a[with_owner]
        first_of = with_owner[np.unique(owner_key, return_index=True)[1]]
        cost = np.ones(len(b), np.int64)
        cost[first_of[~np.isin(b[first_of] * G.n + a[first_of], seen)]] = 2
        if extra_cost is not None:
            cost += extra_cost
        group = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
        spent = np.cumsum(cost) - np.repeat(np.r_[0, np.cumsum(cost)[group[1:] - 1]], np.diff(np.r_[group, len(b)]))
        return spent <= room[b], a

    def add_owners(b, v, a, hop):
        """The owners of the rows just taken become tokens at hop + 1, unless already tokens."""
        nonlocal seen, count
        w = np.flatnonzero(a >= 0)
        okey, idx = np.unique(b[w] * G.n + a[w], return_index=True)
        new = ~np.isin(okey, seen, assume_unique=True)
        w = w[idx[new]]
        if len(w):
            out_b.append(b[w])
            out_v.append(a[w])
            out_h.append(np.full(len(w), min(hop + 1, max_hops), np.int8))
            out_p.append(len(TABLES) * (1 + G.ntype[v[w]]) + G.ntype[a[w]])
            count += np.bincount(b[w], minlength=B)
            seen = np.union1d(seen, okey[new])

    def related_of(b, v, h, also_exclude=None):
        """related_rows of the rows (b, v) at hops h, newest source first per example, leaving out
        tokens (and also_exclude). Returns (example, node, hop, path, source index into b)."""
        o = np.flatnonzero(np.isin(G.ntype[v], related_types))
        o = o[np.lexsort((-G.time[v[o]], b[o]))]
        group = np.r_[0, np.flatnonzero(np.diff(b[o])) + 1] if len(o) else np.zeros(0, np.int64)
        rank = np.arange(len(o)) - np.repeat(group, np.diff(np.r_[group, len(o)]))
        exclude = seen if also_exclude is None else np.union1d(seen, also_exclude)
        rb, rv, step, rp, src = related_rows(G, b[o], v[o], rank, visible, related, rng, exclude, cap1)
        src = o[src]
        return rb, rv, np.minimum(h[src] + step, max_hops), rp, src

    def force(b, v, h, half=False):
        """Take the related rows of the rows (b, v, hops h) just taken, then theirs, while they fit
        (in half of the room left on the first round if half)."""
        nonlocal seen, count
        while len(b):
            rb, rv, rh, rp, _ = related_of(b, v, h)
            if not len(rb):
                return
            room = K - n_seed - count
            take, a = charge(rb, rv, room // 2 if half else room)
            half = False
            rb, rv, rh, rp, a = rb[take], rv[take], rh[take], rp[take], a[take]
            if not len(rb):
                return
            out_b.append(rb)
            out_v.append(rv)
            out_h.append(rh.astype(np.int8))
            out_p.append(rp)
            count += np.bincount(rb, minlength=B)
            seen = np.union1d(seen, rb * G.n + rv)
            for h_ in np.unique(rh):  # the owner of a row at hop h is at hop h + 1, as in the walk
                m = rh == h_
                add_owners(rb[m], rv[m], a[m], int(h_))
                m &= G.ntype[rv] != users
                eb, ev = extra.get(int(h_), (np.zeros(0, np.int64), np.zeros(0, np.int64)))
                extra[int(h_)] = (np.r_[eb, rb[m]], np.r_[ev, rv[m]])
            b, v, h = rb, rv, rh

    related_types = np.array([TABLES.index(t) for t in RELATED])
    extra = {}  # hop -> (example, node): related rows that join the frontier after that hop
    b_, a_ = expand(G.aptr, G.aadj, np.arange(B), seeds)  # own rows by seed, table, then time
    m = visible[a_]
    b_, a_ = b_[m], a_[m]
    last = np.r_[(b_[1:] != b_[:-1]) | (G.ntype[a_[1:]] != G.ntype[a_[:-1]]), True][:len(a_)]  # per chain
    anchor_b, anchor_v = b_[last], a_[last]
    other_allowed = (context_filter(G, seeds, b_, a_, visible) if others == "context"
                     else lambda b, v: np.ones(len(v), bool))
    roots = seeds.copy()
    seed_b, seed_v, seed_r = np.arange(B), seeds, np.zeros(B, np.int64)  # example, node, rank (0 = newest)
    if start in ("latest", "latest-per-type") and len(anchor_b):
        # latest: the num_seeds newest of all the user's visible actions; latest-per-type: the newest visible
        # action of each type (the anchors), so up to one seed per activity table
        cb, cv = (b_, a_) if start == "latest" else (anchor_b, anchor_v)
        o = np.lexsort((cv, G.time[cv], cb))[::-1]  # per example, newest first
        sb, sv = cb[o], cv[o]
        group = np.r_[0, np.flatnonzero(np.diff(sb)) + 1]
        rank = np.arange(len(sb)) - np.repeat(group, np.diff(np.r_[group, len(sb)]))
        pick_seed = rank < (num_seeds if start == "latest" else len(ACTOR_COL))
        none = np.setdiff1d(np.arange(B), sb)  # no visible action: the user stays the seed
        seed_b = np.r_[sb[pick_seed], none]
        seed_v = np.r_[sv[pick_seed], seeds[none]]
        seed_r = np.r_[rank[pick_seed], np.zeros(len(none), np.int64)]
        o = np.lexsort((seed_r, seed_b))
        seed_b, seed_v, seed_r = seed_b[o], seed_v[o], seed_r[o]
        roots = seed_v[seed_r == 0]
    keep = ~np.isin(anchor_b * G.n + anchor_v, seed_b * G.n + seed_v)
    anchor_b, anchor_v = anchor_b[keep], anchor_v[keep]
    if anchors == "none":  # no latest-row-of-each-chain injection into hop 1
        anchor_b, anchor_v = anchor_b[:0], anchor_v[:0]
    n_seed = np.bincount(seed_b, minlength=B)

    seen = np.sort(seed_b * G.n + seed_v)
    count = np.zeros(B, np.int64)
    fb, fv = seed_b, seed_v  # frontier: example and node
    out_b, out_v, out_h, out_p = [], [], [], []
    own = np.flatnonzero((roots != seeds) & visible[seeds] & (n_seed < K))  # action seeds: the user is a token
    if len(own):
        seen = np.union1d(seen, own * G.n + seeds[own])
        count[own] += 1
        fb, fv = np.r_[fb, own], np.r_[fv, seeds[own]]
        out_b.append(own)
        out_v.append(seeds[own])
        out_h.append(np.ones(len(own), np.int8))
        out_p.append(G.ntype[seeds[own]])
    if related > 0:
        force(seed_b, seed_v, np.zeros(len(seed_b), np.int64), half=True)
    for hop in range(1, max_hops + 1):
        i, v = expand(G.ptr, G.adj, np.arange(len(fb)), fv)
        m = visible[v]
        m[m] = other_allowed(fb[i[m]], v[m])
        i, v = cap(i[m], v[m], cap1, rng, recent)
        b, par = fb[i], fv[i]
        if hop == 1:
            b, v, par = np.r_[b, anchor_b], np.r_[v, anchor_v], np.r_[par, roots[anchor_b]]
        key = b * G.n + v
        key, idx = np.unique(key, return_index=True)
        b, v, par = b[idx], v[idx], par[idx]
        new = ~np.isin(key, seen, assume_unique=True)
        b, v, par, key = b[new], v[new], par[new], key[new]
        tie = rng.random(len(b))
        o = np.lexsort((tie, b) if recent is None else (tie, -recent[v], b))
        b, v, par, key = b[o], v[o], par[o], key[o]
        # In this random order a row costs 1, or 2 if it is the first to bring in a new owner, plus
        # (related) its related rows that are neither tokens nor rows of this hop.
        bundle = None
        if related > 0:
            src = related_of(b, v, np.full(len(b), hop), key)[4]
            bundle = np.bincount(src, minlength=len(b))
        take, a = charge(b, v, K - n_seed - count, bundle)
        b, v, par, key, a = b[take], v[take], par[take], key[take], a[take]
        if len(b):
            out_b.append(b)
            out_v.append(v)
            out_h.append(np.full(len(b), hop, np.int8))
            out_p.append(G.ntype[v] if hop == 1 else len(TABLES) * (1 + G.ntype[par]) + G.ntype[v])
            count += np.bincount(b, minlength=B)
            seen = np.union1d(seen, key)
            add_owners(b, v, a, hop)  # the owners of the rows just taken, unless already tokens
            if related > 0:
                force(b, v, np.full(len(b), hop))
        elif not any(h >= hop for h in extra):
            break
        go_on = G.ntype[v] != users  # the walk stops at other users
        fb, fv = b[go_on], v[go_on]
        if hop in extra:  # related rows taken before the walk go on from this hop
            eb, ev = extra.pop(hop)
            fb, fv = np.r_[fb, eb], np.r_[fv, ev]
    b = np.concatenate(out_b) if out_b else np.zeros(0, np.int64)
    v, hop, path = (np.concatenate(x) if out_b else np.zeros(0, np.int64) for x in (out_v, out_h, out_p))
    tie = rng.random(len(b))
    order = np.lexsort((tie, b) if recent is None else (tie, -recent[v], b))
    v, hop, path = v[order], hop[order], path[order]

    cnt = np.bincount(b, minlength=B)
    start_ = np.cumsum(cnt) - cnt
    col = np.arange(K - 1)[None, :]
    pick = np.where(col < cnt[:, None], col, (rng.random((B, K - 1)) * cnt[:, None]).astype(np.int64))
    # Columns after the seeds of each example read pick from its start; the seeds overwrite the rest.
    pick = np.take_along_axis(pick, np.maximum(np.arange(1, K)[None, :] - n_seed[:, None], 0), axis=1)
    nodes = np.empty((B, K), np.int64)
    hops = np.zeros((B, K), np.int8)
    paths = np.full((B, K), N_PATHS, np.int8)
    has = cnt > 0
    pos = (start_[:, None] + pick)[has]
    nodes[has, 1:], hops[has, 1:], paths[has, 1:] = v[pos], hop[pos], path[pos]
    if (~has).any():
        nodes[~has, 1:] = rng.choice(np.flatnonzero(visible), size=((~has).sum(), K - 1))
        hops[~has, 1:] = max_hops + 1
        paths[~has, 1:] = N_PATHS + 1
    nodes[seed_b, seed_r], hops[seed_b, seed_r], paths[seed_b, seed_r] = seed_v, 0, N_PATHS
    return nodes, hops, paths


def action_chain(G, nodes, by_repo):
    """(seed row, token i, token j) linking each seed's activity tokens of the same actor (and
    repository if by_repo) in row-time order, i the earlier one. Only sampled, visible tokens
    are chained, so an edge never reveals a row the example cannot see."""
    K = nodes.shape[1]
    flat = nodes.ravel()
    group = G.repo[flat] if by_repo else np.zeros(len(flat), np.int64)
    pos = np.flatnonzero((G.actor[flat] >= 0) & (group >= 0))
    row, actor, group = pos // K, G.actor[flat[pos]], group[pos]
    o = np.lexsort((G.time[flat[pos]], group, actor, row))
    pos, row, actor, group = pos[o], row[o], actor[o], group[o]
    nxt = ((row[1:] == row[:-1]) & (actor[1:] == actor[:-1]) & (group[1:] == group[:-1])
           & (flat[pos[1:]] != flat[pos[:-1]]))
    a, b = pos[:-1][nxt], pos[1:][nxt]
    return a // K, a % K, b % K


def subgraph_edges(G, nodes, action_edges="none", users=None, visible=None):
    """Edges among each seed's K tokens: (seed row, token i, token j, relation), both directions.

    The relation of the edge from the referencing row to the referenced one is its FK index, the
    reverse edge has N_REL added. Action chains (--action-edges) come after them: 2 * N_REL +
    offset from the earlier action to the later one, plus N_ACTION_REL for the reverse.

    With users (the example's user node) and visible, on the chain graph: the actor FK edges that
    graph leaves out, per example. A row of another user links to that user; a row of the example's
    user only if it is the latest visible row of its chain."""
    B, K = nodes.shape
    flat = nodes.ravel()
    p, f = expand(G.fptr, G.fadj, np.arange(B * K), flat)
    rel = expand(G.fptr, G.frel, np.arange(B * K), flat)[1]
    row = p // K
    keys = np.repeat(np.arange(B), K) * G.n + flat
    order = np.argsort(keys)
    q = np.minimum(np.searchsorted(keys[order], row * G.n + f), len(keys) - 1)
    hit = keys[order][q] == row * G.n + f
    i, j, r, rel = p[hit] % K, order[q[hit]] % K, row[hit], rel[hit]
    if users is not None and G.structure == "chain":
        row_all = np.repeat(np.arange(B), K)
        actor, nxt = G.actor[flat], G.next_row[flat]
        pos = np.flatnonzero((actor >= 0) & ((actor != users[row_all]) | (nxt < 0) | ~visible[nxt]))
        target = row_all[pos] * G.n + actor[pos]
        q = np.minimum(np.searchsorted(keys[order], target), len(keys) - 1)
        hit = keys[order][q] == target
        i = np.r_[i, pos[hit] % K]
        j = np.r_[j, order[q[hit]] % K]
        r = np.r_[r, row_all[pos[hit]]]
        rel = np.r_[rel, G.actor_rel[G.ntype[flat[pos[hit]]]]]
    rs, is_, js, rels = [r, r], [i, j], [j, i], [rel, np.where(rel < N_REL, rel + N_REL, rel + 1)]
    for offset, by_repo in ACTION_CHAINS[action_edges]:
        r, i, j = action_chain(G, nodes, by_repo)
        fwd = np.full(len(r), 2 * N_REL + offset, np.int64)
        rs += [r, r]
        is_ += [i, j]
        js += [j, i]
        rels += [fwd, fwd + N_ACTION_REL]
    return np.concatenate(rs), np.concatenate(is_), np.concatenate(js), np.concatenate(rels)


# --- Model (model.py, local_module.py, codebook.py, encoders.py) -----------------------------

class PositionalEncoding(nn.Module):
    """torch_geometric.nn.PositionalEncoding: sinusoids of the relative time."""

    def __init__(self, d, base_freq=1e-4):
        super().__init__()
        self.register_buffer("frequency", torch.logspace(0, 1, d // 2, base_freq))

    def forward(self, x):
        out = x.view(-1, 1) * self.frequency.view(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


class TimeEncoder(nn.Module):
    """NeighborTimeEncoder: positional encoding + linear, a learned vector for masked times."""

    def __init__(self, d):
        super().__init__()
        self.pos = PositionalEncoding(d)
        self.linear = nn.Linear(d, d)
        self.mask_vector = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_vector, std=0.02)

    def forward(self, rel_time):
        B, K = rel_time.shape
        out = self.linear(self.pos(rel_time.reshape(-1))).view(B, K, -1)
        mask = (rel_time < 0).unsqueeze(-1).float()
        return (1 - mask) * out + mask * self.mask_vector


class LearnedTimeEncoder(TimeEncoder):
    """--time-emb func: TimeEncoder with learned frequencies and phases instead of the fixed sinusoids,
    cos(w log(1 + days) + b) (TGAT's functional encoding on the log age, as FuncTimeEncoder in
    experiments/gnn), then the same linear layer and mask vector."""

    def __init__(self, d):
        super().__init__(d)
        del self.pos
        self.w = nn.Parameter(torch.logspace(-1, 1, d))
        self.b = nn.Parameter(torch.zeros(d))

    def pos(self, rel_time):
        return torch.cos(torch.log1p(rel_time.clamp(min=0)).unsqueeze(-1) * self.w + self.b)


TIME_ENCODERS = {"sin": TimeEncoder, "func": LearnedTimeEncoder}


class TableEncoder(nn.Module):
    """Stand-in for torch_frame's ResNet: input layer, residual blocks, LayerNorm-ReLU-Linear."""

    def __init__(self, d_in, out, hidden=128, layers=4, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)


class RowEncoder(nn.Module):
    """NeighborTfsEncoder: each token's row through the encoder of its table."""

    def __init__(self, G, cutoffs, week_day, channels, dropout):
        super().__init__()
        self.encoders = nn.ModuleList([
            TableEncoder(G.feats[t].shape[1] + G.later[t].shape[1], channels, dropout=dropout)
            for t in range(len(TABLES))])
        for t in range(len(TABLES)):
            self.register_buffer(f"feat{t}", torch.tensor(G.feats[t]), persistent=False)
            self.register_buffer(f"later{t}", torch.tensor(G.later[t]), persistent=False)
        self.register_buffer("ntype", torch.tensor(G.ntype), persistent=False)
        self.register_buffer("offset", torch.tensor(G.offset[:-1]), persistent=False)
        self.register_buffer("node_time", torch.tensor(G.time), persistent=False)
        self.register_buffer("cutoffs", torch.tensor(cutoffs), persistent=False)  # [weeks, tables]
        self.register_buffer("week_day", torch.tensor(week_day), persistent=False)
        self.channels = channels

    def forward(self, nodes, week):
        flat = nodes.reshape(-1)
        types = self.ntype[flat]
        local = flat - self.offset[types]
        wk = week.repeat_interleave(nodes.shape[1])
        out = torch.zeros(len(flat), self.channels, device=flat.device)
        for t, enc in enumerate(self.encoders):
            m = torch.nonzero(types == t).squeeze(1)
            if not len(m):
                continue
            x = getattr(self, f"feat{t}")[local[m]]
            later = getattr(self, f"later{t}")
            if later.shape[1]:
                happened = later[local[m]] < self.cutoffs[wk[m], t][:, None]
                x = torch.cat([x, happened.float()], 1)
            out[m] = enc(x)
        return out.view(*nodes.shape, -1), types.view(nodes.shape)

    def rel_time(self, nodes, week):
        t = self.week_day[week][:, None] - self.node_time[nodes]
        user = self.ntype[nodes[:, 0]] == TABLES.index("Users")  # chain --chain-seed latest: an action
        t[:, 0] = torch.where(user, torch.zeros_like(t[:, 0]), t[:, 0])  # a user seed has time 0, as in utils.py
        return t


class GNNPEEncoder(nn.Module):
    """GIN over each sampled subgraph with random scalar inputs (encoders.py, pe_dim=0).

    relational=True makes it relation-aware as in GINE: each message is ReLU(x_src + e_rel), with
    a learned embedding per layer of the FK relation and direction of the edge."""

    def __init__(self, d, layers=4, relational=False, n_rel=2 * N_REL):
        super().__init__()
        h = d // 4
        self.rel = (nn.ModuleList([nn.Embedding(n_rel, h) for _ in range(layers)])
                    if relational else None)
        self.input_proj = nn.Linear(1, h)
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.BatchNorm1d(2 * h), nn.ReLU(),
                                                 nn.Linear(2 * h, h)) for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))  # GINConv(train_eps=True)
        self.bns = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(layers)])
        self.final_transform = nn.Linear(h, d)
        for lin in [self.input_proj, self.final_transform]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, edge_index, edge_type, B, K):
        x = self.input_proj(torch.randn(B * K, 1, device=edge_index.device))
        src, dst = edge_index
        for i, mlp in enumerate(self.mlps):
            msg = x[src] if self.rel is None else F.relu(x[src] + self.rel[i](edge_type))
            agg = torch.zeros_like(x).index_add_(0, dst, msg)
            x = x + F.relu(self.bns[i](mlp((1 + self.eps[i]) * x + agg)))
        return self.final_transform(x).view(B, K, -1)


class AttentionBias(nn.Module):
    """Per-head bias of the local attention logits (Graphormer-style spatial and edge encoding):
    one term for the hop pair of the two tokens, plus one for the FK relation and direction of
    every edge between them. Zero at the start, so training starts from plain attention."""

    def __init__(self, heads, n_rel=2 * N_REL, max_hop=MAX_HOP):
        super().__init__()
        self.max_hop = max_hop
        self.hop = nn.Embedding((max_hop + 1) ** 2, heads)
        self.rel = nn.Embedding(n_rel, heads)
        nn.init.zeros_(self.hop.weight)
        nn.init.zeros_(self.rel.weight)

    def forward(self, hops, edge_index, edge_type):
        K = hops.shape[1]
        bias = self.hop(hops[:, :, None] * (self.max_hop + 1) + hops[:, None, :])  # [B, K, K, heads]
        src, dst = edge_index
        bias = bias.index_put((src // K, src % K, dst % K), self.rel(edge_type), accumulate=True)
        return bias.permute(0, 3, 1, 2)  # [B, heads, K, K]


class RecencyBias(nn.Module):
    """--recency-attn: logit bias -decay * log(1 + D / days) on attending to an action D days older than
    the example's newest action token, so its weight is multiplied by (1 + D / days) ^ -decay. Tokens
    that are not actions get no bias. Decay starts at 1, one per head and one for the readout."""

    def __init__(self, heads, days):
        super().__init__()
        self.days = days
        self.raw_decay = nn.Parameter(torch.full((heads + 1,), math.log(math.e - 1)))  # softplus gives 1
        self.register_buffer("is_action", torch.tensor([t in LAG_KEY for t in TABLES]), persistent=False)

    def decay(self):
        return F.softplus(self.raw_decay)

    def forward(self, age, types):
        action = self.is_action[types]
        newest = age.masked_fill(~action, math.inf).amin(1, keepdim=True)
        older = ((age - newest).clamp(min=0) / self.days).log1p().masked_fill(~action, 0)
        decay = self.decay()
        attn = -decay[:-1].view(1, -1, 1, 1) * older[:, None, None, :]  # [B, heads, 1, K]: towards each token
        return attn, -decay[-1] * older.unsqueeze(-1)  # readout: [B, K, 1]


class FeedForwardNetwork(nn.Module):
    def __init__(self, d, ffn, dropout):
        super().__init__()
        self.bn_in, self.bn_out = nn.BatchNorm1d(d), nn.BatchNorm1d(d)
        self.ffn_net = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(ffn, d), nn.Dropout(dropout))

    def forward(self, x):
        x = self.bn_in(x.transpose(1, 2)).transpose(1, 2)
        x = self.ffn_net(x)
        return self.bn_out(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Pre-LayerNorm transformer layer (local_module.py)."""

    def __init__(self, d, ffn, dropout, attn_dropout, heads):
        super().__init__()
        self.heads, self.attn_dropout = heads, attn_dropout
        self.self_attention_norm = nn.LayerNorm(d)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        self.self_attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = FeedForwardNetwork(d, ffn, dropout)

    def forward(self, x, bias=None):
        B, L, D = x.shape
        h = self.self_attention_norm(x)
        q, k, v = (p(h).view(B, L, self.heads, D // self.heads).transpose(1, 2)
                   for p in [self.q_proj, self.k_proj, self.v_proj])
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.self_attention_dropout(self.out_proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.ffn(self.ffn_norm(x))


class LocalModule(nn.Module):
    """Transformer over the K tokens; readout = seed token + attention-weighted neighbours.
    With seed_mask (--num-seeds > 1) the seed token is the mean of the masked tokens."""

    def __init__(self, d, layers, heads, dropout, attn_dropout):
        super().__init__()
        self.att_embeddings_nope = nn.Linear(d, d)
        self.layers = nn.ModuleList([EncoderLayer(d, 2 * d, dropout, attn_dropout, heads)
                                     for _ in range(layers)])
        self.final_ln = nn.LayerNorm(d)
        self.attn_layer = nn.Linear(2 * d, 1)

    def forward(self, x_set, bias=None, seed_mask=None, readout_bias=None):
        x = self.att_embeddings_nope(x_set)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.final_ln(x)
        if seed_mask is not None:
            m = seed_mask.unsqueeze(-1)
            node = (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
            logits = self.attn_layer(torch.cat([node.expand_as(x), x], 2))
            if readout_bias is not None:
                logits = logits + readout_bias
            logits = logits.masked_fill(m, -math.inf)
            return (node + (x * F.softmax(logits, dim=1)).sum(1, keepdim=True)).squeeze(1)
        node, nbr = x[:, :1], x[:, 1:]
        logits = self.attn_layer(torch.cat([node.expand_as(nbr), nbr], 2))
        if readout_bias is not None:
            logits = logits + readout_bias[:, 1:]
        w = F.softmax(logits, dim=1)
        return (node + (nbr * w).sum(1, keepdim=True)).squeeze(1)


class VectorQuantizerEMA(nn.Module):
    """Codebook of centroids updated by exponential moving averages (codebook.py)."""

    def __init__(self, num_embeddings, dim, decay=0.99):
        super().__init__()
        self.num_embeddings, self.decay = num_embeddings, decay
        self.register_buffer("_embedding", torch.randn(num_embeddings, dim))
        self.register_buffer("_embedding_output", torch.randn(num_embeddings, dim))
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", torch.randn(num_embeddings, dim))
        self.bn = nn.BatchNorm1d(dim, affine=False)

    def get_k(self):
        """Centroids in the input scale; keys and values of the global attention."""
        return self._embedding_output

    @torch.no_grad()
    def update(self, x):
        x = self.bn(x)
        dist = (x ** 2).sum(1, keepdim=True) + (self._embedding ** 2).sum(1) - 2 * x @ self._embedding.t()
        idx = dist.argmin(1)
        enc = F.one_hot(idx, self.num_embeddings).float()
        self._ema_cluster_size.mul_(self.decay).add_((1 - self.decay) * enc.sum(0))
        n = self._ema_cluster_size.sum()
        self._ema_cluster_size.copy_((self._ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n)
        self._ema_w.mul_(self.decay).add_((1 - self.decay) * enc.t() @ x)
        self._embedding.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(0)
        self._embedding_output.copy_(self._embedding * std + self.bn.running_mean.unsqueeze(0))
        return idx


class RelGTLayer(nn.Module):
    def __init__(self, d, global_dim, local_layers, heads, dropout, attn_dropout, conv_type,
                 num_centroids, num_seeds):
        super().__init__()
        self.d, self.heads, self.conv_type, self.attn_dropout = d, heads, conv_type, attn_dropout
        self.num_centroids = num_centroids
        if conv_type != "global":
            self.local_module = LocalModule(d, local_layers, heads, dropout, attn_dropout)
            self.layer_norm_local = nn.LayerNorm(d)
        if conv_type != "local":
            self.vq = VectorQuantizerEMA(num_centroids, global_dim)
            self.register_buffer("c_idx", torch.randint(0, num_centroids, (num_seeds,)))
            self.lin_proj_g = nn.Linear(d, global_dim)
            self.lin_key_g, self.lin_query_g, self.lin_value_g = (nn.Linear(global_dim, d) for _ in range(3))
            self.layer_norm_global = nn.LayerNorm(d)

    def global_forward(self, x, seed_idx):
        q_x = self.lin_proj_g(x)
        codebook = self.vq.get_k().detach().clone()  # updated in place below
        k, v = self.lin_key_g(codebook), self.lin_value_g(codebook)
        q = self.lin_query_g(q_x)
        q, k, v = (t.view(t.shape[0], self.heads, -1).transpose(0, 1) for t in (q, k, v))
        dots = q @ k.transpose(1, 2) / math.sqrt(self.d)
        count = torch.bincount(self.c_idx, minlength=self.num_centroids).float()
        attn = F.softmax(dots + torch.log(count).view(1, 1, -1), dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        out = (attn @ v).transpose(0, 1).reshape(x.shape[0], -1)
        if self.training:
            self.c_idx[seed_idx] = self.vq.update(q_x.detach())
        return out

    def forward(self, x_set, x, seed_idx, bias=None, seed_mask=None, readout_bias=None):
        out = []
        if self.conv_type != "global":
            out.append(self.layer_norm_local(self.local_module(x_set, bias, seed_mask, readout_bias)))
        if self.conv_type != "local":
            out.append(self.layer_norm_global(self.global_forward(x, seed_idx)))
        return torch.cat(out, 1)


class RelGT(nn.Module):
    ELEMENTS = ["type", "hop", "time", "tfs", "gnn", "path"]

    def __init__(self, G, cutoffs, week_day, num_seeds, n_out, channels=128, local_layers=1, heads=4,
                 dropout=0.3, num_centroids=1024, conv_type="full", ablate="none", n_seed_feats=0,
                 path_token=False, rel_pe=False, attn_bias=False, action_edges=False, chain=False,
                 max_hop=MAX_HOP, multi_seed=False, recency_days=None, time_emb="sin"):
        super().__init__()
        c = channels
        self.multi_seed = multi_seed  # several hop-0 tokens per example (--num-seeds > 1)
        n_rel = N_ALL_REL if chain else 2 * N_REL + (2 * N_ACTION_REL if action_edges else 0)
        self.rows = RowEncoder(G, cutoffs, week_day, c, dropout)
        self.seed_encoder = (nn.Sequential(nn.Linear(n_seed_feats, c), nn.ReLU(), nn.Linear(c, c))
                             if n_seed_feats else None)
        self.type_encoder = nn.Embedding(len(TABLES) + 1, c)
        self.hop_encoder = nn.Embedding(max_hop + 2, c)  # hops + 1: seed, ..., fallback
        self.time_encoder = TIME_ENCODERS[time_emb](c)
        self.pe_encoder = GNNPEEncoder(c, relational=rel_pe, n_rel=n_rel)
        self.path_encoder = nn.Embedding(N_PATHS + 2, c) if path_token else None
        self.attn_bias = AttentionBias(heads, n_rel, max_hop) if attn_bias and conv_type != "global" else None
        self.recency = RecencyBias(heads, recency_days) if recency_days and conv_type != "global" else None
        self.elements = [k for k in self.ELEMENTS if k != ablate and (k != "path" or path_token)]
        self.norms = nn.ModuleDict({f"{k}_norm": nn.LayerNorm(c) for k in self.elements})
        self.in_mixture = nn.Sequential(nn.Linear(len(self.elements) * c, 2 * c), nn.ReLU(), nn.Linear(2 * c, c))
        self.conv = RelGTLayer(c, c // 2, local_layers, heads, dropout, dropout, conv_type, num_centroids,
                               num_seeds)
        h = 2 if conv_type == "full" else 1
        self.ff = nn.Sequential(nn.BatchNorm1d(h * c), nn.Linear(h * c, 2 * c), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(2 * c, c), nn.Dropout(dropout), nn.BatchNorm1d(c))
        # torch_geometric MLP(num_layers=2): Linear, BatchNorm, ReLU, Linear
        self.head = nn.Sequential(nn.Linear(c, c), nn.BatchNorm1d(c), nn.ReLU(), nn.Linear(c, n_out))

    def forward(self, nodes, hops, paths, week, seed_idx, edge_index, edge_type, seed_x=None):
        B, K = nodes.shape
        tfs, types = self.rows(nodes, week)
        age = self.rows.rel_time(nodes, week)
        parts = {
            "type": self.type_encoder(types),
            "hop": self.hop_encoder(hops + 1),
            "time": self.time_encoder(age),
            "tfs": tfs,
            "gnn": self.pe_encoder(edge_index, edge_type, B, K),
        }
        if self.path_encoder is not None:
            parts["path"] = self.path_encoder(paths)
        x_set = self.in_mixture(torch.cat([self.norms[f"{k}_norm"](parts[k]) for k in self.elements], -1))
        seed_mask = hops == 0 if self.multi_seed else None
        if self.seed_encoder is not None:  # per-user summary features join the seed token
            if seed_mask is not None:
                x_set = x_set + seed_mask.unsqueeze(-1) * self.seed_encoder(seed_x).unsqueeze(1)
            else:
                x_set = torch.cat([x_set[:, :1] + self.seed_encoder(seed_x).unsqueeze(1), x_set[:, 1:]], 1)
        bias = None if self.attn_bias is None else self.attn_bias(hops, edge_index, edge_type)
        readout_bias = None
        if self.recency is not None:
            recent, readout_bias = self.recency(age, types)
            bias = recent if bias is None else bias + recent
        query = x_set[:, 0]  # to the global codebook
        if seed_mask is not None:
            m = seed_mask.unsqueeze(-1)
            query = (x_set * m).sum(1) / m.sum(1)
        out = self.ff(self.conv(x_set, query, seed_idx, bias, seed_mask, readout_bias))
        return self.head(out)


# --- Training ------------------------------------------------------------------------------

class Samples:
    """Tokens, labels and subgraph edges of every (user, week) example, on the device."""

    def __init__(self, nodes, hops, paths, week, cidx, y, edges, device, seed_x=None):
        self.n, self.K = nodes.shape
        self.seed_x = None if seed_x is None else torch.tensor(seed_x, dtype=torch.float32, device=device)
        self.nodes = torch.tensor(nodes, dtype=torch.int32, device=device)
        self.hops = torch.tensor(hops, device=device)
        self.paths = torch.tensor(paths, device=device)
        self.week = torch.tensor(week, device=device)
        self.cidx = torch.tensor(cidx, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        r, i, j, rel = edges
        order = np.argsort(r, kind="stable")
        self.esrc, self.edst, self.erel = i[order], j[order], rel[order]  # int16 token positions, int8
        self.eptr = np.zeros(self.n + 1, np.int64)
        self.eptr[1:] = np.cumsum(np.bincount(r, minlength=self.n))
        self.device = device

    def batch(self, ids):
        start, cnt = self.eptr[ids], self.eptr[ids + 1] - self.eptr[ids]
        pos = np.repeat(start - np.cumsum(cnt) + cnt, cnt) + np.arange(cnt.sum())
        shift = np.repeat(np.arange(len(ids)) * self.K, cnt)
        edge_index = torch.tensor(np.stack([self.esrc[pos] + shift, self.edst[pos] + shift]),  # int64
                                  device=self.device)
        edge_type = torch.tensor(self.erel[pos], dtype=torch.long, device=self.device)
        t = torch.tensor(ids, device=self.device)
        seed_x = None if self.seed_x is None else self.seed_x[t]
        return (self.nodes[t].long(), self.hops[t].long(), self.paths[t].long(), self.week[t], self.cidx[t],
                edge_index, edge_type, seed_x), self.y[t]


def masked_bce(logits, y):
    m = ~torch.isnan(y)
    loss = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
    return (loss * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def predict(model, S, ids, batch_size=2048):
    model.eval()
    torch.manual_seed(0)  # the subgraph encoding draws random inputs
    out = [torch.sigmoid(model(*S.batch(ids[i:i + batch_size])[0])).cpu().numpy()
           for i in range(0, len(ids), batch_size)]
    return np.concatenate(out)


def mean_ap(S, ids, p):
    y = S.y[torch.tensor(ids, device=S.device)].cpu().numpy()
    aps = [average_precision_score(y[~np.isnan(y[:, j]), j], p[~np.isnan(y[:, j]), j])
           for j in range(len(EVENT_COLS)) if (y[~np.isnan(y[:, j]), j] == 1).any()]  # not the auxiliary output
    return float(np.mean(aps))


def fit(make_model, S, train_ids, args, epochs, val_ids=None):
    """Adam + BCE as in main_node_ddp.py; with val_ids, keep the epoch with the best mean AP."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, best_epoch, history = -np.inf, None, epochs, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = rng.permutation(train_ids)
        for step, i in enumerate(range(0, len(perm), args.batch_size)):
            ids = perm[i:i + args.batch_size]
            if (args.max_steps_per_epoch and step >= args.max_steps_per_epoch) or len(ids) < 2:
                break
            inputs, y = S.batch(ids)
            opt.zero_grad()
            loss = masked_bce(model(*inputs), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        msg = f"  epoch {epoch:2d}  train loss {np.mean(losses):.4f}  {time.time() - t0:.0f}s"
        if val_ids is not None:
            ap = mean_ap(S, val_ids, predict(model, S, val_ids))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_map": ap})
            msg += f"  val mean AP {ap:.4f}"
            if ap > best:
                best, best_epoch = ap, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def tokenize(G, groups, cutoffs, K, seed, budget="uniform", order="random", action_edges="none", lookback=None,
             max_hops=None, chain_seed="user", exclude_push=False, num_seeds=1, fk_order="random", related=0,
             others="all", anchors="chain"):
    """Tokens of every example, cached on disk (sampling is the slow part). max_hops, related and others are
    used by the chain structure (sample_walk)."""
    mode = () if budget == "uniform" else (budget, order)
    mode += () if G.structure == "fk" else (("structure", G.structure, "latest-owner", max_hops),)
    mode += () if chain_seed == "user" else (("chain_seed", chain_seed),)
    mode += () if not exclude_push else (("exclude_push",),)
    mode += () if num_seeds == 1 else (("num_seeds", num_seeds),)
    mode += () if fk_order == "random" else (("fk_order", fk_order),)
    mode += () if not related else (("related-all", related),)
    mode += () if others == "all" else (("others", others),)
    mode += () if anchors == "chain" else (("anchors", anchors),)
    mode += () if action_edges == "none" else (("action_edges", action_edges),)
    mode += () if not lookback else (("lookback", lookback),)
    key = hashlib.md5(repr((K, seed, cutoffs.tolist(), [g.tolist() for g in groups], G.n) + mode)
                      .encode()).hexdigest()
    path = CACHE_DIR / f"tokens2_K{K}_{key[:12]}.npz"  # 2: with token paths and edge relations
    if path.exists():
        z = np.load(path)
        print(f"tokens from {path.name}")
        return z["nodes"], z["hops"], z["paths"], (z["er"], z["ei"], z["ej"], z["erel"])
    t0 = time.time()
    rng = np.random.default_rng(seed)
    nodes, hops, paths, er, ei, ej, erel, n = [], [], [], [], [], [], [], 0
    for w, seeds in enumerate(groups):
        if G.structure == "chain":
            nd, hp, pt = sample_walk(G, seeds, cutoffs[w], K, rng, max_hops, lookback=lookback, start=chain_seed,
                                     exclude_push=exclude_push, num_seeds=num_seeds, fk_order=fk_order,
                                     related=related, others=others, anchors=anchors)
        else:
            nd, hp, pt = sample_tokens(G, seeds, cutoffs[w], K, rng, budget=budget, order=order, lookback=lookback,
                                       exclude_push=exclude_push, fk_order=fk_order)
        r, i, j, rel = subgraph_edges(G, nd, action_edges, seeds, visible_nodes(G, cutoffs[w], lookback, exclude_push))
        nodes.append(nd.astype(np.int32))
        hops.append(hp)
        paths.append(pt)
        er.append((r + n).astype(np.int32))
        ei.append(i.astype(np.int16))
        ej.append(j.astype(np.int16))
        erel.append(rel.astype(np.int8))
        n += len(seeds)
    edges = tuple(np.concatenate(a) for a in (er, ei, ej, erel))
    out = np.concatenate(nodes), np.concatenate(hops), np.concatenate(paths), edges
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Written to a temporary file and renamed, so a run sampling the same key at the same time never
    # loads a half-written cache file.
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, nodes=out[0], hops=out[1], paths=out[2], er=edges[0], ei=edges[1], ej=edges[2], erel=edges[3])
    os.replace(tmp, path)
    print(f"sampled {n:,} examples x {K} tokens, {len(edges[0]):,} subgraph edges ({time.time() - t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description="RelGT for the weekly activity labels")
    base.add_period_args(ap)
    ap.add_argument("--num-neighbors", type=int, default=100, help="K tokens per example (paper: 300)")
    ap.add_argument("--path-budget", choices=["uniform", "path"], default="uniform",
                    help="token sampling: uniform over all neighbours (paper) or an even budget per path")
    ap.add_argument("--path-order", choices=["random", "recent"], default="random",
                    help="with --path-budget path: which rows of a path fill its budget")
    ap.add_argument("--fk-order", choices=["random", "recent"], default="random",
                    help="which FK/PK neighbours are kept when over budget: random ones, or the newest rows first")
    ap.add_argument("--path-token", action="store_true",
                    help="add the token's relation path (as in --path-budget) as a token element")
    ap.add_argument("--rel-pe", action="store_true",
                    help="relation-aware subgraph positional encoding (GINE messages with FK relation)")
    ap.add_argument("--attn-bias", action="store_true",
                    help="local attention bias from the hop pair and the FK relation between tokens")
    ap.add_argument("--action-edges", choices=list(ACTION_CHAINS), default="none",
                    help="extra token edges between actions of the same user in time order: sequence = "
                         "each action to the user's next one, repo = next one in the same repository, both")
    ap.add_argument("--lookback-days", type=float, default=None,
                    help="only activity rows from the last N days before each table's cutoff are visible "
                         "to the graph (Users and Repositories always are); default: all history")
    ap.add_argument("--exclude-push", action="store_true",
                    help="hide all Pushes rows from the graph; push is still predicted and --history still "
                         "has its push features")
    ap.add_argument("--graph-structure", choices=["fk", "chain"], default="fk",
                    help="fk: every foreign key is an edge (every action links to its user). chain: a user "
                         "links only to its latest visible action of each type, each action to the previous "
                         "/ next one of the same type; other foreign keys stay; tokens by breadth-first walk, "
                         "another user's action brings that user in")
    ap.add_argument("--max-hops", type=int, default=50,
                    help="with --graph-structure chain: how far the walk may go from the user")
    ap.add_argument("--chain-seed", choices=["user", "latest", "latest-per-type"], default="user",
                    help="with --graph-structure chain: token 0 (where the walk starts and the readout is "
                         "taken) is the user, or the user's latest visible action (--num-seeds of them), or "
                         "latest-per-type: the latest visible action of each action type (up to 6 seeds)")
    ap.add_argument("--num-seeds", type=int, default=1,
                    help="with --graph-structure chain --chain-seed latest: use the user's N latest visible "
                         "actions as seeds (fewer if the user has fewer); counts towards --num-neighbors")
    ap.add_argument("--related-rows", type=int, default=0,
                    help="with --graph-structure chain: every issue / issue comment / PR review / review "
                         "comment token comes with up to N rows related to it through its foreign keys (its "
                         "issue, PR or review, thread, replies), newest first; a walk row is taken only if they "
                         "fit too; 0 = off")
    ap.add_argument("--other-actions", choices=["all", "context"], default="all",
                    help="with --graph-structure chain: which activity rows of OTHER users can be tokens. all: any "
                         "(reached through shared repositories, issues, PRs). context: only comments on the "
                         "user's PRs / issues, reviews of the user's PRs, and issues opened in a repository where "
                         "the user pushed or was assigned an issue; the users themselves, repositories and the "
                         "user's own rows are not filtered")
    ap.add_argument("--chain-anchors", choices=["chain", "none"], default="chain",
                    help="with --graph-structure chain: chain = hop 1 also holds the latest visible row of each of "
                         "the user's chains that is not a seed; none = no such rows, tokens come only from the "
                         "seeds and what the walk reaches from them")
    ap.add_argument("--recency-attn", action="store_true",
                    help="with --graph-structure chain: attend more to recent actions, in the local attention "
                         "and the readout, with a learned decay; the tokens do not change")
    ap.add_argument("--recency-days", type=float, default=7,
                    help="with --recency-attn: an action this many days older than the newest action token "
                         "starts at half the attention weight")
    ap.add_argument("--time-emb", choices=list(TIME_ENCODERS), default="sin",
                    help="token age encoding: sin = fixed sinusoids + linear (the paper), func = learned "
                         "cos(w log(1 + days) + b) + linear")
    ap.add_argument("--channels", type=int, default=128, help="paper: 512")
    ap.add_argument("--num-layers", type=int, default=1, help="local transformer layers (paper: 1-8)")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-centroids", type=int, default=1024)
    ap.add_argument("--conv-type", choices=["full", "local", "global"], default="full")
    ap.add_argument("--ablate", choices=["none"] + RelGT.ELEMENTS, default="none",
                    help="leave one token element out")
    ap.add_argument("--dropout", type=float, default=0.3, help="feed-forward and attention dropout")
    ap.add_argument("--lr", type=float, default=3e-4, help="paper: 1e-4")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20, help="paper: 10")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--habit", action="store_true",
                    help="add the review-commenting habit features to the seed token")
    ap.add_argument("--holdout-weeks", type=int, default=1,
                    help="the newest N labelled weeks are held out of training, validation and the final fit "
                         "and scored at the end (0: no holdout, the final model uses every labelled week)")
    ap.add_argument("--exclude-push-label", action="store_true",
                    help="do not predict push: its label is left out of the loss, validation thresholds and "
                         "AP, and the output has push = 0 for every user (the other 5 labels are unchanged)")
    ap.add_argument("--history", action="store_true",
                    help="add the baseline's per-user count / recency features to the seed token")
    ap.add_argument("--review-aux", action="store_true",
                    help="extra training output: pr_review or pr_review_comment")
    g = ap.add_argument_group("users (test users = those in test_input.csv)")
    for stage, what in [("train", "training"), ("val", "validation / thresholds"), ("final", "final fit")]:
        g.add_argument(f"--{stage}-users", choices=USER_SETS, default="all",
                       help=f"users whose examples are used for {what}")
    g.add_argument("--test-pre-june", action="store_true",
                   help="shortcut: test users only for every stage and --end 2026-06-01 (unless --end is given)")
    ap.add_argument("--name", default="default", help="output sub-directory")
    args = ap.parse_args()
    if args.graph_structure == "chain" and args.path_budget != "uniform":
        ap.error("--path-budget path works with --graph-structure fk only")
    if args.chain_seed != "user" and args.graph_structure != "chain":
        ap.error("--chain-seed latest needs --graph-structure chain")
    if args.num_seeds != 1 and (args.graph_structure != "chain" or args.chain_seed != "latest"):
        ap.error("--num-seeds needs --graph-structure chain --chain-seed latest "
                 "(latest-per-type takes one seed per action type and no --num-seeds)")
    if args.related_rows < 0:
        ap.error("--related-rows must be >= 0")
    if args.related_rows and args.graph_structure != "chain":
        ap.error("--related-rows needs --graph-structure chain")
    if args.other_actions != "all" and args.graph_structure != "chain":
        ap.error("--other-actions context needs --graph-structure chain")
    if args.other_actions != "all" and args.related_rows:
        ap.error("--other-actions context does not work with --related-rows (related rows are not filtered)")
    if args.chain_anchors != "chain" and args.graph_structure != "chain":
        ap.error("--chain-anchors none needs --graph-structure chain")
    if args.recency_attn and args.graph_structure != "chain":
        ap.error("--recency-attn needs --graph-structure chain")
    if args.recency_days <= 0:
        ap.error("--recency-days must be positive")
    if not 1 <= args.num_seeds < args.num_neighbors:
        ap.error("--num-seeds must be in 1..num_neighbors - 1")
    if not 1 <= args.max_hops <= 120:
        ap.error("--max-hops must be in 1..120")
    if args.test_pre_june:
        args.train_users = args.val_users = args.final_users = "test"
        args.end = args.end or "2026-06-01"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = OUT_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort, ev, own, data_end = base.load()
    n = len(cohort)
    lags = {k: base.TEST_START - v for k, v in data_end.items()}
    weeks, train_weeks, val_weeks, final_weeks = base.split_weeks(args, data_end)
    holdout_weeks = weeks[len(weeks) - args.holdout_weeks:] if args.holdout_weeks else []
    if holdout_weeks:  # the newest labelled weeks are kept out of training, validation and the final fit
        inner = argparse.Namespace(**{**vars(args), "end": str(holdout_weeks[0])})
        _, train_weeks, val_weeks, final_weeks = base.split_weeks(inner, data_end)
    print(f"device {device}, {len(train_weeks)} train weeks, validation {[str(w)[:10] for w in val_weeks]}, "
          f"holdout {[str(w)[:10] for w in holdout_weeks]}")

    tables = {w: base.build_week(w, ev, own, lags, data_end, n) for w in weeks}
    test = pd.read_csv(base.TEST_INPUT)
    test_cidx = pd.Series(np.arange(n), index=cohort.github_user_id.values).loc[test.UserID].values
    all_weeks = weeks + [base.TEST_START]

    # Per week and table: the cutoff in days; rows with an earlier row time are visible.
    min_lag = min(lags.values())
    cutoffs = np.array([[((s - lags[LAG_KEY[t]] if t in LAG_KEY else s - min_lag) - base.DATA_START) / DAY
                         for t in TABLES] for s in all_weeks], np.float32)
    week_day = np.array([(s - base.DATA_START) / DAY for s in all_weeks], np.float32)

    # Examples: eligible users of every labelled week, then the test users.
    cidx, week, y = [], [], []
    for i, w in enumerate(weeks):
        m = np.flatnonzero(tables[w].eligible.values)
        cidx.append(m)
        week.append(np.full(len(m), i))
        y.append(tables[w][[f"y_{k}" for k in EVENT_COLS]].values[m])
    cidx.append(test_cidx)
    week.append(np.full(len(test_cidx), len(weeks)))
    y.append(np.full((len(test_cidx), len(EVENT_COLS)), np.nan))
    sizes = np.cumsum([0] + [len(c) for c in cidx])
    y = np.concatenate(y)
    IP = EVENT_COLS.index("push")
    if args.exclude_push_label:  # unknown label: masked_bce and mean_ap skip it
        y[:, IP] = np.nan
    if args.review_aux:  # 1 if either review label is 1, 0 if both are 0, unknown otherwise
        r, c = y[:, IR], y[:, IC]
        union = np.where((r == 1) | (c == 1), 1.0, np.where((r == 0) & (c == 0), 0.0, np.nan))
        y = np.concatenate([y, union[:, None]], 1)
    G = Graph(args.graph_structure)
    seeds = [G.offset[TABLES.index("Users")] + cohort.user_id.values[c] for c in cidx]
    nodes, hops, paths, edges = tokenize(G, seeds, cutoffs, args.num_neighbors, args.seed, args.path_budget,
                                         args.path_order, args.action_edges, args.lookback_days, args.max_hops,
                                         args.chain_seed, args.exclude_push, args.num_seeds, args.fk_order,
                                         args.related_rows, args.other_actions, args.chain_anchors)
    ids = {w: np.arange(sizes[i], sizes[i + 1]) for i, w in enumerate(all_weeks)}
    is_test = np.zeros(n, bool)
    is_test[test_cidx] = True
    user_sets = {"all": np.ones(n, bool), "test": is_test, "exclude-test": ~is_test}
    cid = dict(zip(all_weeks, cidx))  # cohort index of each week's examples

    def pick(wks, mode):
        """Boolean masks (per week) of the examples of the chosen user set."""
        return {w: user_sets[mode][cid[w]] for w in wks}

    def select(wks, mode):
        sel = pick(wks, mode)
        return np.concatenate([ids[w][sel[w]] for w in wks])

    train_ids = select(train_weeks, args.train_users)
    val_ids = select(val_weeks, args.val_users)
    all_ids = select(final_weeks, args.final_users)
    if not len(train_ids) or not len(val_ids) or not len(all_ids):
        raise SystemExit("no examples for the chosen weeks and user sets")
    print(f"{len(train_ids):,} training examples, {len(val_ids):,} validation, {len(ids[base.TEST_START]):,} test")

    # Per-user seed features: log1p, standardised on the training examples.
    feats = []
    if args.history:  # the test week is not in tables, so it is built here without labels
        hist = {**tables, base.TEST_START: base.build_week(base.TEST_START, ev, own, lags, data_end, n,
                                                           with_labels=False)}
        cols = [c for c in hist[base.TEST_START].columns
                if c not in ("eligible", "cidx", "week") and not c.startswith("y_")]
        feats.append(np.log1p(np.concatenate([hist[w][cols].values[c] for w, c in zip(all_weeks, cidx)])
                              .astype(np.float64).clip(0)))
        print(f"{len(cols)} history features on the seed token")
    if args.habit:
        R, C = review.load_review_detail(cohort)
        feats.append(np.log1p(np.concatenate([review.habit_features(w, R, C, lags, n).values[c]
                                              for w, c in zip(all_weeks, cidx)])))
        print(f"{feats[-1].shape[1]} habit features on the seed token")
    seed_x = None
    if feats:
        seed_x = np.concatenate(feats, 1)
        seed_x = (seed_x - seed_x[train_ids].mean(0)) / (seed_x[train_ids].std(0) + 1e-6)
    S = Samples(nodes, hops, paths, np.concatenate(week), np.concatenate(cidx), y, edges, device, seed_x)

    def make_model():
        return RelGT(G, cutoffs, week_day, n, len(EVENT_COLS) + args.review_aux, channels=args.channels,
                     local_layers=args.num_layers, heads=args.num_heads, dropout=args.dropout,
                     num_centroids=args.num_centroids, conv_type=args.conv_type, ablate=args.ablate,
                     n_seed_feats=0 if seed_x is None else seed_x.shape[1], path_token=args.path_token,
                     rel_pe=args.rel_pe, attn_bias=args.attn_bias,
                     action_edges=args.action_edges != "none", chain=args.graph_structure == "chain",
                     max_hop=args.max_hops + 1 if args.graph_structure == "chain" else MAX_HOP,
                     multi_seed=args.num_seeds > 1 or args.chain_seed == "latest-per-type",
                     recency_days=args.recency_days if args.recency_attn else None,
                     time_emb=args.time_emb).to(device)

    print(f"model parameters: {sum(p.numel() for p in make_model().parameters()):,}")
    model, best_epoch, history = fit(make_model, S, train_ids, args, args.epochs, val_ids=val_ids)
    print(f"best epoch {best_epoch}")

    # Thresholds on the pooled validation weeks (same rule as the baseline).
    metrics = {"args": vars(args), "best_epoch": best_epoch, "history": history, "per_label": {},
               "validation": []}
    if model.recency is not None:  # per head, then the readout
        metrics["recency_decay"] = model.recency.decay().tolist()
        print("learned recency decay: " + " ".join(f"{d:.2f}" for d in metrics["recency_decay"]))
    usel = pick(val_weeks, args.val_users)
    val_p = {w: predict(model, S, ids[w][usel[w]])[:, :len(EVENT_COLS)] for w in val_weeks}
    if args.exclude_push_label:  # never predicted
        for w in val_weeks:
            val_p[w][:, IP] = -1.0
    rates, pooled, thr = {}, {}, np.zeros(len(EVENT_COLS))
    for j, k in enumerate(EVENT_COLS):
        if args.exclude_push_label and j == IP:
            thr[j], rates[k], pooled[k] = np.inf, 0.0, np.array([-1.0])
            metrics["per_label"][k] = {"threshold": float("inf"), "val_f1": None, "val_pred_rate": 0.0, "val_ap": None}
            print(f"{k:18s} not predicted")
            continue
        yv = np.concatenate([tables[w][f"y_{k}"].values[tables[w].eligible.values][usel[w]]
                             for w in val_weeks])
        pv = np.concatenate([val_p[w][:, j] for w in val_weeks])
        m = ~np.isnan(yv)
        pooled[k] = pv[m]
        thr[j], f1 = base.best_threshold(yv[m], pv[m])
        rates[k] = float((pv[m] >= thr[j]).mean())
        metrics["per_label"][k] = {"threshold": thr[j], "val_f1": f1, "val_pred_rate": rates[k],
                                   "val_ap": float(average_precision_score(yv[m], pv[m]))}
        print(f"{k:18s} val F1 {f1:.4f}  AP {metrics['per_label'][k]['val_ap']:.4f}  threshold {thr[j]:.3f}  "
              f"predicted {rates[k]:.3f}  positives {yv[m].mean():.3f}")

    # Per-week scores on validation weeks where all 6 labels are known.
    for w in val_weeks:
        t = tables[w]
        m = t.eligible.values
        if t.loc[m, [f"y_{k}" for k in EVENT_COLS]].isna().any().any():
            continue
        gids = cohort.github_user_id.values[m][usel[w]]
        truth = base.to_output(gids, {k: t[f"y_{k}"].values[m][usel[w]] for k in EVENT_COLS})
        pred = base.to_output(gids, {k: val_p[w][:, j] >= metrics["per_label"][k]["threshold"]
                                     for j, k in enumerate(EVENT_COLS)})
        r = {"week": str(w), "users": int(usel[w].sum()), **base.scores(pred, truth)}
        metrics["validation"].append(r)
        print(f"validation {str(w)[:10]}: score {100 * r['score']:.2f}")
    metrics["val_score_mean"] = float(np.mean([r["score"] for r in metrics["validation"]]))
    print(f"mean validation score {100 * metrics['val_score_mean']:.2f}")

    # Thresholds tuned jointly on the score over the same weeks (experiments/gnn_review); kept if better.
    def labels(w):
        return tables[w][[f"y_{k}" for k in EVENT_COLS]].values[tables[w].eligible.values][usel[w]]

    full = [w for w in val_weeks if not np.isnan(labels(w)).any()]
    joint, _ = review.joint_thresholds(np.concatenate([labels(w) for w in full]).astype(bool),
                                       np.concatenate([val_p[w] for w in full]), thr, False)
    if args.exclude_push_label:
        joint[IP] = np.inf
    metrics["variants"] = {}
    for v, t in [("per-label F1 thresholds", thr), ("joint thresholds", joint)]:
        ws = [review.score(labels(w).astype(bool), val_p[w] >= t) for w in full]
        metrics["variants"][v] = {"weekly": ws, "mean": float(np.mean(ws)), "thresholds": t.tolist()}
        print(f"  {v:25s} " + "  ".join(f"{100 * s:.2f}" for s in ws) + f"   mean {100 * np.mean(ws):.2f}")
    metrics["chosen"] = max(metrics["variants"], key=lambda v: metrics["variants"][v]["mean"])
    chosen = np.array(metrics["variants"][metrics["chosen"]]["thresholds"])
    rates = {k: float((pooled[k] >= chosen[j]).mean()) for j, k in enumerate(EVENT_COLS)}
    print(f"chosen: {metrics['chosen']}")

    # Final model: all labelled weeks for the selected number of epochs; the validation share of
    # predicted positives is the decision rule on the test week (as in the baseline).
    final, _, _ = fit(make_model, S, all_ids, args, best_epoch)
    p = predict(final, S, ids[base.TEST_START])
    # Holdout weeks: never seen by either model or the thresholds. "final model" decides like the test
    # week (the validation share of positives per label); "validation model" uses the chosen thresholds.
    hsel = pick(holdout_weeks, args.val_users)
    metrics["holdout"] = []
    dump = {}
    if holdout_weeks:  # the validation model's probabilities on the tune weeks (experiments/ensemble/run.py)
        dump["tune_weeks"] = np.array([str(w)[:10] for w in val_weeks])
        for w in val_weeks:
            t = tables[w]
            m = t.eligible.values
            dump[f"tune_{str(w)[:10]}_gids"] = cohort.github_user_id.values[m][usel[w]]
            dump[f"tune_{str(w)[:10]}_prob"] = val_p[w].astype(np.float32)
            dump[f"tune_{str(w)[:10]}_truth"] = t[[f"y_{k}" for k in EVENT_COLS]].values[m][usel[w]].astype(np.float32)
    for w in holdout_weeks:
        t = tables[w]
        m = t.eligible.values
        if t.loc[m, [f"y_{k}" for k in EVENT_COLS]].isna().any().any():
            continue
        hid = ids[w][hsel[w]]
        gids = cohort.github_user_id.values[m][hsel[w]]
        truth = base.to_output(gids, {k: t[f"y_{k}"].values[m][hsel[w]] for k in EVENT_COLS})
        ph = {"final model": predict(final, S, hid)[:, :len(EVENT_COLS)],
              "validation model": predict(model, S, hid)[:, :len(EVENT_COLS)]}
        preds = {"final model": [np.zeros(len(hid), bool) if args.exclude_push_label and j == IP else
                                 ph["final model"][:, j] >= np.quantile(ph["final model"][:, j], 1 - rates[k])
                                 for j, k in enumerate(EVENT_COLS)],
                 "validation model": [ph["validation model"][:, j] >= chosen[j] for j in range(len(EVENT_COLS))]}
        r = {"week": str(w), "users": int(len(hid))}
        for v, cols in preds.items():
            r[v] = base.scores(base.to_output(gids, dict(zip(EVENT_COLS, cols))), truth)
        metrics["holdout"].append(r)
        dump[f"{str(w)[:10]}_gids"] = gids  # lets holdout predictions of several runs be joined per user
        dump[f"{str(w)[:10]}_truth"] = t[[f"y_{k}" for k in EVENT_COLS]].values[m][hsel[w]]
        for v, cols in preds.items():
            dump[f"{str(w)[:10]}_{v.split()[0]}"] = np.stack(cols, 1)
            dump[f"{str(w)[:10]}_prob_{v.split()[0]}"] = ph[v].astype(np.float32)
        print(f"holdout {str(w)[:10]}: " + "  ".join(f"{v} {100 * r[v]['score']:.2f}" for v in preds))
    if metrics["holdout"]:
        metrics["holdout_score_mean"] = {v: float(np.mean([r[v]["score"] for r in metrics["holdout"]]))
                                         for v in ("final model", "validation model")}
        print("mean holdout score: " + "  ".join(f"{v} {100 * s:.2f}" for v, s in metrics["holdout_score_mean"].items()))
    out = pd.DataFrame({"UserID": test.UserID.values})
    for j, k in enumerate(EVENT_COLS):
        if args.exclude_push_label and j == IP:
            out[k] = 0
            continue
        out[k] = (p[:, j] >= np.quantile(p[:, j], 1 - rates[k])).astype(int)
    out.to_csv(out_dir / "output.csv", index=False)
    # Test-week probabilities of the final model and the validation share of positives per label, so that
    # experiments/relgt/bag_submit.py can bag several seeds into one submission.
    np.savez(out_dir / "test.npz", user_id=test.UserID.values, prob=p[:, :len(EVENT_COLS)].astype(np.float32),
             rates=np.array([rates[k] for k in EVENT_COLS]), labels=np.array(EVENT_COLS))
    if dump:
        np.savez(out_dir / "holdout.npz", **dump)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {out_dir / 'output.csv'}")


if __name__ == "__main__":
    main()
