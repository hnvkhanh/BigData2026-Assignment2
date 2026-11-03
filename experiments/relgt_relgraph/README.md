# RelGT / GNN on a relationship-augmented graph

In the `relgt` / `gnn` experiments each user is linked only to the PRs, issues
and repositories they **acted on in the last 14 days**. Most of the
relationships behind next week's PRs, reviews and comments are not edges in
that graph: repos the user owns, an issue assigned to them, a PR someone
@mentioned them in, a new PR on a repo they review, the people they work with.

This experiment adds those relationships as **typed edges from the user to the
actual entities**. Every edge becomes a token of the same form as the base
tokens: entity embedding, node type, relation type, hop, four node features and
age. RelGT attends over them together with the base neighbourhood, and the GNN
passes relation-specific messages along them. Each relationship gets its own
relation embedding (and, in the GNN, its own message weight matrix), so the model
can learn, for example, that an `assigned` issue matters differently from an
`open_authored_pr`.

An earlier attempt, `experiments/relgt_relational`, compressed the same
relationships into one feature vector on the root token. Here they stay as
graph structure.

## Relationship edges ([`graph.py`](graph.py))

| Relation | Edge to | Up to | Node features (log1p unless noted) | Age = days since |
|---|---|---|---|---|
| `owns` | repo the user owns with activity in 30 d | 4 | events by others, own pushes, open issues / open PRs by others | last event on the repo |
| `assigned` | open issue assigned to the user | 4 | comments in 7 d, issue age, total comments | issue first seen |
| `mentioned_in` | PR or issue where someone @mentioned the user (30 d) | 4 | mentions | last mention |
| `review_candidate` | open PR by someone else, first seen in 30 d, on a repo the user reviewed (90 d) or owns | 4 | user's reviews on the repo, PR age, reviews so far, owns repo (0/1) | PR opened |
| `open_authored_pr` | the user's open PR (first seen in 90 d) | 4 | approvals, reviews and review comments from others, head branch pushed in 7 d (0/1), age | last feedback or opening |
| `open_authored_issue` | the user's open issue | 4 | comments by others, those in 7 d, age | last comment or opening |
| `thread` | issue/PR thread the user joined, with comments by others in 7 d | 4 | new comments by others, user's own comments (30 d) | last comment by others |
| `reviewed_repo` | repo the user reviewed in 90 d | 4 | reviews, new PRs by others in 7 d, open PRs by others | user's last review there |
| `unpaired_branch` | repo with non-mainline branches pushed in 7 d that have no PR from the user yet | 4 | pushes, branches | last push |
| `interacted` | **another user** the user reviewed, was reviewed by, commented with, merged for, or @mentioned / was mentioned by (90 d) | 8 | interactions, partner's new PRs in 7 d, partner is in the cohort (0/1), partner's new issues in 7 d | last interaction |

That makes 44 extra slots per user, alongside the 48 base slots. Users are a new
node type (4) with their own entity embeddings, appended after the PR, issue and
repository ids. On the training data, eligible users get about 12 relationship
edges each.

When a relation has more candidates than slots, the edges kept are the most
recent or most active (see each builder's `score`).

**Leakage rules:** every row used must be acted and available before the cutoff,
and so must the entity an edge points to. `closed_at` and `merged_at` are used
only when earlier than the cutoff, to decide what is still open or recently merged.

**Data limitations:** only cohort users' activity is collected. Partners outside
the cohort (96% of `interacted` partners) appear only through their interactions
with cohort users. `mentioned_in` and `thread` are sparse because `IssueComments`
has few rows from June to August 2026.

## Setup

Everything else matches `experiments.relgt` and `experiments.relgt_relational`:
the 14-day base snapshots, five-action 14-day eligibility, weekly cutoffs from
2026-03-11 16:00 UTC, 70/15/15 chronological split, model sizes, AdamW at 1e-3,
5 epochs, per-label thresholds chosen on calibration weeks, and per-week official
scoring. `--select-epoch` keeps the epoch with the best calibration-week score.

`--relations none` is the control: the same model class, with the extra node and
relation types available but no relationship edges. `--relations a,b` keeps only
the listed relations, for ablations.

```bash
python -m experiments.relgt_relgraph.run backtest --select-epoch                  # RelGT, all relations
python -m experiments.relgt_relgraph.run backtest --architecture gnn --select-epoch
python -m experiments.relgt_relgraph.run backtest --architecture gnn --relations none --select-epoch
python -m experiments.relgt_relgraph.run predict --architecture gnn --select-epoch
python -m pytest experiments/relgt_relgraph
```

The relationship edges are cached in the run folder (`cache/relation_tokens.npz`),
keyed by the data files, start cutoff, graph version and relations.

## Results

The data release has pushes up to 2026-10-26. The split is 22 train / 5
calibration / 5 holdout weeks (holdout from 2026-09-16), with 5 epochs and
`--select-epoch`. Rows are seed means; the per-seed table comes from
`python -m experiments.relgt_relational.summarize experiments/relgt_relgraph/outputs`.

| Model | Relationship edges | Score | Macro-F1 | Micro-F1 | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|---|---|---|---|
| RelGT | none (control) | 0.5610 | 0.5160 | 0.6060 | 0.463 | 0.471 | 0.442 | 0.473 | 0.421 | 0.826 |
| RelGT | all | **0.5724** | **0.5294** | **0.6154** | **0.488** | **0.487** | 0.446 | **0.498** | 0.432 | 0.825 |
| GNN | none (control) | 0.5630 | 0.5162 | 0.6097 | 0.459 | 0.476 | 0.438 | 0.470 | 0.428 | 0.827 |
| GNN | all | **0.5714** | **0.5273** | **0.6156** | **0.489** | **0.491** | 0.439 | **0.484** | 0.436 | 0.825 |

Per seed, the gain over the control is:

| | seed 42 | seed 7 |
|---|---|---|
| RelGT | +0.008 (0.5672 vs 0.5590) | +0.015 (0.5777 vs 0.5630) |
| GNN | +0.004 (0.5631 vs 0.5595) | +0.013 (0.5798 vs 0.5665) |

What the results show:

- **The edges help where the root vector did not.** All four paired runs improve,
  and macro-F1 and micro-F1 rise together. The root-vector version
  (`relgt_relational`) traded micro-F1 for macro-F1 and left the score flat. On the
  same setup it averaged 0.5607 (RelGT) and 0.5657 (GNN), against 0.5724 and 0.5714
  here.
- **The gains are on the relationship-driven actions:** `open_pr` +0.025 to
  +0.030, `open_issue` +0.015, `pr_review` +0.014 to +0.025. `push`, which depends
  mostly on the user's own rhythm, does not change.
- **These are the best graph-model scores in the repository so far.** The
  earlier full comparison had `relgt` at 0.566 and `relgt_frac50` at 0.570; here
  the best single runs reach 0.5798 (GNN) and 0.5777 (RelGT).
- **Epoch selection matters more with the larger graph.** The calibration score
  peaks at epoch 2 and falls to epoch 5 (RelGT, seed 42: 0.5644 → 0.5458), so
  always run with `--select-epoch`.
- **Edge coverage per eligible user-week** (mean edges, share of users with at
  least one): `open_authored_pr` 2.2 / 76%, `owns` 1.9 / 77%, `interacted` 1.4 / 42%,
  `open_authored_issue` 1.0 / 38%, `review_candidate` 1.0 / 35%, `unpaired_branch`
  0.9 / 58%, `reviewed_repo` 0.8 / 43%, `assigned` 0.4 / 17%, `mentioned_in` 0.02 / 2%,
  `thread` ≈ 0. Because the last two are almost always empty, the gain comes from
  the other eight relations. An ablation with `--relations` can confirm which
  ones matter.
