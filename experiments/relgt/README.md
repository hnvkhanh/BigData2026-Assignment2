# RelGT with a chain graph (`experiments/relgt/run.py`)

Relational Graph Transformer (Dwivedi et al. 2025) predicting the 6 weekly activity labels
(open_pr, open_issue, issue_comment, pr_review, pr_review_comment, push) for each (cohort user,
week start `s`). Plain PyTorch port of the official code; examples, labels and time split are the
baseline's (`experiments/baseline/run.py`). This note covers the **chain graph** experiments
(`--graph-structure chain --chain-seed latest --num-seeds 16`), which differ from the plain FK
graph described in the top-level README.

## 1. Node set

One node per row of each of the 8 tables, in one global id space (`Graph.offset`):

Users, Repositories, PullRequests, Issues, IssueComments, PRReviews, PRReviewComments, Pushes.

Each node has a type, a row time (`TIME_COL`: `first_seen_at` or `created_at`, in days since the
data start) and table-specific features (flags, log counts, one-hot categories, text as log
length). Columns only known later (`merged_at`, `closed_at`, `dismissed_at`) enter as
"happened before the cutoff" flags.

**Activity rows** are the six non-entity tables. Each has an *actor* (`ACTOR_COL`):
`author_id` (PRs, issues, issue comments, review comments), `reviewer_id` (reviews), `user_id`
(pushes). Users and Repositories are not actions.

## 2. Edges

### Plain FK graph (`--graph-structure fk`, the baseline for comparison)

One undirected edge per foreign key (`FKEYS`), e.g. `Pushes.user_id -> Users`,
`IssueComments.issue_id -> Issues`. `PullRequests.merged_by_id` is excluded (known only after
the merge). Problem: every action links straight to its user, so a user with thousands of rows
is a huge hub and the 1-hop sample is dominated by whatever table is most frequent (pushes).

### Chain graph (`--graph-structure chain`)

Built in `Graph.__init__`:

1. **Actor FK removed.** An activity row no longer links to its actor.
2. **Chain edges.** Rows are sorted by (actor, row time) per table. Each row gets an edge to the
   same actor's *next* row of the same table (relation `CHAIN_REL`; the reverse direction is
   `CHAIN_REL + 1`). So each user has up to 6 chains, one per action type, ordered in time.
3. **All other FKs stay:** repo, PR, issue, review, reply-to, `pull_request_id`, and `assignee_id`
   (self-assigned issues keep their link to the user, drawn as a plain FK; see decision log below).
4. **User links are made per example** (`subgraph_edges`, because the "latest" row depends on the
   cutoff):
   * the user links to the **latest visible row of each of its chains** (up to 6 links);
   * any activity row of **another** user among the tokens links to that user, and that user joins
     the tokens (the "owner" rule).

Because a table's rows become visible in time order, the visible part of a chain is always a
prefix, so a chain edge never points past the cutoff.

```
User ── latest PR ── prev PR ── prev PR ── ...        (chain: open_pr)
     ── latest Push ─ prev Push ─ ...                  (chain: push)
     ── latest IssueComment ─ ...                      (chain: issue_comment)
         │ repo_id / issue_id / pr_id ...  (kept FKs to Repositories, Issues, PullRequests)
         └── other users' rows ── their owner (User token)
```

The relation ids (`RELATIONS`, `N_REL`) index the FK edge types; chain edges and their reverse are
extra ids, used by the GIN positional encoding.

## 3. Visibility (no leakage)

`visible_nodes`: a node of table T is visible to week `s` only if its row time is before
`s - lag_T`, the staleness table T has at the test week (`LAG_KEY`; Users/Repositories use the
latest). `--exclude-push` hides every Pushes row; `--lookback-days N` hides activity rows older than
N days. Chain and owner edges are only built among visible tokens.

## 4. Token sampling (`sample_walk`)

K = 100 tokens per example (`--num-neighbors`).

1. **Seeds (`--chain-seed latest --num-seeds 16`).** Token 0..15 are the user's 16 latest visible
   actions (any table, newest first; fewer if the user has fewer), all hop 0. The user is a hop-1
   token. A user with no visible action keeps the user as seed. Their time is kept, so the model
   sees how long ago the user was last active.
2. **Breadth-first walk** from all seeds, nearest hops first, up to `--max-hops` (50 in the exp2 and
   fk runs, 20 in exp4). Hop 1 additionally contains the latest visible row of each of the user's chains. Each
   hop's candidates are capped at 1,000 neighbours per node (random).
3. **Budget.** A hop with more new nodes than the remaining budget gives a random subset. A row
   costs 1, or 2 if it is the first to bring in a new owner user (that user also becomes a token, one
   hop further). The walk does not continue *from* another user.
4. **Fill.** If fewer than K nodes are reachable, random reached nodes repeat; if none, random
   visible rows (hop `max_hops + 1`).

Because the newest actions are nearest, the walk goes **back in time** along each chain.

## 5. Model

Per token: node type, hop, relative time to `s`, row features (per-table residual MLP), and a GIN
positional encoding over the sampled subgraph (random inputs, `edge_index` from `subgraph_edges`),
concatenated and mixed by an MLP. Then a 1-layer local transformer (128 channels, 4 heads), readout =
mean of the seed tokens + attention-weighted others, global attention to an EMA vector-quantised
codebook, FFN, and a 6-output head. Training: Adam lr 3e-4, batch 512, 15 epochs, epoch chosen on
validation mean AP, thresholds from joint tuning or per-label F1, final model retrained on all
labelled weeks.

## 6. Experiments run

Common settings: chain graph, `--chain-seed latest`, `--num-seeds 16`, K=100, 15 epochs.
`end` is the last label date; "all data" means no `--end` cap.

| Run (`outputs/`) | Data end | Pushes in graph | Max hops | Val mAP | Holdout (final model) |
|---|---|---|---|---|---|
| `exp2_nopush` | 2026-06-01 | no | 50 | 0.5916 | 0.6004 |
| `exp2_all_data` | all | yes | 50 | 0.5195 | 0.4769 |
| `exp4_all_maxhop20` | all | no | 20 | 0.5086 | 0.4623 |
| `fk_exp1` (plain FK graph) | 2026-06-01 | yes | 50 | 0.5846 | 0.5897 |
| `fk_exp2_nopush` (plain FK graph) | 2026-06-01 | no | 50 | 0.5851 | 0.5908 |

Notes:
* Val mAP is the mean average precision over the labels; scores on the `end=2026-06-01` window and on
  the all-data window are not comparable (different weeks and different label rates).
* `fk_*` runs use `--graph-structure fk` (single user seed, user linked directly to every row) with
  otherwise the same settings and window as `exp2_nopush`, so they are the baseline for the chain
  graph. Without pushes, chain beats FK by +0.010 holdout (0.600 vs 0.591). Under FK, pushes in the
  graph give no gain (0.590 vs 0.591): the push hub swamps the 1-hop sample.

## 7. Decision log

* **Training uses the same chain graph as `analysis/relgt_graph.py --interactive`** (user links to the latest
  visible row per table; another user's sampled row brings that user in as a token).
* **Self-assigned issues keep their FK link to the user** (`Issues.assignee_id`), drawn as a plain
  FK even though they may be older rows. Assignment is a different relationship from authorship;
  do not remove it to make user links strictly latest-only.
* Chain results produced before `run.py` was saved on 2026-11-05 15:44 (`chain_latest`,
  `chain_latest_nopush`) used the older graph and are not comparable with later ones.

## 8. Reproduce

```bash
python experiments/relgt/run.py --graph-structure chain --chain-seed latest --num-seeds 16 \
    --epochs 15 --end 2026-06-01 --exclude-push --name exp2_nopush
python experiments/relgt/run.py --graph-structure chain --chain-seed latest --num-seeds 16 \
    --epochs 15 --exclude-push --max-hops 20 --name exp4_all_maxhop20
# plain FK graph baselines
python experiments/relgt/run.py --graph-structure fk --epochs 15 --end 2026-06-01 --name fk_exp1
python experiments/relgt/run.py --graph-structure fk --epochs 15 --end 2026-06-01     --exclude-push --name fk_exp2_nopush
```

Each run writes `output.csv`, `metrics.json` (all args are stored in it) and, with `--holdout-weeks`,
`holdout.npz` to `experiments/relgt/outputs/<name>/`. The npz has, per holdout week `<date>`, the user ids
(`<date>_gids`), labels (`<date>_truth`), the decisions and probabilities of the final and validation
models (`<date>_final`, `<date>_prob_final`, `<date>_validation`, `<date>_prob_validation`), and for each
tune week `tune_<date>_{gids,prob,truth}` (validation model; since 2026-11-16, for
`experiments/ensemble/run.py`). Tokens are cached in `experiments/relgt/cache/`.
`python analysis/relgt_graph.py --interactive` draws the graph for a given user and week.
