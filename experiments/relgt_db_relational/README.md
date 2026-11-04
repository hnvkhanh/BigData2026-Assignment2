# relgt_db on the database plus relationship tables

`relgt_db` (`experiments/relgt/db_run.py`) applies RelGT to the database itself:
every row is a node, every foreign key an edge, and each user-week samples
K = 128 rows from the user's 1- and 2-hop neighbourhood. The relationships behind
next week's PRs, reviews and comments are mostly **not** within two hops of
the user in the raw schema:

- a new PR by someone else on a repo the user owns is user → push → repo → PR
  (three hops), and ownership is a login match, not a key;
- an @mention is text inside a body;
- a new PR on a repo the user reviews is user → review → repo → PR;
- the people a user works with are user → review → PR → author.

This experiment models them the way a RelBench database would: as **extra
tables with foreign keys** ([`tables.py`](tables.py)). RelGT then treats them
like any other table (own column encoder, node type, hop, time, and subgraph
structure). Nothing is turned into a hand-made feature vector.

| Table | Rows | One row per | Foreign keys | Columns |
|---|---|---|---|---|
| `RepoOwners` | 67,833 | repository a user owns (`owner_login` = `login`) | user, repo | known time |
| `Notifications` | 365,101 | event by someone else that concerns the user | user, PR, issue | kind, acted / known time |
| `Collaborations` | 41,732 | pair of users who interacted (review, comment, merge, mention) | user A, user B | first interaction time |

Notification kinds (counts over the whole release): `pr_on_reviewed_repo` 215,695,
`comment_on_authored` 55,315, `pr_on_owned_repo` 55,175, `mention` 19,445,
`issue_on_owned_repo` 8,954, `review_on_authored_pr` 5,772,
`review_comment_on_authored_pr` 4,447, `thread_reply` 273,
`authored_pr_merged_by_other` 25. (Merged PRs with an author other than the merger are
rare because 73% of merged PRs have no recorded author.)

A notification puts the PR or issue it is about two hops from the user. A
collaboration row puts the partner user two hops away. `Collaborations` is
symmetric, so it means the same from either side. `Notifications` has no key to
the actor, so a row never appears in the actor's own neighbourhood, where it
would mean something else.

**Leakage:** every row's time column is the moment it is known: the later of
the source event's availability (`first_seen_at` for PRs, issues and reviews,
`created_at` for comments, `merged_at` for merges) and the availability of the
rows it references. The `relgt_db` sampler already drops rows and edges not
known strictly before the seed time.

### Reserved tokens (`--reserve N`)

Upstream RelGT samples the K − 1 tokens uniformly from the 1/2-hop union. Pushes
are about half of that union, so with uniform sampling relationship rows get
about 21 of the 128 tokens and their targets fewer still. `--reserve N`
([`sampler.py`](sampler.py)) first fills up to N tokens with the seed's
relationship rows, most recent first and alternating between tables. Each row is
followed by the rows it points to: the PR or issue at hop 2, or at hop 1 when the
user is directly linked to it. The remaining tokens are sampled as upstream, never
repeating a reserved row.

## Run

```bash
python -m experiments.relgt_db_relational.run backtest --epochs 10                 # all tables, upstream sampling
python -m experiments.relgt_db_relational.run backtest --epochs 10 --reserve 32
python -m experiments.relgt_db_relational.run backtest --epochs 10 --tables none   # control = plain relgt_db
python -m experiments.relgt_db_relational.run backtest --epochs 10 --tables Notifications --reserve 32
python -m experiments.relgt_db_relational.run predict --epochs 10 --reserve 32
python -m pytest experiments/relgt_db_relational
```

All other options are those of `experiments.relgt.run` (`--num-neighbors`,
`--hidden-dim`, `--conv-type`, `--context-mode`, ...). Everything except the
database and sampler is `relgt_db`: user-weeks, labels, five-actions-in-14-days
eligibility, Wednesday 16:00 cutoffs from 2026-03-11, the 70/15/15 split, the
model, the best-calibration-AUPRC epoch and per-label thresholds. Outputs match
`relgt_db` and record `relationship_tables` and `reserve` in the metrics.

## Results

Split: 22 train / 5 calibration / 5 holdout weeks (holdout from 2026-09-16), K =
128, 10 epochs, defaults otherwise. Seeds 42 and 7; the grid is
[`outputs/run_grid.sh`](outputs/run_grid.sh). The **score** is the official
weekly (macro-F1 + micro-F1) / 2. **AUPRC** is the threshold-free macro average
precision on the same holdout rows.

| Variant | Score s42 | Score s7 | **Score mean** | Macro-F1 | Micro-F1 | **Holdout AUPRC mean** |
|---|---|---|---|---|---|---|
| control (`--tables none`) | 0.5706 | 0.5531 | **0.5619** | 0.5102 | 0.6136 | 0.5060 |
| tables, upstream sampling | 0.5372 | 0.5624 | 0.5498 | 0.5009 | 0.5987 | 0.5093 |
| tables, `--reserve 32` | 0.5211 | 0.5605 | 0.5408 | 0.4953 | 0.5863 | **0.5302** |

Holdout AUPRC per action (seed means):

| Variant | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| control | 0.465 | 0.447 | 0.428 | 0.467 | 0.406 | 0.824 |
| `--reserve 32` | **0.500** | **0.455** | **0.458** | **0.510** | **0.416** | **0.844** |

What the results show:

- **With reserved tokens, the relationship tables rank users better.** Holdout
  AUPRC rises by 0.024 on average (+0.010 and +0.038 by seed), and it rises for
  every action. The largest gains are on `pr_review` (+0.043), `open_pr` (+0.035)
  and `issue_comment` (+0.030), the actions the relationships were meant to help.
  The calibration-week AUPRC rises too, from about 0.50 to 0.55–0.56.
- **With upstream uniform sampling the tables do almost nothing** (AUPRC +0.003).
  At about 21 random tokens out of 128, the relationship rows are swamped by pushes.
- **The official F1 score does not follow, because the thresholds do not transfer.**
  Per-label thresholds are chosen on the calibration weeks (2026-08-19 to
  09-15). Those weeks include the two densest weeks of PR and issue collection
  since May: 7–10k notifications a week, against 300–1,600 in July–August and
  1–5k in the holdout weeks (see the table below). A model that relies on the
  relationship rows produces probabilities that shift with that density, so
  thresholds set on the dense weeks are too permissive or too strict on the
  holdout weeks. The F1 score falls even though the ranking improved. The
  control relies mostly on pushes, whose collection is steady, so it is less exposed.
- **Seed variance is large.** The control alone moves by 0.018 between seeds,
  and the selected epoch ranges from 1 to 7. Two seeds are enough to see the AUPRC
  gain, which holds at both seeds, but not to measure small score differences.

Notifications per week, near the split (all kinds):

| Week (Wed–Tue) | 07-29 | 08-05 | 08-12 | 08-19 | 08-26 | **09-02** | **09-09** | 09-16 | 09-23 | 09-30 | 10-07 | 10-14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rows | 320 | 1,628 | 670 | 840 | 803 | **11,231** | **13,141** | 6,775 | 3,028 | 5,187 | 2,697 | 2,663 |

Next steps that follow from this:

1. **Make the thresholds robust to the density shift**, for example by keeping each
   label's calibration-week positive *rate* per week (a quantile threshold) instead
   of a fixed probability. That would turn the AUPRC gain into an F1 gain if
   thresholding is the only cause.
2. **Add per-row density context**, such as the number of notifications in the
   user's window or the week's collection volume, so the model can separate "quiet
   user" from "thin data week".
3. Ablate tables (`--tables Notifications`, ...) and the reserve size (16 / 64).
