# Experiment: temporal heterogeneous GNN (`experiments/gnn/run.py --graph-structure hetero`)

The user–repo GNN reduces every user–repo pair to one edge of windowed counts, and its time
encoding (first / last age of the pair) adds at most noise. This model keeps the events and the
objects they act on, TGAT-style. One full graph per week, built only from rows visible before
each table's cutoff (same weeks, lags, eligibility and split as the baseline):

- *Nodes*: the 9,000 cohort users (the 60 baseline features), and the PRs, issues and
  repositories reached by an edge (test week: 24k PRs, 17k issues, 22k repos). An object's
  features are its visible cohort events per activity and its distinct cohort users, plus days
  since first seen for a PR / issue and the organisation flag for a repo.
- *Edges, one per event* (90-day window, `--history-days`):
  - user → PR / issue / pushed repo per activity, keeping the latest 32 per user and activity
    (`--het-max-events`);
  - repo → user `received`;
  - PR / issue → its repository.

  Every relation also runs reversed with its own weights (~300k edges per week).
- *Time embedding*: each edge's age (days before its cutoff; for object → repo, days since the
  object was first seen) goes through a learned `cos(w·log(1+age) + b)` encoding
  (`--het-time-dim 16`, one encoder each for events, received and membership).
- *Model* (`TemporalHeteroGNN`), 2 layers, dim 64:
  - message `W_r[h_src ‖ φ(age)]`;
  - attention `a_r·LeakyReLU(Q h_dst + m)`, softmaxed over a node's incoming edges of each
    relation;
  - relations summed per node type, then residual + LayerNorm;
  - head on the user nodes.

  Training (AdamW lr 2e-3, early stopping, thresholds) is the same as for the user–repo GNN.
- `--het-time none` feeds zeros in place of φ. The architecture stays the same, so the
  comparison isolates the time embedding.

```bash
python experiments/gnn/run.py --graph-structure hetero --holdout                  # -> outputs/all_het_ho/
python experiments/gnn/run.py --graph-structure hetero --het-time none --holdout  # -> outputs/all_het_notime_ho/
bash runs/gnn_ho1107.sh   # het and het_notime x seeds 0-2 under --holdout -> outputs/het_ho_s*, het_notime_ho_s*
```

**Results** (the 11-07 holdout protocol of [ego_gnn.md](ego_gnn.md), 3 seeds, runs `het_ho_s0-2` and `het_notime_ho_s0-2`):

| Model | Run | Holdout score, mean (min–max) | Best epoch |
|---|---|---|---|
| Temporal hetero GNN | `het` | **38.71** (37.98–39.59) | 11, 5, 12 |
| Same, time embedding off | `het_notime` | 37.42 (36.90–37.78) | 11, 5, 12 |

Mean holdout F1 per activity:

| Activity | het | het_notime | Difference |
|---|---|---|---|
| open_pr | 0.380 | 0.371 | +0.009 |
| open_issue | 0.432 | 0.418 | +0.014 |
| issue_comment | 0.393 | 0.382 | +0.011 |
| pr_review | 0.386 | 0.383 | +0.003 |
| pr_review_comment | 0.335 | 0.303 | +0.032 |

**Result: the time embedding is worth about 1.3 points, but the hetero GNN does not beat the
earlier models.**
- With the time embedding the score is 38.71, against 37.42 without it. It is higher on every
  activity, mostly on pr_review_comment (+0.032); pr_review barely moves (+0.003). The seed
  ranges do not overlap (37.98–39.59 against 36.90–37.78), but with 3 seeds and a seed spread
  of ~1–2 points this is suggestive, not conclusive.
- Against the models of the 11-07 holdout table in [ego_gnn.md](ego_gnn.md), 38.71 is level with the user–repo GNN (38.92)
  and the ego GNN with user features (38.80), within the seed spread. Keeping the individual
  events and objects instead of one windowed edge per user–repo pair brings no measurable gain.
- Without the time embedding the hetero GNN is about as good as the plain ego GNN (36.53)
  and 1.5 points under the user–repo GNN.
- Both variants have the same best epochs (11, 5, 12) per seed, and train for as long as the
  user–repo GNN (5–12 epochs), unlike the ego models.
- One holdout week, 3 seeds; tuning-week scores were not recorded here.

## 14-day window (`--history-days 14`)

The same two models when they only see the last 14 days before each cutoff:
- event edges, received edges and object / repo counts come from the 14 days;
- a PR's or issue's days since first seen are clipped at 14;
- the user features are recomputed from the 14 days (39 instead of 60: the 30d / 90d / all-time
  counts are dropped).

Eligibility and labels are unchanged, so the runs are scored on the same users as above. The
test-week graph shrinks to 2.2k PRs, 1.7k issues and 6.2k repos (~77k edges per week on average,
against ~300k).

```bash
bash runs/h14_ho1107.sh   # het_h14 and het_notime_h14 x seeds 0-2 under --holdout; summary next to het / het_notime
```

**Results** (holdout protocol, 3 seeds, runs `het_h14_ho_s0-2` and `het_notime_h14_ho_s0-2`):

| Model | 90 days | 14 days | Change | Best epoch (14 days) |
|---|---|---|---|---|
| Temporal hetero GNN | **38.71** (37.98–39.59) | 36.01 (35.54–36.43) | -2.70 | 11, 6, 4 |
| Same, time embedding off | 37.42 (36.90–37.78) | 36.67 (36.33–36.89) | -0.75 | 4, 6, 12 |

Mean holdout F1 per activity:

| Activity | het | het_h14 | het_notime | het_notime_h14 |
|---|---|---|---|---|
| open_pr | 0.380 | 0.357 | 0.371 | 0.350 |
| open_issue | 0.432 | 0.403 | 0.418 | 0.405 |
| issue_comment | 0.393 | 0.352 | 0.382 | 0.374 |
| pr_review | 0.386 | 0.375 | 0.383 | 0.390 |
| pr_review_comment | 0.335 | 0.290 | 0.303 | 0.300 |

**Result: the 14-day window hurts, and it removes the benefit of the time embedding.**
- The time embedding model loses 2.7 points (38.71 to 36.01), on every activity: most on
  pr_review_comment (-0.045) and issue_comment (-0.041). The seed ranges do not overlap.
- Without the embedding the loss is 0.75 points (37.42 to 36.67) and the ranges overlap, so
  that drop is within the seed spread. pr_review even gains slightly (+0.007).
- At 14 days the order flips: the model with time embedding scores 0.66 under the one without
  (36.01 against 36.67, ranges overlapping, so no real difference). The 1.3-point gain at 90 days
  disappears. A 14-day window leaves little age variation for the embedding to use, and the
  longer history it exploited at 90 days is gone.
- The extra history is useful, mainly to the model that can place it in time. A shorter window
  gives no gain and no cheaper way to match the 90-day score, although the graph is about 4x
  smaller.
- Best epochs are early and uneven (4-12). One holdout week, 3 seeds.

## Event graph: links between actions (`--het-graph event`)

The graph above only joins actions where their edges meet at the same user or object. Nothing
records that a review comment belongs to a review, that a push updated a PR, or which action
came after which. `--het-graph event` (`hetero_event_graph`) makes every kept event a node of
its own: the same latest 32 per user and activity, under the same visibility rules. Events can
then link to each other directly. Week 2026-09-26 has 9,000 users, 20k PRs, 13k issues,
29k repos and 177k events, with about 550k edges:

| Relation | From → to | Edge age | Edges (09-26) |
|---|---|---|---|
| `<activity>` (6) | user → own event | days before the cutoff | 177k |
| `<activity>_on` (6) | event → PR / issue / repo | days before the cutoff | 175k |
| `issue_comment_on_pr` | issue comment on a PR → the PR node | days before the cutoff | 1.8k |
| `push_to_pr` | push → the PR it likely updates | days before the cutoff | 6.7k |
| `next` | a user's event → that user's next event | gap (days) | 170k |
| `in_review` | review comment → its review | gap (days) | 2.5k |
| `reply_to` | review comment → the comment it answers | gap (days) | 77 |
| `received` | another user's event on my PR / issue → me | days before the `received` cutoff | 80 |
| `pr_in`, `issue_in` | PR / issue → its repo | days since first seen | 33k |

- **Where the links come from.**
  - `in_review` comes from `PRReviewComments.pull_request_review_id`, and `reply_to` from
    `in_reply_to_id`. Both need the two events in the graph.
  - An issue comment on a PR moves to the PR node when `Issues.pull_request_id` names the PR
    (about a third of PR issues do).
  - Text is masked in the data, so `push_to_pr` is a heuristic: a same-repo PR by the pusher
    whose masked head branch equals the pushed branch and that is open at the push. It links
    9% of pushes, and 37% of those links are ambiguous (the closest PR is taken).
- **Not linked.** An issue and the PR that fixes it can't be joined: PR bodies are masked
  (`#000`) and there is no closing-reference column.
- **Event node features.** One-hot activity, log age, whether it is on the user's own PR or
  issue, and whether it has a link (moved to its PR, linked to a PR, or part of a kept review).
- **Model.** Same `TemporalHeteroGNN` as above, with a fourth time-encoder family for gaps
  between events. It uses **4 layers** by default, because user → event → PR → event → other
  user is 4 hops. `--het-layers N` overrides this, and the folder name gets `_l<N>`.
- **Memory.** The 4 layers store about 12 GB of per-edge activations, which leaves no headroom on a
  12 GB card, so `--het-checkpoint on` (the default with `--het-graph event`) keeps only each layer's
  input and recomputes its activations in the backward pass. Same model and losses, peak 5.55 GB.
  Run one job per GPU.

```bash
# one run (holdout week 09-27 -> 10-04, all 6 activities, push included) -> outputs/all_het_ev_ho0926/
python experiments/gnn/run.py --graph-structure hetero --het-graph event --holdout --holdout-week "2026-09-26 16:00"
# ablation without the time embedding -> outputs/all_het_ev_notime_ho0926/
python experiments/gnn/run.py --graph-structure hetero --het-graph event --het-time none --holdout --holdout-week "2026-09-26 16:00"
# 3 seeds each, next to the object graph; prints the summary table at the end
CONFIGS="het_ev het_ev_notime" bash runs/ho0927.sh   # -> outputs/het_ev[_notime]_ho0926_s{0,1,2}/
# both GPUs at once: one config per card
GPU=0 CONFIGS="het_ev" bash runs/ho0927.sh &  GPU=1 CONFIGS="het_ev_notime" bash runs/ho0927.sh
# draw one user's 4-hop field (event-to-event links dashed)
python analysis/model_graph.py --model hetero-event --week "2026-09-26 16:00"          # app at http://127.0.0.1:8052
python analysis/model_graph.py --html --model hetero-event --week "2026-09-26 16:00"   # -> figures/graph_hetero-event_<login>_<date>.html
```

**Results** (holdout week 09-27 → 10-04, all 6 activities, 3 seeds, finished 2026-11-16; score = mean, range in brackets):

| Config | Score | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | Best epochs |
|---|---|---|---|---|---|---|---|---|
| `het_ev` | 51.71 (50.85-52.40) | 0.410 | 0.411 | 0.422 | 0.461 | 0.445 | 0.666 | 1, 6, 12 |
| `het_ev_notime` | 51.97 (51.45-52.26) | 0.403 | 0.434 | 0.408 | 0.459 | 0.444 | 0.668 | 1, 6, 12 |
| `het` (object graph, for comparison) | 52.77 (52.21-53.69) | 0.411 | 0.445 | 0.425 | 0.486 | 0.461 | 0.669 | 17, 6, 16 |
| `het_notime` | 52.34 (51.96-52.84) | 0.405 | 0.445 | 0.395 | 0.456 | 0.458 | 0.675 | 5, 12, 5 |

6-activity score by push rule (`het_ev` / `het_ev_notime`): tune 51.71 / 51.97, prev_week 50.89 / 50.93,
none 31.09 / 30.70.

- **The event graph does not help.** `het_ev` is about 1 point below `het`, and `het_ev_notime` about
  0.4 below `het_notime`. The gap to `het` is the size of the seed ranges, but the direction is the same
  with and without the time embedding, and no activity gains clearly (pr_review is lowest, 0.461 vs 0.486).
  Direct event links (`in_review`, `reply_to`, `push_to_pr`) cover few events (2.5k, 77, 6.7k of 177k),
  so they add little signal for the extra size.
- **Time embedding.** With the event graph, removing it changes the score by +0.26, within noise
  (`het`: -0.43 the other way). No clear effect in either graph.
- **Training is erratic.** Best epochs jump between 1 and 12, so the selected epoch is noisy.
- **Push rule.** `tune` is best, `prev_week` loses about 0.8-1.0, and `none` falls to about 31, so push
  carries the score here too.

For the best of the other GNNs on this week, `all_ego_uf` scores 52.98, `all_ego_uf_rel` 53.06
and `user_repo` 52.29 (3 seeds each). The top four configurations overlap within seed noise.
