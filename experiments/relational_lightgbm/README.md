# LightGBM with cross-action relational features

Most models predict `push` well but the other five actions poorly. Those actions
depend less on a user's own rhythm and more on **relationships**: who owns the
repo, who authored the PR, who reviewed there before, who is assigned the issue,
and what other people just did on those objects. This experiment encodes such
relationships as explicit per-user features and adds them to the LightGBM
baseline.

Everything except the new features is the LightGBM baseline
(`experiments/lightgbm_baseline/run.py`): the same weekly cutoffs (Wednesday
16:00 UTC from 2026-03-11), the same 5-actions-in-90-days eligibility, the same
baseline features, LightGBM settings, 70/15/15 chronological split, fixed
thresholds (0.30, push 0.45) and per-week official scoring. Scores therefore
compare directly with `lgbm`, and `--groups none` reproduces the baseline
through this code path.

## Run

```bash
python -m experiments.relational_lightgbm.run backtest
python -m experiments.relational_lightgbm.run backtest --groups none              # baseline control
python -m experiments.relational_lightgbm.run backtest --groups reviewer,sequence  # ablation
python -m experiments.relational_lightgbm.run predict
python -m pytest experiments/relational_lightgbm
```

A backtest writes `backtest_metrics.json`, `holdout_probabilities.csv`
(`actual_<label>` and `relational_lightgbm_<label>`), `feature_importance.csv`
(normalised gain per label), `hypothesis_lift.csv` and `models/<label>.txt`.
Prediction writes `predictions.csv`, `probabilities.csv`, `metadata.json` and the models.

## Features

All counts are for one user at one cutoff `t`. "Others" means any user other than the
one being predicted. Windows: 7 days (`_7d`), 30 days (`_30d`), and open items
first seen within 90 days.

| Group | Feature | Hypothesis |
|---|---|---|
| `owner` | `own_push_share_30d`, `own_repos_pushed_30d` | users push to repos they own |
| | `own_repo_new_issues_by_others_7d`, `own_repo_open_issues_by_others` | owners push, open PRs and comment after an issue is opened on their repo |
| | `own_repo_issue_comments_by_others_7d` | owners reply to discussion on their repo |
| | `own_repo_new_prs_by_others_7d`, `own_repo_open_prs_by_others` | owners review incoming PRs |
| `sequence` | `feature_branch_pushes_7d` | PRs follow pushes to non-mainline branches |
| | `unpaired_branches_7d` | feature branches pushed in the last 7 days with no PR from this user yet (matched on `head_ref`) |
| | `new_issues_on_pushed_repos_7d` | pushes follow new issues on repos the user works on |
| `assigned` | `assigned_open_issues`, `assigned_new_issues_7d` | assignees push and open PRs |
| `reviewer` | `reviewed_repos_90d` | past reviewers on a repo review there again |
| | `new_prs_on_reviewed_repos_7d`, `open_prs_on_reviewed_repos` | proxy for "requested as reviewer": new or open PRs by others on repos the user reviewed in the last 90 days |
| | `open_prs_reviewed_by_user` | PRs the user reviewed in the last 30 days that are still open (follow-up reviews or comments) |
| `author` | `open_authored_prs`, `open_authored_issues` | authors keep commenting on their open PRs and issues |
| | `reviews_received_7d`, `review_comments_received_7d`, `issue_comments_received_7d` | authors reply to feedback from others |
| `repo_activity` | `active_repos_30d`, `neighbor_*_7d` | activity by others on repos the user touched in the last 30 days |
| `mentions` | `mentions_received_7d`, `mentions_received_30d` | `@login` of the user in someone else's issue body, issue comment, review or review comment |
| `threads` | `thread_new_comments_7d` | new comments by others in issue/PR threads the user opened or commented on |
| `review_cycle` | `own_prs_merged_7d`, `merged_others_prs_7d` | merges (`merged_at` earlier than `t`) of the user's PRs, and PRs by others that the user merged |
| | `approvals_received_30d` | approvals by others on the user's PRs |
| | `open_prs_with_recent_push` | the user's open PRs whose head branch they pushed to in the last 7 days |
| `habit` | `<label>_active_weeks_8w`, `any_active_weeks_8w` | number of the last 8 weeks in which the user did each action |

The first six groups test the hypotheses this experiment started from (`--groups hypotheses`).
The last four came out of the data exploration below. `--groups all` (the default) uses
all ten groups.

### Leakage rules

- Every row must be acted **and** available before `t` (PRs and issues: `first_seen_at`;
  reviews: `submitted_at` and `first_seen_at`; comments and pushes: `created_at`).
- A PR or issue counts as open at `t` if `closed_at` is null or not earlier than `t`.
  That value is known only later, so it is used only when it is earlier than `t`.
- A PR whose opening falls before the data window uses `first_seen_at` as its opening time.

### Data limitations

- **Ownership** compares `Repositories.owner_login` with `Users.login`, case-insensitively.
  Both hold the latest value seen, so a user or repo renamed later may be
  misattributed. Organisation repos have no single owner and never count.
- **Requested reviewers are not in the data.** The `reviewer` group approximates them
  with the repos where the user has reviewed before.
- **Only cohort users' activity is collected.** "Activity by others" is therefore
  only what other cohort users did, and `*_received_7d` and
  `own_repo_issue_comments_by_others_7d` are rarely non-zero.
- `Pushes.is_new_ref` is `False` on every row in this release, so this experiment
  has no "new branch" feature.

`hypothesis_lift.csv` checks each hypothesis on the **training weeks only**. For
each pair of a feature and the label it should raise, it gives
`P(label | feature > 0)` and `P(label | feature = 0)`.

## Exploring the data for more relationships

```bash
python -m experiments.relational_lightgbm.explore
```

The exploration uses only the 22 training weeks (162,730 user-weeks). It writes
`relationship_lift.csv`, with every feature × label lift, and `transition_lift.csv`.
Besides the plain lift, it reports `new_lift`: the same lift measured only on
user-weeks where the user did **not** do the label in the previous 7 days. A
high `new_lift` means the relationship predicts something the user's own recent
activity does not.

Action transitions: the lift of doing the column action next week, given the row
action in the last 7 days.

| from \ next week | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| open_pr | 4.96 | 3.04 | 3.26 | 3.04 | 3.25 | 1.63 |
| open_issue | 2.77 | 11.36 | 4.90 | 3.00 | 3.34 | 1.52 |
| issue_comment | 3.04 | 5.19 | 8.69 | 4.90 | 5.81 | 1.60 |
| pr_review | 2.68 | 2.92 | 4.56 | 11.10 | 12.52 | 1.61 |
| pr_review_comment | 2.82 | 3.17 | 4.95 | 10.06 | 15.46 | 1.61 |
| push | 4.60 | 3.60 | 4.32 | 4.72 | 5.00 | 3.00 |

Reviews and review comments are tightly coupled in both directions, and pushes
precede PRs. Relationships found by mining the data, each measured with `new_lift`:

| Relationship | Feature → label | Support | new_lift |
|---|---|---|---|
| weekly habit | `pr_review_comment_active_weeks_8w` → pr_review_comment | 23,487 | 14.7× |
| weekly habit | `pr_review_active_weeks_8w` → pr_review | 31,470 | 11.4× |
| weekly habit | `open_issue_active_weeks_8w` → open_issue | 31,432 | 6.7× |
| being @mentioned | `mentions_received_30d` → issue_comment / review comment / review | 1,460 | 5.5× / 5.1× / 4.3× |
| activity in a joined thread | `thread_new_comments_7d` → issue_comment | 248 | 4.1× |
| merging others' PRs | `merged_others_prs_7d` → open_pr | 16,922 | 3.5× |
| own PR just merged | `own_prs_merged_7d` → open_pr | 5,468 | 3.0× |
| open PR still being pushed | `open_prs_with_recent_push` → open_pr / review comment | 19,701 | 2.4× / 2.3× |

Two other candidates were tried and dropped, because only cohort users' activity
is collected and they almost never fire: replies to the user's review comments
(8 user-weeks) and "changes requested" on the user's PRs (25).

## Results

Data release with pushes up to 2026-10-26. The split is 22 train / 5 calibration /
5 holdout weeks (holdout from 2026-09-16), with fixed thresholds. Values are holdout
means over weeks.

| | `--groups none` (baseline) | `--groups hypotheses` | `--groups all` |
|---|---|---|---|
| **score** | 0.5339 | **0.5401** | **0.5408** |
| macro-F1 | 0.4858 | 0.4941 | 0.4917 |
| micro-F1 | 0.5819 | 0.5862 | 0.5898 |
| open_pr | 0.4535 | 0.4643 | 0.4594 |
| open_issue | 0.4664 | 0.4724 | 0.4727 |
| issue_comment | 0.4399 | 0.4435 | 0.4492 |
| pr_review | 0.4464 | 0.4651 | 0.4612 |
| pr_review_comment | 0.4275 | 0.4382 | 0.4211 |
| push | 0.6813 | 0.6810 | 0.6863 |

The mined groups add almost nothing for LightGBM (+0.0007). The baseline
features already include each action's recency and longer-window shares, which
overlap heavily with the 8-week habits. Fed to RelGT/GNN, which see only 14
days of history, the features raise macro-F1 but leave the combined score flat
(see `experiments/relgt_relational`).

With the hypothesis groups, the model scores higher in every one of the 5 holdout weeks
(+0.004 to +0.009).

Lift on training weeks (`hypothesis_lift.csv`), as `P(label | feature > 0)` / `P(label | feature = 0)`:

| Hypothesis | Feature → label | Rows with feature | Lift |
|---|---|---|---|
| past reviewer reviews again on the same repo | `reviewed_repos_90d` → pr_review | 53,951 | 18.6× |
| past reviewer and new PRs on that repo | `new_prs_on_reviewed_repos_7d` → pr_review_comment / pr_review | 18,009 | 12.9× / 11.4× |
| owner comments on new issues on own repo | `own_repo_new_issues_by_others_7d` → issue_comment | 902 | 6.5× |
| PR follows feature-branch pushes | `unpaired_branches_7d` → open_pr | 56,136 | 3.4× |
| PR after an issue on own repo | `own_repo_new_issues_by_others_7d` → open_pr | 902 | 2.2× |
| PR author comments on own PR | `open_authored_prs` → issue_comment | 110,047 | 2.0× |
| push to own repos | `own_push_share_30d` → push | 107,098 | 1.5× |
| assignee opens PR / pushes | `assigned_open_issues` → open_pr / push | 21,366 | 1.5× / 1.3× |

Every hypothesis holds in the data, but the score gain is small. The reason is that
the baseline's per-action recency (`<label>_days_since_last`) already captures most of
the same users. For example, someone who reviewed on a repo recently also has a small
`pr_review_days_since_last`. The relational features carry 13–14% of total gain for the
non-push labels and about 5% for push. `pr_review` benefits most:
`open_prs_reviewed_by_user` and `open_prs_on_reviewed_repos` are its 2nd and 3rd most
important features.
