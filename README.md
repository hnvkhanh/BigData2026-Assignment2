# Assignment 2: Predicting GitHub User Activity

Predict, for each user in `test_input.csv`, which of 6 activities (`open_pr`, `open_issue`,
`issue_comment`, `pr_review`, `pr_review_comment`, `push`) they perform during
2026-11-22 00:00 – 11-29 00:00 Taiwan time (2026-11-21 16:00 UTC + 7 days).

## Setup

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
```

`assignment2/` (the course data) is not committed; it must sit at the repo root. The tables are read
from `data/db/`, built by the course's `assignment2/scripts/preprocess.py --out data` from GH Archive
hour files (`scripts/download_gharchive.py --out data/gharchive --cohort-only` fetches them).

## Repository layout

```
ghactivity/              shared library: data loading, per-week features and labels, outages, data periods, scoring
experiments/<name>/      one experiment per folder, run as python experiments/<name>/run.py; outputs/ next to it (not committed)
  baseline/              LightGBM baseline; writes the submission file
  gnn/                   GNNs: ego graph (default), user-repo, temporal hetero, chain graph
  gnn_review/            hetero GNN using the review / review-comment link
  relations/             cross-table feature groups screened with LightGBM
  relgt/                 Relational Graph Transformer (README.md: its design)
  missing_data/          ways of handling GH Archive's missing events
  segments/              quiet vs active error split, capture drops by user type
  ensemble/              combines the saved probabilities of holdout runs (seed bagging, model mean, per-activity pick)
runs/                    multi-seed / multi-GPU run scripts; lib.sh holds their shared setup
analysis/                Dash apps and figure scripts (EDA, missing events, model graphs, schema)
scripts/                 utilities: download GH Archive, collect / summarize scores, clean archived runs
docs/                    experiment write-ups (docs/experiments/), EDA guide, action relationships, assignment PDF
figures/                 generated figures
archive/                 old code kept for reference, not maintained
```

Every script is run from anywhere; paths resolve from the repo root. Experiments import the shared
code as `import ghactivity as base` and each other as `from experiments.gnn import run as gnn`.

## Run

```bash
python experiments/baseline/run.py
python assignment2/scripts/evaluate.py --pred experiments/baseline/outputs/none/output.csv --input assignment2/submission/test_input.csv
```

Writes `experiments/baseline/outputs/<gap-mode>/output.csv` (rename to
`output_{studentID}.csv` for submission) and `metrics.json` with validation scores. Takes
about 30 s. Options: `--gap-mode {none,mask,drop,downweight,recent}`, `--gap-weight`,
`--seed` and the data-period options below (see `--help`).

**Data period** (baseline, GNN, GNN-review, relations and RelGT runners). A week counts as inside a period when its label window
lies fully in `[start, end]` (UTC). `--start`/`--end` bound every stage; `--train-start/-end`,
`--val-start/-end` and `--test-start/-end` override them for one stage (`test` = the weeks
the final models are fitted on). Without `--val-start/--val-end`, validation is the last
`--val-weeks` weeks; training weeks never overlap validation.

```bash
python experiments/baseline/run.py --start 2026-03-01 --end 2026-10-28
python experiments/baseline/run.py --train-start 2026-09-09 --val-start 2026-10-07 --val-end 2026-10-28
```

### Multi-seed runs (`runs/`)

Most scripts run configs × seeds 0–2 one run at a time on a free GPU (`GPU=<index>` forces one);
`gnn_ho1107.sh` keeps GPUs 0 and 1 busy at once, and the smoke scripts use seed 42. Every run logs to
`experiments/gnn/outputs/logs/<run>.log`, and each script ends with a summary table
(`scripts/summarize_runs.py`). `CONFIGS="..."` picks a subset; see each script's header and
`runs/lib.sh` (Python choice, GPU picking).

| Script | Runs | Scored week |
|---|---|---|
| `runs/ho0927.sh` | ego, hetero, event-graph and user–repo GNNs | 09-27 → 10-04, all 6 activities |
| `runs/qprev_ho0927.sh` | ego GNN, quiet users copy their previous week | 09-27 → 10-04 |
| `runs/chain_ho0927.sh` | RelGT and GNN on the chain graph | 09-27 → 10-04 |
| `runs/chain_ftime_ho0927.sh` | the same with a learned time embedding | 09-27 → 10-04 |
| `runs/ensemble_ho0927.sh` | ego, hetero, chain and chain + history reruns that save probabilities, then their ensembles (`SMOKE=1` for a quick check) | 09-27 → 10-04 |
| `runs/gnn_ho1107.sh` | ego, user–repo and hetero GNNs, two GPUs at once | 11-07 (no push labels) |
| `runs/h14_ho1107.sh` | hetero GNN that sees only the last 14 days | 11-07 |
| `runs/ego_time_ho1107.sh` | ego GNN with the hetero model's time encoding | 11-07 |
| `runs/smoke_may.sh`, `runs/smoke_may.ps1` | the best configs retrained on data before June (short smoke runs by default; the `.ps1` queues 10 configs over several GPUs) | 05-24 → 05-31 |

```bash
CONFIGS="het all_ego_uf" bash runs/ho0927.sh
python scripts/summarize_runs.py --suffix ho0926 het het_notime all_ego_uf user_repo   # the table again
```

## Results

Holdout week 09-27 → 10-04 (6,064 eligible users, official 6-activity score, 3 seeds, mean and
range). Training labels end by 09-19, the week 09-19 → 09-26 sets early stopping and the share of
users predicted positive, and the holdout week is used for nothing else.

| Config | Model | Score | Write-up |
|---|---|---|---|
| `relgt_chain` (final model) | RelGT on the chain graph, retrained through the tuning week | 53.12 (52.72–53.59) | [holdout_0927](docs/experiments/holdout_0927.md) |
| `ego_uf_qprev` | ego GNN + user features, quiet users copy their previous week\* | 53.08 (52.63–53.70) | [holdout_0927](docs/experiments/holdout_0927.md) |
| `all_ego_uf_rel` | ego GNN + user features + relative (capture-independent) features | 53.06 (52.85–53.47) | [gnn_missing_data_fixes](docs/experiments/gnn_missing_data_fixes.md) |
| `all_ego_uf` | ego GNN + user features | 52.98 (52.39–53.33) | [holdout_0927](docs/experiments/holdout_0927.md) |
| `het` | temporal heterogeneous GNN | 52.77 (52.21–53.69) | [hetero_gnn](docs/experiments/hetero_gnn.md) |
| `gnn_chain` | GNN on the chain graph | 52.48 (51.90–52.94) | [holdout_0927](docs/experiments/holdout_0927.md) |
| `het_notime` | hetero GNN without the time embedding | 52.34 (51.96–52.84) | [hetero_gnn](docs/experiments/hetero_gnn.md) |
| `user_repo` | user–repo GNN | 52.29 (52.07–52.47) | [user_repo_gnn](docs/experiments/user_repo_gnn.md) |
| `het_ev_notime` | hetero GNN on the event graph, no time embedding | 51.97 (51.45–52.26) | [hetero_gnn](docs/experiments/hetero_gnn.md) |
| `het_ev` | hetero GNN on the event graph | 51.71 (50.85–52.40) | [hetero_gnn](docs/experiments/hetero_gnn.md) |

\* An upper bound: quiet users get their real labels of the previous week, which the test week
cannot see.

The seed ranges of the top configs overlap, so this one week does not rank them. For reference, the course README reports 49.61 for *copy last week* and 54.82 for its
GNN with time encoding on the same week, with its own setup.

## Documentation

| Doc | Content |
|---|---|
| [docs/eda.md](docs/eda.md) | the interactive EDA app (`analysis/eda.py`) |
| [docs/experiments/baseline.md](docs/experiments/baseline.md) | LightGBM baseline: samples, no-leakage rules, features, validation score |
| [docs/experiments/outages.md](docs/experiments/outages.md) | training weeks hit by collection outages (`--gap-mode`) |
| [docs/experiments/missing_data.md](docs/experiments/missing_data.md) | handling GH Archive's missing events (`experiments/missing_data/run.py`) |
| [docs/experiments/ego_gnn.md](docs/experiments/ego_gnn.md) | ego-graph GNN, separate quiet / active models, holdout week 11-07 |
| [docs/experiments/holdout_0927.md](docs/experiments/holdout_0927.md) | holdout week 09-27 with push: GNNs, chain-graph models, quiet-user rule, where the score is lost |
| [docs/experiments/gnn_missing_data_fixes.md](docs/experiments/gnn_missing_data_fixes.md) | the LightGBM missing-data fixes in the GNN (`--relative-features`, `--push-share`) |
| [docs/experiments/user_repo_gnn.md](docs/experiments/user_repo_gnn.md) | user–repo GNN, all labels vs push only |
| [docs/experiments/hetero_gnn.md](docs/experiments/hetero_gnn.md) | temporal heterogeneous GNN, 14-day window, event graph |
| [docs/experiments/ego_time.md](docs/experiments/ego_time.md) | ego GNN with the hetero model's time encoding |
| [docs/experiments/relgt.md](docs/experiments/relgt.md) | Relational Graph Transformer results |
| [docs/experiments/ensemble.md](docs/experiments/ensemble.md) | ensemble of the tied graph models and the chain model with history features (`experiments/ensemble/run.py`) |
| [experiments/relgt/README.md](experiments/relgt/README.md) | RelGT design: node set, chain graph, visibility, token sampling |
| [docs/action_relationships.md](docs/action_relationships.md) | how the actions and tables relate (`experiments/gnn_review/`, `experiments/relations/`) |
| [docs/requirement_for_task.pdf](docs/requirement_for_task.pdf) | the assignment |

## Other tools

- `scripts/download_gharchive.py` (fetch GH Archive hours), `scripts/collect_scores.py` (gather every
  metrics file into one JSON), `scripts/summarize_runs.py` (holdout scores of multi-seed runs),
  `scripts/delete_archive_pt.py` (deletes `.pt` files and cache dirs under `results/archive_old_data`).
- `analysis/missing_data.py` (explorer app for GH Archive's missing events), `analysis/model_graph.py`
  (one user's example as each graph model sees it), `analysis/relgt_graph.py` (RelGT token graphs),
  `analysis/schema_graph.py` (draws `figures/schema_graph.*`).
- `experiments/segments/run.py`: quiet vs active error split of a holdout run and the capture drops by
  user type; see *Where the score is lost* in [holdout_0927](docs/experiments/holdout_0927.md).
- [archive/](archive/README.md): the first RelGT-inspired experiment.

## Data notes

- Only cohort users' own activity is collected, so *received* activity only covers
  interactions between cohort users and is sparse.
