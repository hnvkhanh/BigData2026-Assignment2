# Assignment 2: Predicting GitHub User Activity (write-up)

Predict, for each user in `test_input.csv`, which of 6 activities (`open_pr`, `open_issue`,
`issue_comment`, `pr_review`, `pr_review_comment`, `push`) they perform during
2026-11-22 00:00 – 11-29 00:00 Taiwan time (2026-11-21 16:00 UTC + 7 days). Score: the official
6-activity F1-based score of the course (`assignment2/scripts/evaluate.py`).

To run the best RelGT variant (chain graph + `--fk-order recent`) see [README.md](README.md).

## Summary

- Starting point: a LightGBM baseline on per-user count / recency features. GH Archive misses events in
  some periods, so part of the work handles that (outage weeks, relative features, quiet-user rules).
- Main line: graph models over the relational tables: ego-graph, user-repo and temporal heterogeneous GNNs,
  a chain-graph GNN / RelGNN, and a Relational Graph Transformer (RelGT) on the chain or foreign-key graph.
- Evaluation: a holdout week (09-27 → 10-04, 6,064 users). Training labels end by 09-19, the week before sets
  early stopping and the share of users predicted positive per activity, and the holdout week is used for
  nothing else. Every config has 3 seeds.
- Result: all graph models land at 52–54 on that week and their seed ranges overlap. The one gain that
  survives a paired bootstrap is averaging the seeds of RelGT on the chain graph (54.42 against 53.15 for the
  same bag of the chain GNN). On RelGT token sampling, `--fk-order recent` helps the weaker fk graph most and
  ties on the chain graph (bags 54.06 for chain + recent, 54.42 for chain, 54.10 for fk + recent).
- What did not work: picking a model per activity by tune week (`select` overfits it), mixing models (no better
  than the best bag), history features on the chain model (inside seed noise).
- Limits: one holdout week, so differences under about 1 point are not resolved, and the submission pipeline
  does not bag seeds.

## Approach

1. **Data.** Tables in `data/db/` built from GH Archive hour files for the course's cohort users; only the
   cohort's own activity is collected, so *received* activity is sparse. Weekly examples run from Saturday
   16:00 UTC; eligible users and labels follow the course's definition.
2. **Baseline** ([baseline](docs/experiments/baseline.md)): LightGBM, no-leakage feature windows, outage
   handling (`--gap-mode`, [outages](docs/experiments/outages.md)).
3. **Graph models**: ego GNN, user-repo GNN, hetero GNN with a time embedding
   ([hetero_gnn](docs/experiments/hetero_gnn.md)), GNN / RelGNN on the chain graph
   ([relgnn](docs/experiments/relgnn.md)), RelGT ([relgt](docs/experiments/relgt.md), design in
   [experiments/relgt/README.md](experiments/relgt/README.md)).
4. **Decision rule**: per activity, predict positive for the share of users that maximises F1 on the tune
   week; push uses that share or the previous week's push share.
5. **Analysis**: where the score is lost (quiet vs active users, capture drops) in
   [holdout_0927](docs/experiments/holdout_0927.md); seed bagging and model mixing in
   [ensemble](docs/experiments/ensemble.md).

## Setup

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # CUDA 13.0 build, as used
pip install -r requirements.txt
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
python scripts/download_gharchive.py --out data/gharchive --cohort-only --start 2025-12-01 --end 2026-11-16T09:00
python assignment2/scripts/preprocess.py --raw data/gharchive --out data  # writes the tables to data/db/
```

`assignment2/` (the course data) is not committed; it must sit at the repo root. The tables are read
from `data/db/`. The download range above (UTC, `--end` exclusive) is the one the submission was built from;
keep it to reproduce the submitted output, since the training / validation weeks and the visibility cutoffs
follow the last moment in the data. `--cohort-only` keeps only the lines `preprocess.py` uses, which saves disk.

## Repository layout

```
ghactivity/              shared library: data loading, per-week features and labels, outages, data periods, scoring
experiments/<name>/      one experiment per folder, run as python experiments/<name>/run.py; outputs/ next to it (not committed)
  baseline/              LightGBM baseline; writes the submission file
  gnn/                   GNNs: ego graph (default), user-repo, temporal hetero, chain / fk row graph (RowGNN or RelGNN)
  gnn_review/            hetero GNN using the review / review-comment link
  relations/             cross-table feature groups screened with LightGBM
  relgt/                 Relational Graph Transformer (README.md: its design)
  relgnn/                RelGNN model (atomic routes, composite message passing), trained by gnn/run.py --token-model relgnn
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
| `runs/gnn_chain_bag_ho0927.ps1` | chain GNN, seeds 0-2, validation and final (refit) stages at the recorded best epoch, then its bag and its mean with `relgt_chain` | 09-27 → 10-04 |
| `runs/sampling_ho0927.ps1` | RelGT token sampling: chain vs fk graph x `--fk-order` random vs recent, seeds 0-2 on several GPUs, then their summaries and bags | 09-27 → 10-04 |
| `runs/test_users_ho0927.ps1` | chain RelGT trained on the test users only, one seed, scored against `relgt_chain` on the same users | 09-27 → 10-04 |
| `runs/relgnn_ho0927.ps1` | RelGNN on the chain graph (user or seed readout) and the fk graph, seeds 0-2 on several GPUs, validation and final stages, then summaries and bags with `gnn_chain` and `relgt_chain` | 09-27 → 10-04 |
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
| `gnn_chain` (final model) | GNN on the chain graph, 1 epoch, retrained through the tuning week | 53.22 (52.73–53.52) | [gnn_chain_bag](docs/experiments/gnn_chain_bag.md) |
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

The seed ranges of the top configs overlap, so this one week does not rank them. Averaging the
three seeds of `relgt_chain` scores 54.42, +1.27 [+0.29, +2.30] over the same bag of `gnn_chain`
(53.15), which gains nothing from bagging ([gnn_chain_bag](docs/experiments/gnn_chain_bag.md),
[ensemble](docs/experiments/ensemble.md)). For reference, the course README reports 49.61 for *copy last week* and 54.82 for its
GNN with time encoding on the same week, with its own setup.

### RelGT token sampling (seed bags, holdout week 09-27)

Same week and protocol, 3 seeds per config; `recent` keeps the newest rows where tokens compete
([sampling](docs/experiments/sampling.md), [ensemble](docs/experiments/ensemble.md)).

| Config | Graph, `--fk-order` | Single seeds (mean) | Seed bag |
|---|---|---|---|
| `relgt_chain` | chain, random | 51.72–53.85 (52.86) | 54.42 |
| `relgt_chain_recent` | chain, recent | 52.91–53.71 (53.36) | 54.06 |
| `relgt_fk` | fk, random | 49.83–50.61 (50.10) | 52.66 (−1.40 [−2.49, −0.32] vs chain_recent) |
| `relgt_fk_recent` | fk, recent | 52.07–52.60 (52.27) | 54.10 |

Mixing the four bags gives 53.37 and picking per activity 54.32, neither better than the best bag.

## Documentation

| Doc | Content |
|---|---|
| [README.md](README.md) | how to run the chain + recent RelGT experiment (smoke, holdout, test-week prediction) |
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
| [docs/experiments/gnn_chain_bag.md](docs/experiments/gnn_chain_bag.md) | seed bagging of the chain GNN, with a final (refit) stage (`runs/gnn_chain_bag_ho0927.ps1`) |
| [docs/experiments/sampling.md](docs/experiments/sampling.md) | RelGT token sampling: graph structure x `--fk-order` (`runs/sampling_ho0927.ps1`) |
| [docs/experiments/relgt_ablation.md](docs/experiments/relgt_ablation.md) | ablation of RelGT's sampling components: chain graph, latest seed, recent fill (`runs/relgt_ablation_ho0927.ps1`) |
| [docs/experiments/relgt_seedtype.md](docs/experiments/relgt_seedtype.md) | RelGT seeded at the latest action of each action type (up to 6 seeds) instead of the 16 newest (`runs/relgt_seedtype_ho0927.ps1`); finished, not adopted (bag 53.6 / 52.8 vs 54.4 / 54.1 for the 16-seed pair) |
| [docs/experiments/test_users.md](docs/experiments/test_users.md) | the chain RelGT trained on the test users only (`runs/test_users_ho0927.ps1`) |
| [experiments/relgt/README.md](experiments/relgt/README.md) | RelGT design: node set, chain graph, visibility, token sampling |
| [docs/experiments/relgnn.md](docs/experiments/relgnn.md) | RelGNN on the chain and fk graphs vs `gnn_chain` / `relgt_chain` (`runs/relgnn_ho0927.ps1`) |
| [experiments/relgnn/README.md](experiments/relgnn/README.md) | RelGNN design: atomic routes of this schema, composite message passing, differences from the official code |
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
