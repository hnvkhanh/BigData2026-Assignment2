# Relational GNN action specialists

This experiment trains relational GNN specialists on the same chronological
user-week examples. The push model is shared across variants:

- `push_specialist_model.pt` receives the full six-relation graph history and
  predicts only `push`.

Two non-push variants are available:

- **`all_non_push`**: `non_push_specialist_model.pt` sees all five non-push
  action relations and predicts the five non-push labels together.
- **`grouped`**: `pr_specialist_model.pt` sees only PR-related history and
  predicts `open_pr`, `pr_review`, and `pr_review_comment`; meanwhile
  `issue_specialist_model.pt` sees only issue-related history and predicts
  `open_issue` and `issue_comment`.
- **`both`** (default): trains both variants, so their combined holdout scores
  can be compared on the same examples.

Grouped histories are filtered before graph construction, so each specialist
does not receive unrelated action nodes. Repository context is derived only
from the selected action group. The push specialist still sees every action.
The disjoint predictions are assembled by label into
`combined_specialists` (all-non-push variant) or `combined_grouped` (grouped
variant); probabilities are not averaged. Thresholds are selected on
calibration weeks, and combined six-label metrics are evaluated on the
chronological holdout. `holdout_probabilities.csv` includes actual labels,
both combined branches when available, and specialist probabilities, making it
usable with `exam_fail_case.py`.

All variants use 14-day graph snapshots and the same eligible user-week rows
and chronological split.

Run from the repository root:

```powershell
python -m experiments.relgt_action_specialists.run
```

For a quick smoke backtest:

```powershell
python -m experiments.relgt_action_specialists.run --epochs 1 --batch-size 64
```

Choose a single variant with `--variant all_non_push` or `--variant grouped`.
Omit it or use `--variant both` to train both for comparison.

Options include `--db-dir`, `--output-dir`, `--start-cutoff`, `--device`,
`--epochs`, `--batch-size`, `--hidden-dim`, `--num-layers`, `--dropout`,
`--learning-rate`, `--seed`, and `--variant`.
