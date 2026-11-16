# Interactive EDA (`analysis/eda.py`)

```bash
python analysis/eda.py        # then open http://127.0.0.1:8050 (startup ~3 s)
```

A Dash + Plotly app over the cohort's activity, with the same activity definitions and
test-aligned weeks as the baseline. Global filters (date range, all / test / non-test users,
activities) drive the tabs: daily activity per table with the outages shaded, weekly label
rates and co-occurrence, persistence (P(active) by weeks since last active, week-to-week
transitions), quiet users, per-user concentration, time-of-day heatmaps, top repositories,
data-quality checks (staleness before the test week, how late rows appear, null rates), a
per-table column explorer and a per-user timeline. The shifted test selection rule reproduces
the 4,465 test users exactly.

The *Quiet users* tab covers the quiet segment of `experiments/gnn/run.py --segment activity`.
It uses the models' examples and per-table cutoffs and reproduces the 1,714 quiet test users.
The window and minimum number of rows can be changed (`--ego-lookback`,
`--active-min-events`). It shows:
- quiet share and return rates per week;
- label rates and the share of positives, quiet vs active;
- what predicts a return, against a break-even line (F1 / 2);
- activity hidden between the cutoff and the week start;
- how often each user is quiet;
- how the test week's quiet users compare.

Weeks distorted by the May cohort sampling or the summer outage are left out by default.
