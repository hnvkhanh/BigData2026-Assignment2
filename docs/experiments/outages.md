# Experiment: training weeks hit by collection outages

Earlier setup (test week 11-12, validation weeks 10-07 / 10-14), not rerun. On the current GH
Archive data the outages are re-measured, and these modes compared again, in [Handling GH Archive's missing events](missing_data.md).

Daily cohort event counts show outages (listed in `GAPS` in `run.py`):

- **Non-push, 2026-06-01 → 09-07:** issues, PRs, comments and reviews drop to ~5–20% of
  their usual level and end with a backfill burst (~1,400 events/day on 9/4–9/6).
- **Push:** almost nothing 08-18 → 09-12 (≤ ~20% of the usual volume); days missing on 10-07, 10-15 → 10-18 and 10-23.
- October is quieter than spring for every activity but steadily so; the test week is in this
  regime, so it is not treated as an outage.

A training week is *affected* if its label week overlaps an outage, or if more than 25% of its
30-day feature window does. Modes (`--gap-mode`):

| Mode | Training weeks used | Validation score, mean of 3 seeds (min–max) |
|---|---|---|
| `none`: keep everything | 38 | **49.60** (49.46–49.69) |
| `downweight`: affected weeks weight 0.2 | 38 | 49.44 (49.24–49.74) |
| `mask`: drop a week only for labels whose label week has an outage | 23 non-push / 32 push | 49.25 (49.09–49.44) |
| `drop`: drop affected weeks | 19 | 49.19 (48.94–49.46) |
| `recent`: only weeks after the summer outage (09-09 on) | 4 | 49.17 (49.14–49.19) |

**Result: none of the treatments beats training on all weeks.** All differences (≤ 0.4) are
inside the seed-to-seed spread (up to 0.5), so `none` stays the default. A likely reason: the
model mostly ranks users by their own recent activity, and the decision rule is set on recent
weeks (threshold / predicted share), so a week-wide drop in activity shifts every user's
features and labels together and still gives useful ranking examples. Even 4 post-outage weeks
reach almost the same score, so more weeks add little.

Caveats: only 2 validation weeks (~4.5k users each) are scored, and those weeks contain push
outages themselves (10-07, 10-15 → 10-18), so push labels there are slightly understated.
