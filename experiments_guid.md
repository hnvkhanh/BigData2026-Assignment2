# Experiment implementation guide

Use this guide when adding a model or variant under `experiments/`. The goal is
for an experiment to be reproducible, leakage-safe, comparable with existing
baselines, and analyzable from its saved outputs.

## 1. Start with a precise experiment definition

Before writing the model, document:

- The prediction unit, normally one user at a weekly forecast cutoff.
- The target labels and forecast horizon. This task uses six labels
  (`open_pr`, `open_issue`, `issue_comment`, `pr_review`,
  `pr_review_comment`, `push`) over the seven days following the cutoff.
- Every input feature, its lookback window, aggregation, and availability rule.
- Eligibility filters separately from model features. For example, a 90-day
  activity filter may determine which users enter the dataset without making
  90-day aggregates model inputs.
- The architecture and how it differs from existing experiments.
- Whether a run is a backtest, prediction workflow, or both.

Do not silently change the target definition, split, eligible user set, or
feature windows between a baseline and its comparison variant.

## 2. Keep historical inputs cutoff-safe

For each historical example, construct features only from records known before
that example's cutoff:

- Require the action timestamp to be earlier than the cutoff.
- Require the event row and referenced entity to have been available by the
  cutoff. Use the repository's `available_at` and `entity_available_at`
  fields where applicable.
- Use a half-open future label window, `[cutoff, cutoff + 7 days)`, so actions
  exactly at the cutoff belong to the target, not the history.
- Do not use future actions, future entity metadata, holdout labels, or
  holdout metrics for training, feature selection, calibration, or model
  selection.

Reuse shared definitions and builders where possible, especially the label
order, `TrainingExamples`, graph snapshots, event loader, and chronological
split in `experiments/relgt/data.py` and `experiments/relgt/run.py`. If an
experiment needs different inputs, build those inputs explicitly from
cutoff-safe source data. For example, a specialist that excludes push must
remove push events before snapshot construction, including any graph context
derived only from push events.

Record the actual input restrictions, windows, eligibility policy, and target
labels in the experiment README and metrics metadata. Keep documentation in
sync with code constants: do not claim a 90-day feature window if the runner
constructs 14 days of data.

## 3. Use chronological evaluation

Use weekly cutoffs in time order; never randomly split user-week rows across
time. Unless there is a specific justified design, follow the repository's
chronological split convention: the latest cutoff week for holdout (the task predicts
a single week), the week before it for calibration, and every earlier week for
training. Keep all users for a cutoff in
the same temporal partition.

- Fit model parameters only on the training partition.
- Select probability thresholds and, if applicable, fusion weights on the
  calibration partition only.
- Measure and report final performance on holdout only after choices are
  fixed.
- If an experiment has multiple model variants, use the same examples,
  cutoffs, and split so results are comparable.
- Fail with a clear error if a required partition is empty or there are too
  few cutoffs for the split.

Store thresholds, split boundaries, counts of examples/cutoffs, and the
threshold-selection method with the metrics. Do not select variants by
repeatedly inspecting the holdout; reserve it for final comparisons.

## 4. Specify features and model I/O clearly

Use stable label order from the shared `LABELS` definition. Make each model's
output dimension match the labels it predicts, and explicitly map partial
outputs back to label names. For a specialist or grouped experiment, validate
that combined models cover every expected label exactly once before computing
overall metrics.

Feature construction should:

- Use explicit functions with clear inputs and outputs, such as a feature
  builder returning a typed array, token structure, or dataframe.
- State array dimensions and dtypes; validate matching row counts and label
  dimensions before training.
- Keep user IDs and cutoffs for alignment and analysis, but exclude identifiers
  from model features unless their use is intentional and documented.
- Handle missing/no-activity history deterministically, using the existing
  conventions where applicable.
- Avoid building features from the current prediction batch if the value is
  intended to be a global statistic. Derive statistics from the full
  cutoff-available population instead.
- Apply transforms consistently during training, calibration, holdout, and
  prediction.

When fusing model components, distinguish representation fusion inside one
jointly trained model from an ensemble of independently trained models.
Document which it is, the exact probability/representation operation, branch
weights if any, and where weights are selected.

## 5. Organize implementation and CLI

Prefer one package per standalone experiment:

```text
experiments/<experiment_name>/
    __init__.py
    run.py
    model.py                 # when the architecture merits a separate module
    test_<experiment_name>.py
    README.md
    requirements.txt         # only for experiment-specific dependencies
    outputs/                  # generated at runtime; do not commit run outputs
```

Follow existing project helpers and conventions instead of copying shared
loaders, label definitions, metric calculations, split logic, or device
selection. Keep model architecture code separate from orchestration when that
makes the model easier to test.

Provide a command-line entry point with sensible, documented defaults. Common
options include data/output directories, start cutoff, device, random seed,
epochs, batch size, and model-specific hyperparameters. Validate invalid
values early with actionable CLI errors. Use a unique timestamped output
directory by default and allow an explicit `--output-dir`; do not overwrite
another run's outputs implicitly.

Support both `backtest` and `predict` when the experiment is intended to
produce submission predictions. If it is intentionally backtest-only, state
that limitation in its README. Keep commands for runnable experiments in the
repository-root `cmd.md`.

## 6. Save comparable and analyzable results

At minimum, a backtest should save:

- `backtest_metrics.json`: experiment name, architecture/variant, model
  settings, random seed, split boundaries, example counts, thresholds,
  per-label metrics, macro-F1, micro-F1, and the assignment score.
- `holdout_probabilities.csv`: one row per held-out user and cutoff, actual
  labels as `actual_<label>`, and predicted probabilities named
  `<branch>_<label>`.
- Trained model checkpoint(s), with filenames identifying the model/branch.

Use the common identity columns `user_id` and `cutoff`, and preserve the
shared label names. For a combined or specialist run, save branch probabilities
as well as the combined output so comparisons and `exam_fail_case.py` can
inspect each branch. Verify that every holdout row aligns to the correct
actual labels; do not rely on row order alone when joining model outputs.

For prediction runs, save evaluator-format `predictions.csv`, unthresholded
`probabilities.csv`, model checkpoint(s), and metadata containing cutoff,
thresholds, feature/architecture settings, and relevant training/calibration
counts. Never include target labels that are unavailable at prediction time.

If examples or features are cached, keep caches inside the run folder. Include
data-file signatures and all feature-defining parameters (lookback, eligibility
window, feature version, and similar settings) in the cache key/metadata.
Reject stale or incompatible caches rather than silently reusing them.

## 7. Make experiments testable and reproducible

Set Python, NumPy, and framework seeds before model initialization and
shuffling. Record the seed and effective settings in saved metadata. Avoid
hard-coded machine-specific paths, silent dependency fallbacks, and broad
exception handling that hides failures.

Add focused tests for the experiment's behavior, including as applicable:

- Feature values, window boundaries, and cutoff/availability filtering.
- Model output shapes for full-label and specialist/grouped variants.
- Loss/training and inference on a small synthetic batch.
- Label ordering and output-to-label mapping.
- Calibration thresholds or fusion behavior using calibration examples only.
- Holdout files, required columns, probability ranges, and metadata.
- Invalid CLI settings and incompatible options.

Before considering the experiment complete:

1. Run its focused tests and a short CPU smoke backtest.
2. Check CLI help and reject invalid options.
3. Check syntax, formatting, and whitespace with the repository's configured
   tools.
4. Inspect the saved metrics and holdout CSV from the smoke run.
5. Confirm the README and `cmd.md` match the actual code and outputs.

Keep test fixtures small and synthetic; do not require access to the full
assignment dataset for unit tests. Do not commit generated checkpoints, caches,
or run outputs.