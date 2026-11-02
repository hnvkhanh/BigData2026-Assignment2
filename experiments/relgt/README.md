# RelGT-inspired graph branch

This experiment adds a relational graph model alongside the existing LightGBM
baseline. It is an assignment-specific implementation inspired by the
[Relational Graph Transformer (RelGT)](https://github.com/snap-stanford/relgt)
paper and its public implementation; it is **not** a drop-in run of Stanford's
RelBench training code.

The design references the upstream [`model.py`](https://github.com/snap-stanford/relgt/blob/main/model.py),
[`encoders.py`](https://github.com/snap-stanford/relgt/blob/main/encoders.py),
and [`local_module.py`](https://github.com/snap-stanford/relgt/blob/main/local_module.py):
typed/time-aware neighborhood tokens, local self-attention, and global
centroid-attention readout are adapted to the assignment-specific snapshots.

## What this implementation models

At each historical weekly cutoff, it creates a temporal heterogeneous graph
from cohort actions in the preceding 90 days:

- Target users connect to typed Pull Request, Issue, and Repository nodes through
  six activity relations (open, comment, review, inline review comment, push).
- PR and Issue nodes connect to their Repository at hop 2. Repository nodes are
  also reached directly by Push relations.
- The target node and sampled neighbors are tokenized with entity features,
  node type, relation type, hop distance, relative time, and local structure.
  At most four recent entity neighbors per activity relation and 24 repositories
  are retained for each user.
- A local Transformer encodes each sampled neighborhood. Learnable centroids
  attend to that neighborhood, and a global attention readout combines the
  centroid representation with the target user representation.
- One model predicts all six next-week labels with binary cross-entropy.

This follows RelGT's five-component tokenization and local/global attention
design, with a much smaller sampled graph and PyTorch-only implementation so it
can run on Apple Silicon MPS. It does not reproduce RelGT's vector-quantized
centroid codebook, full RelBench feature encoders, or all cross-table edges.
The branch is therefore a transparent RelGT-inspired experiment, not a claim of
an exact RelGT reproduction.

Historical snapshots require both the event row and its referenced node to be
available before the cutoff. PRs, Issues, and PRReviews use `first_seen_at`;
IssueComments, PRReviewComments, and Pushes use `created_at`. Action timestamps
are used for history windows and future labels. PR conversations are excluded
from the `open_issue` label, matching the assignment definition.

## Environment

The root `requirements.txt` contains the existing data and LightGBM
dependencies. This branch additionally requires PyTorch:

```bash
python -m pip install -r requirements.txt -r experiments/relgt/requirements.txt
```

On an Apple Silicon Mac, a recent standard PyTorch install includes MPS support.
At startup the script reports the number of visible CUDA GPUs and, when
`nvidia-smi` is available, each GPU's utilization and memory use. `--device
auto` selects the CUDA GPU with the lowest reported utilization (breaking ties
by memory use); if CUDA is unavailable, it selects MPS, then CPU. You can
explicitly use `--device cuda` to select the least-loaded CUDA GPU, or choose
`--device mps` / `--device cpu`. If `nvidia-smi` cannot report utilization, it
warns and uses free GPU memory as the selection fallback. The code does not
require PyTorch Geometric or a Stanford RelGT clone.

The optional `hybrid` mode also requires the pinned S2P2/EasyTPP dependency:

```bash
python -m pip install -r experiments/lightgbm_baseline/s2p2-requirements.txt
```

In hybrid mode RelGT learns from the temporal graph snapshots while S2P2 learns
from each user's six-mark action sequence. Both branches train on the same
chronological training weeks. A shared probability-fusion weight is selected
from `0.0` to `1.0` on calibration weeks, along with per-label thresholds; the
holdout is used only for final evaluation. S2P2 uses CPU when CUDA is
unavailable because its upstream implementation warns against MPS. On CUDA,
automatic device selection chooses the least-utilized GPU.

The `relgt_lightgbm` mode combines RelGT with the LightGBM baseline and uses
the root `requirements.txt` dependencies; it does not require EasyTPP. It builds
the baseline's cutoff-safe tabular features, aligns them to RelGT user-week
examples, and selects a shared probability-fusion weight and label thresholds
using calibration weeks only. The LightGBM features include recent total
actions, per-action shares, and per-day action frequencies over 7-, 30-, 90-,
and 365-day windows, plus the cutoff-available `is_bot` feature. The frequency-
only neural network in the baseline remains restricted to its frequency inputs.

The `relgt_lightgbm_s2p2` mode combines all three branches. It requires both
the base/RelGT dependencies and the optional pinned S2P2/EasyTPP dependency.
The three non-negative probability weights are selected on a simplex grid with
0.25 increments using calibration weeks only, together with per-label
thresholds. The S2P2 dependency is checked before training starts.

## Run a chronological backtest

From the project root:

```bash
python -m experiments.relgt.run backtest
```

The default settings train for five epochs, calibrate per-label F1 thresholds
on later weeks, and score a final chronological holdout. Every invocation
creates a unique timestamped run folder, such as
`experiments/relgt/outputs/backtest_20261102_001234_12345/`. Its report and
temporal graph cache are kept inside that folder, so concurrent runs do not
overwrite one another.

For a quick first run with fewer training epochs:

```bash
python -m experiments.relgt.run backtest --epochs 1 --batch-size 128
```

Run the combined RelGT + S2P2 model with:

```bash
python -m experiments.relgt.run backtest \
  --temporal-model hybrid \
  --device auto \
  --epochs 1 \
  --s2p2-epochs 1
```

The run report includes the calibration-selected RelGT/S2P2 fusion weights.
`holdout_probabilities.csv` records the aligned RelGT, S2P2, and fused
probabilities for each holdout user-week, and the run folder contains both
trained model weights. S2P2 training settings can be adjusted with
`--s2p2-batch-size`, `--s2p2-max-sequence-length`, `--s2p2-hidden-size`,
`--s2p2-state-size`, `--s2p2-num-layers`, and `--s2p2-integral-samples`.

Run RelGT combined with LightGBM with:

```bash
python -m experiments.relgt.run backtest \
  --temporal-model relgt_lightgbm \
  --epochs 1 \
  --n-estimators 300
```

Fusion is calibrated on the same chronological calibration weeks used for
threshold selection; the holdout is reserved for final evaluation. The report
includes the fused score and separate RelGT and LightGBM holdout metrics.
`holdout_probabilities.csv` contains the aligned branch and fused
probabilities. The run saves `relgt_model.pt` and one
`lightgbm_<label>.txt` file per trained label. LightGBM settings can be
adjusted with `--n-estimators`, `--learning-rate`, `--num-leaves`,
`--min-child-samples`, `--subsample`, `--colsample-bytree`, and `--reg-lambda`.

Run all three models together with:

```bash
python -m experiments.relgt.run backtest \
  --temporal-model relgt_lightgbm_s2p2 \
  --epochs 1 \
  --s2p2-epochs 1 \
  --n-estimators 300
```

The backtest report records all three fusion weights and the individual branch
holdout metrics. `holdout_probabilities.csv` includes RelGT, LightGBM, S2P2,
and fused probabilities. Both secondary branch weights and the RelGT weights
are saved in the run folder.

Each run gets its own cache to keep concurrent runs isolated. Use
`--output-dir` to choose a specific folder when you want to organize or rerun
an experiment:

```bash
python -m experiments.relgt.run backtest \
  --epochs 5 --hidden-dim 64 --num-heads 4 --num-layers 2 \
  --output-dir experiments/relgt/outputs/baseline_64d
```

Other useful options include `--start-cutoff`, `--device`, `--batch-size`,
`--hidden-dim`, `--num-heads`, `--num-layers`, `--dropout`, and `--seed`.
`--hidden-dim` must be divisible by `--num-heads`.

## Generate predictions

Use the latest assignment data release and its `test_input.csv`:

```bash
python -m experiments.relgt.run predict
```

By default this uses the assignment cutoff `2026-11-11 16:00:00 UTC` and
creates a unique timestamped folder under `experiments/relgt/outputs/`. It
writes:

- `predictions.csv`: evaluator-format binary labels.
- `probabilities.csv`: unthresholded per-label
  probabilities, for later comparison or probability-level fusion.
- `model.pt` for RelGT-only runs, or `relgt_model.pt` plus the selected
  secondary branch's weights for combined runs, and `metadata.json`.

Select the RelGT + LightGBM ensemble for prediction with:

```bash
python -m experiments.relgt.run predict \
  --temporal-model relgt_lightgbm \
  --output-dir experiments/relgt/outputs/relgt_lightgbm_prediction
```

The output includes fused predictions and probabilities, separate
`relgt_<label>_probability` and `lightgbm_<label>_probability` columns,
`relgt_model.pt`, LightGBM model files, and metadata containing the calibrated
fusion weights and LightGBM settings.

To generate predictions with all three models, use:

```bash
python -m experiments.relgt.run predict \
  --temporal-model relgt_lightgbm_s2p2 \
  --output-dir experiments/relgt/outputs/relgt_lightgbm_s2p2_prediction
```

The probabilities file includes branch-specific RelGT, LightGBM, and S2P2
columns as well as fused probabilities. The output folder contains
`relgt_model.pt`, `s2p2_model.pt`, LightGBM model files, and metadata with the
calibration-selected three-branch weights.

Training and calibration examples for prediction are restricted to weekly
samples whose complete seven-day label window ends on or before the requested
prediction cutoff. RelGT's graph snapshots use only actions and entity rows
available before each sample cutoff.

Override the cutoff, test-user input, or output location with flags:

```bash
python -m experiments.relgt.run predict \
  --cutoff "2026-11-11 16:00:00" \
  --test-input assignment2/evaluation/test_input.csv \
  --output-dir experiments/relgt/outputs/candidate
```

Pass an explicit `--output-dir` to both the prediction and evaluator commands
when you want a stable, known path. Omit it for timestamped, collision-free
outputs when launching multiple experiments.

Validate the output format and user IDs:

```bash
python assignment2/evaluation/evaluate.py \
  --pred experiments/relgt/outputs/predictions.csv \
  --input assignment2/evaluation/test_input.csv
```

Run final prediction only after the assignment's latest data release is
available. Use a separate output directory to keep candidate metrics and
predictions from overwriting one another.

## Compare with LightGBM

Compare LightGBM, RelGT, RelGT+S2P2, RelGT+LightGBM, and
RelGT+LightGBM+S2P2 runs using the same cutoff dates and chronological split.
Compare the assignment score
`(macro-F1 + micro-F1) / 2` on the untouched holdout. Combined models are
experiments, not assumed improvements; retain one only if it improves the
measured holdout score over the individual branches.
