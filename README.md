# How to run the chain + recent RelGT experiment

This guide runs `relgt_chain_recent`: the Relational Graph Transformer (`experiments/relgt/run.py`) on the
**chain graph**, seeded at the user's 16 latest actions, with `--fk-order recent` (where tokens compete for the
K = 100 slots, the newest rows win instead of random ones). The method, results and the other configs are in
[WRITEUP.md](WRITEUP.md) and [docs/experiments/sampling.md](docs/experiments/sampling.md).

## 1. Setup

```bash
pip install -r requirements.txt            # needs a CUDA build of torch for a reasonable run time
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
```

- `assignment2/` (course data) must sit at the repo root and the tables must exist in `data/db/` (built with
  `assignment2/scripts/preprocess.py --out data`, from GH Archive hours fetched by
  `python scripts/download_gharchive.py --out data/gharchive --cohort-only`). See *Setup* in [WRITEUP.md](WRITEUP.md).
- One GPU with a few GB free is enough. A run takes about 15-25 min of training, plus a one-off token sampling
  of about 6.5 min per seed for this config (cached in `experiments/relgt/cache/`, so a rerun is faster).
- Python: the scripts use `python` from the active environment; PowerShell scripts fall back to
  `.venv-win\Scripts\python.exe` if no virtualenv is active (`-Python <path>` overrides it).

## 2. Pipeline check (a few minutes)

```powershell
.\runs\sampling_ho0927.ps1 -Smoke -Configs relgt_chain_recent
```

One seed, 1 epoch of 30 steps, run name `smoke_relgt_chain_recent_ens0926_s0`. It should end with a
`mean holdout score` line and a summary table.

## 3. Reproduce the holdout result (week 2026-09-27 to 10-04 Taiwan time)

```powershell
.\runs\sampling_ho0927.ps1 -Configs relgt_chain_recent                  # seeds 0-2, GPUs 0 and 1
.\runs\sampling_ho0927.ps1 -Configs relgt_chain_recent -Seeds 0 -Gpus 0 # one seed on one GPU
.\runs\sampling_ho0927.ps1 -Configs relgt_chain_recent -SkipRuns        # only the summaries again
```

Options: `-Gpus`, `-Seeds`, `-HoWeek "2026-09-26 16:00"` (UTC start of the scored week), `-Python`,
`-FreeMiB` (a GPU with more memory than this in use is skipped until it frees up), `-EchoEvery` (seconds
between log echoes). Ctrl+C stops the running jobs too.

What it does: trains on weeks whose labels end by 09-19, tunes on 09-19 to 09-26 (early stopping and the share
of users predicted positive), scores the holdout week, and refits the final model through 09-26. Each seed is
saved to `experiments/relgt/outputs/relgt_chain_recent_ens0926_s<seed>/` (`metrics.json`, `holdout.npz` with the
probabilities, `output.csv`); logs go to `experiments/gnn/outputs/logs/<run>.log`.

The same single run without the script (Windows PowerShell or bash, from the repo root):

```bash
python experiments/relgt/run.py --graph-structure chain --chain-seed latest --num-seeds 16 --fk-order recent \
    --epochs 15 --end "2026-10-03 16:00" --holdout-weeks 1 --val-start "2026-09-19 16:00" \
    --val-end "2026-09-26 16:00" --seed 0 --name relgt_chain_recent_ens0926_s0
```

Expected (seed 0-2, official 6-activity score): 52.91, 53.71, 53.47 (mean 53.36), seed bag 54.06. GPU
nondeterminism moves a single run by a few tenths.

### Compare and bag the seeds

`sampling_ho0927.ps1` ends with both of these (run them yourself after `-SkipRuns` or your own runs):

```bash
python scripts/summarize_runs.py --suffix ens0926 relgt_chain relgt_chain_recent
python experiments/ensemble/run.py --suffix ens0926 relgt_chain relgt_chain_recent
```

The second command averages the seeds of each config (the bag) and compares them with the paired bootstrap;
it writes `experiments/ensemble/outputs/<members>_ens0926/metrics.json`. It needs the reference runs
`relgt_chain_ens0926_s0-2` from `bash runs/ensemble_ho0927.sh` (or `CONFIGS="relgt_chain" QUEUE_A="relgt_chain"
QUEUE_B="" bash runs/ensemble_ho0927.sh`, ~30 min per seed) if you want that comparison.

## 4. Predict the test week with the best model (seed bag)

The best model is the **seed bag**: three RelGT runs (seeds 0-2) of the same config, averaged. On the holdout
week the bag of `relgt_chain_recent` scores 54.06 against 53.36 for the mean of its single seeds.

**One command** (from the repo root, Windows PowerShell):

```powershell
.\run_submission.ps1                  # seeds 0-2 on GPU 0, one after the other (~1.5 h)
.\run_submission.ps1 -Gpus 0,1        # seeds spread over two GPUs
.\run_submission.ps1 -StudentId 123456   # also copies the result to output_123456.csv
.\run_submission.ps1 -Smoke           # pipeline check: seed 0, 1 epoch of 30 steps
.\run_submission.ps1 -SkipTrain       # only bag and check runs that already exist
```

It does the following; run the steps by hand if you prefer.

1. **Trains one final model per seed** on every labelled week (no holdout):
   `python experiments/relgt/run.py --graph-structure chain --chain-seed latest --num-seeds 16 --fk-order recent --epochs 15 --holdout-weeks 0 --seed <s> --name relgt_chain_recent_test_s<s>`.
   Each run writes `output.csv` (that seed alone) and `test.npz` (its test-week probabilities and validation
   share of positives). About 30 min per seed (see section 1); logs go to `experiments/relgt/outputs/logs/`.
   The share of users predicted positive per activity comes from the validation weeks (the last 3 labelled
   weeks by default; `--val-weeks`, `--val-start`/`--val-end` change that).
2. **Bags the seeds**:
   `python experiments/relgt/bag_submit.py relgt_chain_recent_test_s0 relgt_chain_recent_test_s1 relgt_chain_recent_test_s2 --name relgt_chain_recent_bag`.
   Per activity it ranks each seed's probabilities, averages the ranks over the seeds, and predicts positive
   for the top share of users, the share being the mean of the seeds' validation shares. This is the bag of
   `experiments/ensemble/run.py` (rank averaging), except that the share is the mean of the per-seed shares
   instead of being re-tuned on the bag's own tune week, which `test.npz` does not carry.
3. **Checks the format** with `assignment2/scripts/evaluate.py` (below).

**The result.** The file is `experiments/relgt/outputs/relgt_chain_recent_bag/output.csv` (one
row per user of `assignment2/submission/test_input.csv`, six 0/1 columns). Rename it to
`output_{studentID}.csv` (or pass `-StudentId`). The format check by hand:

```bash
python assignment2/scripts/evaluate.py --pred experiments/relgt/outputs/relgt_chain_recent_bag/output.csv \
    --input assignment2/submission/test_input.csv
```

`evaluate.py` only checks the format here: the test labels do not exist yet. A single run's `output.csv`
(step 1 above) is a valid submission too, but it is the weaker, unbagged model. The 54.06 above is a holdout-week
result of the ensemble script; the submission bag has not been scored on a holdout week.

## 5. Troubleshooting

- *Out of memory*: lower `--batch-size` (default 512), or give the job a free GPU with `-Gpus`.
- *`unknown config`*: `-Configs` takes `relgt_chain`, `relgt_chain_recent`, `relgt_fk`, `relgt_fk_recent`.
- *The script waits with "gpu busy"*: another program uses more than `-FreeMiB` MiB on it; free the card or
  raise `-FreeMiB`.
- *Rerun from scratch*: delete `experiments/relgt/outputs/<run>/` (and `experiments/relgt/cache/` to resample
  tokens).
- *`SKIP_RUNS=1` with the bash ensemble script* did not take effect when set through PowerShell `$env:`; use
  `sampling_ho0927.ps1 -SkipRuns` or call the python commands above directly.
