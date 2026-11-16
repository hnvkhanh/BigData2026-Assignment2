#!/usr/bin/env bash
# Smoke test: retrain the 3 best configs of the 09-27 holdout (all_ego_uf_rel 53.06, all_ego_uf 52.98,
# het 52.77) on data before June only, seed 42, one GPU, one run after the other.
# Holdout week 2026-05-23 16:00 UTC (05-24 -> 05-31 Taiwan time), tune week 05-16, so training labels
# end by 05-16 and features / tuning never see June or later. The week is fully labelled, so it is
# scored on all 6 activities (push included).
# Short runs: EPOCHS (default 2) epochs, ego runs capped at STEPS (default 30) batches per epoch.
# The scores only check that the pipeline runs end to end; they are not comparable to full runs.
#   GPU=0 bash script_smoke_may.sh                      # all 3 on GPU 0
#   GPU=1 CONFIGS="het" EPOCHS=1 bash script_smoke_may.sh
HO_WEEK=${HO_WEEK:-"2026-05-23 16:00"}
SUF=ho$(echo "$HO_WEEK" | cut -c6-10 | tr -d -)
SEED=${SEED:-42}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-2}
STEPS=${STEPS:-30}
LOG_DIR=experiments/gnn/outputs/logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1            # Windows: UTF-8 for prints piped through tee and for open()
export CUDA_VISIBLE_DEVICES=$GPU
# Python: the active venv, else the Windows venv (.venv-win, Git Bash), else the WSL / Linux venv (.venv),
# else python on PATH
if [ -z "$PYTHON" ]; then
  if [ -n "$VIRTUAL_ENV" ]; then PYTHON=python
  elif [ -x .venv-win/Scripts/python.exe ]; then PYTHON=.venv-win/Scripts/python.exe
  elif [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python
  else PYTHON=python; fi
fi
echo "python: $PYTHON"

declare -A CFG=(
  [all_ego_uf_rel]="--user-features --relative-features --max-steps-per-epoch $STEPS"
  [all_ego_uf]="--user-features --max-steps-per-epoch $STEPS"
  [het]="--graph-structure hetero --task all"
)
CONFIGS=${CONFIGS:-"all_ego_uf_rel all_ego_uf het"}

for c in $CONFIGS; do
  name=smoke_${c}_${SUF}_s$SEED
  log=$LOG_DIR/$name.log
  echo "$(date '+%F %T') [gpu$GPU $c] START -> $log"
  "$PYTHON" experiments/gnn/run.py --holdout --holdout-week "$HO_WEEK" ${CFG[$c]} \
    --epochs "$EPOCHS" --seed "$SEED" --name "$name" --no-progress 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%F %T') [gpu$GPU $c] OK    $(grep '^HOLDOUT .*users' "$log" | tail -1)"
  else
    echo "$(date '+%F %T') [gpu$GPU $c] FAILED (exit $rc), see $log"
  fi
done

"$PYTHON" -c "
import json,glob,os
for f in sorted(glob.glob('experiments/gnn/outputs/smoke_*_${SUF}_s${SEED}/metrics.json')):
    m=json.load(open(f)); h=m['holdout']
    print(f'{os.path.basename(os.path.dirname(f)):34s} score {100*h[\"score\"]:.2f}  best_ep {m[\"best_epoch\"]}  '
          + '  '.join(f'{k} {v:.3f}' for k,v in h['f1'].items()))
"
