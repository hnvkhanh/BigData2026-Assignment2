#!/usr/bin/env bash
# Smoke test: retrain the 3 best configs of the 09-27 holdout (all_ego_uf_rel 53.06, all_ego_uf 52.98,
# het 52.77) on data before June only, seed 42, one GPU, one run after the other.
# Holdout week 2026-05-23 16:00 UTC (05-24 -> 05-31 Taiwan time), tune week 05-16, so training labels
# end by 05-16 and features / tuning never see June or later. The week is fully labelled, so it is
# scored on all 6 activities (push included).
# Short runs: EPOCHS (default 2) epochs, ego runs capped at STEPS (default 30) batches per epoch.
# The scores only check that the pipeline runs end to end; they are not comparable to full runs.
#   GPU=0 bash runs/smoke_may.sh                      # all 3 on GPU 0
#   GPU=1 CONFIGS="het" EPOCHS=1 bash runs/smoke_may.sh
# runs/smoke_may.ps1 does the same for 10 configs from PowerShell, with a job queue over several GPUs.
# Python, logs: see runs/lib.sh.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HO_WEEK=${HO_WEEK:-"2026-05-23 16:00"}
SUF=ho$(echo "$HO_WEEK" | cut -c6-10 | tr -d -)
SEED=${SEED:-42}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-2}
STEPS=${STEPS:-30}
echo "python: $PYTHON"

declare -A CFG=(
  [all_ego_uf_rel]="--user-features --relative-features --max-steps-per-epoch $STEPS"
  [all_ego_uf]="--user-features --max-steps-per-epoch $STEPS"
  [het]="--graph-structure hetero --task all"
)
CONFIGS=${CONFIGS:-"all_ego_uf_rel all_ego_uf het"}

for c in $CONFIGS; do
  name=smoke_${c}_${SUF}_s$SEED
  run_logged "[gpu$GPU $c]" "" "$LOG_DIR/$name.log" \
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/gnn/run.py --holdout --holdout-week "$HO_WEEK" ${CFG[$c]} \
    --epochs "$EPOCHS" --seed "$SEED" --name "$name" --no-progress
done

"$PYTHON" scripts/summarize_runs.py --suffix "$SUF" --sort "smoke_*"
