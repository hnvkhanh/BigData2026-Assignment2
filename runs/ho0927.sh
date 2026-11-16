#!/usr/bin/env bash
# Holdout runs on the week 2026-09-27 to 10-04 (Taiwan time; 09-26 16:00 UTC), scored on all 6 activities,
# push included (that week has push labels; week 11-07 does not). Protocol: train on weeks whose labels
# end by 09-19, tune (early stopping, share of predicted positives) on 09-19 -> 09-26, score the
# holdout week, no retrain. Configs x seeds 0-2, one run at a time on one GPU.
#
# Before each run the script picks a free GPU (GPU=<index> forces one); GPU picking, Python and logs:
# see runs/lib.sh.
# CONFIGS picks the configs to run, e.g.  CONFIGS="het user_repo" bash runs/ho0927.sh
# het_ev / het_ev_notime (--het-graph event) peak at ~5.5 GB with activation checkpointing (the default): one per GPU.
# Output folders are named <cfg>_ho0926_s<seed>: the suffix is the UTC start date of the scored week.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HO_WEEK=${HO_WEEK:-"2026-09-26 16:00"}   # UTC start of the scored week (Saturday 16:00)
SUF=ho$(echo "$HO_WEEK" | cut -c6-10 | tr -d -)

declare -A CFG=(
  [het]="--graph-structure hetero --task all"
  [het_notime]="--graph-structure hetero --task all --het-time none"
  [het_ev]="--graph-structure hetero --het-graph event --task all"
  [het_ev_notime]="--graph-structure hetero --het-graph event --task all --het-time none"
  [all_ego_uf]="--user-features"
  [all_ego_uf_pprev]="--user-features --push-share prev-week"
  [all_ego_uf_rel]="--user-features --relative-features"
  [all_ego_uf_rel_pprev]="--user-features --relative-features --push-share prev-week"
  [user_repo]="--graph-structure user-repo"
)
ALL="het het_notime het_ev het_ev_notime all_ego_uf all_ego_uf_pprev all_ego_uf_rel all_ego_uf_rel_pprev user_repo"
CONFIGS=${CONFIGS:-"het het_notime all_ego_uf user_repo"}
jobs=(); for c in $CONFIGS; do for s in 0 1 2; do jobs+=("$c $s"); done; done

n=${#jobs[@]} i=0
for j in "${jobs[@]}"; do
  set -- $j
  i=$((i + 1))
  gpu=$(pick_gpu) || exit 1
  name=${1}_${SUF}_s$2
  run_logged "[gpu$gpu $1 s$2]" "$i/$n" "$LOG_DIR/$name.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" experiments/gnn/run.py --holdout --holdout-week "$HO_WEEK" ${CFG[$1]} \
    --seed "$2" --name "$name"
done
echo "$(date '+%F %T') all runs finished"

"$PYTHON" scripts/summarize_runs.py --suffix "$SUF" $ALL
