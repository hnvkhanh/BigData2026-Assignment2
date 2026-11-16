#!/usr/bin/env bash
# Holdout runs of the GNN experiment on the default holdout week 11-07 (--holdout), configs x seeds 0-2,
# one queue per GPU (GPUs 0 and 1). Each run's output goes to its log file and to the console, prefixed
# with [gpu cfg seed]. Output folders are named <cfg>_ho_s<seed>.
# CONFIGS picks the configs to run (default: the hetero ones; the ego / user-repo runs are done), e.g.
#   CONFIGS="het" bash runs/gnn_ho1107.sh
# Python, logs: see runs/lib.sh.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

declare -A CFG=(
  [all_ego]=""
  [all_ego_seg]="--segment activity"
  [all_ego_uf]="--user-features"
  [all_ego_uf_seg]="--user-features --segment activity"
  [user_repo]="--graph-structure user-repo --task all"
  [het]="--graph-structure hetero --task all"
  [het_notime]="--graph-structure hetero --task all --het-time none"
)
ALL="all_ego all_ego_uf all_ego_seg all_ego_uf_seg user_repo het het_notime"
CONFIGS=${CONFIGS:-"het het_notime"}
jobs=(); for c in $CONFIGS; do for s in 0 1 2; do jobs+=("$c $s"); done; done
half=$(( (${#jobs[@]} + 1) / 2 ))

run_queue() {  # $1 = GPU, rest = "cfg seed" jobs
  local gpu=$1; shift
  local n=$# i=0
  for j in "$@"; do
    set -- $j
    i=$((i + 1))
    local name=${1}_ho_s$2
    run_logged "[gpu$gpu $1 s$2]" "$i/$n" "$LOG_DIR/$name.log" \
      env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" experiments/gnn/run.py --holdout ${CFG[$1]} --seed "$2" --name "$name"
  done
  echo "$(date '+%F %T') [gpu$gpu] queue finished"
}

run_queue 0 "${jobs[@]:0:$half}" &
run_queue 1 "${jobs[@]:$half}" &
wait
echo "$(date '+%F %T') all runs finished"

"$PYTHON" scripts/summarize_runs.py --suffix ho $ALL
