#!/usr/bin/env bash
# Holdout runs (week 11-07, --holdout) of the ego GNN + user features with the hetero model's time
# encoding (--ego-time func) and its ablation (--ego-time none), seeds 0-2, one after another on a single
# GPU. Each run's output goes to its log file and to the console, prefixed with [gpu cfg seed].
#   bash runs/ego_time_ho1107.sh                        # GPU 0, both configs
#   GPU=1 CONFIGS="ego_uf_ftime" bash runs/ego_time_ho1107.sh
# The summary also lists all_ego_uf (the original sinusoidal encoding) from its earlier runs.
# Python, logs: see runs/lib.sh.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

declare -A CFG=(
  [ego_uf_ftime]="--user-features --ego-time func"
  [ego_uf_notime]="--user-features --ego-time none"
)
ALL="all_ego_uf ego_uf_ftime ego_uf_notime"
CONFIGS=${CONFIGS:-"ego_uf_ftime ego_uf_notime"}
GPU=${GPU:-0}
jobs=(); for c in $CONFIGS; do for s in 0 1 2; do jobs+=("$c $s"); done; done

n=${#jobs[@]} i=0
for j in "${jobs[@]}"; do
  set -- $j
  i=$((i + 1))
  name=${1}_ho_s$2
  run_logged "[gpu$GPU $1 s$2]" "$i/$n" "$LOG_DIR/$name.log" \
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiments/gnn/run.py --holdout ${CFG[$1]} --seed "$2" --name "$name"
done
echo "$(date '+%F %T') all runs finished"

"$PYTHON" scripts/summarize_runs.py --suffix ho $ALL
