#!/usr/bin/env bash
# Holdout runs of the hetero GNN that only sees the last 14 days before each cutoff
# (--history-days 14: event edges, object / repo features, PR / issue ages and the user features),
# configs x seeds 0-2, one run at a time on one GPU. Same protocol and log layout as runs/gnn_ho1107.sh;
# the summary prints the 90-day het runs too, so the effect of the window shows in one table.
# Before each run the script picks a free GPU (GPU=<index> forces one); GPU picking, Python and logs:
# see runs/lib.sh.
# CONFIGS picks the configs to run, e.g.  CONFIGS="het_h14" bash runs/h14_ho1107.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

declare -A CFG=(
  [het_h14]="--graph-structure hetero --task all --history-days 14"
  [het_notime_h14]="--graph-structure hetero --task all --het-time none --history-days 14"
)
ALL="het het_notime het_h14 het_notime_h14"   # summary: 90-day runs (runs/gnn_ho1107.sh) next to the 14-day ones
CONFIGS=${CONFIGS:-"het_h14 het_notime_h14"}
jobs=(); for c in $CONFIGS; do for s in 0 1 2; do jobs+=("$c $s"); done; done

n=${#jobs[@]} i=0
for j in "${jobs[@]}"; do
  set -- $j
  i=$((i + 1))
  gpu=$(pick_gpu) || exit 1
  name=${1}_ho_s$2
  run_logged "[gpu$gpu $1 s$2]" "$i/$n" "$LOG_DIR/$name.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" experiments/gnn/run.py --holdout ${CFG[$1]} --seed "$2" --name "$name"
done
echo "$(date '+%F %T') all runs finished"

"$PYTHON" scripts/summarize_runs.py --suffix ho $ALL
