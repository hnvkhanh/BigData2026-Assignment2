#!/usr/bin/env bash
# Ego GNN + user features with --quiet-rule prev-week, closest match to archive/old_experiments/relgt_inspired:
# active = >= 5 visible events in the 14-day ego window (--active-min-events 5, the old eligible_only
# rule); the GNN is trained, tuned and predicts on the active users only. The old run scored active users
# only, so compare with it on the 'F1 active' column. The overall score adds the quiet users: every quiet
# user is predicted to do exactly what they did in the previous week (its real labels, which the test week
# cannot see, so the score is an upper bound). CONFIGS="ego_uf_qprev" reruns the >= 1 event version.
# Compare with all_ego_uf of runs/ho0927.sh (same model, quiet users predicted by the model).
#
# Holdout runs on the week 2026-09-27 to 10-04 (Taiwan time; 09-26 16:00 UTC), scored on all 6 activities,
# push included (that week has push labels; week 11-07 does not). Protocol: train on weeks whose labels
# end by 09-19, tune (early stopping, share of predicted positives) on 09-19 -> 09-26, score the
# holdout week, no retrain. Seeds 0-2, one run at a time on one GPU.
#
# Before each run the script picks a free GPU (GPU=<index> forces one); GPU picking, Python and logs:
# see runs/lib.sh.
# SEEDS picks the seeds, e.g.  SEEDS="0" bash runs/qprev_ho0927.sh
# Output folders are named <cfg>_ho0926_s<seed>: the suffix is the UTC start date of the scored week.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HO_WEEK=${HO_WEEK:-"2026-09-26 16:00"}   # UTC start of the scored week (Saturday 16:00)
SUF=ho$(echo "$HO_WEEK" | cut -c6-10 | tr -d -)

declare -A CFG=(
  [ego_uf_qprev]="--user-features --quiet-rule prev-week"                           # active = >= 1 event
  [ego_uf_qprev5]="--user-features --quiet-rule prev-week --active-min-events 5"   # old experiment's rule
)
CONFIGS=${CONFIGS:-"ego_uf_qprev5"}
SEEDS=${SEEDS:-"0 1 2"}
jobs=(); for c in $CONFIGS; do for s in $SEEDS; do jobs+=("$c $s"); done; done

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

# all_ego_uf: the reference runs of runs/ho0927.sh
"$PYTHON" scripts/summarize_runs.py --suffix "$SUF" --segments all_ego_uf ego_uf_qprev ego_uf_qprev5
