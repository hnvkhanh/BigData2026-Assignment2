#!/usr/bin/env bash
# Ensemble experiment on the holdout week 2026-09-27 -> 10-04 (Taiwan time; 09-26 16:00 UTC), all 6 activities.
# Reruns the tied configs so that their per-user probabilities are saved (holdout.npz), adds the chain model
# with the baseline's history features on the seed tokens, then combines them (experiments/ensemble/run.py).
# Same protocol as runs/ho0927.sh and runs/chain_ho0927.sh: train on weeks whose labels end by 09-19, tune on
# 09-19 -> 09-26, score the holdout week.
#   all_ego_uf        experiments/gnn/run.py --user-features                          (~5 min a seed)
#   het               experiments/gnn/run.py --graph-structure hetero                 (~5 min)
#   relgt_chain       experiments/relgt/run.py, chain graph, 16 seeds, 15 epochs       (~30 min)
#   relgt_chain_hist  the same with --history (60 count / recency features per seed token)
# Two queues, one per GPU (GPUS="0 1"; a single index runs both queues on it, one after the other):
#   queue A: relgt_chain s0-2;  queue B: all_ego_uf and het s0-2, then relgt_chain_hist s0-2.
# Output folders: <cfg>_ens0926_s<seed> (a new suffix, so the documented ho0926 runs stay untouched).
# Then scripts/summarize_runs.py and the ensembles (experiments/ensemble/outputs/<members>_ens0926*/).
#   CONFIGS="relgt_chain_hist" bash runs/ensemble_ho0927.sh    # a subset
#   SMOKE=1 bash runs/ensemble_ho0927.sh                        # pipeline check: seed 0, 1 epoch of 30 steps,
#                                                               #   folders smoke_<cfg>_ens0926_s0
#   SKIP_RUNS=1 bash runs/ensemble_ho0927.sh                    # only the summary and the ensembles
# Python, logs (experiments/gnn/outputs/logs/<run>.log): see runs/lib.sh.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HO_WEEK=${HO_WEEK:-"2026-09-26 16:00"}   # UTC start of the scored week (Saturday 16:00)
SUF=ens$(echo "$HO_WEEK" | cut -c6-10 | tr -d -)
TUNE=$("$PYTHON" -c "import pandas as pd; print(pd.Timestamp('$HO_WEEK') - pd.Timedelta(days=7))")
HO_END=$("$PYTHON" -c "import pandas as pd; print(pd.Timestamp('$HO_WEEK') + pd.Timedelta(days=7))")
GPUS=${GPUS:-"0 1"}
CONFIGS=${CONFIGS:-"all_ego_uf het relgt_chain relgt_chain_hist"}
QUEUE_A=${QUEUE_A:-"relgt_chain"}
QUEUE_B=${QUEUE_B:-"all_ego_uf het relgt_chain_hist"}
SEEDS=${SEEDS:-"0 1 2"}
PFX=""
FAST_GNN="" FAST_RELGT=""
if [ -n "$SMOKE" ]; then
  PFX="smoke_"; SEEDS="0"
  FAST_GNN="--epochs 1 --max-steps-per-epoch 30"; FAST_RELGT="--epochs 1 --max-steps-per-epoch 30"
fi
CHAIN="--graph-structure chain --chain-seed latest --num-seeds 16"

cmd() {  # $1 = config, $2 = seed, $3 = run name: sets CMD
  case $1 in
    all_ego_uf)       CMD=("$PYTHON" experiments/gnn/run.py --holdout --holdout-week "$HO_WEEK" --user-features
                           $FAST_GNN --seed "$2" --name "$3") ;;
    het)              CMD=("$PYTHON" experiments/gnn/run.py --holdout --holdout-week "$HO_WEEK" --graph-structure hetero
                           --task all $FAST_GNN --seed "$2" --name "$3") ;;
    relgt_chain)      CMD=("$PYTHON" experiments/relgt/run.py $CHAIN --epochs 15 --end "$HO_END" --holdout-weeks 1
                           --val-start "$TUNE" --val-end "$HO_WEEK" $FAST_RELGT --seed "$2" --name "$3") ;;
    relgt_chain_hist) CMD=("$PYTHON" experiments/relgt/run.py $CHAIN --history --epochs 15 --end "$HO_END" --holdout-weeks 1
                           --val-start "$TUNE" --val-end "$HO_WEEK" $FAST_RELGT --seed "$2" --name "$3") ;;
    *)                echo "unknown config $1" >&2; return 1 ;;
  esac
}

in_configs() { for c in $CONFIGS; do [ "$c" = "$1" ] && return 0; done; return 1; }

run_queue() {  # $1 = GPU, $2 = tag, rest = configs; each config's seeds in turn
  local gpu=$1 tag=$2; shift 2
  local jobs=() c s
  for c in "$@"; do in_configs "$c" || continue; for s in $SEEDS; do jobs+=("$c $s"); done; done
  local n=${#jobs[@]} i=0 j
  for j in "${jobs[@]}"; do
    set -- $j
    i=$((i + 1))
    local name=${PFX}${1}_${SUF}_s$2
    cmd "$1" "$2" "$name" || continue
    run_logged "[gpu$gpu $tag ${1} s$2]" "$i/$n" "$LOG_DIR/$name.log" env CUDA_VISIBLE_DEVICES="$gpu" "${CMD[@]}"
  done
  echo "$(date '+%F %T') [gpu$gpu $tag] queue finished"
}

if [ -z "$SKIP_RUNS" ]; then
  set -- $GPUS
  if [ $# -ge 2 ]; then
    run_queue "$1" A $QUEUE_A &
    run_queue "$2" B $QUEUE_B &
    wait
  else
    run_queue "$1" A $QUEUE_A
    run_queue "$1" B $QUEUE_B
  fi
  echo "$(date '+%F %T') all runs finished"
fi

M=(); for c in $CONFIGS; do M+=("${PFX}$c"); done
"$PYTHON" scripts/summarize_runs.py --suffix "$SUF" "${M[@]}"

ens() { echo; echo "== ensemble: $*"; "$PYTHON" experiments/ensemble/run.py --suffix "$SUF" "$@"; }
ens "${M[@]}"                                       # every config, push by the tune-week share
ens "${M[@]}" --push-rule prev-week                 # the submission's push rule
ens "${M[@]}" --rule joint                          # shares by coordinate ascent of the score
if [ ${#M[@]} -gt 2 ]; then
  ens "${PFX}all_ego_uf" "${PFX}het" "${PFX}relgt_chain"                  # the three documented configs
  ens "${PFX}relgt_chain" "${PFX}relgt_chain_hist" --relgt-variant validation   # history on the seed token
fi
