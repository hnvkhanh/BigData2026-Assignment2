#!/usr/bin/env bash
# Learned time embedding for the chain-graph models: runs/chain_ho0927.sh with the fixed sinusoidal
# token-age encoding replaced by a learned cos(w log(1 + days) + b) + linear (relgt --time-emb func,
# gnn --chain-time func). Everything else is the same as runs/chain_ho0927.sh:
# Chain-graph models on the holdout week 2026-09-27 -> 10-04 (Taiwan time; 09-26 16:00 UTC), scored on
# all 6 activities, push included. Same protocol as runs/ho0927.sh: train on weeks whose labels end
# by 09-19, tune on 09-19 -> 09-26, score the holdout week.
#   relgt_chain_ftime: experiments/relgt/run.py, chain graph, 16 latest actions as seeds, 15 epochs (the exp2
#                settings of experiments/relgt/README.md, pushes kept). It reports two holdout scores:
#                "validation model" (trained to 09-19, thresholds of 09-19) and "final model"
#                (retrained through 09-26, the validation share of predicted positives).
#   gnn_chain_ftime:   experiments/gnn/run.py --graph-structure chain, same seeds / tokens settings, --holdout
#                (trained to 09-19, the tune week's share of predicted positives, no retrain).
# Seeds 0-2. Runs go one at a time, each on a single GPU: the free GPU picked when the run starts
# (under GPU_FREE_MIB, here default 2000 MiB; FREE_MB is still accepted); if none is free it waits.
# GPU=<index> forces one GPU. CONFIGS picks a subset, e.g.
#   CONFIGS="gnn_chain_ftime" bash runs/chain_ftime_ho0927.sh
# Each run's output goes to its log file and to the console, prefixed with [gpu cfg seed]. Python, logs:
# see runs/lib.sh.
GPU_FREE_MIB=${GPU_FREE_MIB:-${FREE_MB:-2000}}   # a GPU with less memory in use than this counts as free
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

HO_WEEK=${HO_WEEK:-"2026-09-26 16:00"}   # UTC start of the scored week (Saturday 16:00)
SUF=ho$(echo "$HO_WEEK" | cut -c6-10 | tr -d -)
# Tune week = the week before, holdout end = a week after (RelGT takes them as a data period).
TUNE=$("$PYTHON" -c "import pandas as pd; print(pd.Timestamp('$HO_WEEK') - pd.Timedelta(days=7))")
HO_END=$("$PYTHON" -c "import pandas as pd; print(pd.Timestamp('$HO_WEEK') + pd.Timedelta(days=7))")
CONFIGS=${CONFIGS:-"relgt_chain_ftime gnn_chain_ftime"}
CHAIN="--graph-structure chain --chain-seed latest --num-seeds 16"

cmd() {  # $1 = config, $2 = seed, $3 = run name: sets CMD, the python command line
  case $1 in
    relgt_chain_ftime) CMD=("$PYTHON" experiments/relgt/run.py $CHAIN --time-emb func --epochs 15 --end "$HO_END"
                            --holdout-weeks 1 --val-start "$TUNE" --val-end "$HO_WEEK" --seed "$2" --name "$3") ;;
    gnn_chain_ftime)   CMD=("$PYTHON" experiments/gnn/run.py $CHAIN --chain-time func --holdout --holdout-week "$HO_WEEK"
                            --seed "$2" --name "$3") ;;
    *)                 echo "unknown config $1" >&2; return 1 ;;
  esac
}

# One run at a time, on one GPU: every config's seeds 0-2, in CONFIGS order.
n=$(( $(echo $CONFIGS | wc -w) * 3 )); i=0
for c in $CONFIGS; do
  for s in 0 1 2; do
    i=$((i + 1))
    name=${c}_${SUF}_s$s
    cmd "$c" "$s" "$name" || continue
    gpu=$(pick_gpu) || exit 1
    run_logged "[gpu$gpu $c s$s]" "$i/$n" "$LOG_DIR/$name.log" env CUDA_VISIBLE_DEVICES="$gpu" "${CMD[@]}"
  done
done
echo "$(date '+%F %T') all runs finished"

"$PYTHON" scripts/summarize_runs.py --suffix "$SUF" relgt_chain relgt_chain_ftime gnn_chain gnn_chain_ftime
