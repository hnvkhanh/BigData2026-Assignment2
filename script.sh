#!/usr/bin/env bash
# Holdout runs of the GNN experiment (configs x seeds 0-2), one queue per GPU.
# Each run's output goes to its log file and to the console, prefixed with [gpu cfg seed].
# CONFIGS picks the configs to run (default: the hetero ones; the others are done), e.g.
#   CONFIGS="user_repo het" bash script.sh
LOG_DIR=experiments/gnn/outputs/logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1      # print lines as they happen, not when the buffer fills
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-10}   # one progress-bar update per 10 s, so the console is readable

declare -A CFG=(
  [het]="--graph-structure hetero --task all"
  [het_notime]="--graph-structure hetero --task all --het-time none"
)
ALL="het het_notime"
CONFIGS=${CONFIGS:-"het het_notime"}
jobs=(); for c in $CONFIGS; do for s in 0 1 2; do jobs+=("$c $s"); done; done
half=$(( (${#jobs[@]} + 1) / 2 ))

run_queue() {  # $1 = GPU, rest = "cfg seed" jobs
  local gpu=$1; shift
  local n=$# i=0
  for j in "$@"; do
    set -- $j
    i=$((i + 1))
    local name=${1}_ho_s$2 tag="[gpu$gpu $1 s$2]"
    local log=$LOG_DIR/$name.log
    echo "$(date '+%F %T') $tag START ($i/$n) -> $log"
    # tqdm redraws its bar with \r: turn those into new lines so each update shows on the console
    CUDA_VISIBLE_DEVICES=$gpu python experiments/gnn/run.py --holdout ${CFG[$1]} --seed $2 --name "$name" 2>&1 \
      | tee "$log" | stdbuf -o0 tr '\r' '\n' | sed -u "/^\s*$/d; s/^/$tag /"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
      echo "$(date '+%F %T') $tag DONE  $(grep '^HOLDOUT .*users' "$log" | tail -1)"
    else
      echo "$(date '+%F %T') $tag FAILED (exit $rc), see $log"
    fi
  done
  echo "$(date '+%F %T') [gpu$gpu] queue finished"
}

run_queue 0 "${jobs[@]:0:$half}" &
run_queue 1 "${jobs[@]:$half}" &
wait
echo "$(date '+%F %T') all runs finished"

python -c "
import json,glob,numpy as np
for c in '$ALL'.split():
    ms=[json.load(open(f)) for f in sorted(glob.glob(f'experiments/gnn/outputs/{c}_ho_s*/metrics.json'))]
    if not ms: continue
    sc=[100*m['holdout']['score'] for m in ms]; f1={k:np.mean([m['holdout']['f1'][k] for m in ms]) for k in ms[0]['holdout']['f1']}
    print(f'{c:16s} n={len(ms)} score {np.mean(sc):.2f} ({min(sc):.2f}-{max(sc):.2f}) best_ep {[m[\"best_epoch\"] for m in ms]}  '+'  '.join(f'{k} {v:.3f}' for k,v in f1.items()))
"