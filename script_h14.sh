#!/usr/bin/env bash
# Holdout runs of the hetero GNN that only sees the last 14 days before each cutoff
# (--history-days 14: event edges, object / repo features, PR / issue ages and the user features),
# configs x seeds 0-2, one run at a time on one GPU. Same protocol and log layout as script.sh; the
# summary prints the 90-day het runs too, so the effect of the window shows in one table.
#
# Before each run the script picks a free GPU: the one with the least memory in use, if that is
# under GPU_FREE_MIB (default 1500 MiB; idle cards here use ~250-700 MiB). nvidia-smi on WSL does not
# list the processes on a card, so memory in use is the test. If no GPU is free it waits and checks
# again every GPU_WAIT seconds (default 60). GPU=<index> skips the check and uses that card.
# CONFIGS picks the configs to run, e.g.  CONFIGS="het_h14" bash script_h14.sh
LOG_DIR=experiments/gnn/outputs/logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1      # print lines as they happen, not when the buffer fills
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-10}   # one progress-bar update per 10 s, so the console is readable
GPU_FREE_MIB=${GPU_FREE_MIB:-1500}
GPU_WAIT=${GPU_WAIT:-60}

declare -A CFG=(
  [het_h14]="--graph-structure hetero --task all --history-days 14"
  [het_notime_h14]="--graph-structure hetero --task all --het-time none --history-days 14"
)
ALL="het het_notime het_h14 het_notime_h14"   # summary: 90-day runs (script.sh) next to the 14-day ones
CONFIGS=${CONFIGS:-"het_h14 het_notime_h14"}
jobs=(); for c in $CONFIGS; do for s in 0 1 2; do jobs+=("$c $s"); done; done

pick_gpu() {  # prints the index of a free GPU, waiting until there is one
  if [ -n "$GPU" ]; then echo "$GPU"; return; fi
  while true; do
    local best
    best=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1)
    if [ -z "$best" ]; then
      echo "$(date '+%F %T') nvidia-smi gave no GPU" >&2; exit 1
    fi
    local idx=${best%%,*} used=${best##*, }
    if [ "$used" -lt "$GPU_FREE_MIB" ]; then echo "$idx"; return; fi
    echo "$(date '+%F %T') no free GPU (least used: gpu$idx, $used MiB >= $GPU_FREE_MIB MiB); retry in ${GPU_WAIT}s" >&2
    sleep "$GPU_WAIT"
  done
}

n=${#jobs[@]} i=0
for j in "${jobs[@]}"; do
  set -- $j
  i=$((i + 1))
  gpu=$(pick_gpu) || exit 1
  name=${1}_ho_s$2 tag="[gpu$gpu $1 s$2]"
  log=$LOG_DIR/$name.log
  echo "$(date '+%F %T') $tag START ($i/$n) -> $log"
  # tqdm redraws its bar with \r: turn those into new lines so each update shows on the console
  CUDA_VISIBLE_DEVICES=$gpu python experiments/gnn/run.py --holdout ${CFG[$1]} --seed $2 --name "$name" 2>&1 \
    | tee "$log" | stdbuf -o0 tr '\r' '\n' | sed -u "/^\s*$/d; s/^/$tag /"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%F %T') $tag DONE  $(grep '^HOLDOUT .*users' "$log" | tail -1)"
  else
    echo "$(date '+%F %T') $tag FAILED (exit $rc), see $log"
  fi
done
echo "$(date '+%F %T') all runs finished"

python -c "
import json,glob,numpy as np
for c in '$ALL'.split():
    ms=[json.load(open(f)) for f in sorted(glob.glob(f'experiments/gnn/outputs/{c}_ho_s*/metrics.json'))]
    if not ms: continue
    sc=[100*m['holdout']['score'] for m in ms]; f1={k:np.mean([m['holdout']['f1'][k] for m in ms]) for k in ms[0]['holdout']['f1']}
    print(f'{c:16s} n={len(ms)} score {np.mean(sc):.2f} ({min(sc):.2f}-{max(sc):.2f}) best_ep {[m[\"best_epoch\"] for m in ms]}  '+'  '.join(f'{k} {v:.3f}' for k,v in f1.items()))
"
