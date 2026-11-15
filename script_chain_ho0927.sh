#!/usr/bin/env bash
# Chain-graph models on the holdout week 2026-09-27 -> 10-04 (Taiwan time; 09-26 16:00 UTC), scored on
# all 6 activities, push included. Same protocol as script_ho0927.sh: train on weeks whose labels end
# by 09-19, tune on 09-19 -> 09-26, score the holdout week.
#   relgt_chain: experiments/relgt/run.py, chain graph, 16 latest actions as seeds, 15 epochs (the exp2
#                settings of experiments/relgt/README.md, pushes kept). It reports two holdout scores:
#                "validation model" (trained to 09-19, thresholds of 09-19) and "final model"
#                (retrained through 09-26, the validation share of predicted positives).
#   gnn_chain:   experiments/gnn/run.py --graph-structure chain, same seeds / tokens settings, --holdout
#                (trained to 09-19, the tune week's share of predicted positives, no retrain).
# Seeds 0-2. Runs go one at a time, each on a single GPU: the free GPU (least memory in use,
# under FREE_MB, default 2000 MiB) picked when the run starts; if none is free it waits. GPU=<index>
# forces one GPU. Each run loads the 2.6M-node graph and ~1 GB of tokens. CONFIGS picks a subset, e.g.
#   CONFIGS="gnn_chain" bash script_chain_ho0927.sh
# Each run's output goes to its log file and to the console, prefixed with [gpu cfg seed].
LOG_DIR=experiments/gnn/outputs/logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1      # print lines as they happen, not when the buffer fills
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-10}   # one progress-bar update per 10 s, so the console is readable

HO_WEEK=${HO_WEEK:-"2026-09-26 16:00"}   # UTC start of the scored week (Saturday 16:00)
SUF=ho$(echo "$HO_WEEK" | cut -c6-10 | tr -d -)
# Tune week = the week before, holdout end = a week after (RelGT takes them as a data period).
TUNE=$(python -c "import pandas as pd; print(pd.Timestamp('$HO_WEEK') - pd.Timedelta(days=7))")
HO_END=$(python -c "import pandas as pd; print(pd.Timestamp('$HO_WEEK') + pd.Timedelta(days=7))")
FREE_MB=${FREE_MB:-2000}   # a GPU with less memory in use than this counts as free
CONFIGS=${CONFIGS:-"relgt_chain gnn_chain"}
CHAIN="--graph-structure chain --chain-seed latest --num-seeds 16"

cmd() {  # $1 = config, $2 = seed, $3 = run name: the python command line
  case $1 in
    relgt_chain) echo "python experiments/relgt/run.py $CHAIN --epochs 15 --end '$HO_END' --holdout-weeks 1" \
                      "--val-start '$TUNE' --val-end '$HO_WEEK' --seed $2 --name $3" ;;
    gnn_chain)   echo "python experiments/gnn/run.py $CHAIN --holdout --holdout-week '$HO_WEEK' --seed $2 --name $3" ;;
  esac
}

pick_gpu() {  # print the GPU to use: the one with the least memory in use among those under FREE_MB,
              # waiting (re-checking every minute) while every GPU is busy. GPU=<index> forces one.
  if [ -n "$GPU" ]; then echo "$GPU"; return; fi
  local g
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', *' -v max="$FREE_MB" '$2 < max {print $2, $1}' | sort -n | head -1 | cut -d' ' -f2)
    if [ -n "$g" ]; then echo "$g"; return; fi
    echo "$(date '+%F %T') all GPUs busy (>= $FREE_MB MiB in use), waiting" >&2
    sleep 60
  done
}

run_one() {  # $1 = config, $2 = seed, $3 = position ("i/n"): one run on the free GPU picked at its start
  local cfg=$1 s=$2 pos=$3 gpu
  gpu=$(pick_gpu)
  local name=${cfg}_${SUF}_s$s tag="[gpu$gpu $cfg s$s]"
  local log=$LOG_DIR/$name.log
  echo "$(date '+%F %T') $tag START ($pos) -> $log"
  # tqdm redraws its bar with \r: turn those into new lines so each update shows on the console
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$(cmd "$cfg" "$s" "$name")" 2>&1 \
    | tee "$log" | stdbuf -o0 tr '\r' '\n' | sed -u "/^\s*$/d; s/^/$tag /"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%F %T') $tag DONE  $(grep -E '^HOLDOUT .*users|^mean holdout score' "$log" | tail -1)"
  else
    echo "$(date '+%F %T') $tag FAILED (exit $rc), see $log"
  fi
}

# One run at a time, on one GPU: every config's seeds 0-2, in CONFIGS order.
n=$(( $(echo $CONFIGS | wc -w) * 3 )); i=0
for c in $CONFIGS; do
  for s in 0 1 2; do
    i=$((i + 1))
    run_one "$c" "$s" "$i/$n"
  done
done
echo "$(date '+%F %T') all runs finished"

python -c "
import json,glob,numpy as np
def row(name, runs, get):
    if not runs: return
    ms=[json.load(open(f)) for f in runs]
    hs=[get(m) for m in ms]
    sc=[100*h['score'] for h in hs]; f1={k:np.mean([h[k] for h in hs]) for k in hs[0] if k!='score'}
    print(f'{name:28s} n={len(ms)} score {np.mean(sc):.2f} ({min(sc):.2f}-{max(sc):.2f}) best_ep {[m[\"best_epoch\"] for m in ms]}  '
          +'  '.join(f'{k} {v:.3f}' for k,v in f1.items()))
rel=sorted(glob.glob('experiments/relgt/outputs/relgt_chain_${SUF}_s*/metrics.json'))
for v in ('validation model','final model'):
    row(f'relgt_chain ({v})', rel, lambda m,v=v: {**m['holdout'][0][v]['per_event'], 'score': m['holdout'][0][v]['score']})
gnn=lambda m: {**m['holdout']['f1'], 'score': m['holdout']['score']}
row('gnn_chain', sorted(glob.glob('experiments/gnn/outputs/gnn_chain_${SUF}_s*/metrics.json')), gnn)
print('-- script_ho0927.sh, same week --')
for c in 'all_ego_uf het het_notime user_repo'.split():
    row(c, sorted(glob.glob(f'experiments/gnn/outputs/{c}_${SUF}_s*/metrics.json')), gnn)
"
