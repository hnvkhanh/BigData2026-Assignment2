#!/usr/bin/env bash
# Holdout runs on the week 2026-09-27 to 10-04 (Taiwan time; 09-26 16:00 UTC), scored on all 6 activities,
# push included (that week has push labels; week 11-07 does not). Protocol: train on weeks whose labels
# end by 09-19, tune (early stopping, share of predicted positives) on 09-19 -> 09-26, score the
# holdout week, no retrain. Configs x seeds 0-2, one run at a time on one GPU.
#
# Before each run the script picks a free GPU: the one with the least memory in use, if that is
# under GPU_FREE_MIB (default 1500 MiB; idle cards here use ~250-700 MiB). nvidia-smi on WSL does not
# list the processes on a card, so memory in use is the test. If no GPU is free it waits and checks
# again every GPU_WAIT seconds (default 60). GPU=<index> skips the check and uses that card.
# CONFIGS picks the configs to run, e.g.  CONFIGS="het user_repo" bash script_ho0927.sh
# het_ev / het_ev_notime (--het-graph event) use ~11 GB of a 12 GB card: one per GPU, nothing else on it.
# Output folders are named <cfg>_ho0926_s<seed>: the suffix is the UTC start date of the scored week.
HO_WEEK=${HO_WEEK:-"2026-09-26 16:00"}   # UTC start of the scored week (Saturday 16:00)
SUF=ho$(echo "$HO_WEEK" | cut -c6-10 | tr -d -)
LOG_DIR=experiments/gnn/outputs/logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1      # print lines as they happen, not when the buffer fills
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-10}   # one progress-bar update per 10 s, so the console is readable
GPU_FREE_MIB=${GPU_FREE_MIB:-1500}
GPU_WAIT=${GPU_WAIT:-60}

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
  name=${1}_${SUF}_s$2 tag="[gpu$gpu $1 s$2]"
  log=$LOG_DIR/$name.log
  echo "$(date '+%F %T') $tag START ($i/$n) -> $log"
  # tqdm redraws its bar with \r: turn those into new lines so each update shows on the console
  CUDA_VISIBLE_DEVICES=$gpu python experiments/gnn/run.py --holdout --holdout-week "$HO_WEEK" ${CFG[$1]} --seed $2 --name "$name" 2>&1 \
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
    ms=[json.load(open(f)) for f in sorted(glob.glob(f'experiments/gnn/outputs/{c}_${SUF}_s*/metrics.json'))]
    if not ms: continue
    sc=[100*m['holdout']['score'] for m in ms]; f1={k:np.mean([m['holdout']['f1'][k] for m in ms]) for k in ms[0]['holdout']['f1']}
    print(f'{c:22s} n={len(ms)} score {np.mean(sc):.2f} ({min(sc):.2f}-{max(sc):.2f}) best_ep {[m[\"best_epoch\"] for m in ms]}  '+'  '.join(f'{k} {v:.3f}' for k,v in f1.items()))
    hps=[m.get('holdout_push') for m in ms if m.get('holdout_push')]
    if hps:
        print(f'{\"\":22s} 6-activity score by push rule: '+'  '.join(f'{o} {np.mean([100*h[o][\"score\"] for h in hps]):.2f} (share {np.mean([h[o][\"share\"] for h in hps]):.3f})' for o in hps[0] if isinstance(hps[0][o], dict)))
"
