# Shared setup and helpers of the run scripts in runs/. Each script starts with
#   source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
#
# - moves to the repo root, so the scripts can be started from any folder;
# - PYTHON: $PYTHON if set, else the active venv, else the repo's venv (.venv-win under Git Bash,
#   .venv on WSL / Linux), else python on PATH;
# - LOG_DIR: experiments/gnn/outputs/logs, one <run name>.log per run;
# - pick_gpu: prints a free GPU (GPU=<index> forces one); GPU_FREE_MIB (default 1500) and GPU_WAIT
#   (default 60 s) tune it. A script can set GPU_FREE_MIB before sourcing this file;
# - run_logged: runs one command with its output in a log file and on the console.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

if [ -z "$PYTHON" ]; then
  case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) _venv_python=.venv-win/Scripts/python.exe ;;
    *) _venv_python=.venv/bin/python ;;
  esac
  if [ -n "$VIRTUAL_ENV" ]; then PYTHON=python
  elif [ -x "$_venv_python" ]; then PYTHON=$_venv_python
  else PYTHON=python; fi
fi

LOG_DIR=experiments/gnn/outputs/logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1      # print lines as they happen, not when the buffer fills
export PYTHONUTF8=1            # Windows: UTF-8 for prints piped through tee and for open()
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-10}   # one progress-bar update per 10 s, so the console is readable
GPU_FREE_MIB=${GPU_FREE_MIB:-1500}
GPU_WAIT=${GPU_WAIT:-60}

# Print the index of a free GPU: the one with the least memory in use, if that is under GPU_FREE_MIB
# (idle cards here use ~250-700 MiB). nvidia-smi on WSL does not list the processes on a card, so
# memory in use is the test. While no GPU is free it waits, checking again every GPU_WAIT seconds.
# GPU=<index> skips the check and uses that card.
pick_gpu() {
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

# run_logged <tag> <position or ""> <log> <command...>
# Runs the command (e.g. env CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/gnn/run.py ...): its output
# goes to <log> and to the console, each line prefixed with <tag>. Then prints DONE with the score line
# of the log, or FAILED. Returns the command's exit code.
run_logged() {
  local tag=$1 pos=$2 log=$3
  shift 3
  echo "$(date '+%F %T') $tag START${pos:+ ($pos)} -> $log"
  # tqdm redraws its bar with \r: turn those into new lines so each update shows on the console
  "$@" 2>&1 | tee "$log" | stdbuf -o0 tr '\r' '\n' | sed -u "/^\s*$/d; s/^/$tag /"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "$(date '+%F %T') $tag DONE  $(grep -E '^HOLDOUT .*users|^mean holdout score' "$log" | tail -1)"
  else
    echo "$(date '+%F %T') $tag FAILED (exit $rc), see $log"
  fi
  return "$rc"
}
