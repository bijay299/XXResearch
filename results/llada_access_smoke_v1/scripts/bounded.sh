#!/bin/bash
# bounded.sh <timeout_seconds> <label> <command...>
# Enforces: wall-clock timeout via coreutils timeout (SIGTERM then SIGKILL after 30s grace).
# Charges: wall-clock of the attempt to the GPU ledger when label starts with gpu_.
W=/data/bijaypandey/llada_smoke
TO=$1; LABEL=$2; shift 2
T0=$(date +%s)
timeout --kill-after=30s "${TO}s" "$@"
RC=$?
T1=$(date +%s)
DUR=$(( T1 - T0 ))
if [[ "$LABEL" == gpu_* ]]; then
  PREV=$(cat $W/GPU_SECONDS 2>/dev/null || echo 0)
  echo $(( PREV + DUR )) > $W/GPU_SECONDS
fi
echo "[bounded] label=$LABEL rc=$RC duration_s=$DUR timeout_s=$TO charged_gpu=$([[ "$LABEL" == gpu_* ]] && echo yes || echo no)" | tee -a $W/logs/bounded.log
exit $RC
