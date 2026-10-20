#!/bin/bash
# attempt.sh <phase> <label> <gpu|cpu> <timeout_s> <command...>
# - Allocates a unique attempt id; each attempt gets its OWN log (never overwritten).
# - Timeout is min(requested, remaining GPU budget) for gpu attempts.
# - Charges wall-clock of EVERY attempt incl. failures to the GPU ledger when kind=gpu.
W=/data/bijaypandey/llada_rwku
PHASE=$1; LABEL=$2; KIND=$3; TO=$4; shift 4
N=$(( $(wc -l < $W/ATTEMPTS.tsv) ))
AID=$(printf "a%03d" $N)
LOG=$W/logs/${AID}_${PHASE}_${LABEL}.log
if [ -e "$LOG" ]; then echo "REFUSING: log exists $LOG" >&2; exit 90; fi
if [ "$KIND" = gpu ]; then
  G=$(cat $W/GPU_SECONDS); REM=$(( 6*3600 - G ))
  if [ "$REM" -le 60 ]; then echo "GPU BUDGET EXHAUSTED (used ${G}s)" >&2; exit 91; fi
  [ "$TO" -gt "$REM" ] && TO=$REM && echo "[attempt] timeout clamped to remaining GPU budget ${REM}s"
fi
S_ISO=$(date -Is); T0=$(date +%s)
timeout --kill-after=30s "${TO}s" "$@" > "$LOG" 2>&1
RC=$?
T1=$(date +%s); DUR=$((T1-T0))
if [ "$KIND" = gpu ]; then
  P=$(cat $W/GPU_SECONDS); echo $((P+DUR)) > $W/GPU_SECONDS
fi
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$AID" "$PHASE" "$LABEL" "$S_ISO" "$DUR" "$RC" \
  "$([ "$KIND" = gpu ] && echo yes || echo no)" "$(basename $LOG)" "timeout=${TO}s" >> $W/ATTEMPTS.tsv
echo "[attempt] id=$AID rc=$RC dur=${DUR}s kind=$KIND log=$(basename $LOG)"
exit $RC
