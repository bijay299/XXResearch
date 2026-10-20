#!/bin/bash
# attempt.sh <stage> <phase> <label> <gpu|cpu> <n_gpus> <timeout_s> <cmd...>
# - Atomic unique attempt id (mkdir), so concurrent launches cannot collide.
# - Own log per attempt; refuses to overwrite.
# - GPU timeout clamped to remaining aggregate GPU budget; charges n_gpus * wall.
# - Measures CPU core-seconds via /usr/bin/time for every attempt.
# - Propagates the child's exit code (failed checks => nonzero).
W=/data/bijaypandey/llada_emdu
STAGE=$1; PHASE=$2; LABEL=$3; KIND=$4; NG=$5; TO=$6; shift 6
for i in $(seq 1 9999); do
  AID=$(printf "b%04d" $i)
  if mkdir "$W/.aid/$AID" 2>/dev/null; then break; fi
done
LOG=$W/logs/${AID}_${STAGE}_${PHASE}_${LABEL}.log
[ -e "$LOG" ] && { echo "REFUSING: log exists $LOG" >&2; exit 90; }
if [ "$KIND" = gpu ]; then
  G=$(cat $W/GPU_SECONDS); REM=$(( 6*3600 - G )); REMW=$(( REM / NG ))
  if [ "$REMW" -le 60 ]; then echo "GPU BUDGET EXHAUSTED (used ${G}s, ${NG} gpu(s))" >&2; exit 91; fi
  if [ "$TO" -gt "$REMW" ]; then TO=$REMW; echo "[attempt] timeout clamped to remaining-budget wall ${REMW}s (${NG} gpu)"; fi
fi
TF=$W/tmp/${AID}.time
S_ISO=$(date -Is); T0=$(date +%s)
/usr/bin/time -f "%U %S" -o "$TF" timeout --kill-after=30s "${TO}s" "$@" > "$LOG" 2>&1
RC=$?
T1=$(date +%s); DUR=$((T1-T0))
CPU=$(awk '{printf "%d", $1+$2}' "$TF" 2>/dev/null || echo 0)
if [ "$KIND" = gpu ]; then
  P=$(cat $W/GPU_SECONDS); echo $(( P + DUR*NG )) > $W/GPU_SECONDS
fi
PC=$(cat $W/CPU_SECONDS_NEW); echo $(( PC + CPU )) > $W/CPU_SECONDS_NEW
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$AID" "$STAGE" "$PHASE" "$LABEL" "$S_ISO" \
  "$DUR" "$RC" "${KIND}x${NG}" "$CPU" "$(basename $LOG)" "timeout=${TO}s charged=$(( DUR*NG ))s" >> $W/ATTEMPTS.tsv
echo "[attempt] id=$AID rc=$RC wall=${DUR}s gpu_charged=$([ "$KIND" = gpu ] && echo $((DUR*NG)) || echo 0)s cpu=${CPU}s log=$(basename $LOG)"
exit $RC
