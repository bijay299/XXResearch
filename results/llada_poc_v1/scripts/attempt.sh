#!/bin/bash
# attempt.sh <phase> <label> <gpu|cpu> <ngpu> <timeout_s> <cmd...>
# Fresh POC ledger. Charges failures and engineering runs. Enforces remaining-budget
# timeouts AND the 05:00 ET Oct 21 GPU stop gate. Atomic attempt ids. Own log each.
W=/data/bijaypandey/llada_poc
PHASE=$1; LABEL=$2; KIND=$3; NG=$4; TO=$5; shift 5
for i in $(seq 1 9999); do AID=$(printf "p%03d" $i); mkdir "$W/.aid/$AID" 2>/dev/null && break; done
LOG=$W/logs/${AID}_${PHASE}_${LABEL}.log
[ -e "$LOG" ] && { echo "REFUSING: log exists $LOG" >&2; exit 90; }
if [ "$KIND" = gpu ]; then
  G=$(cat $W/GPU_SECONDS); REM=$(( 6*3600 - G )); REMW=$(( REM / NG ))
  GATE=$(( $(date -d "2026-10-21 05:00:00 EDT" +%s) - $(date +%s) ))
  [ "$GATE" -le 60 ] && { echo "GPU STOP GATE PASSED (05:00 ET Oct 21)" >&2; exit 92; }
  [ "$REMW" -le 30 ] && { echo "GPU BUDGET EXHAUSTED (used ${G} device-s)" >&2; exit 91; }
  [ "$TO" -gt "$REMW" ] && { TO=$REMW; echo "[attempt] timeout clamped to remaining GPU budget wall ${REMW}s"; }
  [ "$TO" -gt "$GATE" ] && { TO=$GATE; echo "[attempt] timeout clamped to 05:00 ET stop gate ${GATE}s"; }
fi
TF=$W/tmp/${AID}.time; S=$(TZ=America/New_York date -Is); T0=$(date +%s)
/usr/bin/time -f "%U %S" -o "$TF" timeout --kill-after=30s "${TO}s" "$@" > "$LOG" 2>&1
RC=$?; T1=$(date +%s); DUR=$((T1-T0))
CPU=$(awk '{printf "%d", $1+$2}' "$TF" 2>/dev/null || echo 0)
CH=0; [ "$KIND" = gpu ] && CH=$(( DUR*NG ))
[ "$KIND" = gpu ] && echo $(( $(cat $W/GPU_SECONDS) + CH )) > $W/GPU_SECONDS
echo $(( $(cat $W/CPU_SECONDS) + CPU )) > $W/CPU_SECONDS
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$AID" "$PHASE" "$LABEL" "$S" "$DUR" "$RC" "$KIND" "$NG" "$CH" "$CPU" "$(basename $LOG)" "timeout=${TO}s" >> $W/ATTEMPTS.tsv
echo "[attempt] id=$AID rc=$RC wall=${DUR}s gpu_charged=${CH}s cpu=${CPU}s log=$(basename $LOG)"
exit $RC
