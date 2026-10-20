#!/bin/bash
# attempt.sh <phase> <label> <gpu|cpu> <ngpu> <timeout_s> <cmd...>
# Charges every attempt including failures. Atomic attempt ids, one log per attempt.
# No budget cap (lifted by the user); base-10 forced throughout -- a leading-zero
# literal would be read as OCTAL by shell arithmetic, the accounting bug documented
# twice in the prior studies.
W=/data/bijaypandey/mdlm_unlearn
PHASE=$1; LABEL=$2; KIND=$3; NG=$4; TO=$5; shift 5
for i in $(seq 1 9999); do AID=$(printf "p%03d" $i); mkdir "$W/.aid/$AID" 2>/dev/null && break; done
LOG=$W/logs/${AID}_${PHASE}_${LABEL}.log
[ -e "$LOG" ] && { echo "REFUSING: log exists $LOG" >&2; exit 90; }
TF=$W/tmp/${AID}.time; S=$(TZ=America/New_York date -Is); T0=$(date +%s)
/usr/bin/time -f "%U %S" -o "$TF" timeout --kill-after=30s "${TO}s" "$@" > "$LOG" 2>&1
RC=$?; T1=$(date +%s); DUR=$((T1-T0))
CPU=$(awk '{printf "%d", $1+$2}' "$TF" 2>/dev/null | tail -1 | sed 's/^0*\([0-9]\)/\1/' || echo 0)
CPU=${CPU:-0}
CH=0; [ "$KIND" = gpu ] && CH=$(( DUR*NG ))
[ "$KIND" = gpu ] && echo $(( 10#$(cat $W/GPU_SECONDS) + 10#${CH} )) > $W/GPU_SECONDS
echo $(( 10#$(cat $W/CPU_SECONDS) + 10#${CPU} )) > $W/CPU_SECONDS
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$AID" "$PHASE" "$LABEL" "$S" "$DUR" "$RC" "$KIND" "$NG" "$CH" "$CPU" "$(basename $LOG)" "timeout=${TO}s" >> $W/ATTEMPTS.tsv
echo "[attempt] id=$AID rc=$RC wall=${DUR}s gpu_charged=${CH}s cpu=${CPU}s log=$(basename $LOG)"
exit $RC
