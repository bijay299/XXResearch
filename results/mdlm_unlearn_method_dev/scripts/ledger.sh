#!/bin/bash
# Resource ledger. Reports the scheduler counters AND the ATTEMPTS.tsv-derived totals.
#
# WHY BOTH: GPU_SECONDS/CPU_SECONDS are only updated by scripts/scheduler.sh. Runs launched
# directly (all H18 work) append to ATTEMPTS.tsv but never touch those counters, so the counter
# figure silently understated usage by several device-hours while `attempts` kept rising. Showing
# both makes the gap visible instead of reporting a stale number as if it were current.
W=/data/bijaypandey/mdlm_unlearn
G=$(cat $W/GPU_SECONDS); C=$(cat $W/CPU_SECONDS)
ST=$(date -d "$(cat $W/PROJECT_START)" +%s); EL=$(( $(date +%s) - ST ))
N=$(( $(wc -l < $W/ATTEMPTS.tsv) - 1 ))
KB=$(du -sk $W | awk '{print $1}')
read AG AC < <(awk -F'\t' 'NR>1 {g+=$9; c+=$10} END{printf "%d %d", g, c}' $W/ATTEMPTS.tsv)
awk -v el=$EL -v g=$G -v c=$C -v n=$N -v kb=$KB -v ag=$AG -v ac=$AC 'BEGIN{
printf "elapsed %.2f h | gpu %.3f device-h | cpu %.3f core-h | attempts %d | storage %.3f GiB\n",
  el/3600, g/3600, c/3600, n, kb/1048576;
printf "  from ATTEMPTS.tsv: gpu %.3f device-h | cpu %.3f core-h", ag/3600, ac/3600;
if (ag > g) printf "   (counters understate by %.3f device-h: direct-launched runs)", (ag-g)/3600;
printf "\n"}'
