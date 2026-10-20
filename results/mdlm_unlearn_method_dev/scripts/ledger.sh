#!/bin/bash
W=/data/bijaypandey/mdlm_unlearn
G=$(cat $W/GPU_SECONDS); C=$(cat $W/CPU_SECONDS)
ST=$(date -d "$(cat $W/PROJECT_START)" +%s); EL=$(( $(date +%s) - ST ))
N=$(( $(wc -l < $W/ATTEMPTS.tsv) - 1 ))
KB=$(du -sk $W | awk '{print $1}')
awk -v el=$EL -v g=$G -v c=$C -v n=$N -v kb=$KB 'BEGIN{
printf "elapsed %.2f h | gpu %.3f device-h | cpu %.3f core-h | attempts %d | storage %.3f GiB\n",
el/3600, g/3600, c/3600, n, kb/1048576}'
