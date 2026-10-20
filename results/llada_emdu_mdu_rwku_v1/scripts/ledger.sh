#!/bin/bash
W=/data/bijaypandey/llada_emdu
S=$(cat $W/START_EPOCH); N=$(date +%s); E=$((N-S))
G=$(cat $W/GPU_SECONDS); GC=$(cat $W/GPU_SECONDS_CARRIED); C=$(cat $W/CPU_SECONDS_NEW)
ST=$(( $(du -sb /data/bijaypandey/llada_rwku 2>/dev/null|cut -f1) + $(du -sb $W 2>/dev/null|cut -f1) ))
python3 - "$E" "$G" "$GC" "$C" "$ST" <<'PY'
import sys
e,g,gc,c,st=(int(x) for x in sys.argv[1:6])
print(f"=== SHARED LEDGER (carried forward, clock NOT restarted) ===")
print(f"elapsed        : {e/3600:7.3f} h / 24 h      remaining {24-e/3600:7.3f} h")
print(f"GPU total      : {g:6d} s = {g/3600:.4f} h / 6 h   remaining {6*3600-g:6d} s = {(6*3600-g)/3600:.4f} h")
print(f"  of which carried from llada_rwku_baseline_v1 : {gc} s")
print(f"  charged this task                            : {g-gc} s")
print(f"CPU (new work) : {c:6d} core-s = {c/3600:.4f} core-h / 20 core-h   [MEASURED from here on]")
print(f"  historical CPU (previous study)              : UNMETERED - no retrospective charge invented")
print(f"storage        : {st/2**30:7.3f} GiB / 10 GiB")
PY
echo "attempts: $(( $(wc -l < $W/ATTEMPTS.tsv) - 1 ))"
