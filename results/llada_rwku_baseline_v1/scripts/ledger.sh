#!/bin/bash
W=/data/bijaypandey/llada_rwku
S=$(cat $W/START_EPOCH); N=$(date +%s); E=$((N-S))
G=$(cat $W/GPU_SECONDS 2>/dev/null||echo 0)
echo "=== LEDGER $(date -Is) ==="
python3 - "$E" "$G" "$(du -sb $W 2>/dev/null|cut -f1)" <<'PY'
import sys
e,g,st=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
print(f"elapsed        : {e/3600:7.3f} h / 24 h cap   ({24-e/3600:.2f} h remaining)")
print(f"gpu (measured) : {g/3600:7.3f} h /  6 h cap   ({6-g/3600:.2f} h remaining)  [{g} s]")
print(f"storage        : {st/2**30:7.3f} GiB / 10 GiB cap  (excludes reused checkpoint)")
PY
echo "attempts logged: $(( $(wc -l < $W/ATTEMPTS.tsv) - 1 ))"
