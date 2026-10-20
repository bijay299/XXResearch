#!/bin/bash
S=/data/bijaypandey/spex
python3 - "$(cat $S/GPU_SECONDS)" "$(cat $S/CPU_SECONDS)" "$(du -sb $S|cut -f1)" "$(cat $S/START_EPOCH)" <<'PY'
import sys,datetime,zoneinfo
g,c,st,t0=(int(x) for x in sys.argv[1:5])
ny=zoneinfo.ZoneInfo('America/New_York'); now=datetime.datetime.now(ny)
el=now.timestamp()-t0
print(f"=== SPEX SPRINT LEDGER  {now:%H:%M %Z}  elapsed {el/3600:.2f} h of 4.00 ===")
print(f"  GPU-device-h {g/3600:7.3f}   CPU core-h {c/3600:7.3f}   storage {st/2**30:6.3f} GiB")
PY
echo "  attempts: $(( $(wc -l < $S/ATTEMPTS.tsv) - 1 ))"
