#!/bin/bash
W=/data/bijaypandey/llada_poc
G=$(cat $W/GPU_SECONDS); C=$(cat $W/CPU_SECONDS)
ST=$(du -sb $W 2>/dev/null|cut -f1)
python3 - "$G" "$C" "$ST" <<'PY'
import sys,datetime,zoneinfo
g,c,st=(int(x) for x in sys.argv[1:4])
ny=zoneinfo.ZoneInfo('America/New_York'); now=datetime.datetime.now(ny)
stop=datetime.datetime(2026,10,21,5,0,tzinfo=ny)
print(f"=== POC LEDGER (fresh task, separate from all closed studies) {now:%m-%d %H:%M %Z} ===")
print(f"GPU-device-h : {g/3600:7.4f} / 6.0000   remaining {(6*3600-g)/3600:7.4f} h  [{g} device-s]")
print(f"CPU core-h   : {c/3600:7.4f} / 8.0000   remaining {(8*3600-c)/3600:7.4f} h  [measured]")
print(f"storage      : {st/2**30:7.4f} / 10 GiB")
print(f"GPU stop gate: {(stop-now).total_seconds()/3600:7.2f} h remaining until 05:00 ET Oct 21")
PY
echo "attempts: $(( $(wc -l < $W/ATTEMPTS.tsv) - 1 ))"
