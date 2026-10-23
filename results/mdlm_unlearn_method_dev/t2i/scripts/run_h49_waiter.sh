#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
for i in $(seq 1 240); do
  if grep -q '\[h48\] done' outputs/log_h48.txt 2>/dev/null; then exec bash scripts/run_h49_parachute.sh 3; fi
  if grep -q '\[h47\] complete' outputs/log_h47.txt 2>/dev/null; then exec bash scripts/run_h49_parachute.sh 2; fi
  sleep 30
done
