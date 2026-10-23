#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
for i in $(seq 1 240); do
  if grep -q '\[h63-church\] done' outputs/log_h63a.txt 2>/dev/null; then exec bash scripts/run_h64_frag.sh 2 0.3; fi
  if grep -q '\[h63-van_gogh\] done' outputs/log_h63b.txt 2>/dev/null; then exec bash scripts/run_h64_frag.sh 3 0.5; fi
  sleep 30
done
