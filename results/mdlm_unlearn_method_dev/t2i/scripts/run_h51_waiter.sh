#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
for i in $(seq 1 240); do
  if grep -q '\[h50\] done' outputs/log_h50.txt 2>/dev/null; then exec bash scripts/run_h51_forget_fisher.sh 3; fi
  if grep -q '\[h49\] done' outputs/log_h49.txt 2>/dev/null; then exec bash scripts/run_h51_forget_fisher.sh 2; fi
  sleep 30
done
