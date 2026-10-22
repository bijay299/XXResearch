#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
for i in $(seq 1 120); do
  grep -q '\[esd-attacks\] done' outputs/log_esd_attacks.txt 2>/dev/null && break
  sleep 30
done
# alpha=0 is the discriminating control; 0.3 fills in the low end of the ladder
exec bash scripts/run_undo_sweep.sh 3 "0.0|retain_fisher" "0.3|retain_fisher"
