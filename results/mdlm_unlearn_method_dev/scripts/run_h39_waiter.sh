#!/bin/bash
set -u
for i in $(seq 1 180); do
  grep -q '\[train-control-dmg\] done' /data/bijaypandey/t2i_unlearn/outputs/log_tcd.txt 2>/dev/null && break
  sleep 20
done
exec bash /data/bijaypandey/mdlm_unlearn/scripts/run_h39_trainability.sh 3
