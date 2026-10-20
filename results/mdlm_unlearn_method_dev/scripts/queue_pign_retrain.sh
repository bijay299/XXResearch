#!/bin/bash
# Waits for a free GPU among {2,3} without ever exceeding two concurrent, then builds
# P_ign^retrain: a reference trained on the 64 RETAIN facts only.
set -u
W=/data/bijaypandey/mdlm_unlearn
while true; do
  for G in 2 3; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)
    if [ "$U" -lt 2000 ]; then
      echo "[queue] GPU $G free (${U} MiB) -> launching P_ign^retrain"
      exec bash $W/scripts/launch_py.sh $G ignref retain_only 7200 memorize_v2.py \
        --seed 20261111 --train-split retain --steps 1200 \
        --out $W/adapters/ignref_retain_only \
        --report $W/outputs/ignref_retain_only_report.json
    fi
  done
  sleep 60
done
