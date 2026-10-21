#!/bin/bash
# Waits for a free GPU among {2,3} and runs the Phase B target-mode decomposition.
# Cheap (forward passes only); queued so no GPU-second sits idle between training jobs.
set -u
W=/data/bijaypandey/mdlm_unlearn
for i in $(seq 1 480); do
  for G in 2 3; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null || echo 99999)
    if [ "$U" -lt 2000 ]; then
      echo "[queue] GPU $G free (${U} MiB) -> Phase B target-mode decomposition"
      exec bash $W/scripts/launch_py.sh $G phaseB target_modes 3600 measure_target_modes.py
    fi
  done
  sleep 45
done
echo "[queue] timed out after 6h without free capacity"
