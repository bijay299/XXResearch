#!/bin/bash
# Runs the decisive cheap experiments in priority order as GPU capacity frees, never
# exceeding two concurrent (the project's hard cap; GPUs 0-1 belong to other users).
#
#   1. h10_oracle.py  -- the H10-M0 oracle. Forces every position whose argmax is a
#      fact-answer token to be committed LAST, with no training. That is an exact upper
#      bound on any trained H10, so it can kill the mechanism outright in ~7 GPU-min.
#      Includes a parity gate: with the intervention off it must reproduce the pinned
#      sampler bit-identically, or it aborts.
#   2. measure_target_modes.py -- Phase B, the target-mode decomposition that tests H6'
#      (the failure mode is set by the target's own mode). Forward passes only.
set -u
W=/data/bijaypandey/mdlm_unlearn

run_when_free () {   # $1 label   $2 script   $3 timeout_s
  local label="$1" script="$2" to="$3"
  for i in $(seq 1 540); do
    for G in 2 3; do
      U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G" 2>/dev/null || echo 99999)
      if [ "$U" -lt 2000 ]; then
        echo "[queue] GPU $G free (${U} MiB) -> $label"
        bash "$W/scripts/launch_py.sh" "$G" "$label" "$label" "$to" "$script"
        echo "[queue] $label finished rc=$?"
        return 0
      fi
    done
    sleep 40
  done
  echo "[queue] TIMED OUT waiting for capacity: $label"
  return 1
}

run_when_free h10oracle h10_oracle.py 3600
run_when_free phaseB    measure_target_modes.py 3600
echo "[queue] all cheap experiments attempted"
