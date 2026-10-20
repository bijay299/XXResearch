#!/bin/bash
# launch_probe.sh <gpu 2|3> <start>
set -u
source /data/bijaypandey/anvil/env.sh
case "$1" in 2|3) ;; *) echo "REFUSING: GPUs 2 and 3 only (0 and 1 belong to another user)" >&2; exit 91;; esac
export CUDA_VISIBLE_DEVICES=$1
exec /data/bijaypandey/anvil/attempt.sh probe "relearn_$2_gpu$1" gpu 1 5400 \
  /data/bijaypandey/llada_smoke/venv_llada/bin/python \
  /data/bijaypandey/anvil/scripts/relearn_probe.py --start "$2"
