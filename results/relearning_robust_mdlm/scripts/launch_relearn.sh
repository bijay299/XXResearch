#!/bin/bash
# launch_relearn.sh <gpu 2|3> <tag> <start> [extra args...]
set -u
source /data/bijaypandey/anvil/env.sh
case "$1" in 2|3) ;; *) echo "REFUSING: GPUs 2 and 3 only" >&2; exit 91;; esac
export CUDA_VISIBLE_DEVICES=$1
G=$1; TAG=$2; ST=$3; shift 3
exec /data/bijaypandey/anvil/attempt.sh probe "relearn_${TAG}_gpu${G}" gpu 1 5400 \
  /data/bijaypandey/llada_smoke/venv_llada/bin/python \
  /data/bijaypandey/anvil/scripts/relearn_probe.py --start "$ST" --tag "$TAG" "$@"
