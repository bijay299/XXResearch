#!/bin/bash
# launch_flatten.sh <gpu 2|3> <arm> [extra args...]
set -u
source /data/bijaypandey/anvil/env.sh
case "$1" in 2|3) ;; *) echo "REFUSING: GPUs 2 and 3 only" >&2; exit 91;; esac
export CUDA_VISIBLE_DEVICES=$1
G=$1; ARM=$2; shift 2
exec /data/bijaypandey/anvil/attempt.sh flat "${ARM}_gpu${G}" gpu 1 5400 \
  /data/bijaypandey/llada_smoke/venv_llada/bin/python \
  /data/bijaypandey/anvil/scripts/flatten.py --arm "$ARM" \
  --out /data/bijaypandey/anvil/adapters/"${ARM}" "$@"
