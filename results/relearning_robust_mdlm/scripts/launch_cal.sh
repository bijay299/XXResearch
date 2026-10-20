#!/bin/bash
set -u
source /data/bijaypandey/anvil/env.sh
case "$1" in 2|3) ;; *) echo "REFUSING: GPUs 2 and 3 only" >&2; exit 91;; esac
export CUDA_VISIBLE_DEVICES=$1
G=$1; LR=$2; LG=${3:-1.0}
exec /data/bijaypandey/anvil/attempt.sh cal "lr${LR}_lamG${LG}_gpu${G}" gpu 1 1800 \
  /data/bijaypandey/llada_smoke/venv_llada/bin/python \
  /data/bijaypandey/anvil/scripts/flatten.py --arm anvil --lam-G "$LG" --lr "$LR" \
  --steps 30 --ckpt-at 30 --probe-every 10 \
  --out /data/bijaypandey/anvil/adapters/cal_lr${LR}_lamG${LG}
