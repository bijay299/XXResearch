#!/bin/bash
set -u
source /data/bijaypandey/mdlm_unlearn/env.sh
case "$1" in 2|3) ;; *) echo "REFUSING: GPUs 2-3 only" >&2; exit 91;; esac
export CUDA_VISIBLE_DEVICES=$1
G=$1; PH=$2; LB=$3; TO=$4; SC=$5; shift 5
exec /data/bijaypandey/mdlm_unlearn/attempt.sh "$PH" "${LB}_gpu${G}" gpu 1 "$TO" \
  /data/bijaypandey/llada_smoke/venv_llada/bin/python \
  "/data/bijaypandey/mdlm_unlearn/scripts/$SC" "$@"
