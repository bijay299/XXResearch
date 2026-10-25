#!/bin/bash
# Closes the gap FINDING_four_point_frontier S3 declares explicitly:
#   "MDU lambda=2 was NOT run. It was queued and then CANCELLED at the user's explicit instruction
#    to refocus on the t2i line -- not because it was uninformative. So the frontier has a hole
#    between lam=1 (0.440/0.524) and lam=3 (0.322/0.406). Linear interpolation puts lam=2 near
#    TV 0.38 / retain 0.46, which AFC's (0.298/0.515) would still dominate -- but that is an
#    interpolation, not a measurement, and the lam=4 episode is a direct demonstration that a single
#    unmeasured point can look different from expectation."
#
# The t2i line is now complete (eleven defences reversed, all five competitors discharged), so this
# is the outstanding item. ~3 GPU-hours. If lam=2 lands where interpolation says, AFC's frontier
# dominance holds on five measured MDU points; if it lands near AFC, the dominance claim narrows.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
mkdir -p .labels logs
build() { local S=$1 L="mdu_l2_s${1}_fr"
  mkdir ".labels/$L" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$2
  [ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_lam.py --arm mdu --lam-forget 2 \
    --lam-retain 1 --seed "$S" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "outputs/arm_$L.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1
  echo "$L done $(date -Is)" >>logs/l2_gap.log; }
( for S in 0 1 2 3 4 5; do build "$S" 0; done; echo "L2 A DONE $(date -Is)" >>logs/l2_gap.log ) &
( for S in 6 7 8 9 10 11; do build "$S" 1; done; echo "L2 B DONE $(date -Is)" >>logs/l2_gap.log ) &
wait
