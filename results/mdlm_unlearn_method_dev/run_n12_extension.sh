#!/bin/bash
# The corrected n=8 comparison leaves ONE standing claim: AFC retains +0.1408 more capability at
# matched disclosure (0.5336 vs 0.3928), holding on 7 of 8 paired seeds. Every other claim this
# project made died when the seed count grew -- n=3 stability died at n=5, the n=5 comparative died
# when the baseline's lambda was checked, the n=3 TV advantage died at n=8. The retain delta is
# ~2.3 pooled sd, so it should survive, but "should survive" is exactly what was said before.
#
# So: extend BOTH columns to n=12 (seeds 8-11) under the explicit, recorded settings whose bridges
# and five exact seed reproductions established them as faithful today. Pre-committed reading:
#   retain delta holds at n=12 -> it is the contribution, and it is the thing to write up;
#   retain delta shrinks        -> this project has no surviving advantage over MDU and that is the
#                                  result, reported as such.
# TV is reported at n=12 too, but no TV claim is being made: at n=8 the gap was 0.0016.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
A=/data/bijaypandey/mdlm_unlearn/adapters

afc() { export CUDA_VISIBLE_DEVICES=$2; L="afc_l1_s$1_n12"
  [ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_h16.py --mode none --lam-forget 1 \
    --lam-retain 1 --seed "$1" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "outputs/arm_$L.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1; }

mdu() { export CUDA_VISIBLE_DEVICES=$2; L="mdu_l3_s$1_n12"
  [ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_lam.py --arm mdu --lam-forget 3 \
    --lam-retain 1 --seed "$1" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "outputs/arm_$L.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1; }

# GPU 2 is free now: take the AFC half immediately.
( for S in 8 9 10 11; do afc "$S" 2; done; echo "N12 AFC DONE $(date -Is)" >>logs/n12_afc.log ) &
# GPU 3 is held by the gated RTT chain; wait for it rather than exceeding two GPUs.
( while pgrep -f "train_rtt.py" >/dev/null; do sleep 120; done
  for S in 8 9 10 11; do mdu "$S" 3; done; echo "N12 MDU DONE $(date -Is)" >>logs/n12_mdu.log ) &
wait
