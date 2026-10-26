#!/bin/bash
# FINDING57 section 4 names its own missing control: a RANDOM-mask prune of the CLEAN model at
# matched sparsity. Post-attack, re-zeroing any 3% of FF weights is harmless -- but pre-attack FRP's
# selection plainly did work (church 0.1333 vs the base model's 1.000). Without this arm we cannot
# say how much of that erasure came from FRP's forget-critical RANKING versus from removing rho of
# feed-forward capacity at all. This closes that gap.
#
# NOT FRP: --random-mask replaces Algorithm 1's TopK selection only. Every other step, including
# the Wanda-style calibration, is identical, so the arms differ in selection and nothing else.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home

# never more than two of our GPUs at a time: wait for the TI push to release them
while pgrep -f "attack_ti.py --concept church" >/dev/null; do sleep 60; done

build_and_eval() {  # $1 concept  $2 rho  $3 rand-seed  $4 arm label  $5 log
  if [ ! -f "adapters/$4/unet.pt" ]; then
    echo "=== BUILD $4 concept=$1 rho=$2 rand-seed=$3 $(date -Is)" >>"$5"
    venv_t2i/bin/python scripts/erase_frp.py --concept "$1" --rho "$2" --beta 0.05 --lmbda 0.05 \
      --seed 42 --random-mask --rand-seed "$3" --label "$4" >>"$5" 2>&1 || return 1
  fi
  [ -f "outputs/gen_$4.json" ] || venv_t2i/bin/python scripts/gen_eval.py \
    --label "$4" --unet "adapters/$4/unet.pt" --target "$1" >>"$5" 2>&1
}

( export CUDA_VISIBLE_DEVICES=0
  for S in 0 1; do build_and_eval church 0.03 "$S" "F57rand_church_r3_rs${S}" logs/f57_a.log; done
  echo "F57 A DONE $(date -Is)" >>logs/f57_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  # matched sparsity on the second concept, so the control is not single-concept
  build_and_eval parachute 0.05 0 "F57rand_parachute_r5_rs0" logs/f57_b.log
  echo "F57 B DONE $(date -Is)" >>logs/f57_b.log ) &
wait
