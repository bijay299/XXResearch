#!/bin/bash
# MDU half of the n=12 extension, placed on GPU 0 rather than GPU 3. Another user (jibekgupta) is
# now running on GPUs 2 and 3 -- sharing GPU 3 with my in-flight RTT attack -- and GPUs 0/1 are
# empty, so queueing more of my work onto their card would be the wrong call. run_n12_extension.sh's
# chained MDU half targets GPU 3 and guards on output existence, so whatever this finishes first it
# will skip. Two of my GPUs at a time, as always.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
export CUDA_VISIBLE_DEVICES=0
for S in 8 9 10 11; do
  L="mdu_l3_s${S}_n12"
  [ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_lam.py --arm mdu --lam-forget 3 \
    --lam-retain 1 --seed "$S" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "outputs/arm_$L.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1
done
echo "N12 MDU GPU0 DONE $(date -Is)" >>logs/n12_mdu.log
