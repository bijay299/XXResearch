#!/bin/bash
# BRIDGE CONTROL. The lambda=3 rebuilds do NOT reproduce mdu_s3..s7, so those extension seeds were
# not lambda=3. But that comparison only becomes interpretable if MY rebuild protocol matches the
# protocol behind the lambda=3 seeds 0-2 arms (ll_mdu_f3, sel_mdu_l3_s1/s2) -- whose lr, rank and
# steps are ALSO unrecorded. train_lam.py defaults are lr 1e-4, rank 16, steps 300.
#
# So: rebuild seed 0 at explicit lambda=3 with the same defaults and compare against ll_mdu_f3
# (TV 0.2825, retain 0.4042, DISC 0.2333, DEG 0.0500).
#   - reproduces  -> the rebuild protocol matches, and mdu_l3_s3..s7 are directly comparable to
#                    seeds 0-2, so the whole n=8 MDU column can be rebuilt on one footing;
#   - does not    -> the seeds 0-2 arms differ in some other unrecorded hyperparameter too, and NO
#                    cross-seed MDU comparison in this project is currently sound.
# Either way it is recorded under a new label; nothing existing is touched.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
export CUDA_VISIBLE_DEVICES=3
L=mdu_l3_s0_bridge
[ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_lam.py --arm mdu --lam-forget 3 --lam-retain 1 \
  --seed 0 --out "adapters/$L" >>"logs/${L}.log" 2>&1
[ -f "outputs/arm_${L}.json" ] || $PY scripts/eval_arm_partition.py \
  --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/${L}.log" 2>&1
echo "BRIDGE DONE $(date -Is)" >>logs/${L}.log
