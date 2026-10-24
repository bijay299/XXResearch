#!/bin/bash
# The gated re-run of the named RTT control failed instantly: adapters/mdu_pub/ckpt_300 has no
# adapter_config.json. A survey shows only the 12 checkpoints built TODAY still exist -- every
# historical arm checkpoint (mdu_pub, h16_none_f1, ll_mdu_f3, seed_afc_s1/s2, afc_s3..s7,
# mdu_s3..s7) is gone from disk. So the control cannot be re-attacked on its original weights; its
# arm has to be rebuilt first.
#
# mdu_pub is the MDU lambda_f=8 arm. Both bridge controls today showed train_lam.py reproduces a
# published arm EXACTLY from explicit settings, so a rebuild is the right instrument -- but it is a
# REBUILD, recorded under a new label, and must not be relabelled as the original.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
export CUDA_VISIBLE_DEVICES=3
A=/data/bijaypandey/mdlm_unlearn/adapters

# 1. rebuild the control's arm at explicit lambda_f=8, and evaluate it so the rebuild can be checked
#    against mdu_pub's published row (TV->pristine 0.1908, retain 0.0911, DISC 0.0667, DEG 0.1500).
L=mdu_l8_s0_rebuild
[ -d "adapters/$L/ckpt_300" ] || $PY scripts/train_lam.py --arm mdu --lam-forget 8 --lam-retain 1 \
  --seed 0 --out "adapters/$L" >>"logs/$L.log" 2>&1
[ -f "outputs/arm_$L.json" ] || $PY scripts/eval_arm_partition.py \
  --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1

# 2. the GATED attack at the control's own recorded hyperparameters, read out of
#    rtt_mdu_l8_lr1e5.json: t_frac 0.2, probe-steps 0,10,25,50,100,200, lr 1e-5, rank 16, batch 2,
#    gen 64, n_eval 60, seed 0. train_rtt.py now computes attack_validity (threshold +0.20 on T).
[ -f "outputs/rtt_mdu_l8_lr1e5_gated.json" ] || $PY scripts/train_rtt.py --arm "adapters/$L/ckpt_300" \
  --t-frac 0.2 --probe-steps "0,10,25,50,100,200" --lr 1e-5 --rank 16 --batch 2 \
  --gen-length 64 --n-eval 60 --seed 0 --label mdu_l8_lr1e5_gated >>logs/mdu_l8_lr1e5_gated.log 2>&1

# 3. the matched AFC arm at the identical budget. Its original checkpoint is gone too, so use the
#    seed-0 bridge -- which reproduced h16_none_f1 exactly, so it is the right stand-in and is
#    labelled as the bridge build, not as h16_none_f1.
[ -f "outputs/rtt_afc_l1_lr1e5_gated.json" ] || $PY scripts/train_rtt.py \
  --arm "$A/afc_l1_s0_bridge/ckpt_300" \
  --t-frac 0.2 --probe-steps "0,10,25,50,100,200" --lr 1e-5 --rank 16 --batch 2 \
  --gen-length 64 --n-eval 60 --seed 0 --label afc_l1_lr1e5_gated >>logs/afc_l1_lr1e5_gated.log 2>&1
echo "RTT REBUILD CHAIN DONE $(date -Is)" >>logs/rtt_gated.log
