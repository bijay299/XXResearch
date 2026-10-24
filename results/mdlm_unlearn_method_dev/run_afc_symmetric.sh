#!/bin/bash
# SYMMETRIC CONTROL. I just corrected the MDU baseline for unrecorded hyperparameters. The same
# scrutiny has to fall on MY OWN method's arms: the AFC lambda=1 seeds (h16_none_f1, seed_afc_s1/s2,
# afc_s3..s7) have their mode/rank/localize/lam_forget/lr equally UNRECORDED in the arm JSONs, the
# logs and ATTEMPTS.tsv. train_h16.py defaults are mode=proj rank=8 lam_forget=1.0 lr=1e-4
# lora_rank=16 steps=300 -- but the selected AFC arm is "none_f1" by its own label, i.e. --mode none.
#
# Stage 1 (BRIDGE, seed 0): rebuild at explicit --mode none --lam-forget 1 and check it reproduces
#   h16_none_f1 (TV 0.2288, retain 0.5328, DISC 0.2083, DEG 0.0500), exactly as the MDU bridge
#   reproduced ll_mdu_f3. If it does NOT reproduce, the AFC column is as confounded as MDU's was and
#   the corrected n=8 comparison is not yet sound either -- in which case say so, do not publish it
#   as settled.
# Stage 2 (seeds 3-7): rebuild under the same explicit settings, so BOTH columns are on one
#   recorded footing rather than only the baseline.
# New labels throughout; nothing existing is touched.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python

build() {  # $1 seed  $2 gpu  $3 label
  export CUDA_VISIBLE_DEVICES=$2
  [ -d "adapters/$3/ckpt_300" ] || $PY scripts/train_h16.py --mode none --lam-forget 1 \
    --lam-retain 1 --seed "$1" --out "adapters/$3" >>"logs/$3.log" 2>&1
  [ -f "outputs/arm_$3.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$3/ckpt_300" --label "$3" >>"logs/$3.log" 2>&1
}

# Stage 1: the bridge alone, so a failure is seen before five more builds are spent on it.
build 0 2 afc_l1_s0_bridge
$PY - <<'PY' >>logs/afc_bridge_verdict.log 2>&1
import json
try:
    d=json.load(open("outputs/arm_afc_l1_s0_bridge.json")); fp=d["forget_profile"]
    got=(round(d["TV_forget_to"]["pristine_base"],4), round(d["retain_ASCR_mean"],4),
         round(fp["DISCLOSE"],4), round(fp["DEGENERATE"],4))
    ref=(0.2288, 0.5328, 0.2083, 0.0500)
    print("bridge", got, "ref", ref, "REPRODUCES" if got==ref else "DOES NOT REPRODUCE")
except Exception as e:
    print("bridge missing:", e)
PY
# Stage 2: the extension seeds, on two GPUs.
( for S in 3 5 7; do build "$S" 2 "afc_l1_s${S}_rec"; done; echo "AFC A DONE $(date -Is)" >>logs/afc_rec_a.log ) &
( for S in 4 6;   do build "$S" 3 "afc_l1_s${S}_rec"; done; echo "AFC B DONE $(date -Is)" >>logs/afc_rec_b.log ) &
wait
