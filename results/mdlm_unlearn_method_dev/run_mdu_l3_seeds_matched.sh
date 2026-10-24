#!/bin/bash
# WHY: the n=5/n=8 seed extension's MDU arms (mdu_s3..mdu_s7) have NO RECORDED lambda_f. Nothing in
# outputs/log_mdu_s*.txt, logs/p24*_D_mdu_s*.log, ATTEMPTS.tsv or the arm JSONs records it, and
# train_lam.py's --lam-forget DEFAULT IS 8.0 while the headline baseline is lambda_f=3. Their
# DISCLOSE profile is also inconsistent with the lambda=3 seeds 0-2: 0.267-0.458 (mean 0.267 over
# eight) against 0.058-0.233 (mean 0.133 over three). So the n=8 comparison is NOT at matched
# disclosure, which is the condition the headline AFC-vs-MDU claim rests on, and RESULTS.md's
# retraction may be comparing AFC against a different lambda than it says.
#
# This rebuilds MDU seeds 3-7 at an EXPLICIT --lam-forget 3 under NEW labels (mdu_l3_sN), so:
#   - no old measurement is overwritten or relabelled; mdu_s3..s7 stay exactly as they are;
#   - the lambda is recorded in the label and on the command line this time.
# It does NOT assume the originals were wrong. If the rebuilds reproduce mdu_s3..s7, the extension
# was lambda=3 after all and only the provenance was missing. If they do not, the n=5/n=8
# comparison in RESULTS.md is confounded and has to be corrected.
set -u
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
cd /data/bijaypandey/mdlm_unlearn

# Respect the two-GPU cap: wait for the t2i phase's jobs to release the devices first.
while pgrep -f "attack_ti.py --concept church" >/dev/null || pgrep -f "erase_frp.py" >/dev/null; do
  sleep 120
done

build() {  # $1 seed  $2 gpu
  S=$1; L="mdu_l3_s${S}"
  export CUDA_VISIBLE_DEVICES=$2
  if [ ! -d "adapters/$L/ckpt_300" ]; then
    $PY scripts/train_lam.py --arm mdu --lam-forget 3 --lam-retain 1 \
      --seed "$S" --out "adapters/$L" >>"logs/${L}.log" 2>&1 || return 1
  fi
  [ -f "outputs/arm_${L}.json" ] || $PY scripts/eval_arm_partition.py \
    --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/${L}.log" 2>&1
}

( for S in 3 5 7; do build "$S" 2; done; echo "MDU_L3 A DONE $(date -Is)" >>logs/mdu_l3_a.log ) &
( for S in 4 6;   do build "$S" 3; done; echo "MDU_L3 B DONE $(date -Is)" >>logs/mdu_l3_b.log ) &
wait
