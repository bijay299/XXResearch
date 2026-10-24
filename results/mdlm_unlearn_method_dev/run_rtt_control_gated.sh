#!/bin/bash
# Closes the standing hole in every status report: the NAMED RTT positive control, mdu_l8_lr1e5,
# carries NO attack_validity block. It was produced before the gate existed, so T -- recovery on the
# facts the attacker is TRAINED ON -- was never measured for it. With T unmeasured its 8.99% recovery
# cannot be called a failure-to-recover rather than a broken attack, and AFC's 0% at the same budget
# is therefore an UNVALIDATED attack, not a robustness result.
#
# scripts/train_rtt.py now computes attack_validity (line 229, threshold +0.20 absolute on T). So
# re-run the control at its OWN recorded hyperparameters -- t_frac 0.2, probe-steps 0,10,25,50,100,200,
# lr 1e-5, rank 16, batch 2, gen 64, n_eval 60, seed 0, read straight out of rtt_mdu_l8_lr1e5.json --
# and the AFC lambda=1 arm at the identical settings, under NEW labels. The originals stay as they are.
#
#   control gate PASSES -> the budget is valid, and AFC's number at it becomes interpretable
#   control gate FAILS  -> the budget is too weak to attack anything, and NO robustness claim can be
#                          made from this budget for either arm. That is the honest outcome and is
#                          to be reported as such, not quietly dropped in favour of the budgets that
#                          do pass.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
export CUDA_VISIBLE_DEVICES=3
A=/data/bijaypandey/mdlm_unlearn/adapters
run() {  # $1 arm dir  $2 label
  [ -f "outputs/rtt_$2.json" ] || $PY scripts/train_rtt.py --arm "$1" \
    --t-frac 0.2 --probe-steps "0,10,25,50,100,200" --lr 1e-5 --rank 16 --batch 2 \
    --gen-length 64 --n-eval 60 --seed 0 --label "$2" >>"logs/$2.log" 2>&1
}
run "$A/mdu_pub/ckpt_300"   mdu_l8_lr1e5_gated      # the control, same arm as the original
run "$A/h16_none_f1/ckpt_300" afc_l1_lr1e5_gated    # AFC lambda=1 at the identical budget
echo "RTT GATED DONE $(date -Is)" >>logs/rtt_gated.log
