#!/bin/bash
# AFC -- "anchor-first on the conditional-answer anchor": the factorial cross neither MDU nor
# EMDU evaluated.
#
#   MDU  (2605.18253) : anchor-LAST  KL + random-answer anchor + uniform weights
#   EMDU (2610.25447) : anchor-FIRST KL + FULL-ANSWER-MASKED anchor + uniform weights
#   AFC  (this)       : anchor-FIRST KL + random-answer anchor + uniform weights
#
# EMDU changed the KL direction AND the anchor's answer span together. Separating them matters:
# at lam_forget=1 AFC reaches DISCLOSE 0.208 / DEGENERATE 0.050 / retain ASCR 0.533 /
# TV->pristine 0.229, which dominates MDU lam=1 (0.317 / 0.075 / 0.524 / 0.407) and is the best
# TV in the study, while EMDU's FAM variant collapses to DEGENERATE 1.000. So the direction is
# the useful half of EMDU's change and FAM is the harmful half.
#
# This maps AFC's curve so the comparison against MDU is curve-vs-curve, not point-vs-point.
# train_h16.py --mode none IS AFC (anchor-first, uniform, random-answer anchor, no projection).
set -u
cd /data/bijaypandey/mdlm_unlearn
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python

busy () { ps -eo args= | awk -v py="$PY" '$1 == py && (/train_h16\.py/ || /train_atwu\.py/)' | wc -l; }

run () {   # mode rank lam_forget label [extra...]
  while [ "$(busy)" -ne 0 ]; do sleep 20; done
  m=$1; k=$2; lf=$3; lab=$4; shift 4
  [ -f "adapters/${lab}/report.json" ] && { echo "[queue_afc] skip ${lab}"; return 0; }
  ./scripts/launch_py.sh 3 D "$lab" 7200 train_h16.py \
    --mode "$m" --rank "$k" --lam-forget "$lf" --steps 300 "$@" \
    --out /data/bijaypandey/mdlm_unlearn/adapters/"$lab" \
    > "outputs/log_${lab}.txt" 2>&1
  sleep 5
}

# AFC curve: the operating points that matter are where retain survives.
run none 8 0.5 afc_f0p5
run none 8 2   afc_f2
run none 8 4   afc_f4
run none 8 8   afc_f8
# AFC + the validated position selector, to see whether localization helps THIS loss
run none 8 1   afc_loc_f1 --localize
run none 8 2   afc_loc_f2 --localize
echo "[queue_afc] done"
