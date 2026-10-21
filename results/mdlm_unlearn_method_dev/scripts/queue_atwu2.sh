#!/bin/bash
# ATWU -- the mandatory SOTA baseline. Nothing may be called state of the art until this runs.
# Published 8B/TOFU settings (arXiv 2606.06320 Table 6): alpha 0.50, gamma 3.00, beta 7.00,
# scorer-lr 0.050, model-lr 1.5e-5, lam_ent 1, lam_pop 15. Their budget rho is supplementary-
# only, so it is SWEPT and ATWU's BEST rho is what we report against.
#
# Also runs the selector head-to-head on one anchor, which is legitimate because ATWU states it
# "can be combined with diverse forget losses":
#   satga     + learned      ATWU as published
#   satga     + divergence   ATWU's loss, our free selector
#   anchor_kl + learned      ATWU's learned selector on the anchor loss
#
# Guard tests argv[0] against the venv python binary (see queue_h16c.sh header for the two
# earlier guard bugs this avoids).
set -u
cd /data/bijaypandey/mdlm_unlearn
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python

busy () { ps -eo args= | awk -v py="$PY" '$1 == py && (/train_atwu\.py/ || /train_h16\.py/)' | wc -l; }

run () {   # floss selector steps label [extra...]
  while [ "$(busy)" -ne 0 ]; do sleep 20; done
  fl=$1; se=$2; st=$3; lab=$4; shift 4
  [ -f "adapters/${lab}/report.json" ] && { echo "[queue_atwu2] skip ${lab}"; return 0; }
  ./scripts/launch_py.sh 3 D "$lab" 7200 train_atwu.py \
    --forget-loss "$fl" --selector "$se" --steps "$st" "$@" \
    --out /data/bijaypandey/mdlm_unlearn/adapters/"$lab" \
    > "outputs/log_${lab}.txt" 2>&1
  sleep 5
}

run satga learned 20 atwu_smoke --ckpt-at 20
if [ ! -f adapters/atwu_smoke/report.json ]; then
  echo "[queue_atwu2] SMOKE FAILED -- stopping before the batch"; exit 1
fi
echo "[queue_atwu2] smoke OK"

# rho sweep: ATWU's own budget, the one hyperparameter not recoverable from the paper text
run satga learned 300 atwu_rho10 --rho 0.10
run satga learned 300 atwu_rho25 --rho 0.25
run satga learned 300 atwu_rho50 --rho 0.50
# selector head-to-head, matched budgets
run satga     divergence 300 atwu_satga_div --topk-frac 0.25
run anchor_kl learned    300 atwu_anchor_lrn --rho 0.25
echo "[queue_atwu2] done"
