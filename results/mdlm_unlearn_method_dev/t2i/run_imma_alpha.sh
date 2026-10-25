#!/bin/bash
# IMMA at my pre-declared alpha=1e-5 / I=1500 destroys the model: target 0.0000 AND retain8 0.0000
# on BOTH arms (K1 FAIL). The loss history shows this is NOT divergence -- L_upper stayed in
# 0.03-1.04 and L_inner in 0.07-0.55 throughout. So the objective is bounded while the 44.0M
# cross-attention weights drift far from pretraining under 1500 Adam ascent steps, and the model
# outputs garbage for everything, target and retain alike.
#
# alpha and I are UNSPECIFIED in the paper (PREREG_H82 S2 pre-declared them). The paper's own
# framing is that IMMA learns a "poor model INITIALIZATION" for adaptation -- degraded, not
# destroyed -- so a much smaller budget is the faithful reading. This sweeps it before anything is
# concluded about the method. Reported as a PORT hyperparameter failure, not as "IMMA destroys
# models".
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
build() {  # $1 alpha  $2 steps  $3 gpu
  L="imma_church_frp_a$1_s$2"
  mkdir ".labels/$L" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$3
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_imma.py --concept church \
    --teacher adapters/frp_church_r3_l05/unet.pt --alpha "$1" --steps "$2" \
    --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "adapters/$L/unet.pt" ] || { echo "$L BUILD FAILED" >>logs/imma_alpha.log; return 1; }
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R" >>logs/imma_alpha.log
}
( build 1e-7 1500 0; build 1e-6 300 0; echo "IMMA_A A DONE $(date -Is)" >>logs/imma_alpha.log ) &
( build 1e-6 1500 1; build 1e-5 300 1; echo "IMMA_A B DONE $(date -Is)" >>logs/imma_alpha.log ) &
wait
