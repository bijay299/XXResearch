#!/bin/bash
# The usable lambda window for UCE on SD-1.5 is now BRACKETED BY MEASUREMENT, not assumed:
#   lam <= 1.0  -> over-erases: target 0.0000 but retain8 0.4167-0.5333  (below K1)
#   lam = 10000 -> W_new -> W_old: target 1.0000 retain8 1.0000          (NO erasure at all)
# A usable arm, if one exists, is strictly in between. Phase 1 locates it with build+eval only
# (no attacks: attacking a non-erased arm is vacuous -- see ERRATUM_uce_lambda_scale.md section 6).
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export BASE_MODEL=runwayml/stable-diffusion-v1-5 BASE_VARIANT=fp16
export BASE_REV=451f4fe16113bff5a5d2269ed5ad43b0592e9a14
mkdir -p .labels logs

run_arm () {
  LAM="$1"; G="$2"
  L="uce_sd15_church_lam${LAM}"
  mkdir ".labels/$L" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$G
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_uce.py --concept church \
    --lam "$LAM" --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  [ -f "adapters/$L/unet.pt" ] || { echo "$L BUILD FAILED -- not a kill-rule result" >>logs/uce_cm.log; return 0; }
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  [ -f "outputs/gen_$L.json" ] || { echo "$L NO EVAL OUTPUT -- no K1 verdict" >>logs/uce_cm.log; return 0; }
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])")
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])")
  echo "$L as-is: target=$T retain8=$R" >>logs/uce_cm.log
}

for cfg in "10 0" "100 1"; do set -- $cfg; run_arm "$1" "$2" & done; wait
for cfg in "30 0" "300 1"; do set -- $cfg; run_arm "$1" "$2" & done; wait
echo "UCE SD15 BRACKET PHASE1 DONE $(date -Is)" >>logs/uce_cm.log
