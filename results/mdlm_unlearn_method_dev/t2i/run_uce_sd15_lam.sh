#!/bin/bash
# UCE's SD-1.5 arm fell to retain8 0.5333 (below K1) at the default lam=0.1 -- the regularisation
# strength toward W_old, tuned on SD-1.4. Higher lam preserves more of the original matrices, so this
# sweeps it to recover a usable SD-1.5 arm and give the closed-form class a second measured point.
# Reported as a hyperparameter-transfer fix, not as a verdict on UCE.
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
for cfg in "0.5 0" "1.0 1"; do
  set -- $cfg; LAM=$1; G=$2
  L="uce_sd15_church_lam${LAM}"
  mkdir ".labels/$L" 2>/dev/null || continue
  ( export CUDA_VISIBLE_DEVICES=$G
    [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_uce.py --concept church \
      --lam "$LAM" --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
    [ -f "adapters/$L/unet.pt" ] || { echo "$L BUILD FAILED" >>logs/uce_cm.log; exit 0; }
    [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
      --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
    R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
    T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
    echo "$L as-is: target=$T retain8=$R" >>logs/uce_cm.log
    awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$L retain8=$R below K1 -- not attacked" >>logs/uce_cm.log; exit 0; }
    for SD in 0 1 6; do
      K="H89L_${L}_sd${SD}_lr1e-5"
      mkdir ".labels/$K" 2>/dev/null || continue
      [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
        --unet "adapters/$L/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" >>logs/uce_cm_atk.log 2>&1
    done
    echo "$L grid done $(date -Is)" >>logs/uce_cm.log ) &
done
wait
echo "UCE SD15 LAM DONE $(date -Is)" >>logs/uce_cm.log
