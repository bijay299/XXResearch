#!/bin/bash
# TI max-over-valid has climbed 0.400 -> 0.800 with budget, mirroring LoRA's 0.867 -> 1.000. lr 2e-2
# is the productive direction (4000 steps at 2e-2 gave 0.800; at 1e-2 only 0.533), so this pushes
# that axis harder across seeds rather than adding more of the weaker setting.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
A=imma_church_frp_a1e-6_s1500
ti() { K="H82TQ_${A}_s$1_lr$2_sd$3"; mkdir ".labels/$K" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$4
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept church \
    --unet "adapters/$A/unet.pt" --steps "$1" --lr "$2" --seed "$3" --label "$K" \
    >>logs/imma_ti_push2.log 2>&1; }
ti 6000 2e-2 6 1; ti 4000 3e-2 0 1; ti 6000 2e-2 1 1
echo "TIQ DONE $(date -Is)" >>logs/imma_ti_push2.log
