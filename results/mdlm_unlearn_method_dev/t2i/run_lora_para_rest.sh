#!/bin/bash
# Church got 6 grid points, parachute only 3. PROTOCOL_FROZEN_v1 requires the SAME grid per arm,
# so complete parachute's all_attn legs.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
rest() {
  local pref=$1; shift
  for spec in "0 1e-3 all_attn" "1 1e-3 all_attn" "0 3e-3 all_attn"; do
    set -- $spec "$@"; local SD=$1 LR=$2 SC=$3; shift 3
    local L="${pref}_sd${SD}_lr${LR}_${SC}"
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_lora.py --concept parachute "$@" \
      --steps 300 --lr "$LR" --rank 16 --alpha 16 --scope "$SC" --seed "$SD" --label "$L" \
      >>logs/gridp2_${pref}.log 2>&1
  done
  echo "REST $pref DONE $(date -Is)" >>logs/gridp2_${pref}.log
}
( export CUDA_VISIBLE_DEVICES=0; rest H71P_advu --text-encoder "$TE" ) &
( export CUDA_VISIBLE_DEVICES=1; rest H71P_frp --unet adapters/frp_parachute_r5_l05/unet.pt ) &
wait
