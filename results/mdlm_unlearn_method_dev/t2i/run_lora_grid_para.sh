#!/bin/bash
# Second concept for the LoRA grid, launched BEFORE the finding is written.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
grid() {
  local pref=$1; shift
  for spec in "0 1e-3 cross" "1 1e-3 cross" "6 1e-3 cross"; do
    set -- $spec "$@"; local SD=$1 LR=$2 SC=$3; shift 3
    local L="${pref}_sd${SD}_lr${LR}_${SC}"
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_lora.py --concept parachute "$@" \
      --steps 300 --lr "$LR" --rank 16 --alpha 16 --scope "$SC" --seed "$SD" --label "$L" \
      >>logs/gridp_${pref}.log 2>&1
  done
  echo "GRIDP $pref DONE $(date -Is)" >>logs/gridp_${pref}.log
}
( export CUDA_VISIBLE_DEVICES=0; grid H71P_advu --text-encoder "$TE" ) &
( export CUDA_VISIBLE_DEVICES=1; grid H71P_frp --unet adapters/frp_parachute_r5_l05/unet.pt ) &
wait
