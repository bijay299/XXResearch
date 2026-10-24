#!/bin/bash
# PREREG_H71 grid. Two valid LoRA attacks are NOT enough: max-over-valid is monotone in attacks
# run (FINDING35 S4), and PROTOCOL_FROZEN_v1 requires a symmetric six-point grid applied
# IDENTICALLY to every arm. Grid fixed here before the numbers are read:
#   seeds {0,1,6} at lr 1e-3 rank 16 cross
#   seeds {0,1}   at lr 1e-3 rank 16 all_attn
#   seed  {0}     at lr 3e-3 rank 16 all_attn
# Applied to BOTH the official AdvUnlearn arm and the FRP arm.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
grid() {  # $1=label-prefix  $2...=model args
  local pref=$1; shift
  for spec in "0 1e-3 cross" "1 1e-3 cross" "6 1e-3 cross" "0 1e-3 all_attn" "1 1e-3 all_attn" "0 3e-3 all_attn"; do
    set -- $spec "$@"; local SD=$1 LR=$2 SC=$3; shift 3
    local L="${pref}_sd${SD}_lr${LR}_${SC}"
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_lora.py --concept church "$@" \
      --steps 300 --lr "$LR" --rank 16 --alpha 16 --scope "$SC" --seed "$SD" --label "$L" \
      >>logs/grid_${pref}.log 2>&1
  done
  echo "GRID $pref DONE $(date -Is)" >>logs/grid_${pref}.log
}
( export CUDA_VISIBLE_DEVICES=0; grid H71G_advu --text-encoder "$TE" ) &
( export CUDA_VISIBLE_DEVICES=1; grid H71G_frp --unet adapters/frp_church_r3_l05/unet.pt ) &
wait
echo "GRIDS DONE $(date -Is)" >>logs/grid_chain.log
