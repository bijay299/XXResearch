#!/bin/bash
# Third arm family under the LoRA adversary: the damage+distil arm. Completes arm coverage
# (AdvUnlearn = text-encoder erasure, FRP = pruning, h68rf = noise+distillation).
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for SD in 0 1 6; do
    L=H71D_h68rf_sd${SD}_lr1e-3_cross
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_lora.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps 300 --lr 1e-3 --rank 16 --alpha 16 \
      --scope cross --seed $SD --label "$L" >>logs/h71d_a.log 2>&1
  done
  echo "D A DONE $(date -Is)" >>logs/h71d_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for SD in 0 1; do
    L=H71D_h68rf_sd${SD}_lr1e-3_all_attn
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_lora.py --concept church \
      --unet adapters/h68v2_rf_a0.3_church/unet.pt --steps 300 --lr 1e-3 --rank 16 --alpha 16 \
      --scope all_attn --seed $SD --label "$L" >>logs/h71d_b.log 2>&1
  done
  echo "D B DONE $(date -Is)" >>logs/h71d_b.log ) &
wait
