#!/bin/bash
# PREREG_H71: LoRA adversary against the two best-utility erased arms. Same T/H split, same
# FINDING20 gate, same utility probe as the full-fine-tuning attacks, so the two adversaries are
# directly comparable.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
( export CUDA_VISIBLE_DEVICES=0
  venv_t2i/bin/python scripts/attack_lora.py --concept church --text-encoder "$TE" \
    --steps 300 --lr 1e-4 --rank 4 --alpha 4 --scope cross --seed 0 \
    --label H71_lora_advunlearn_church_r4_cross >>logs/h71_lora_a.log 2>&1
  echo "LORA A DONE $(date -Is)" >>logs/h71_lora_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  venv_t2i/bin/python scripts/attack_lora.py --concept church \
    --unet adapters/frp_church_r3_l05/unet.pt --steps 300 --lr 1e-4 --rank 4 --alpha 4 \
    --scope cross --seed 0 --label H71_lora_frp_church_r4_cross >>logs/h71_lora_b.log 2>&1
  echo "LORA B DONE $(date -Is)" >>logs/h71_lora_b.log ) &
wait
